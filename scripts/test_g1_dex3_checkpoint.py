"""Pass/fail sanity test for a trained G1 Dex3 checkpoint, run before putting it on the robot.

Checks:
  1. The checkpoint has params and norm stats.
  2. The policy loads.
  3. A robot-client style observation returns a finite [action_horizon, 28] action chunk.
  4. A 43-dim whole-body state gives the same actions as the 28-dim upper-body state (same noise).
  5. On real frames from a held-out split (default: test), the predicted actions are
     a. finite,
     b. in the right action space: the first arm action stays close to the current arm state. Outputs are
        absolute joint targets; a delta/absolute mix-up shows up here as a jump of several tenths of a rad,
     c. inside the joint range seen in the training actions (plus a margin).
  6. Inference latency (reported, not checked).

Prints PASS/FAIL per check and exits with code 1 if any check fails. For accuracy numbers (MAE per joint group and
horizon), use scripts/eval_g1_dex3_offline.py.

Example:
    HF_LEROBOT_HOME=/mnt/drive2/rajat_ws/datasets uv run scripts/test_g1_dex3_checkpoint.py \
        --checkpoint-dir checkpoints/pi05_g1_dex3/run1/49999
"""

import dataclasses
import glob
import pathlib
import sys
import time

from lerobot.common.datasets import lerobot_dataset
import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi.policies import g1_dex3_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# Arm dims of the 28-dim action, see `g1_dex3_policy.G1_JOINTS`.
ARM_DIMS = np.r_[0:7, 14:21]


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    config_name: str = "pi05_g1_dex3"
    # config_name: str = 'single_pickplace'
    # Dataset root holding the train/val/test splits, relative to HF_LEROBOT_HOME.
    dataset: str = "splitted_lerobot_and_school"
    split: str = "test"
    # Number of real frames to test, spread evenly over the split.
    num_frames: int = 20
    # Max allowed median |first predicted arm action - current arm state| (rad). Recorded data is ~0.03-0.08.
    max_arm_jump: float = 0.2
    # Allowed overshoot (rad) beyond the min/max of the training actions.
    envelope_margin: float = 0.1
    # Max fraction of predicted action values outside the training envelope.
    max_outside_fraction: float = 0.01


class Report:
    def __init__(self):
        self.failed: list[str] = []

    def check(self, name: str, *, ok: bool, detail: str = "") -> bool:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        if not ok:
            self.failed.append(name)
        return ok


def _to_hwc_uint8(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).clip(0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0)) if img.shape[0] == 3 else img


def _training_action_envelope(root: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.full(g1_dex3_policy.G1_DIM, np.inf), np.full(g1_dex3_policy.G1_DIM, -np.inf)
    for path in glob.glob(str(root / "data" / "*" / "*.parquet")):
        actions = np.stack(pq.read_table(path, columns=["action"])["action"].to_numpy(zero_copy_only=False))
        lo, hi = np.minimum(lo, actions.min(0)), np.maximum(hi, actions.max(0))
    return lo, hi


def main(args: Args) -> int:
    report = Report()
    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    horizon = config.model.action_horizon
    expected_shape = (horizon, g1_dex3_policy.G1_DIM)

    # 1. Checkpoint contents.
    norm_stats_path = checkpoint_dir / "assets" / data_config.asset_id / "norm_stats.json"
    ok = report.check(
        "checkpoint files",
        ok=(checkpoint_dir / "params").is_dir() and norm_stats_path.is_file(),
        detail=f"params/ and {norm_stats_path.relative_to(checkpoint_dir)}",
    )
    if not ok:
        return 1

    # 2. Load.
    start = time.perf_counter()
    try:
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    except Exception as e:
        report.check("load policy", ok=False, detail=repr(e))
        return 1
    report.check("load policy", ok=True, detail=f"{time.perf_counter() - start:.1f}s")

    # 3. Robot-client style example.
    example = g1_dex3_policy.make_g1_dex3_example()
    actions = policy.infer(example)["actions"]
    report.check(
        "client example output",
        ok=actions.shape == expected_shape and bool(np.isfinite(actions).all()),
        detail=f"shape {actions.shape}, expected {expected_shape}",
    )

    # 4. 43-dim whole-body state == 28-dim upper-body state.
    noise = np.random.default_rng(0).standard_normal((horizon, config.model.action_dim)).astype(np.float32)
    whole_body = dict(example, **{"observation/state": np.zeros(43, dtype=np.float32)})
    whole_body["observation/state"][15:43] = example["observation/state"]
    diff = np.abs(policy.infer(example, noise=noise)["actions"] - policy.infer(whole_body, noise=noise)["actions"])
    report.check("43-dim state == 28-dim state", ok=float(diff.max()) < 1e-4, detail=f"max diff {diff.max():.2e}")

    # 5. Real held-out frames.
    repo_id = f"{args.dataset}/{args.split}"
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id, delta_timestamps={"action": [t / meta.fps for t in range(horizon)]}
    )
    lo, hi = _training_action_envelope(lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id).root)

    preds, states, targets, latencies = [], [], [], []
    for idx in np.linspace(0, len(dataset) - 1, args.num_frames).astype(int):
        sample = dataset[int(idx)]
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        obs = {
            "observation/ego_view": _to_hwc_uint8(sample["observation.images.ego_view"]),
            "observation/ego_left": _to_hwc_uint8(sample["observation.images.ego_left"]),
            "observation/ego_right": _to_hwc_uint8(sample["observation.images.ego_right"]),
            "observation/state": state,
            "prompt": sample["task"],
        }
        start = time.perf_counter()
        preds.append(policy.infer(obs)["actions"])
        latencies.append(time.perf_counter() - start)
        states.append(state)
        targets.append(np.asarray(sample["action"]))
    preds, states, targets = np.stack(preds), np.stack(states), np.stack(targets)

    report.check(f"{args.split} frames finite", ok=bool(np.isfinite(preds).all()), detail=f"{len(preds)} frames")

    arm_jump = float(np.median(np.abs(preds[:, 0, ARM_DIMS] - states[:, ARM_DIMS])))
    data_jump = float(np.median(np.abs(targets[:, 0, ARM_DIMS] - states[:, ARM_DIMS])))
    report.check(
        "action space (first arm action near current state)",
        ok=arm_jump <= args.max_arm_jump,
        detail=f"median jump {arm_jump:.3f} rad (recorded data: {data_jump:.3f}, limit {args.max_arm_jump})",
    )

    outside = (preds < lo - args.envelope_margin) | (preds > hi + args.envelope_margin)
    worst = [g1_dex3_policy.G1_JOINTS[j] for j in np.argsort(-outside.mean((0, 1)))[:3] if outside[..., j].any()]
    report.check(
        "inside training action envelope",
        ok=outside.mean() <= args.max_outside_fraction,
        detail=f"{outside.mean():.2%} outside (limit {args.max_outside_fraction:.0%})"
        + (f", worst: {worst}" if worst else ""),
    )

    # 6. Latency (first call includes JIT compilation, so skip it).
    if len(latencies) > 1:
        print(f"[INFO] inference latency: median {np.median(latencies[1:]) * 1000:.0f} ms")
    mae = np.abs(preds[:, :10] - targets[:, :10]).mean()
    print(f"[INFO] MAE vs recorded actions, first 10 steps: {mae:.4f} rad (see eval_g1_dex3_offline.py for details)")

    print(f"\n{'ALL CHECKS PASSED' if not report.failed else f'FAILED: {report.failed}'}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
