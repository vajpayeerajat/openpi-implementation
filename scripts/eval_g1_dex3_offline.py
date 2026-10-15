"""Offline evaluation of a G1 Dex3 checkpoint on a held-out LeRobot split (val / test).

openpi has no validation loop, so this replays held-out frames through the trained policy and compares the
predicted action chunk against the recorded one. Observations are built in the robot-client key format
(`observation/ego_view`, ..., `observation/state`, `prompt`), so this also exercises the serving path.

Example:
    HF_LEROBOT_HOME=/mnt/drive2/rajat_ws/datasets uv run scripts/eval_g1_dex3_offline.py \
        --checkpoint-dir checkpoints/pi05_g1_dex3/my_run/30000 --split val
"""

import dataclasses
import json
import logging
import pathlib

from lerobot.common.datasets import lerobot_dataset
import numpy as np
import tqdm
import tyro

from openpi.policies import g1_dex3_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# Joint groups inside the 28-dim action, see `g1_dex3_policy.G1_JOINTS`.
GROUPS = {
    "left_arm": slice(0, 7),
    "left_hand": slice(7, 14),
    "right_arm": slice(14, 21),
    "right_hand": slice(21, 28),
}


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    config_name: str = "pi05_g1_dex3"
    # Dataset root that holds the train/val/test splits, relative to HF_LEROBOT_HOME.
    dataset: str = "splitted_lerobot_and_school"
    split: str = "val"
    # Evaluate every `stride`-th frame of every episode.
    stride: int = 20
    # Only evaluate the first N episodes (None = all).
    max_episodes: int | None = None
    # Horizons (in steps) at which MAE is also reported, e.g. 10 = first 0.5 s at 20 fps.
    report_steps: tuple[int, ...] = (1, 10, 25, 50)
    output: str | None = None


def _to_hwc_uint8(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).clip(0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0)) if img.shape[0] == 3 else img


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)
    horizon = config.model.action_horizon

    repo_id = f"{args.dataset}/{args.split}"
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id, delta_timestamps={"action": [t / meta.fps for t in range(horizon)]}
    )

    episodes = range(meta.total_episodes)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    indices = [
        i
        for ep in episodes
        for i in range(dataset.episode_data_index["from"][ep].item(), dataset.episode_data_index["to"][ep].item())[
            :: args.stride
        ]
    ]

    abs_err = []  # [N, horizon, 28]
    valid = []  # [N, horizon]
    for idx in tqdm.tqdm(indices, desc=f"Evaluating {repo_id}"):
        sample = dataset[idx]
        obs = {
            "observation/ego_view": _to_hwc_uint8(sample["observation.images.ego_view"]),
            "observation/ego_left": _to_hwc_uint8(sample["observation.images.ego_left"]),
            "observation/ego_right": _to_hwc_uint8(sample["observation.images.ego_right"]),
            "observation/state": np.asarray(sample["observation.state"], dtype=np.float32),
            "prompt": sample["task"],
        }
        pred = policy.infer(obs)["actions"]
        target = np.asarray(sample["action"])
        abs_err.append(np.abs(pred[:horizon] - target))
        valid.append(~np.asarray(sample["action_is_pad"]))

    abs_err = np.stack(abs_err)
    valid = np.stack(valid)

    def mae(steps: int, dims: slice) -> float:
        err = abs_err[:, :steps, dims].mean(-1)
        mask = valid[:, :steps]
        return float(err[mask].mean())

    results = {
        "checkpoint_dir": args.checkpoint_dir,
        "split": args.split,
        "num_frames": len(indices),
        "mae_rad": {
            f"first_{s}_steps": {
                "all": mae(s, slice(None)),
                **{name: mae(s, dims) for name, dims in GROUPS.items()},
            }
            for s in args.report_steps
            if s <= horizon
        },
        "per_joint_mae_rad": dict(
            zip(g1_dex3_policy.G1_JOINTS, (abs_err[valid].mean(0)).round(5).tolist(), strict=True)
        ),
    }
    print(json.dumps(results["mae_rad"], indent=2))

    output = pathlib.Path(args.output or pathlib.Path(args.checkpoint_dir) / f"offline_eval_{args.split}.json")
    output.write_text(json.dumps(results, indent=2))
    logging.info(f"Wrote {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
