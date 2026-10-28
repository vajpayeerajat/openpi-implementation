"""Evaluation of a G1 Dex3 checkpoint with DINO tokens in the action expert (`dino_fusion="expert_stream"`).

Reports, on a held-out LeRobot split:
* offline action MSE (overall and per action dim), against the recorded action chunk;
* counterfactual sensitivity: mean |delta action| (same noise) when
    (a) the DINO tokens are zeroed (token gate forced to 0; the action tokens still attend to them),
    (b) the DINO tokens come from a random other frame (SigLIP / language / state unchanged),
    (c) the DINO attention gate is forced to 0 (exactly the model without DINO);
* added inference latency per control step (one `sample_actions` call, batch 1), measured on GPU with warmup, for
  the base model, expert_stream on all cameras and expert_stream on the wrist cameras only (random weights; latency
  does not depend on them).

Example:
    HF_LEROBOT_HOME=/mnt/drive2/rajat_ws/datasets uv run scripts/eval_dino_expert.py \
        --checkpoint-dir checkpoints/pi05_g1_dex3_dino_expert/my_run/30000 --split val
    # latency only:
    uv run scripts/eval_dino_expert.py --no-accuracy
"""

import dataclasses
import json
import logging
import pathlib
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from lerobot.common.datasets import lerobot_dataset
import numpy as np
import tqdm
import tyro

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.policies import g1_dex3_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    checkpoint_dir: str | None = None
    config_name: str = "pi05_g1_dex3_dino_expert"
    # Dataset root that holds the train/val/test splits, relative to HF_LEROBOT_HOME.
    dataset: str = "splitted_lerobot_and_school"
    split: str = "val"
    # Evaluate every `stride`-th frame of every episode.
    stride: int = 20
    # Only evaluate the first N episodes (None = all).
    max_episodes: int | None = None
    num_steps: int = 10
    seed: int = 0
    accuracy: bool = True
    latency: bool = True
    # Configs timed for the latency report (base first: the others are reported relative to it).
    latency_configs: tuple[str, ...] = (
        "pi05_g1_dex3",
        "pi05_g1_dex3_dino_expert",
        "pi05_g1_dex3_dino_expert_wrist",
    )
    latency_iters: int = 20
    output: str | None = None


def _to_hwc_uint8(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).clip(0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0)) if img.shape[0] == 3 else img


def _with_param(state: nnx.State, name: str, value: float) -> nnx.State:
    return nnx_utils.state_map(state, nnx_utils.PathRegex(name), lambda v: v.replace(jnp.full_like(v.value, value)))


def accuracy(args: Args) -> dict:
    config = _config.get_config(args.config_name)
    assert config.model.dino_fusion == "expert_stream", "counterfactuals need dino_fusion='expert_stream'"
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)
    horizon = config.model.action_horizon
    graphdef, state = nnx.split(policy._model)  # noqa: SLF001
    variants = {
        "full": state,
        "tokens_zeroed": _with_param(state, "dino_gate", 0.0),
        "attn_gate_zero": _with_param(state, "dino_attn_gate", 0.0),
    }

    @jax.jit
    def sample(state, rng, observation, dino_observation, noise):
        model = nnx.merge(graphdef, state)
        # DINO tokens from `dino_observation`, everything else from `observation`.
        object.__setattr__(model, "embed_dino", lambda _: _pi0.Pi0.embed_dino(model, dino_observation))
        return model.sample_actions(rng, observation, num_steps=args.num_steps, noise=noise)

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
    np_rng = np.random.default_rng(args.seed)
    other_indices = np_rng.permutation(indices)

    def observation(sample) -> tuple[_model.Observation, np.ndarray]:
        obs = {
            "observation/ego_view": _to_hwc_uint8(sample["observation.images.ego_view"]),
            "observation/ego_left": _to_hwc_uint8(sample["observation.images.ego_left"]),
            "observation/ego_right": _to_hwc_uint8(sample["observation.images.ego_right"]),
            "observation/state": np.asarray(sample["observation.state"], dtype=np.float32),
            "prompt": sample["task"],
        }
        inputs = policy._input_transform(obs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return _model.Observation.from_dict(inputs), np.asarray(inputs["state"])

    def to_actions(state, actions) -> np.ndarray:
        outputs = policy._output_transform({"state": state[0], "actions": np.asarray(actions[0])})  # noqa: SLF001
        return outputs["actions"][:horizon]

    sq_err, valid, deltas = [], [], {"tokens_zeroed": [], "other_frame": [], "attn_gate_zero": []}
    rng = jax.random.key(args.seed)
    for idx, other_idx in tqdm.tqdm(list(zip(indices, other_indices, strict=True)), desc=f"Evaluating {repo_id}"):
        sample_ = dataset[idx]
        obs, state_in = observation(sample_)
        other_obs, _ = observation(dataset[int(other_idx)])
        rng, sample_rng, noise_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_rng, (1, horizon, config.model.action_dim))

        full = to_actions(state_in, sample(variants["full"], sample_rng, obs, obs, noise))
        target = np.asarray(sample_["action"])
        sq_err.append(np.square(full - target))
        valid.append(~np.asarray(sample_["action_is_pad"]))
        for name in ("tokens_zeroed", "attn_gate_zero"):
            pred = to_actions(state_in, sample(variants[name], sample_rng, obs, obs, noise))
            deltas[name].append(np.abs(pred - full).mean())
        pred = to_actions(state_in, sample(variants["full"], sample_rng, obs, other_obs, noise))
        deltas["other_frame"].append(np.abs(pred - full).mean())

    sq_err, valid = np.stack(sq_err), np.stack(valid)
    per_dim = sq_err[valid].mean(0)
    params = state.to_pure_dict()
    return {
        "checkpoint_dir": args.checkpoint_dir,
        "split": args.split,
        "num_frames": len(indices),
        "mse": float(per_dim.mean()),
        "per_dim_mse": dict(zip(g1_dex3_policy.G1_JOINTS, per_dim.round(6).tolist(), strict=True)),
        "mean_abs_delta_action": {k: float(np.mean(v)) for k, v in deltas.items()},
        "tanh_dino_gate": {
            "mean": float(np.tanh(params["dino_gate"]).mean()),
            "absmax": float(np.abs(np.tanh(params["dino_gate"])).max()),
        },
        "tanh_dino_attn_gate": float(np.tanh(params["dino_attn_gate"])),
    }


def latency(args: Args) -> dict:
    results = {}
    for name in args.latency_configs:
        model_config = _config.get_config(name).model
        model = model_config.create(jax.random.key(0))
        # inference runs in bfloat16 (see `BaseModelConfig.load`).
        graphdef, state = nnx.split(model)
        state = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, state)
        model = nnx.merge(graphdef, state)
        sample = nnx_utils.module_jit(model.sample_actions)
        obs = model_config.fake_obs(1)
        rng = jax.random.key(0)
        for _ in range(3):
            jax.block_until_ready(sample(rng, obs, num_steps=args.num_steps))
        times = []
        for _ in range(args.latency_iters):
            start = time.perf_counter()
            jax.block_until_ready(sample(rng, obs, num_steps=args.num_steps))
            times.append(time.perf_counter() - start)
        results[name] = {"median_ms": 1000 * float(np.median(times)), "p90_ms": 1000 * float(np.percentile(times, 90))}
        logging.info(f"{name}: {results[name]}")
        del model, sample, state
    base = results[args.latency_configs[0]]["median_ms"]
    for value in results.values():
        value["added_ms"] = value["median_ms"] - base
    return {"device": str(jax.devices()[0]), "num_steps": args.num_steps, "per_config": results}


def main(args: Args) -> None:
    results = {}
    if args.accuracy:
        assert args.checkpoint_dir is not None, "--checkpoint-dir is needed for the accuracy report"
        results["accuracy"] = accuracy(args)
    if args.latency:
        results["latency"] = latency(args)
    print(json.dumps(results, indent=2))
    if args.output or args.checkpoint_dir:
        output = pathlib.Path(args.output or pathlib.Path(args.checkpoint_dir) / f"dino_eval_{args.split}.json")
        output.write_text(json.dumps(results, indent=2))
        logging.info(f"Wrote {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
