"""Fast normalization statistics for local LeRobot v2.1 datasets, read from parquet only.

`compute_norm_stats.py` loads samples through `LeRobotDataset`, which decodes every camera video for every frame
even though only state and actions are needed (>10 h for the 557k-frame G1 dataset). This script reads the
parquet files directly, builds the same action chunks LeRobot does (`action_horizon` steps, clamped at the
episode end), and pushes them through the config's own repack + data transforms with placeholder images, so the
resulting stats match `compute_norm_stats.py`.

One deliberate difference: dims whose q01/q99 range is degenerate are widened to min/max (see
`_widen_degenerate_quantiles`), otherwise pi05's quantile normalization blows rare values up to ~1e6.

Example:
    HF_LEROBOT_HOME=/mnt/drive2/rajat_ws/datasets uv run scripts/compute_norm_stats_g1_dex3.py --config-name pi05_g1_dex3
"""

import json

from lerobot.common.datasets import lerobot_dataset
import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def _widen_degenerate_quantiles(stats: normalize.RunningStats, max_abs_normalized: float) -> normalize.NormStats:
    """Falls back to min/max for dims whose q01/q99 range is too narrow for the data.

    A joint that is still in >98% of frames (e.g. the left hand, active in 0.5% of G1 frames) gets q01 == q99, and
    quantile normalization `(x - q01) / (q99 - q01 + 1e-6)` then maps its rare active values to ~1e6.
    """
    norm_stats = stats.get_statistics()
    q01, q99 = norm_stats.q01.copy(), norm_stats.q99.copy()
    lo, hi = stats._min, stats._max  # noqa: SLF001
    scale = q99 - q01 + 1e-6
    extreme = np.maximum(np.abs((lo - q01) / scale * 2 - 1), np.abs((hi - q01) / scale * 2 - 1))
    widen = extreme > max_abs_normalized
    if widen.any():
        print(
            f"Using min/max instead of q01/q99 for dims {np.flatnonzero(widen).tolist()} (max |x| {extreme.max():.3g})"
        )
        q01[widen], q99[widen] = lo[widen], hi[widen]
    return normalize.NormStats(mean=norm_stats.mean, std=norm_stats.std, q01=q01, q99=q99)


def main(config_name: str, batch_size: int = 1024, max_abs_normalized: float = 10.0):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    horizon = config.model.action_horizon
    (action_key,) = data_config.action_sequence_keys

    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    transforms = [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
    placeholder_image = np.zeros((8, 8, 3), dtype=np.uint8)
    image_keys = dict.fromkeys(meta.video_keys + meta.image_keys, placeholder_image)

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    episodes = [json.loads(line) for line in (meta.root / "meta" / "episodes.jsonl").read_text().splitlines()]
    for ep in tqdm.tqdm(episodes, desc="Computing stats"):
        table = pq.read_table(meta.root / meta.get_data_file_path(ep["episode_index"]))
        state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
        action = np.stack(table[action_key].to_numpy(zero_copy_only=False)).astype(np.float32)
        # LeRobot clamps delta-timestamp queries to the episode, i.e. it repeats the last action.
        chunk_idx = np.minimum(np.arange(len(action))[:, None] + np.arange(horizon)[None], len(action) - 1)

        out = {"state": [], "actions": []}
        for t in range(len(action)):
            sample = {
                "observation.state": state[t].copy(),
                action_key: action[chunk_idx[t]],
                "prompt": "",
                **image_keys,
            }
            for transform in transforms:
                sample = transform(sample)
            out["state"].append(sample["state"])
            out["actions"].append(sample["actions"])
            if len(out["state"]) == batch_size or t == len(action) - 1:
                for key, s in stats.items():
                    s.update(np.stack(out[key]))
                out = {"state": [], "actions": []}

    norm_stats = {key: _widen_degenerate_quantiles(s, max_abs_normalized) for key, s in stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
