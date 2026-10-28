"""Checks how much the DINO fusion of a checkpoint has learned (both "prefix_add" and "expert_stream").

Prints the norms of the DINO fusion params. With `--config-name`, also runs one real data batch and prints, per
camera, the RMS of the DINO term relative to the SigLIP image tokens:
* "prefix_add": ||dino_proj(LN(dino))|| / ||siglip||, the term added to the SigLIP tokens (exactly 0 at init);
* "expert_stream": ||gated DINO tokens|| / ||siglip|| (also 0 at init, since the token gate starts at 0).
A ratio that is still ~0 after training means the DINO features barely reach the model.

Examples:
    uv run scripts/dino_param_norms.py checkpoints/pi05_g1_dex3_dinov3/my_run/30000
    HF_LEROBOT_HOME=... uv run scripts/dino_param_norms.py checkpoints/pi05_g1_dex3_dinov3/my_run/30000 \
        --config-name pi05_g1_dex3_dinov3
"""

import dataclasses
import pathlib

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args:
    # Checkpoint step dir (containing `params/`), or the `params` dir itself.
    checkpoint_dir: str
    # Train config of the checkpoint. If set, also measures the DINO / SigLIP ratio on one real batch.
    config_name: str | None = None
    batch_size: int = 8


def _rms(x) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(x, jnp.float32)))))


def print_param_norms(params: dict) -> None:
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    for path, raw in sorted(flat.items()):
        if not path.startswith("dino_"):
            continue
        value = np.asarray(raw, np.float32)
        print(f"{path:30s} shape={value.shape!s:14s} fro={np.linalg.norm(value):.6g} absmax={np.abs(value).max():.6g}")
        if path in ("dino_gate", "dino_attn_gate"):
            print(
                f"{'  tanh(' + path + ')':30s} mean={np.tanh(value).mean():.6g} absmax={np.abs(np.tanh(value)).max():.6g}"
            )


def print_ratios(config: _config.TrainConfig, params: dict, batch_size: int) -> None:
    config = dataclasses.replace(config, batch_size=batch_size, num_workers=0)
    fusion = config.model.dino_fusion
    assert fusion in ("prefix_add", "expert_stream"), f"{config.name} has no DINO fusion"
    model = config.model.load(params)
    observation, _ = next(iter(_data_loader.create_data_loader(config, num_batches=1)))
    observation = _model.preprocess_observation(None, observation, train=False)

    if fusion == "expert_stream":
        gated, _, _, _ = model.embed_dino(observation)
        num_per_camera = gated.shape[1] // len(model.dino_cameras)

    print(f"\nDINO / SigLIP RMS ratio ({fusion}, {batch_size} frames):")
    for name in observation.images:
        siglip, _ = model.PaliGemma.img(observation.images[name], train=False)
        if fusion == "prefix_add":
            dino = model.dino_proj(model.dino_norm(jax.lax.stop_gradient(model.dino(observation.images[name]))))
        elif name in model.dino_cameras:
            j = model.dino_cameras.index(name)
            dino = gated[:, j * num_per_camera : (j + 1) * num_per_camera]
        else:
            print(f"  {name:20s} (no DINO tokens)")
            continue
        print(
            f"  {name:20s} ||dino||={_rms(dino):.6g}  ||siglip||={_rms(siglip):.6g}  ratio={_rms(dino) / max(_rms(siglip), 1e-12):.6g}"
        )


def main(args: Args) -> None:
    params_path = pathlib.Path(args.checkpoint_dir)
    if params_path.name != "params":
        params_path = params_path / "params"
    params = _model.restore_params(params_path, restore_type=np.ndarray)
    print_param_norms(params)
    if args.config_name is not None:
        params = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), params)
        print_ratios(_config.get_config(args.config_name), params, args.batch_size)


if __name__ == "__main__":
    main(tyro.cli(Args))
