"""Prints the norms of the DINO fusion params of a checkpoint (both "prefix_add" and "expert_stream").

A "prefix_add" `dino_proj.kernel` that is still close to zero means the DINO features barely reach the model.

Example:
    uv run scripts/dino_param_norms.py checkpoints/pi05_g1_dex3_dinov3/my_run/30000/params
"""

import sys

import flax.traverse_util
import numpy as np

from openpi.models import model as _model


def main(params_path: str) -> None:
    params = _model.restore_params(params_path, restore_type=np.ndarray)
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


if __name__ == "__main__":
    main(sys.argv[1])
