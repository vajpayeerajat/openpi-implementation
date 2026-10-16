"""Print the architecture of a training config's model on CPU, without loading or allocating any weights.

The model is built with `jax.eval_shape`, so only parameter shapes exist (a 3B-parameter pi05 costs no memory).

Example:
    JAX_PLATFORMS=cpu uv run scripts/print_architecture.py --config-name pi05_g1_dex3 --depth 4
"""

import collections
import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import siglip as _siglip
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_g1_dex3"
    # Depth of the module tree to print (0 = no tree).
    depth: int = 3
    # Print every parameter with its shape.
    full: bool = False
    # Print the data transform pipeline (needs the norm stats in assets/).
    transforms: bool = True
    # Print input / intermediate / output tensor shapes, traced through the real model code.
    shapes: bool = True
    # Batch size used for the shape trace.
    batch_size: int = 1


def _component(path: tuple[str, ...]) -> str:
    """Which part of pi05 a parameter belongs to."""
    if path[:2] == ("PaliGemma", "img"):
        return "Vision encoder (SigLIP So400m/14) + projection"
    if path[:2] == ("PaliGemma", "llm"):
        if any(p.endswith("_1") for p in path[2:]):
            return "Action expert (Gemma, weights with suffix _1)"
        if "embedder" in path:
            return "VLM token embedder (Gemma vocab)"
        return "VLM language model (PaliGemma Gemma)"
    return f"Action head: {path[0]}"


def _shape(x) -> str:
    return f"{tuple(x.shape)} {x.dtype}"


def print_shapes(mc, abstract, batch_size: int) -> None:
    """Traces the model on shapes only (jax.eval_shape): nothing is computed or allocated."""
    obs, actions = mc.inputs_spec(batch_size=batch_size)
    graphdef, state = nnx.split(abstract)

    def trace(fn, *inputs):
        return jax.eval_shape(lambda st, *xs: fn(nnx.merge(graphdef, st), *xs), state, *inputs)

    print(f"\n=== Input / output shapes (batch {batch_size}) ===")
    print("  Model inputs (after the data and model transforms):")
    rows = [(f"image[{n}]", img, "values in [-1, 1]") for n, img in obs.images.items()]
    rows += [(f"image_mask[{n}]", m, "") for n, m in obs.image_masks.items()]
    rows.append(("state", obs.state, "normalized, padded to action_dim"))
    if obs.tokenized_prompt is not None:
        rows.append(("tokenized_prompt", obs.tokenized_prompt, "prompt (+ discretized state for pi05)"))
        rows.append(("tokenized_prompt_mask", obs.tokenized_prompt_mask, ""))
    rows.append(("actions (training target)", actions, "normalized, arm deltas, padded"))
    for label, x, note in rows:
        print(f"    {label:35s} {_shape(x):30s} {note}")

    first = next(iter(obs.images))
    img_tokens = trace(lambda m, o: m.PaliGemma.img(o.images[first], train=False)[0], obs)
    prefix, prefix_mask, _ = trace(lambda m, o: m.embed_prefix(o), obs)
    t = jax.ShapeDtypeStruct((batch_size,), jnp.float32)
    suffix = trace(lambda m, o, a, tt: m.embed_suffix(o, a, tt), obs, actions, t)
    loss = trace(lambda m, o, a: m.compute_loss(jax.random.key(0), o, a, train=False), obs, actions)
    out = trace(lambda m, o: m.sample_actions(jax.random.key(0), o, num_steps=1), obs)

    print("  Intermediate:")
    print(f"    one camera -> SigLIP + projection   {_shape(img_tokens)}")
    print(f"    prefix tokens (images + text)       {_shape(prefix)}   mask {tuple(prefix_mask.shape)}")
    print(f"    suffix tokens (noisy actions)       {_shape(suffix[0])}   action-expert width")
    if len(suffix) > 3 and suffix[3] is not None:
        print(f"    time conditioning (adaRMS)          {_shape(suffix[3])}")
    print(f"    training loss per sample and step   {_shape(loss)}")
    print("  Model output:")
    print(f"    sample_actions                      {_shape(out)}   normalized; see robot output below")

    print("  Layer sizes:")
    for label, variant in (("VLM language model", mc.paligemma_variant), ("Action expert", mc.action_expert_variant)):
        g = _gemma.get_config(variant)
        print(f"    {label} ({variant}): {g.depth} layers, width {g.width}, mlp {g.mlp_dim}, "
              f"{g.num_heads} heads x {g.head_dim} (kv heads {g.num_kv_heads})")  # fmt: skip
    v = _siglip.decode_variant("So400m/14")
    print(f"    SigLIP So400m/14: {v['depth']} layers, width {v['width']}, mlp {v['mlp_dim']}, {v['num_heads']} heads, "
          f"patch 14 -> {img_tokens.shape[1]} tokens, projected to {img_tokens.shape[2]}")  # fmt: skip


def _fmt(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    mc = config.model

    print(f"=== Config: {config.name} ===")
    for field in ("model_type", "paligemma_variant", "action_expert_variant", "action_dim", "action_horizon",
                  "max_token_len", "discrete_state_input", "dtype"):  # fmt: skip
        if hasattr(mc, field):
            print(f"  {field:24s} {getattr(mc, field)}")
    print(f"  {'image resolution':24s} {_model.IMAGE_RESOLUTION} (cameras: {', '.join(_model.IMAGE_KEYS)})")
    print(f"  {'batch_size':24s} {config.batch_size}")

    # Shapes only: no weights are created.
    abstract = nnx.eval_shape(lambda: mc.create(jax.random.key(0)))
    state = nnx.state(abstract)
    params = traverse_util.flatten_dict(state.to_pure_dict())
    trainable = set(traverse_util.flatten_dict(nnx.state(abstract, config.trainable_filter).to_pure_dict()))

    def size(v) -> int:
        return int(np.prod(v.shape))

    def nbytes(v) -> int:
        return size(v) * np.dtype(v.dtype).itemsize

    tokens_per_image = (_model.IMAGE_RESOLUTION[0] // 14) ** 2
    n_cams = len(_model.IMAGE_KEYS)
    prefix = n_cams * tokens_per_image + mc.max_token_len
    print("\n=== Data flow (per sample) ===")
    print(f"  {n_cams} cameras x {_model.IMAGE_RESOLUTION[0]}x{_model.IMAGE_RESOLUTION[1]} RGB"
          f" -> SigLIP -> {n_cams} x {tokens_per_image} image tokens")  # fmt: skip
    if getattr(mc, "discrete_state_input", False):
        print(f"  prompt + discretized state as text  -> up to {mc.max_token_len} text tokens")
    else:
        print(f"  prompt -> up to {mc.max_token_len} text tokens; state -> continuous token")
    print(f"  prefix = {prefix} tokens (bidirectional) -> VLM language model -> KV cache")
    print(f"  noisy action chunk [{mc.action_horizon} x {mc.action_dim}] + flow-matching time"
          f" -> action expert (attends to prefix) -> velocity [{mc.action_horizon} x {mc.action_dim}]")  # fmt: skip

    groups = collections.defaultdict(lambda: [0, 0, 0])  # params, trainable params, bytes
    for path, v in params.items():
        g = groups[_component(path)]
        g[0] += size(v)
        g[1] += size(v) if path in trainable else 0
        g[2] += nbytes(v)
    total = sum(g[0] for g in groups.values())
    total_train = sum(g[1] for g in groups.values())
    print("\n=== Parameters by component ===")
    print(f"  {'component':52s} {'params':>9s} {'trainable':>10s} {'memory':>9s}")
    for name, (n, t, b) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:52s} {_fmt(n):>9s} {_fmt(t):>10s} {b / 2**30:8.2f}G")
    print(f"  {'TOTAL':52s} {_fmt(total):>9s} {_fmt(total_train):>10s} "
          f"{sum(g[2] for g in groups.values()) / 2**30:8.2f}G")  # fmt: skip
    print(f"  Frozen: {_fmt(total - total_train)} (freeze_filter of the config)")

    # Rough GPU memory for the training state, before activations (all float32 here).
    p_gb = sum(nbytes(v) for v in params.values()) / 2**30
    t_gb = sum(nbytes(v) for k, v in params.items() if k in trainable) / 2**30
    ema_gb = p_gb if config.ema_decay is not None else 0.0
    print(f"  Training state: params {p_gb:.1f}G + EMA {ema_gb:.1f}G + AdamW moments {2 * t_gb:.1f}G "
          f"+ gradients {t_gb:.1f}G = {p_gb + ema_gb + 3 * t_gb:.1f}G before activations")  # fmt: skip

    if args.shapes:
        print_shapes(mc, abstract, args.batch_size)

    if args.depth > 0:
        print(f"\n=== Module tree (depth {args.depth}) ===")
        tree = collections.defaultdict(int)
        for path, v in params.items():
            for d in range(1, min(args.depth, len(path)) + 1):
                tree[path[:d]] += size(v)
        for path in sorted(tree):
            leaf = params.get(path)
            shape = f"  {tuple(leaf.shape)} {leaf.dtype}" if leaf is not None else ""
            print(f"  {'  ' * (len(path) - 1)}{path[-1]}  [{_fmt(tree[path])}]{shape}")

    if args.full:
        print("\n=== All parameters ===")
        for path, v in sorted(params.items()):
            mark = "" if path in trainable else "  (frozen)"
            print(f"  {'/'.join(path):110s} {tuple(v.shape)!s:24s} {v.dtype}{mark}")

    if args.transforms:
        data_config = config.data.create(config.assets_dirs, mc)
        print("\n=== Data pipeline (training) ===")
        for stage in ("repack_transforms", "train_transforms", "data_transforms", "model_transforms"):
            group = getattr(data_config, stage, None)
            if group is None:
                continue
            names = [type(t).__name__ for t in group.inputs]
            if stage == "data_transforms":
                names.append("Normalize (quantile)" if data_config.use_quantile_norm else "Normalize (z-score)")
            print(f"  {stage:18s} {' -> '.join(names) or '-'}")
        print(f"  {'outputs (inference)':18s} "
              f"{' -> '.join(type(t).__name__ for t in [*data_config.data_transforms.outputs]) or '-'}")  # fmt: skip
        # What the robot receives: model output -> unnormalize -> output transforms.
        robot = {
            "state": np.zeros(mc.action_dim, np.float32),
            "actions": np.zeros((mc.action_horizon, mc.action_dim), np.float32),
        }
        for transform in data_config.data_transforms.outputs:
            robot = transform(robot)
        print(f"  {'robot output':18s} actions {tuple(np.asarray(robot['actions']).shape)} (absolute joint targets)")


if __name__ == "__main__":
    main(tyro.cli(Args))
