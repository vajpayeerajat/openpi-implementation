"""Attention + Grad-CAM heat maps for a pi05 G1 Dex3 checkpoint on an inference video (no ground truth needed).

Two maps per frame, both computed from what the model itself does on that frame:

  1. Attention   action-expert -> prefix attention. While the model samples its action chunk (same flow-matching
                 loop as serving), the softmax weights from the 50 action tokens to every prefix token are captured in
                 every layer and denoising step, then averaged over heads, queries, layers (--layers) and steps.
                 Answers: "where does the action expert read from?"
  2. Grad-CAM    exactly like rajat_gradcam_analysis.py, but the target is the model's OWN sampled chunk instead of a
                 demonstrated one (as classification Grad-CAM uses the predicted class): score = -flow-matching loss
                 of the sampled chunk over fixed noise/time draws, gradients w.r.t. the prefix tokens. Answers:
                 "which inputs, if changed, would change this decision?"

Besides the per-camera maps, both methods give the share of attention / attribution per input group
(each camera, task words, state digits, template), so you can see e.g. whether it stops looking at the object.

Inputs (pick one):
  --records DIR   policy records written by `serve_policy.py --record` (step_*.npy). Best: exact model inputs,
                  all 3 cameras and the real robot state.
  --video EGO.mp4 the head-camera video (the robot's ego_view stream, not an external camera), optionally
                  --left_video / --right_video. Cameras that are not given are masked out (the model never saw that in
                  training, so treat their absence as a distribution shift). The state is not in a video: pass
                  --states (npy/csv, one row of 28 or 43 values per video frame) or the mid-range state is used, which
                  puts a fake proprioception into the prompt -> the maps then describe a slightly different situation.

Output (--out_dir): heatmaps.mp4, frames/*.png, per_frame.csv, timeline.png.

Usage:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run rajat_attention_video.py \
      --config_name pi05_g1_dex3_dinov3 \
      --checkpoint checkpoints/pi05-g1-dex3-dinov3-b16-96gb/checkpoints/31000 \
      --video trials/P1-001_head.mp4 --prompt "Pick up the pink octopus and place it inside the orange basket."
  # robot client recording (inference_recordings/inference_NN): prompt + state at every policy call, head camera
  uv run rajat_attention_video.py --config_name pi05_g1_pickplace \
      --checkpoint /mnt/nas/rajat_ws/checkpoints/praveen_school_demo_ckpts/20000 \
      --recording /mnt/nas/rajat_ws/inference_recordings/inference_11
  # from policy records (prompt + state + 3 cameras taken from the records)
  uv run rajat_attention_video.py --config_name pi05_g1_dex3_statenoise \
      --checkpoint <ckpt>/26000 --records policy_records

Only the prefix-fusion models (pi05_g1_dex3, _statenoise, _dinov3) are supported; the dino_expert configs of the
dino_add_in_action_expert branch change the action-expert input and are not handled here.
"""

import argparse
import contextlib
import os
from pathlib import Path
import re
import time as _time
import warnings

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.linen as nn
import numpy as np
import pandas as pd

import rajat_gradcam_analysis as G  # noqa: E402  (shared backend, transforms, plotting helpers)

import matplotlib.pyplot as plt  # noqa: E402  (G sets the Agg backend)

RAW_CAMS = ("observation/ego_view", "observation/ego_left", "observation/ego_right")  # policy-level keys
GROUPS = G.INPUT_GROUPS  # ego_view, left_wrist, right_wrist, prompt:task, prompt:state, prompt:template


# ============================================================================
# Attention capture: a drop-in copy of gemma.Attention that also returns the attention probs (mean over heads and
# queries) through the kv-cache output, so they come out of the layer scan stacked as [layers, B, S].
# ============================================================================

_CAPTURE = [False]


def _install_capturing_attention():
    import einops
    import jax
    import jax.numpy as jnp

    import openpi.models.gemma as gm
    from openpi.models import lora

    base = gm.Attention

    class CapturingAttention(base):
        @nn.compact
        def __call__(self, xs, positions, attn_mask, kv_cache):
            dtype = next(x.dtype for x in xs if x is not None)
            qkvs = []
            for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
                if x is None:
                    continue
                if config.num_kv_heads == config.num_heads:
                    qkv_einsum = lora.Einsum(
                        shape=(3, config.num_heads, config.width, config.head_dim), name=gm._name("qkv_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                        lora_config=config.lora_configs.get("attn"))
                    qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
                else:
                    q_einsum = lora.Einsum(
                        shape=(config.num_heads, config.width, config.head_dim), name=gm._name("q_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                        lora_config=config.lora_configs.get("attn"))
                    q = q_einsum("BTD,NDH->BTNH", x)
                    kv_einsum = lora.Einsum(
                        shape=(2, config.num_kv_heads, config.width, config.head_dim), name=gm._name("kv_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                        lora_config=config.lora_configs.get("attn"))
                    k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                    qkvs.append((q, k, v))
            q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
            q = gm._apply_rope(q, positions=positions)
            q *= self.configs[0].head_dim ** -0.5
            k = gm._apply_rope(k, positions=positions)
            if kv_cache is not None:
                cache_k, cache_v = kv_cache[:2]
                k = jnp.concatenate([cache_k, k], axis=1)
                v = jnp.concatenate([cache_v, v], axis=1)
            q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
            logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
            masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, -2.3819763e38)
            probs32 = jax.nn.softmax(masked_logits, axis=-1)
            probs = probs32.astype(dtype)
            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
            encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
            out, start = [], 0
            for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
                if x is not None:
                    end = start + x.shape[1]
                    out_einsum = lora.Einsum(
                        shape=(config.num_heads, config.head_dim, config.width), name=gm._name("attn_vec_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                        lora_config=config.lora_configs.get("attn"))
                    out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                    start = end
                else:
                    out.append(None)
            if _CAPTURE[0]:
                return out, (k, v, probs32.mean(axis=(1, 2, 3)))  # [B, S]: mean over kv heads, heads, queries
            return out, (k, v)

    CapturingAttention.__name__ = "Attention"
    gm.Attention = CapturingAttention  # Block looks the class up at call time; parameter paths are unchanged


def make_attention_sampler(backend, num_steps):
    """jit fn(state, obs, noise) -> (normalized action chunk [B,H,D], attention [steps, layers, B, prefix_len])."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    import openpi.shared.array_typing as at
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    from openpi.models.pi0 import posemb_sincos

    graphdef = backend._graphdef  # noqa: SLF001

    def fn(state, obs, noise):
        with at.disable_typechecking():
            m = nnx.merge(graphdef, state)
            tokens, mask, ar = m.embed_prefix(_model.preprocess_observation(None, obs, train=False))
            p = tokens.shape[1]
            _CAPTURE[0] = False
            _, kv = m.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
            b, h = noise.shape[:2]
            smask = jnp.ones((b, h), jnp.bool_)
            full = jnp.concatenate([jnp.repeat(mask[:, None], h, axis=1),
                                    make_attn_mask(smask, jnp.array([True] + [False] * (h - 1)))], axis=-1)
            pos = jnp.sum(mask, -1)[:, None] + jnp.cumsum(smask, -1) - 1
            x, attns = noise, []
            try:
                _CAPTURE[0] = True
                for i in range(num_steps):  # same Euler steps as Pi0.sample_actions (t: 1 -> 0)
                    t = jnp.full((b,), 1.0 - i / num_steps, jnp.float32)
                    act = m.action_in_proj(x)
                    temb = posemb_sincos(t, m.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
                    cond = nnx.swish(m.time_mlp_out(nnx.swish(m.time_mlp_in(temb))))
                    (_, out), cache = m.PaliGemma.llm([None, act], mask=full, positions=pos, kv_cache=kv,
                                                      adarms_cond=[None, cond])
                    v = m.action_out_proj(out[:, -h:])
                    x = x - v / num_steps
                    attns.append(cache[2][..., :p])  # [layers, B, prefix]
            finally:
                _CAPTURE[0] = False
            return x, jnp.stack(attns)

    return jax.jit(fn)


# ============================================================================
# Frame sources
# ============================================================================


def _video_meta(path):
    import imageio.v3 as iio

    meta = iio.immeta(path, plugin="pyav")
    return float(meta.get("fps") or G.FPS)


def _iter_video(path):
    import imageio.v3 as iio

    yield from iio.imiter(path, plugin="pyav")


def _load_states(path):
    if path.endswith(".npy"):
        s = np.load(path)
    else:
        df = pd.read_csv(path)
        s = df.select_dtypes("number").to_numpy()
        if s.shape[1] not in (28, 43):  # e.g. an extra index/time column
            s = s[:, -28:] if s.shape[1] > 28 and s.shape[1] < 43 else s[:, -43:]
    if s.ndim != 2 or s.shape[1] not in (28, 43):
        raise ValueError(f"{path}: expected [T, 28] or [T, 43] states, got {s.shape}")
    return s.astype(np.float32)


def frames_from_video(args, mid_state):
    """Yields (time_s, {raw cam key: image or None}, raw state, prompt) every --every seconds."""
    fps = _video_meta(args.video)
    stride = max(1, round(args.every * fps))
    states = _load_states(args.states) if args.states else None
    paths = [args.video, args.left_video, args.right_video]
    iters = [_iter_video(p) if p else None for p in paths]
    n_out = 0
    for i, ego in enumerate(iters[0]):
        others = [next(it, None) if it is not None else None for it in iters[1:]]  # assumes same fps, synced start
        if i % stride or (args.start and i / fps < args.start):
            continue
        if states is None:
            state = mid_state
        else:
            state = states[min(i, len(states) - 1)]
        yield i / fps, dict(zip(RAW_CAMS, [ego, *others], strict=True)), state, args.prompt
        n_out += 1
        if args.max_frames and n_out >= args.max_frames:
            return


def frames_from_records(args):
    files = sorted(Path(args.records).glob("step_*.npy"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not files:
        raise FileNotFoundError(f"no step_*.npy in {args.records}")
    stride = max(1, args.record_stride)
    n_out = 0
    for j, f in enumerate(files[::stride]):
        d = np.load(f, allow_pickle=True).item()
        cams = {k: d.get(f"inputs/{k}") for k in RAW_CAMS}
        prompt = args.prompt or d.get("inputs/prompt")
        if prompt is None:
            raise ValueError(f"{f}: no inputs/prompt in the record, pass --prompt")
        # The step index is the inference call; real time depends on the client's replan rate.
        yield j * stride, cams, d["inputs/observation/state"], str(prompt)
        n_out += 1
        if args.max_frames and n_out >= args.max_frames:
            return


# ============================================================================
# Per-frame analysis
# ============================================================================


UPPER_BODY_STATE_COLS = [f"state_{g}_{i}" for g in ("left_arm", "left_hand", "right_arm", "right_hand") for i in range(7)]


def frames_from_recording(args):
    """Robot client recordings (inference_recordings/inference_NN: meta.json, steps.csv, video.mp4 = head camera).

    Analyses the ticks where the client called the policy (new_inference == 1), with the 28-dim upper-body state the
    robot had at that tick. Only the head camera is recorded, so the wrist cameras are masked out.
    """
    import json

    rec = Path(args.recording)
    meta = json.loads((rec / "meta.json").read_text())
    steps = pd.read_csv(rec / "steps.csv")
    # Default: only the ticks where the client called the policy. --every_frame: every recorded tick, i.e. what the
    # model would have attended to had it been called at that frame (between calls the robot executes the old chunk).
    calls = steps if args.every_frame else steps[steps["new_inference"] == 1]
    calls = calls.iloc[:: max(1, args.record_stride)]
    if args.start:
        calls = calls[calls["t_elapsed"] >= args.start]
    if args.max_frames:
        calls = calls.iloc[: args.max_frames]
    prompt = args.prompt or meta["config"]["prompt"]
    want = dict(zip(calls["video_frame"].astype(int), calls.index, strict=True))
    last = max(want) if want else -1
    for i, frame in enumerate(_iter_video(str(rec / "video.mp4"))):
        if i in want:
            r = steps.loc[want[i]]
            cams = {RAW_CAMS[0]: frame, RAW_CAMS[1]: None, RAW_CAMS[2]: None}
            yield float(r["t_elapsed"]), cams, r[UPPER_BODY_STATE_COLS].to_numpy(np.float32), prompt
        if i >= last:
            return


def baseline_config():
    """pi05_g1_pickplace (the praveen_school_demo baseline, trained in the old repo) expressed with this repo's configs:
    same model and G1 Dex3 transforms as pi05_g1_dex3, but max_token_len 200 and a 32-dim state (see
    build_input_transform). Only the pi05 default prompt length and the state padding differ."""
    import dataclasses

    from openpi.training import config as _config

    base = _config.get_config("pi05_g1_dex3")
    return dataclasses.replace(base, name="pi05_g1_pickplace", model=dataclasses.replace(base.model, max_token_len=200))


def build_input_transform(config, checkpoint, asset_dir):
    """Like G.build_transforms, but without the LeRobot repack (we feed policy-level keys, as the server does)."""
    from openpi import transforms as _transforms
    from openpi.shared import normalize as _normalize

    assets = [Path(asset_dir)] if asset_dir else [p.parent for p in Path(checkpoint, "assets").rglob("norm_stats.json")]
    if len(assets) != 1:
        raise FileNotFoundError(f"Expected one norm_stats.json under {checkpoint}/assets, found {assets}; pass --asset")
    norm_stats = _normalize.load(assets[0])
    data_cfg = config.data.create(config.assets_dirs, config.model)
    state_dim = len(norm_stats["state"].q01)
    # The baseline (pi05_g1_pickplace) padded the 28-dim state to 32 before normalizing, so its norm stats and its
    # prompt state string have 32 values; the dex3 configs keep 28.
    pad = [_transforms.PadStatesAndActions(state_dim)] if state_dim != G.REAL_DIM else []
    inp = _transforms.compose([
        *data_cfg.data_transforms.inputs,
        *pad,
        _transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
        *data_cfg.model_transforms.inputs,
    ])
    q01, q99 = norm_stats["state"].q01, norm_stats["state"].q99
    return inp, ((q01 + q99) / 2).astype(np.float32)[: G.REAL_DIM], state_dim


def prepare(inp, cams, state, prompt):
    present = {k: v is not None for k, v in cams.items()}
    ref = next(v for v in cams.values() if v is not None)
    raw = {k: (v if v is not None else np.zeros_like(np.asarray(ref))) for k, v in cams.items()}
    item = inp({**raw, "observation/state": np.asarray(state, np.float32).copy(), "prompt": prompt})
    item["image_mask"] = dict(item["image_mask"])
    for raw_key, key in zip(RAW_CAMS, G.MODEL_CAMERAS, strict=True):
        if not present[raw_key]:  # camera not provided: mask it out of the prefix
            item["image_mask"][key] = np.False_
    return item, present


def shares_from_scores(r, backend, tgroups):
    return G.input_group_shares(np.asarray(r, np.float64), backend, tgroups)


def attention_maps(att, backend, tok_valid, clip_pct):
    """Per-camera attention maps. A few "sink" patches usually get most of the attention; values are capped at the
    clip_pct percentile of the camera's patches so the rest of the map stays visible (shares are not affected)."""
    maps = {}
    for c in G.CAMERA_NAMES:
        m = att[backend.cam_index[c]] * tok_valid[c]
        if tok_valid[c].any() and clip_pct < 100:
            m = np.minimum(m, np.percentile(m[tok_valid[c]], clip_pct))
        maps[c] = m.reshape(backend.grid, backend.grid)
    return maps


def gradcam_maps(act, grad, backend, tok_valid):
    maps = {}
    for c in G.CAMERA_NAMES:
        idx, v = backend.cam_index[c], tok_valid[c]
        if not v.any():
            maps[c] = np.zeros((backend.grid, backend.grid), np.float32)
            continue
        m = act[idx] @ grad[idx][v].mean(0)
        maps[c] = (np.maximum(m, 0.0) * v).reshape(backend.grid, backend.grid)
    return maps


# ============================================================================
# Rendering
# ============================================================================

ROW_LABELS = {"attn": "Attention (action expert -> image)", "cam": "Grad-CAM (own prediction)"}
GROUP_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#999999"]


def render(images, present, maps, shares, title):
    fig = plt.figure(figsize=(15, 7.2), dpi=90)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.08, hspace=0.18)
    for ri, method in enumerate(("attn", "cam")):
        for ci, cam in enumerate(G.CAMERA_NAMES):
            ax = fig.add_subplot(gs[ri, ci])
            if not present[cam]:
                ax.imshow(np.full((10, 10, 3), 0.85))
                ax.text(5, 5, "not provided\n(masked)", ha="center", va="center", fontsize=9, color="0.3")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(cam, fontsize=9)
                continue
            m = maps[method][cam]
            G._show(ax, images[cam], m if m.max() > 0 else None, title=cam if ri == 0 else None)  # noqa: SLF001
            if ci == 0:
                ax.set_ylabel(ROW_LABELS[method], fontsize=9)
        ax = fig.add_subplot(gs[ri, 3])
        vals = [shares[method][g] for g in GROUPS]
        ax.barh(range(len(GROUPS))[::-1], vals, color=GROUP_COLORS)
        ax.set_yticks(range(len(GROUPS))[::-1], GROUPS, fontsize=8)
        ax.set_xlim(0, max(0.05, max(vals) * 1.15))
        for y, v in zip(range(len(GROUPS))[::-1], vals, strict=True):
            ax.text(v, y, f" {100 * v:.0f}%", va="center", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title("share of " + ("attention" if method == "attn" else "attribution"), fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def timeline(df, out_path, xlabel):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, method, label in zip(axes, ("attn", "cam"), ("attention share", "Grad-CAM share"), strict=True):
        for g, c in zip(GROUPS, GROUP_COLORS, strict=True):
            ax.plot(df["t"], df[f"{method}_{g}"], label=g, color=c, lw=1.6)
        ax.set_ylabel(label)
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, ncol=3, frameon=False)
    axes[1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def _layer_slice(spec, n):
    if not spec:
        return slice(0, n)
    a, _, b = spec.partition("-")
    return slice(int(a), int(b or a) + 1)


def run(args):
    import imageio
    import jax

    if not (args.records or args.video or args.recording):
        raise SystemExit("pass --recording, --records or --video (with --prompt)")
    if args.video and not args.prompt:
        raise SystemExit("--video needs --prompt")

    _install_capturing_attention()
    config_name = args.config_name or G.DEFAULT_CONFIG
    if config_name == "pi05_g1_pickplace":
        config = baseline_config()
        Backend = G.Pi05Backend
    else:
        Backend, config, config_name = G.detect_backend(args.checkpoint, config_name)
    if getattr(config.model, "dino_fusion", "prefix_add") != "prefix_add":
        raise SystemExit(f"{config_name}: dino_fusion={config.model.dino_fusion} is not supported (prefix fusion only)")
    if not args.fp32:
        # Load the weights in bfloat16, as serve_policy.py does (and half the memory of float32).
        import functools

        import jax.numpy as jnp

        from openpi.models import model as _model

        _model.restore_params = functools.partial(_model.restore_params, dtype=jnp.bfloat16)
    backend = Backend(config, args.checkpoint, args.num_steps)
    inp, mid_state, state_dim = build_input_transform(config, args.checkpoint, args.asset)
    sampler = make_attention_sampler(backend, args.num_steps)
    horizon, adim = config.model.action_horizon, config.model.action_dim
    noise, times = G.noise_and_times(args.n_noise, horizon, adim, args.seed)
    gmat = G.group_matrix(adim)[[G.CAM_TARGETS.index(args.target)]]  # only the selected joint group (1 backward pass)
    sample_noise = np.random.default_rng(args.seed + 1).standard_normal((1, horizon, adim)).astype(np.float32)

    src = Path(args.recording or args.records or args.video)
    step = Path(args.checkpoint.rstrip("/")).name
    run_name = Path(args.checkpoint.rstrip("/")).parent.name
    out_dir = Path(args.out_dir or f"./plots/attention/{config_name}__{run_name}__{step}__{src.stem}")
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    if args.recording:
        frames, xlabel = frames_from_recording(args), "time (s)"
        print("[info] recording: head camera only, wrist cameras masked out; state from steps.csv")
    elif args.records:
        frames, xlabel = frames_from_records(args), "inference call"
    else:
        frames, xlabel = frames_from_video(args, mid_state), "time (s)"
        if not args.states:
            print("[warn] no --states: the mid-range state is fed to the model (state is part of the pi05 prompt)")
        missing = [n for n, p in zip(("left", "right"), (args.left_video, args.right_video), strict=True) if not p]
        if missing:
            print(f"[warn] no {'/'.join(missing)} wrist video: those cameras are masked out (never seen in training)")

    out_fps = args.out_fps or (G.FPS / max(1, args.record_stride) if args.recording and args.every_frame else 4.0)
    writer = imageio.get_writer(out_dir / "heatmaps.mp4", fps=out_fps, macro_block_size=1)
    rows, t0, last_png_slot = [], _time.time(), None
    lsl = None
    for fi, (t, cams, state, prompt) in enumerate(frames):
        item, present_raw = prepare(inp, cams, state, prompt)
        present = {c: present_raw[k] for c, k in zip(G.CAMERA_NAMES, RAW_CAMS, strict=True)}
        batch = G.to_model_batch([item])
        obs = backend._obs(batch)  # noqa: SLF001

        # 1) sample the chunk the robot would get, capturing action-expert attention on the way
        if fi == 0:
            print("[jit] first frame: compiling the sampler and the Grad-CAM jacobian (slow once)", flush=True)
        x0, att = sampler(backend.state, obs, jax.numpy.asarray(sample_noise))
        att = np.asarray(att, np.float32)  # [steps, layers, 1, prefix]
        lsl = lsl or _layer_slice(args.layers, att.shape[1])
        att = att[:, lsl, 0].mean((0, 1))
        x0 = np.array(x0, np.float32)
        x0[..., G.REAL_DIM:] = 0.0  # padded dims are zero in training targets

        # 2) Grad-CAM with the own prediction as target
        if fi == 0:
            print("[jit] sampler done", flush=True)
        tokens, pmask, ar = backend._embed(backend.state, obs)  # noqa: SLF001
        act, extra = np.asarray(tokens[0], np.float32), []
        if args.no_gradcam:
            grad = np.zeros_like(act)
        else:
            jac = backend._jac(tokens, backend.state, pmask, ar, jax.numpy.asarray(x0), noise, times, gmat)  # noqa: SLF001
            grad = -np.asarray(jac[0, 0], np.float32)

        ids, word_map, tgroups = backend.text_groups(prompt, np.asarray(item["state"])[:state_dim])
        if not np.array_equal(ids, np.asarray(item["tokenized_prompt"])[np.asarray(item["tokenized_prompt_mask"], bool)]):
            raise RuntimeError("Prompt token reconstruction differs from the transformed input")

        images = {c: np.asarray(item["image"][k]) for c, k in zip(G.CAMERA_NAMES, G.MODEL_CAMERAS, strict=True)}
        tok_valid = {c: (G.token_valid(images[c], backend.grid) if present[c] else np.zeros(backend.grid ** 2, bool))
                     for c in G.CAMERA_NAMES}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            maps = {"attn": attention_maps(att, backend, tok_valid, args.attn_clip), "cam": gradcam_maps(act, grad, backend, tok_valid)}
        r_cam = G.token_attribution(act, grad, extra, backend)
        shares = {"attn": shares_from_scores(att, backend, tgroups), "cam": shares_from_scores(r_cam, backend, tgroups)}

        row = {"t": t, "prompt": prompt}
        for m in ("attn", "cam"):
            for g, v in shares[m].items():
                row[f"{m}_{g}"] = v
        for w, v in G.word_shares(np.asarray(att, np.float64), backend, word_map).items():
            row[f"attn_word_{w}"] = v
        for w, v in G.word_shares(r_cam, backend, word_map).items():
            row[f"cam_word_{w}"] = v
        rows.append(row)

        tstr = f"t={t:.2f}s" if not args.records else f"call {t}"
        title = (f"{config_name} @ {step}   {tstr}   Grad-CAM target: {'off' if args.no_gradcam else args.target}"
                 f"{'' if args.records or args.states or args.recording else '   [state: mid-range placeholder]'}\n\"{prompt}\"")
        img = render(images, present, maps, shares, title)
        writer.append_data(img)
        # PNGs: every analysed frame, or (--png_every) the first analysed frame of each png_every-second window
        slot = int(t // args.png_every) if args.png_every and not args.records else fi
        if not args.no_pngs and slot != last_png_slot:
            name = f"{fi:05d}_t{t:07.2f}s.png" if not args.records else f"{fi:05d}.png"
            plt.imsave(out_dir / "frames" / name, img)
            last_png_slot = slot
        el = _time.time() - t0
        print(f"[{fi + 1}] {tstr}  attn img {100 * sum(shares['attn'][c] for c in G.CAMERA_NAMES):.0f}%  "
              f"cam img {100 * sum(shares['cam'][c] for c in G.CAMERA_NAMES):.0f}%  ({el / (fi + 1):.1f} s/frame)")
    writer.close()
    if not rows:
        raise SystemExit("no frames processed")

    df = pd.DataFrame(rows)
    word_cols = [c for c in df.columns if "_word_" in c]
    df[word_cols] = df[word_cols].fillna(0.0)
    df.to_csv(out_dir / "per_frame.csv", index=False)
    timeline(df, out_dir / "timeline.png", xlabel)
    print("\nMean shares over the video:")
    for m, label in (("attn", "attention"), ("cam", "Grad-CAM")):
        print(f"  {label:10s} " + "  ".join(f"{g} {100 * df[f'{m}_{g}'].mean():.1f}%" for g in GROUPS))
    print(f"\nVideo: {out_dir / 'heatmaps.mp4'}\nTimeline: {out_dir / 'timeline.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="checkpoint step dir with params/ and assets/")
    p.add_argument("--config_name", default=None, help=f"config the checkpoint was trained with (default {G.DEFAULT_CONFIG})")
    p.add_argument("--asset", default=None, help="dir with norm_stats.json (default: the one under <checkpoint>/assets)")
    src = p.add_argument_group("input")
    src.add_argument("--video", help="head-camera (ego_view) video")
    src.add_argument("--left_video", help="left wrist video, synced with --video")
    src.add_argument("--right_video", help="right wrist video, synced with --video")
    src.add_argument("--states", help="npy/csv with one 28- or 43-dim state per video frame (default: mid-range)")
    src.add_argument("--recording", help="robot client recording dir (meta.json, steps.csv, video.mp4)")
    src.add_argument("--records", help="policy_records dir from serve_policy.py --record (instead of --video)")
    src.add_argument("--record_stride", type=int, default=1, help="use every n-th record / policy call")
    src.add_argument("--prompt", help="instruction (required with --video; overrides the records' prompt)")
    src.add_argument("--every", type=float, default=0.5, help="seconds between analysed video frames")
    src.add_argument("--start", type=float, default=0.0, help="skip the first seconds of the video")
    src.add_argument("--max_frames", type=int, default=0, help="stop after this many analysed frames (0 = all)")
    p.add_argument("--target", default="all", choices=G.CAM_TARGETS, help="joint group the Grad-CAM score uses")
    p.add_argument("--layers", default="", help="layer range for the attention maps, e.g. '9-17' (default: all)")
    p.add_argument("--attn_clip", type=float, default=98.0,
                   help="cap attention maps at this percentile of a camera's patches (hides attention sinks; 100 = off)")
    p.add_argument("--n_noise", type=int, default=8, help="noise/time draws in the Grad-CAM score")
    p.add_argument("--num_steps", type=int, default=10, help="denoising steps (same as serving)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp32", action="store_true", help="load the weights in float32 (default bfloat16, as served)")
    p.add_argument("--out_fps", type=float, default=None,
                   help="frame rate of heatmaps.mp4 (default: real time with --every_frame, else 4)")
    p.add_argument("--every_frame", action="store_true",
                   help="--recording: analyse every video frame, not only the policy calls (one per 50 frames)")
    p.add_argument("--no_gradcam", action="store_true", help="attention only (about 4x faster)")
    p.add_argument("--no_pngs", action="store_true", help="do not write frames/*.png")
    p.add_argument("--png_every", type=float, default=0.0,
                   help="write a PNG only once per this many seconds (the video keeps every analysed frame); 0 = all")
    p.add_argument("--out_dir", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
