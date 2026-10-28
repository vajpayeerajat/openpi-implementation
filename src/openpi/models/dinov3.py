"""DINOv3 ViT image encoder (frozen feature extractor), matching the HF `DINOv3ViTModel` weights.

Differences from the SigLIP ViT in `siglip.py`:
* 2D RoPE on the patch tokens instead of a learned position embedding (so any input resolution works).
* A CLS token and register tokens are prepended; RoPE skips them.
* LayerScale after attention and MLP; no bias on the key projection.
* Expects ImageNet-normalized images. `Module` takes openpi images in [-1, 1] and does the resize and
  normalization itself.
"""

import math
from typing import Literal

import flax.linen as nn
import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
import safetensors.numpy

Variant = Literal["dummy", "B/16"]

_VARIANTS = {
    # Tiny random encoder for tests (no pretrained weights). Same 16x16 patch grid as B/16.
    "dummy": {"width": 32, "depth": 2, "num_heads": 2, "mlp_dim": 64, "patch_size": 16, "num_registers": 4},
    "B/16": {"width": 768, "depth": 12, "num_heads": 12, "mlp_dim": 3072, "patch_size": 16, "num_registers": 4},
}

# Pretrained weights for each variant (gated on the HF hub: accept the license on the model page first).
HF_REPO_IDS = {
    "B/16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_LN_EPS = 1e-5
_ROPE_BASE = 100.0


def get_config(variant: Variant) -> dict:
    return _VARIANTS[variant]


def rope_sincos(h: int, w: int, head_dim: int) -> tuple[jax.Array, jax.Array]:
    """sin, cos of shape [h * w, head_dim] for patch centers normalized to [-1, 1] (eval mode, no jitter)."""
    inv_freq = 1.0 / _ROPE_BASE ** jnp.arange(0, 1, 4 / head_dim, dtype=jnp.float32)  # [head_dim // 4]
    coords_h = jnp.arange(0.5, h, dtype=jnp.float32) / h
    coords_w = jnp.arange(0.5, w, dtype=jnp.float32) / w
    coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1).reshape(h * w, 2)
    coords = 2.0 * coords - 1.0
    angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]  # [hw, 2, head_dim // 4]
    angles = jnp.tile(angles.reshape(h * w, -1), 2)  # [hw, head_dim]
    return jnp.sin(angles), jnp.cos(angles)


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


class Attention(nn.Module):
    num_heads: int
    num_prefix_tokens: int
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, sin, cos):
        b, n, d = x.shape
        head_dim = d // self.num_heads

        def proj(name, *, use_bias=True):
            y = nn.Dense(d, use_bias=use_bias, dtype=self.dtype_mm, name=name)(x)
            return y.reshape(b, n, self.num_heads, head_dim)

        q, k, v = proj("q"), proj("k", use_bias=False), proj("v")

        # RoPE on the patch tokens only, in float32.
        def rope(t):
            t = t.astype(jnp.float32)
            p = self.num_prefix_tokens
            patches = t[:, p:] * cos[None, :, None, :] + _rotate_half(t[:, p:]) * sin[None, :, None, :]
            return jnp.concatenate([t[:, :p], patches], axis=1).astype(self.dtype_mm)

        q, k = rope(q), rope(k)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * head_dim**-0.5
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(self.dtype_mm)
        out = jnp.einsum("bhqk,bkhd->bqhd", probs, v.astype(self.dtype_mm)).reshape(b, n, d)
        return nn.Dense(d, dtype=self.dtype_mm, name="o")(out)


class Block(nn.Module):
    num_heads: int
    mlp_dim: int
    num_prefix_tokens: int
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, sin, cos):
        d = x.shape[-1]
        ls1 = self.param("ls1", nn.initializers.constant(1e-5), (d,))
        ls2 = self.param("ls2", nn.initializers.constant(1e-5), (d,))

        y = nn.LayerNorm(epsilon=_LN_EPS, dtype=self.dtype_mm, name="norm1")(x)
        y = Attention(self.num_heads, self.num_prefix_tokens, self.dtype_mm, name="attn")(y, sin, cos)
        x = x + ls1 * y

        y = nn.LayerNorm(epsilon=_LN_EPS, dtype=self.dtype_mm, name="norm2")(x)
        y = nn.Dense(self.mlp_dim, dtype=self.dtype_mm, name="fc1")(y)
        y = nn.gelu(y, approximate=False)
        y = nn.Dense(d, dtype=self.dtype_mm, name="fc2")(y)
        return x + ls2 * y


class Module(nn.Module):
    """DINOv3 ViT. Returns the final-norm patch tokens [b, (image_size / patch_size)**2, width]."""

    variant: Variant = "B/16"
    # Images are resized to this square size before the encoder. 256 with patch 16 gives the same 16x16 grid as
    # SigLIP So400m/14 at 224, so patch i of both encoders covers the same image region.
    image_size: int = 256
    dtype_mm: str = "float32"
    # If set, returns the mean of the final-norm outputs of these blocks (0-based, negative counts from the end)
    # instead of the output of the last block.
    layers: tuple[int, ...] | None = None

    @nn.compact
    def __call__(self, image):
        cfg = get_config(self.variant)
        width, patch = cfg["width"], cfg["patch_size"]
        num_prefix = 1 + cfg["num_registers"]

        # openpi images are [-1, 1]; DINOv3 expects ImageNet-normalized [0, 1] images.
        image = jnp.asarray(image, jnp.float32)
        b = image.shape[0]
        image = jax.image.resize(image, (b, self.image_size, self.image_size, 3), method="bilinear")
        image = ((image + 1.0) / 2.0 - jnp.array(_IMAGENET_MEAN)) / jnp.array(_IMAGENET_STD)

        x = nn.Conv(width, (patch, patch), strides=(patch, patch), padding="VALID", name="embedding")(image)
        _, h, w, _ = x.shape
        x = x.reshape(b, h * w, width)

        cls = self.param("cls", nn.initializers.zeros, (1, 1, width))
        registers = self.param("registers", nn.initializers.zeros, (1, cfg["num_registers"], width))
        x = jnp.concatenate([jnp.tile(cls, [b, 1, 1]), jnp.tile(registers, [b, 1, 1]), x], axis=1)
        x = x.astype(self.dtype_mm)

        sin, cos = rope_sincos(h, w, width // cfg["num_heads"])
        outputs = []
        for i in range(cfg["depth"]):
            x = Block(cfg["num_heads"], cfg["mlp_dim"], num_prefix, self.dtype_mm, name=f"encoderblock_{i}")(
                x, sin, cos
            )
            outputs.append(x)
        norm = nn.LayerNorm(epsilon=_LN_EPS, dtype=self.dtype_mm, name="norm")
        if self.layers is None:
            x = norm(x)
        else:
            x = jnp.mean(jnp.stack([norm(outputs[i]) for i in self.layers]), axis=0).astype(self.dtype_mm)
        return x[:, num_prefix:]


def load_hf_params(variant: Variant) -> dict:
    """Downloads the pretrained HF weights and converts them to the param tree of `Module` (numpy, float32)."""
    path = huggingface_hub.hf_hub_download(HF_REPO_IDS[variant], "model.safetensors")
    sd = safetensors.numpy.load_file(path)

    def dense(prefix, *, bias=True):
        out = {"kernel": sd[f"{prefix}.weight"].T}
        if bias:
            out["bias"] = sd[f"{prefix}.bias"]
        return out

    def norm(prefix):
        return {"scale": sd[f"{prefix}.weight"], "bias": sd[f"{prefix}.bias"]}

    params = {
        # torch conv [out, in, kh, kw] -> flax [kh, kw, in, out].
        "embedding": {
            "kernel": sd["embeddings.patch_embeddings.weight"].transpose(2, 3, 1, 0),
            "bias": sd["embeddings.patch_embeddings.bias"],
        },
        "cls": sd["embeddings.cls_token"],
        "registers": sd["embeddings.register_tokens"],
        "norm": norm("norm"),
    }
    for i in range(get_config(variant)["depth"]):
        p = f"layer.{i}"
        params[f"encoderblock_{i}"] = {
            "norm1": norm(f"{p}.norm1"),
            "attn": {
                "q": dense(f"{p}.attention.q_proj"),
                "k": dense(f"{p}.attention.k_proj", bias=False),
                "v": dense(f"{p}.attention.v_proj"),
                "o": dense(f"{p}.attention.o_proj"),
            },
            "ls1": sd[f"{p}.layer_scale1.lambda1"],
            "norm2": norm(f"{p}.norm2"),
            "fc1": dense(f"{p}.mlp.up_proj"),
            "fc2": dense(f"{p}.mlp.down_proj"),
            "ls2": sd[f"{p}.layer_scale2.lambda1"],
        }
    return jax.tree.map(lambda a: np.asarray(a, np.float32), params)
