import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.dinov3 as _dinov3
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# Input size of the DINOv3 encoder (see `dinov3.Module.image_size`) and the SigLIP So400m/14 patch size.
_DINO_IMAGE_SIZE = 256
_SIGLIP_PATCH_SIZE = 14
# Fixed flow timestep whose (learned) time cond, plus `dino_ctx`, conditions the DINO tokens.
_DINO_CTX_TIME = 0.0


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def _rms(x: at.Array) -> at.Array:
    return jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32))))


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.dino_fusion = config.dino_fusion
        self.use_dino = config.dino_fusion == "prefix_add"
        if self.use_dino:
            dino_width = _dinov3.get_config(config.dino_variant)["width"]
            dino = nnx_bridge.ToNNX(_dinov3.Module(variant=config.dino_variant, dtype_mm=config.dtype))
            dino.lazy_init(next(iter(config.fake_obs().images.values())), rngs=rngs)
            self.dino = dino
            self.dino_norm = nnx.LayerNorm(dino_width, rngs=rngs)
            # Zero init: at step 0 the image tokens are exactly the SigLIP tokens.
            self.dino_proj = nnx.Linear(
                dino_width, paligemma_config.width, kernel_init=nnx.initializers.zeros, rngs=rngs
            )
        elif self.dino_fusion == "expert_stream":
            dino_width = _dinov3.get_config(config.dino_variant)["width"]
            dino_layers = None if config.dino_layers is None else tuple(config.dino_layers)
            dino = nnx_bridge.ToNNX(
                _dinov3.Module(variant=config.dino_variant, dtype_mm=config.dtype, layers=dino_layers)
            )
            fake_images = config.fake_obs().images
            dino.lazy_init(next(iter(fake_images.values())), rngs=rngs)
            self.dino = dino
            self.dino_cameras = tuple(fake_images if config.dino_cameras is None else config.dino_cameras)
            self.dino_pool = config.dino_pool
            self.dino_freeze_encoder = config.dino_freeze_encoder
            # The DINO grid must match the SigLIP grid: DINO tokens take the RoPE positions of the SigLIP patches
            # they cover (see `embed_dino`).
            self.dino_grid = _DINO_IMAGE_SIZE // _dinov3.get_config(config.dino_variant)["patch_size"]
            assert self.dino_grid == _model.IMAGE_RESOLUTION[0] // _SIGLIP_PATCH_SIZE
            assert self.dino_grid % self.dino_pool == 0
            width = action_expert_config.width
            self.dino_norm = nnx.LayerNorm(dino_width, rngs=rngs)
            # default kernel init: LeCun normal (an explicit `lecun_normal()` would be a new function object in each
            # model instance, which breaks graphdef equality between eval_shape and init).
            self.dino_proj = nnx.Linear(dino_width, width, rngs=rngs)
            self.dino_cam_emb = nnx.Param(nnx.initializers.normal(0.02)(rngs.params(), (len(self.dino_cameras), width)))
            # Per-channel gate on the tokens and a scalar gate on the action tokens' attention to them, both
            # applied as tanh(gate). The token gate is zero at init; the attention gate starts at
            # tanh^-1(dino_attn_gate_init) (zero by default). The token gate (and then the projection) learn at a
            # rate proportional to tanh(dino_attn_gate).
            self.dino_gate = nnx.Param(jnp.zeros((width,)))
            self.dino_attn_gate = nnx.Param(jnp.full((), np.arctanh(config.dino_attn_gate_init), dtype=jnp.float32))
            if config.pi05:
                # Learned offset to the adaRMS cond of the DINO tokens (the time cond at t=_DINO_CTX_TIME). It does
                # not depend on the flow timestep, so their K/V can be cached across denoising steps.
                self.dino_ctx = nnx.Param(jnp.zeros((width,)))
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            if self.use_dino:
                # Both encoders give a 16x16 patch grid in the same row-major order, so token i of each covers the
                # same image region and they can be summed.
                dino_tokens = jax.lax.stop_gradient(self.dino(obs.images[name]))
                dino_tokens = self.dino_proj(self.dino_norm(dino_tokens))
                if not self.deterministic:
                    # logged by the train loop: size of the DINO term relative to the SigLIP tokens (0 at init).
                    self.sow(
                        nnx.Intermediate,
                        f"dino_siglip_ratio_{name}",
                        _rms(dino_tokens) / jnp.maximum(_rms(image_tokens), 1e-12),
                    )
                # RAJAT :  This is the addition of DINO tokens to SigLIP tokens
                # As of now, its direct element wise addition of the two tokens similar to OpenVLA.
                # TODO : Bi-directional attention between DINO and SigLIP tokens, where
                #        DINO tokens attend to SigLIP tokens and vice versa.
                #        This will help in the training of the DINO tokens.
                image_tokens = image_tokens + dino_tokens.astype(image_tokens.dtype)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @at.typecheck
    def embed_dino(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], np.ndarray]:
        """Action-expert DINO tokens ("expert_stream").

        Returns the tokens, input mask and ar mask, and for each token the index of the prefix (SigLIP) token whose
        RoPE position it takes: the center patch of its pooled cell, so that the action tokens keep their positions.
        """
        image_names = list(obs.images)
        k, grid = self.dino_pool, self.dino_grid
        pooled = grid // k
        cells = np.arange(pooled) * k + k // 2
        patch_idx = (cells[:, None] * grid + cells[None, :]).reshape(-1)

        input_mask = []
        tokens = []
        pos_idx = []
        for i, name in enumerate(self.dino_cameras):
            dino_tokens = self.dino(obs.images[name])
            if self.dino_freeze_encoder:
                dino_tokens = jax.lax.stop_gradient(dino_tokens)
            # [b, grid * grid, w] in row-major order -> k x k average pooling.
            b, _, w = dino_tokens.shape
            dino_tokens = dino_tokens.astype(jnp.float32).reshape(b, pooled, k, pooled, k, w).mean(axis=(2, 4))
            dino_tokens = dino_tokens.reshape(b, pooled * pooled, w)
            dino_tokens = self.dino_proj(self.dino_norm(dino_tokens)) + self.dino_cam_emb.value[i]
            tokens.append(jnp.tanh(self.dino_gate.value) * dino_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=dino_tokens.shape[1]))
            pos_idx.append(image_names.index(name) * grid * grid + patch_idx)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        # the prefix does not attend to DINO tokens; DINO tokens attend to the prefix and to each other.
        ar_mask = jnp.array([True] + [False] * (tokens.shape[1] - 1))
        return tokens, input_mask, ar_mask, np.concatenate(pos_idx)

    def _dino_ctx_cond(self, batch_size: int) -> at.Float[at.Array, "b emb"]:
        """adaRMS cond of the DINO tokens: the time cond at a fixed timestep plus a learned offset."""
        cond = posemb_sincos(
            jnp.full((batch_size,), _DINO_CTX_TIME), self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )
        cond = nnx.swish(self.time_mlp_in(cond))
        cond = nnx.swish(self.time_mlp_out(cond))
        return cond + self.dino_ctx.value

    def _dino_gated_attn(
        self,
        attn_mask: at.Bool[at.Array, "b t s"],
        query_is_suffix: at.Bool[at.Array, " t"],
        key_is_dino: at.Bool[at.Array, " s"],
    ) -> tuple[at.Bool[at.Array, "b t s"], tuple[at.Bool[at.Array, "b t s"], at.Float[at.Array, ""]]]:
        """Moves the suffix -> DINO entries of `attn_mask` into the tanh-gated attention block (see `gemma.Attention`)."""
        gated_mask = attn_mask & query_is_suffix[None, :, None] & key_is_dino[None, None, :]
        return attn_mask & ~gated_mask, (gated_mask, jnp.tanh(self.dino_attn_gate.value))

    def _expert_stream_forward(
        self, observation: _model.Observation, x_t: _model.Actions, time: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b p emb"], at.Float[at.Array, "b ah ad"], dict[str, at.Array]]:
        """Prefix, DINO and suffix tokens in one pass ("expert_stream"). Returns the prefix outputs, v_t, and stats.

        This is the training path; `sample_actions` computes the same thing with the prefix and DINO tokens cached.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        dino_tokens, dino_mask, dino_ar_mask, dino_pos_idx = self.embed_dino(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        num_prefix, num_dino, num_suffix = prefix_tokens.shape[1], dino_tokens.shape[1], suffix_tokens.shape[1]
        input_mask = jnp.concatenate([prefix_mask, dino_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, dino_ar_mask, suffix_ar_mask], axis=0)
        token_idx = jnp.arange(num_prefix + num_dino + num_suffix)
        attn_mask, gated_attn = self._dino_gated_attn(
            make_attn_mask(input_mask, ar_mask),
            token_idx >= num_prefix + num_dino,
            (token_idx >= num_prefix) & (token_idx < num_prefix + num_dino),
        )
        # same positions as without DINO for the prefix and suffix tokens (see `sample_actions`).
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=1) - 1
        positions = jnp.concatenate([prefix_positions, prefix_positions[:, dino_pos_idx], suffix_positions], axis=1)
        if adarms_cond is not None:
            # per-token cond in the expert stream: fixed context for the DINO tokens, flow timestep for the suffix.
            batch_size, width = adarms_cond.shape
            adarms_cond = jnp.concatenate(
                [
                    jnp.broadcast_to(self._dino_ctx_cond(batch_size)[:, None], (batch_size, num_dino, width)),
                    jnp.broadcast_to(adarms_cond[:, None], (batch_size, num_suffix, width)),
                ],
                axis=1,
            )
        (prefix_out, expert_out), _ = self.PaliGemma.llm(
            [prefix_tokens, jnp.concatenate([dino_tokens, suffix_tokens], axis=1)],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            gated_attn=gated_attn,
        )
        v_t = self.action_out_proj(expert_out[:, -self.action_horizon :])

        def rms(x, mask):
            mean_sq = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1)
            return jnp.sqrt(jnp.sum(mean_sq * mask) / jnp.maximum(jnp.sum(mask), 1))

        stats = {"dino_token_rms": rms(dino_tokens, dino_mask), "expert_input_rms": rms(suffix_tokens, suffix_mask)}
        return prefix_out, v_t, stats

    def _fill_kv_cache(
        self, observation: _model.Observation
    ) -> tuple[_gemma.KVCache, at.Bool[at.Array, "b c"], at.Int[at.Array, " b"], int]:
        """Runs the prefix (and the DINO tokens, for "expert_stream") once for `sample_actions`.

        Returns the KV cache, the input mask of the cached tokens, the number of valid prefix tokens (the suffix
        positions start there), and the number of DINO tokens at the end of the cache.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        if self.dino_fusion != "expert_stream":
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
            return kv_cache, prefix_mask, jnp.sum(prefix_mask, axis=-1), 0

        # The DINO tokens only depend on the observation (their adaRMS cond is not the flow timestep), so they are
        # computed here once and their K/V reused by all denoising steps.
        dino_tokens, dino_mask, dino_ar_mask, dino_pos_idx = self.embed_dino(observation)
        cache_mask = jnp.concatenate([prefix_mask, dino_mask], axis=1)
        attn_mask = make_attn_mask(cache_mask, jnp.concatenate([prefix_ar_mask, dino_ar_mask], axis=0))
        adarms_cond = self._dino_ctx_cond(prefix_tokens.shape[0]) if self.pi05 else None
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, dino_tokens],
            mask=attn_mask,
            positions=jnp.concatenate([positions, positions[:, dino_pos_idx]], axis=1),
            adarms_cond=[None, adarms_cond],
        )
        return kv_cache, cache_mask, jnp.sum(prefix_mask, axis=-1), dino_tokens.shape[1]

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if self.dino_fusion == "expert_stream":
            _, v_t, stats = self._expert_stream_forward(observation, x_t, time)
            if train:
                # picked up by the train loop for logging.
                for name, value in stats.items():
                    self.sow(nnx.Intermediate, name, value)
            return jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix (and the DINO tokens, for "expert_stream")
        kv_cache, prefix_mask, num_prefix_valid, num_dino = self._fill_kv_cache(observation)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_mask.shape[1] + suffix_tokens.shape[1],
            )
            gated_attn = None
            if num_dino:
                # the suffix attends to the cached DINO tokens through the gated attention block.
                key_idx = jnp.arange(full_attn_mask.shape[-1])
                full_attn_mask, gated_attn = self._dino_gated_attn(
                    full_attn_mask,
                    jnp.ones(suffix_tokens.shape[1], dtype=jnp.bool_),
                    (key_idx >= prefix_mask.shape[1] - num_dino) & (key_idx < prefix_mask.shape[1]),
                )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = num_prefix_valid[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                gated_attn=gated_attn,
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
