import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.dinov3 as _dinov3
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # DINOv3 encoder variant, used when `dino_fusion` is not "none".
    dino_variant: _dinov3.Variant | None = None
    # How DINOv3 features enter the model:
    # - "none": no DINO.
    # - "prefix_add": a frozen DINOv3 encoder runs on every camera, and its patch tokens are added to the SigLIP image
    #   tokens through a zero-initialized projection (so the model starts out identical to pi0/pi05).
    # - "expert_stream": pooled DINO tokens are extra tokens of the action expert, between the prefix and the suffix.
    #   They attend to the prefix and to each other, the prefix does not see them, and the action tokens see them
    #   through a separately normalized, tanh-gated attention block (zero at init, so the model starts out identical).
    #   Their K/V do not depend on the flow timestep, so they are computed once per `sample_actions` call.
    # For backward compatibility, setting only `dino_variant` selects "prefix_add".
    dino_fusion: Literal["none", "prefix_add", "expert_stream"] = "none"
    # The options below only apply to "expert_stream".
    # Cameras that get DINO tokens (None = all).
    dino_cameras: tuple[str, ...] | None = None
    # k x k average pooling of the 16x16 DINO patch grid (k=2 gives 64 tokens per camera).
    dino_pool: int = 2
    # DINO blocks whose (final-norm) outputs are averaged; None = the final block.
    dino_layers: tuple[int, ...] | None = None
    # Learning rate multiplier for the new DINO fusion params (not the encoder).
    dino_lr_mult: float = 10.0
    dino_freeze_encoder: bool = True
    # tanh(dino_attn_gate) at init: how open the action tokens' attention to the DINO tokens starts. 0 keeps the model
    # identical to pi0/pi05 at init, but then the attention gate and the (also zero) token gate each get gradient only
    # through the other and can stay closed; a small value like 0.2 gives the DINO branch gradient from the start.
    dino_attn_gate_init: float = 0.0

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.dino_variant is not None and self.dino_fusion == "none":
            object.__setattr__(self, "dino_fusion", "prefix_add")
        if self.dino_fusion != "none" and self.dino_variant is None:
            raise ValueError(f"dino_fusion={self.dino_fusion!r} needs a dino_variant.")
        if not -1.0 < self.dino_attn_gate_init < 1.0:
            raise ValueError(f"dino_attn_gate_init={self.dino_attn_gate_init} must be in (-1, 1) (it is a tanh value).")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        freeze = nnx.All(*filters) if filters else nnx.Nothing
        if self.dino_fusion == "prefix_add" or (self.dino_fusion == "expert_stream" and self.dino_freeze_encoder):
            # The DINOv3 encoder is frozen; its fusion layers (dino_norm, dino_proj, ...) are trained.
            dino_filter = nnx_utils.PathRegex("dino/.*")
            freeze = nnx.Any(freeze, dino_filter) if filters else dino_filter
        return freeze

    @override
    def get_lr_mult(self) -> tuple[nnx.filterlib.Filter, float] | None:
        if self.dino_fusion != "expert_stream":
            return None
        # All DINO fusion params ("dino_norm", "dino_proj", "dino_gate", ...), but not the encoder ("dino/...").
        return nnx_utils.PathRegex("dino_.*"), self.dino_lr_mult
