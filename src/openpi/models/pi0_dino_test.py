"""Tests for the DINOv3 fusion modes of pi0/pi05 (`Pi0Config.dino_fusion`), on tiny Gemma/DINO configs (CPU)."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import optimizer as _optimizer

BATCH_SIZE = 2
NUM_STEPS = 3


def _config(**kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        max_token_len=16,
        dtype="float32",
        **kwargs,
    )


_EXPERT = {"dino_variant": "dummy", "dino_fusion": "expert_stream"}


def _obs_and_actions(config: pi0_config.Pi0Config, seed: int = 0) -> tuple[_model.Observation, _model.Actions]:
    obs, actions = config.fake_obs(BATCH_SIZE), config.fake_act(BATCH_SIZE)
    keys = jax.random.split(jax.random.key(seed), len(obs.images) + 2)
    images = {
        name: jax.random.uniform(key, image.shape, minval=-1.0, maxval=1.0)
        for key, (name, image) in zip(keys, obs.images.items(), strict=False)
    }
    # mask out one camera of one example, so masking is exercised too.
    image_masks = {**obs.image_masks, "right_wrist_0_rgb": jnp.array([True, False])}
    state = jax.random.normal(keys[-2], obs.state.shape)
    actions = jax.random.normal(keys[-1], actions.shape)
    return dataclasses.replace(obs, images=images, image_masks=image_masks, state=state), actions


def _create(config: pi0_config.Pi0Config, *, gates: float = 0.0):
    """Model with all (non-encoder) params perturbed, so that zero-initialized parts like adaRMS are exercised."""
    model = config.create(jax.random.key(0))
    params = nnx.state(model, nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex("(PaliGemma/img|dino)/.*"))))
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(1), len(leaves))
    leaves = [x + 0.05 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys, strict=True)]
    nnx.update(model, jax.tree.unflatten(treedef, leaves))
    if config.dino_fusion == "expert_stream":
        model.dino_gate.value = jnp.full_like(model.dino_gate.value, gates)
        model.dino_attn_gate.value = jnp.full_like(model.dino_attn_gate.value, gates)
    return model


def _baseline_with_same_weights(model) -> nnx.Module:
    """The model without DINO, with the shared weights of `model`."""
    baseline = _config().create(jax.random.key(0))
    nnx.update(baseline, nnx.state(model, nnx.Not(nnx_utils.PathRegex("dino.*"))))
    return baseline


def _loss_and_actions(model, obs, actions):
    loss = nnx_utils.module_jit(model.compute_loss)(jax.random.key(2), obs, actions)
    noise = jax.random.normal(jax.random.key(3), actions.shape)
    sampled = nnx_utils.module_jit(model.sample_actions)(jax.random.key(4), obs, num_steps=NUM_STEPS, noise=noise)
    return loss, sampled


def test_identity_at_init():
    config = _config(**_EXPERT)
    model = _create(config, gates=0.0)
    obs, actions = _obs_and_actions(config)

    loss, sampled = _loss_and_actions(model, obs, actions)
    base_loss, base_sampled = _loss_and_actions(_baseline_with_same_weights(model), obs, actions)
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sampled, base_sampled, rtol=1e-6, atol=1e-6)

    # sanity check: with open gates the DINO branch does change the outputs.
    loss, sampled = _loss_and_actions(_create(config, gates=0.5), obs, actions)
    assert np.abs(np.asarray(loss) - np.asarray(base_loss)).max() > 1e-3
    assert np.abs(np.asarray(sampled) - np.asarray(base_sampled)).max() > 1e-3


def test_no_leakage_into_prefix():
    config = _config(**_EXPERT)
    obs, actions = _obs_and_actions(config)

    @nnx.jit
    def prefix_outputs(model):
        # Only the valid prefix tokens are compared. A padding token (here the masked camera) attends to nothing, so
        # its softmax is uniform over all keys, DINO included; but no other token attends to it.
        prefix_out, _, _ = model._expert_stream_forward(obs, actions, jnp.full((BATCH_SIZE,), 0.5))  # noqa: SLF001
        prefix_mask = model.embed_prefix(obs)[1]
        prefix_out = jnp.where(prefix_mask[..., None], prefix_out, 0.0)
        kv_cache, _, _, num_dino = model._fill_kv_cache(obs)  # noqa: SLF001
        kv_cache = jax.tree.map(
            lambda x: jnp.where(prefix_mask[None, :, :, None, None], x[:, :, : x.shape[2] - num_dino], 0.0), kv_cache
        )
        return prefix_out, kv_cache

    model = _create(config, gates=0.5)
    out_a = prefix_outputs(model)
    # different DINO tokens: other projection, gate and camera embedding.
    model.dino_proj.kernel.value = -2.0 * model.dino_proj.kernel.value
    model.dino_gate.value = jnp.full_like(model.dino_gate.value, -1.0)
    model.dino_cam_emb.value = model.dino_cam_emb.value + 1.0
    out_b = prefix_outputs(model)
    for a, b in zip(jax.tree.leaves(out_a), jax.tree.leaves(out_b), strict=True):
        np.testing.assert_array_equal(a, b)


def test_gradient_flows():
    config = _config(**_EXPERT)
    model = _create(config, gates=0.0)
    obs, actions = _obs_and_actions(config)
    trainable = nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))
    tx = _optimizer.create_optimizer(
        _optimizer.AdamW(), _optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=1e-2), lr_mult=config.get_lr_mult()
    )
    opt_state = tx.init(nnx.state(model, trainable))
    encoder_before = nnx.state(model, nnx_utils.PathRegex("dino/.*"))

    @nnx.jit
    def train_step(model, opt_state):
        def loss_fn(model):
            return jnp.mean(model.compute_loss(jax.random.key(5), obs, actions, train=True))

        grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, trainable))(model)
        intermediates = nnx.pop(model, nnx.Intermediate)
        params = nnx.state(model, trainable)
        updates, opt_state = tx.update(grads, opt_state, params)
        nnx.update(model, jax.tree.map(lambda p, u: p + u, params, updates))
        return grads, opt_state, intermediates

    def norm(grads, name):
        return float(optax.global_norm(grads[name]))

    # the encoder is frozen: it gets no gradient at all.
    grads, opt_state, intermediates = train_step(model, opt_state)
    assert "dino" not in grads
    assert {"dino_token_rms", "expert_input_rms"} <= set(intermediates)
    # both gates start at zero: only the attention gate gets gradient at first, then the token gate, then the rest.
    assert norm(grads, "dino_attn_gate") > 0
    for _ in range(2):
        grads, opt_state, _ = train_step(model, opt_state)
    for name in ("dino_attn_gate", "dino_gate", "dino_proj", "dino_norm", "dino_cam_emb", "dino_ctx"):
        assert norm(grads, name) > 0, name
    for a, b in zip(
        jax.tree.leaves(encoder_before), jax.tree.leaves(nnx.state(model, nnx_utils.PathRegex("dino/.*"))), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_attn_gate_init():
    """With both gates at zero the token gate gets no gradient at step 0; an open attention gate gives it some."""
    obs, actions = _obs_and_actions(_config(**_EXPERT))

    def token_gate_grad(attn_gate_init):
        config = _config(**_EXPERT, dino_attn_gate_init=attn_gate_init)
        model = _create(config)
        # _create zeroes both gates; restore the configured init of the attention gate.
        model.dino_attn_gate.value = config.create(jax.random.key(0)).dino_attn_gate.value
        np.testing.assert_allclose(jnp.tanh(model.dino_attn_gate.value), attn_gate_init, rtol=1e-6)
        trainable = nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))

        @nnx.jit
        def grads(model):
            return nnx.grad(
                lambda m: jnp.mean(m.compute_loss(jax.random.key(5), obs, actions)), argnums=nnx.DiffState(0, trainable)
            )(model)

        return float(optax.global_norm(grads(model)["dino_gate"]))

    assert token_gate_grad(0.0) == 0.0
    assert token_gate_grad(0.2) > 0.0


@pytest.mark.parametrize("dino_cameras", [None, ("left_wrist_0_rgb", "right_wrist_0_rgb")])
def test_cache_consistency(dino_cameras):
    config = _config(**_EXPERT, dino_cameras=dino_cameras, dino_layers=(0, -1))
    model = _create(config, gates=0.5)
    obs, actions = _obs_and_actions(config)
    noise = jax.random.normal(jax.random.key(3), actions.shape)

    cached = nnx_utils.module_jit(model.sample_actions)(jax.random.key(4), obs, num_steps=NUM_STEPS, noise=noise)

    @nnx.jit
    def uncached(model):
        # the same Euler steps, with every step one full prefix + DINO + suffix pass (the training layout).
        x_t, time, dt = noise, 1.0, -1.0 / NUM_STEPS
        for _ in range(NUM_STEPS):
            _, v_t, _ = model._expert_stream_forward(obs, x_t, jnp.full((BATCH_SIZE,), time))  # noqa: SLF001
            x_t, time = x_t + dt * v_t, time + dt
        return x_t

    np.testing.assert_allclose(cached, uncached(model), rtol=1e-4, atol=1e-4)


def test_lr_mult():
    config = _config(**_EXPERT)
    model = _create(config, gates=0.5)
    params = nnx.state(model, nnx.Param)
    grads = jax.tree.map(jnp.ones_like, params)
    tx = _optimizer.create_optimizer(_optimizer.SGD(), _optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=1.0))
    tx_mult = _optimizer.create_optimizer(
        _optimizer.SGD(), _optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=1.0), lr_mult=config.get_lr_mult()
    )
    updates, _ = tx.update(grads, tx.init(params), params)
    updates_mult, _ = tx_mult.update(grads, tx_mult.init(params), params)
    for path, u in updates.flat_state().items():
        mult = 10.0 if path[0].startswith("dino_") else 1.0
        np.testing.assert_allclose(dict(updates_mult.flat_state())[path].value, mult * u.value)


def test_backward_compatibility():
    """ "none" and "prefix_add" keep their param trees, freeze filters and (no) lr multiplier.

    Their outputs were also checked to be bit-identical to the code before "expert_stream" was added.
    """

    def param_roots(config):
        model = nnx.eval_shape(config.create, jax.random.key(0))
        return {path[0] for path, _ in nnx.state(model, nnx.Param).flat_state().items()}

    base_roots = param_roots(_config())
    assert not any(r.startswith("dino") for r in base_roots)
    # setting only `dino_variant` is Option A, as before.
    assert _config(dino_variant="dummy").dino_fusion == "prefix_add"
    assert param_roots(_config(dino_variant="dummy")) == base_roots | {"dino", "dino_norm", "dino_proj"}
    assert param_roots(_config(**_EXPERT)) - base_roots == {
        "dino",
        "dino_norm",
        "dino_proj",
        "dino_cam_emb",
        "dino_gate",
        "dino_attn_gate",
        "dino_ctx",
    }
    assert _config().get_lr_mult() is None
    assert _config(dino_variant="dummy").get_lr_mult() is None

    def frozen(config):
        model = nnx.eval_shape(config.create, jax.random.key(0))
        return {
            path[0] for path, _ in nnx.state(model, nnx.All(nnx.Param, config.get_freeze_filter())).flat_state().items()
        }

    assert frozen(_config(dino_variant="dummy")) == {"dino"}
    assert frozen(_config(**_EXPERT)) == {"dino"}
    assert frozen(_config(**_EXPERT, dino_freeze_encoder=False)) == set()
