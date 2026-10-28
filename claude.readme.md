# Task: Implement "Option C" — DINOv3 features injected into the π0.5 action-expert stream (openpi, JAX/Flax NNX)

## Context
This is a fork of Physical Intelligence's openpi. I already have an experimental DINOv3 branch
that ADDS projected DINO features to the SigLIP image tokens before Gemma ("Option A"):

```python
self.use_dino = config.dino_variant is not None
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
```

On hardware this gave no measurable change on pick-and-place. I want to test a different
fusion site: DINO tokens live in the ACTION EXPERT's token stream (processed by the expert's
weights, i.e. the second/"llm_1" expert of the Gemma mixture). They are visible to action tokens
at every layer through the shared attention, and invisible to the VLM prefix.
Goal: better grasp/place precision and OOD robustness for pick-and-place, without touching the
VLM's input distribution and with low added inference latency.

## Phase 1 — Investigate and report BEFORE writing code
Read the code and give me a short written report covering each item below, citing file paths
and line numbers. Do not guess APIs. If something differs from my assumptions, say so.

1. Model files: where the π0 / π0.5 model is defined (embed_prefix, embed_suffix,
   compute_loss, sample_actions), the Gemma mixture-of-experts module, attention-mask
   construction (ar_mask / make_attn_mask or equivalent), and how the KV cache is filled
   and reused during sampling.
2. **Stop-gradient check (critical):** does the training path apply stop_gradient to the
   prefix/backbone keys/values (knowledge-insulation style), or to any prefix outputs?
   Does fine-tuning use only the flow-matching loss, or also a FAST/CE loss?
   Tell me explicitly whether my existing Option-A `dino_proj` can receive gradient from
   the training loss in the current config. Also add a one-off check that prints the norm of
   `dino_proj.kernel` from my latest checkpoint if one is provided, or explain how I can run it.
3. **adaRMS / time conditioning:** in π0.5, how is the flow timestep injected into the
   action expert (adaptive RMSNorm cond, time MLP, etc.)? Is the conditioning per-expert-stream
   or per-token? This determines whether DINO tokens in the expert stream would depend on the
   timestep t and therefore could or could not be cached across denoising steps.
4. How positions / RoPE indices are assigned to prefix vs suffix tokens.
5. How images and cameras are iterated (base, wrist(s)), the image resolution fed to SigLIP,
   and what the DINOv3 module returns (patch tokens? CLS/register tokens? shape?).
6. Where trainable/frozen parameter filters and optimizer param groups are defined
   (freeze filters, LoRA filters, optax setup), and how I could give new params a
   different learning rate.

End Phase 1 with a concrete implementation plan (files to change, new config fields,
mask layout diagram, caching strategy). Then STOP and wait for my "go".

## Phase 2 — Implementation spec (after I approve the plan)

### Config
Add `dino_fusion: Literal["none", "prefix_add", "expert_stream"]` (default "none"),
keeping the existing Option-A behaviour reachable as "prefix_add". Additional fields:
- `dino_cameras: Sequence[str] | None` (None = all cameras; e.g. wrist only)
- `dino_pool: int = 2` (k×k average pooling of the DINO patch grid)
- `dino_layers: Sequence[int] | None` (None = final layer; otherwise average or concat of
  intermediate layers, if the DINOv3 module can expose them. If it can't, skip this and tell me)
- `dino_lr_mult: float = 10.0`
- `dino_freeze_encoder: bool = True`
Existing configs must behave exactly as before when `dino_fusion="none"`.

### Module (expert_stream)
- Frozen DINOv3 encoder. Patch tokens only (drop CLS/registers unless you find a reason).
  Apply DINO's own expected normalization/resolution. Check what the images look like after
  openpi preprocessing, and convert if needed.
- Per camera: reshape patches to a grid → k×k avg-pool → LayerNorm → Linear to
  **action-expert width** (not PaliGemma width). Use LeCun-normal init, plus a learned
  camera/type embedding.
- Gate: `tokens = tanh(gate) * proj(...)`, gate is a per-channel `nnx.Param` initialized to
  zeros. Do NOT also zero-init the projection (zero gate × zero weight = dead branch).
- Optionally (if the attention mask makes it necessary), give DINO tokens a validity mask
  matching the camera image masks.

### Attention layout (required semantics)
- Prefix (images + language) tokens: must NOT attend to DINO tokens.
- DINO tokens: attend to prefix and to each other (bidirectional within the block).
  They must NOT attend to state/action tokens.
- State/action tokens: attend to prefix, DINO tokens, and per the existing rules among themselves.
- DINO tokens are processed by the ACTION EXPERT weights.
Assign RoPE positions consistently between the training path and the sampling path.

### Timestep conditioning and caching
Based on your Phase-1 finding (item 3):
- If expert-stream conditioning would make DINO tokens depend on t: give the DINO block a
  fixed learned conditioning vector (a learned "context" embedding instead of the time
  embedding) so its K/V are t-independent. Then compute the DINO block ONCE per inference
  call together with the prefix KV-cache fill, and reuse it for all denoising steps.
- If that is impossible without invasive changes to the Gemma module, implement the simpler
  version (recompute DINO tokens with the action tokens each denoising step). Clearly
  report the cost, and leave a TODO for the cached version.
The training forward pass and the cached sampling pass must be numerically consistent
(see tests).

### Training
- DINO encoder frozen. New params (norm, proj, camera embedding, gate, and any learned
  conditioning vector) trained at `base_lr * dino_lr_mult` via an optax multi_transform
  (or the repo's existing mechanism). Existing freeze/LoRA filters must still work.
- Make sure no stop_gradient blocks the flow-loss gradient into the DINO branch in
  expert_stream mode. (Prefix-side insulation, if present, must stay as it is.)

### Logging
Log every N steps:
- `tanh(gate)` mean/abs-max
- RMS of the gated DINO tokens vs RMS of the action-expert stream at the input layer
- grad norm of the DINO-branch params

## Tests (pytest, CPU-friendly, tiny/dummy configs where possible)
1. **Identity at init:** with `dino_fusion="expert_stream"` and gate=0, the loss and the
   sampled actions (fixed rng/noise) equal the baseline model with the same weights
   (allclose). Note: DINO tokens add keys to action-token attention even at gate=0.
   Check whether exact equivalence holds. If it doesn't, find the cleanest way to make it
   hold (e.g. also gating the attention contribution, or masking DINO keys while the gate is
   exactly 0), or report the deviation size and why.
2. **No leakage into the VLM:** perturbing DINO inputs leaves the prefix hidden states /
   prefix KV cache bit-identical.
3. **Gradient flows:** after one optimizer step on random data, the gate and proj params have
   non-zero gradients. DINO encoder params have zero updates when frozen.
4. **Cache consistency:** actions from the cached sampling path match the uncached reference
   path (same noise) within tolerance.
5. **Backward compatibility:** `dino_fusion="none"` and `"prefix_add"` reproduce the previous
   outputs.

## Evaluation utilities
Add a small script that, given a checkpoint and a held-out set of episodes, reports:
- offline action MSE (per action dim and overall)
- **counterfactual sensitivity:** mean |Δaction| when DINO tokens are (a) zeroed, (b) replaced
  with DINO tokens from a random other frame, and (c) when gate is forced to 0.
  This tells me whether the policy actually uses DINO.
- added inference latency per control step (DINO forward + extra tokens), measured on GPU
  with warmup, for: baseline, expert_stream (all cams), expert_stream (wrist only)

## Constraints
- Minimal, surgical diffs. Follow repo style and typing. No new dependencies.
- Don't change the default behaviour of existing configs.
- Add new training configs (e.g. `pi05_<mytask>_dino_expert`, `pi05_<mytask>_dino_expert_wrist`)
  cloned from my existing task config. Ask me which base config to clone if unclear.
- After implementation: run the tests, show results, and summarize every changed file
  with a one-line rationale. List any assumption you had to make.