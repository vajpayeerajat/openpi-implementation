# Fine-tuning pi0.5 on the Unitree G1 + Dex3 dataset


to download the repo - 
git clone <>
then 
git submodule update --init


Configs: `pi05_g1_dex3` (full fine-tune) and `pi05_g1_dex3_lora` (LoRA, low memory).
Code: `src/openpi/policies/g1_dex3_policy.py`, `LeRobotG1Dex3DataConfig` in `src/openpi/training/config.py`.

## Dataset

`/mnt/drive2/rajat_ws/datasets/splitted_lerobot_and_school/{train,val,test}`. Each split is a standalone LeRobot
v2.1 dataset. Only `train` is used for training; `val` and `test` are for offline evaluation.

| split | episodes | frames | tasks |
|---|---|---|---|
| train | 1548 | 557,363 | 833 |
| val | 68 | 35,895 | 8 |
| test | 68 | 36,484 | 7 |

* 20 fps. Cameras: `ego_view` (head), `ego_left` and `ego_right` (wrists), h264 640x480.
* `observation.state` / `action`: 28-dim absolute joint positions, `[L_arm 7, L_hand 7, R_arm 7, R_hand 7]`
  (order in `g1_dex3_policy.G1_JOINTS`).
* Prompt comes from the LeRobot task string (`prompt_from_task=True`).

## Design decisions

* **Action space:** arms are trained as deltas from the current state, hands stay absolute:
  `make_bool_mask(7, -7, 7, -7)`, *not* `(14, -14)`. The policy output is converted back to absolute joint
  targets, so the robot always receives absolute positions.
* **`action_horizon=50`**, which is 2.5 s at 20 fps.
* **`max_token_len=256`.** pi0.5 sees the state only as a discretized string in the prompt. The longest
  5-object prompts reach 197 tokens, too close to the default of 200 (truncation only logs a warning).
* **Norm stats are computed with `scripts/compute_norm_stats_g1_dex3.py`**, not `compute_norm_stats.py`:
  * Speed: it reads only the parquet files, about 40 s instead of more than 10 h of video decoding.
  * Degenerate dims: the left hand is still in 99.5% of frames, so q01 == q99 and quantile normalization
    pushed its rare active frames to about 3e6. Those dims fall back to min/max.
    All normalized values now lie within ±10.

## Steps

All commands run from the repo root.

```bash
export HF_LEROBOT_HOME=/mnt/drive2/rajat_ws/datasets   # never set LEROBOT_HOME; lerobot refuses to import

# 1. Norm stats -> assets/pi05_g1_dex3/splitted_lerobot_and_school/train/norm_stats.json (already computed)
uv run scripts/compute_norm_stats_g1_dex3.py --config-name pi05_g1_dex3   # pi05_g1_dex3_lora reuses these

# 2. Train (full fine-tune; needs about 70+ GB of free GPU memory at batch 32)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_g1_dex3 --exp-name=run1
#    Not enough memory: use pi05_g1_dex3_lora, or lower --batch-size.
#    Resume: add --resume. The G1 configs log to TensorBoard only; add --wandb-enabled to also use W&B.
#    TensorBoard logs go to checkpoints/pi05_g1_dex3/run1/tensorboard:
uvx tensorboard --logdir checkpoints/pi05_g1_dex3 --port 6006
#    val/loss: flow-matching loss on a fixed 640-frame subset of the val split, every 500 steps
#    (--val-interval, --val-num-batches). Uses the EMA weights, i.e. what gets checkpointed, so it lags
#    train/loss early on. Same rng and samples every time, so points are directly comparable.
#    Train and val loss on one chart: "Custom Scalars" tab -> loss -> "train vs val".

# 3a. Pass/fail checkpoint test before using it on the robot (exit code 1 on failure): loads, output shape,
#     43- vs 28-dim state, finite actions, correct action space, actions inside the training joint range
uv run scripts/test_g1_dex3_checkpoint.py --checkpoint-dir checkpoints/pi05_g1_dex3/run1/49999

##### demo run
export HF_LEROBOT_HOME=/mnt/drive2/rajat_ws/datasets
uv run scripts/test_g1_dex3_checkpoint.py --config-name pi05_g1_dex3_debug \
    --checkpoint-dir checkpoints/pi05_g1_dex3_debug/sample/9

# Your real runs use the default config name:
uv run scripts/test_g1_dex3_checkpoint.py --checkpoint-dir checkpoints/pi05_g1_dex3/run1/<step>


# 3b. Offline eval on the held-out splits (MAE in radians, per joint group and horizon)
uv run scripts/eval_g1_dex3_offline.py --checkpoint-dir checkpoints/pi05_g1_dex3/run1/49999 --split val
uv run scripts/eval_g1_dex3_offline.py --checkpoint-dir checkpoints/pi05_g1_dex3/run1/49999 --split test

# 4. Serve
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_g1_dex3 \
    --policy.dir=checkpoints/pi05_g1_dex3/run1/49999
```

## Robot client contract

The client sends this observation (see `g1_dex3_policy.make_g1_dex3_example`):

```python
{
    "observation/ego_view":  uint8 [480, 640, 3],
    "observation/ego_left":  uint8 [480, 640, 3],
    "observation/ego_right": uint8 [480, 640, 3],
    "observation/state":     float [28]  # or the full 43-dim whole-body vector; dims 15..42 are used
    "prompt":                str,
}
```

`client.infer(obs)["actions"]` is `[50, 28]` **absolute** joint targets in `G1_JOINTS` order.
