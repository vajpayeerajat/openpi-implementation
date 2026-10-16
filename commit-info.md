Stage 1 is ready: a new config **`pi05_g1_dex3_statenoise`** adds noise (σ = 0.03 rad) to the arm joints of the robot state during training only, and is otherwise identical to stage 0. The checks passed, but nothing has been trained with the real model yet.

**What the checks showed** (CPU only, capped):
- The noise lands only on the arm joints, at the right size. Hands and targets are untouched.
- Arm targets are computed against the noisy state, as planned.
- The noise is used only when training. Validation, the norm-stats path and the robot policy server all see the clean state.
- The training data loader runs with its worker processes.

A 3-step test run of `train.py` with the tiny model hit my 12 GB memory cap and was stopped. So the full `train.py` loop with noise hasn't run end to end. The only change on that path is one line that turns the noise on, and the loader underneath it was checked separately.

**What changed** (91 lines across 5 files, not committed yet):
- `transforms.py`: the new noise transform (`AddStateNoise`).
- `config.py`: a "training only" slot in the data config, a `state_noise_std` setting on the G1 data config (default 0, so stage 0 is unchanged), and the new config. The new config is built as a copy of `pi05_g1_dex3` with only the noise and its name changed, so it automatically picks up any later edits to `pi05_g1_dex3`. It reuses stage 0's norm stats.
- `data_loader.py` and `train.py`: only the training loader turns the noise on.
- `g1_dex3_policy.py`: names the arm-joint mask (`ARM_MASK`), so the noise and the delta actions share it.

**Steps to start training:**
1. **Commit, so the training machine gets the code:**
   ```bash
   git checkout -b noise_in_action_openpi
   git add src/openpi/transforms.py src/openpi/training/config.py src/openpi/training/data_loader.py src/openpi/policies/g1_dex3_policy.py scripts/train.py
   git commit -m "Stage 1: training-only state noise on arm joints (pi05_g1_dex3_statenoise)"
   git push -u origin noise_in_action_openpi
   ```
   Then on the training machine: `git fetch && git checkout noise_in_action_openpi`. It needs `assets/pi05_g1_dex3/splitted_lerobot_and_school/train/norm_stats.json`, which is already in the repo.
2. **Train with the same command as stage 0**, changing only the config name and run name. Keep every override you used for stage 0 (batch size, steps, and so on):
   ```bash
   export HF_LEROBOT_HOME=/mnt/drive2/rajat_ws/datasets
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_g1_dex3_statenoise \
       --exp-name=stage1_noise003 --overwrite --log-interval=50 --keep-period=25000
   ```
3. **Monitor in TensorBoard.** Expect train loss to sit higher than stage 0, because the model can't predict the noise. Compare `val/loss`, which is noise-free.
4. **Evaluate with `eval_instructions.md` against stage 0's best checkpoint.** The key sign of success: removing the cameras in the Grad-CAM analysis should now clearly raise the loss.

To try a different noise level later, add one line next to the existing one in `config.py`, e.g. `_g1_dex3_with_state_noise("pi05_g1_dex3_statenoise006", std=0.06)`.