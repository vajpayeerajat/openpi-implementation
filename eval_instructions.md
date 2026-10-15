# Evaluation protocol: G1 Dex3 pi0.5 checkpoints

Use the same protocol for every checkpoint and every run, so results are comparable.
Change one thing per training run.

```bash
cd /mnt/drive2/rajat_ws/tasks/openpi-implementation
export HF_LEROBOT_HOME=/mnt/drive2/rajat_ws/datasets
CKPT=checkpoints/pi05_g1_dex3/run1/20000      # the checkpoint under test
```

## Part A: offline (no robot)

| Step | Command | Pass / what to record |
|---|---|---|
| 1. Pick candidates | `uvx tensorboard --logdir checkpoints/pi05_g1_dex3` | Take the 2–3 steps with the lowest `val/loss`. Do not just take the last step. |
| 2. Sanity gate | `uv run scripts/test_g1_dex3_checkpoint.py --checkpoint-dir $CKPT` | Must print `ALL CHECKS PASSED`. If not, do not put it on the robot. |
| 3. Accuracy | `uv run scripts/eval_g1_dex3_offline.py --checkpoint-dir $CKPT --split test` | Record the `first_10_steps` MAE for `right_arm` and `right_hand`. |
| 4. Reliance and attention | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run rajat_gradcam_analysis.py --checkpoint $CKPT` | Record from `report.md`: loss change for **no cameras**, **no state**, **other instruction (both octopi visible)**, and the **target-object Grad-CAM lift**. |

Step 4 notes:
- The script is ported from `../openpi` but has **not been run in this repo yet**. Check its first run carefully.
- It loads the full model; run it when the GPU is free.
- Keep `--prompt_prefix` empty, so the model sees the training prompts.

Signs a new checkpoint is better than the previous one:
- removing the cameras **raises** the loss clearly (old 20000: about 0%);
- object lift above 1 (old: 0.77);
- right-arm skill below 0.78 and right-hand skill below 0.77;
- step 2 still passes.

## Part B: hardware (the robot)

### Fixed settings: keep these identical for every checkpoint

| Setting | Value |
|---|---|
| Steps executed per action chunk | ____ (e.g. 10). The same for every checkpoint. |
| Prompt text | Exactly as written below, copied, not retyped |
| Start pose | Arms at the usual rest pose, hand open |
| Repeats per scene | **5** (minimum 3) |
| Time limit per trial | ____ s (e.g. 60). Stop and score as a failure after that. |
| Lighting and room | Note any change in the notes column |

### Scenes

| ID | Scene setup | Prompt (exact) | Tests |
|---|---|---|---|
| S1 | Pink octopus on table, usual position; orange and brown baskets | `Pick up the pink octopus and place it inside the orange basket.` | Basic success |
| S2 | As S1, **baskets swapped** left↔right | `Pick up the pink octopus and place it inside the orange basket.` | Finds the basket visually |
| S3 | Pink octopus at an **unusual position** (e.g. far side or other edge of the table) | `Pick up the pink octopus and place it inside the orange basket.` | Uses vision, not a memorised path |
| S4 | Pink and blue octopus on table | `Pick up the pink octopus and place it inside the orange basket, then pick up the blue octopus and place it inside the brown basket.` | Follows instruction order |
| S5 | Same layout as S4 | `Pick up the blue octopus and place it inside the brown basket, then pick up the pink octopus and place it inside the orange basket.` | Order reversed |
| S6 | Pink octopus **and** a similar-coloured distractor (red or pink object) | `Pick up the pink octopus and place it inside the orange basket.` | Picks the right object, not just the right colour |
| S7 | **No octopus**, only the similar-coloured distractor | `Pick up the pink octopus and place it inside the orange basket.` | Should not pick (target missing) |

Notes on the scenes:
- For S6 and S7, use an object that is **not in the training data**. Apple, banana, penguin, unicorn and the octopi are all in the new train split, so with a model trained on it the apple is not a fair "unseen object" test.
- Swap the S4 and S5 order between repeats, so the result doesn't depend on which one you ran first.

### Scoring each trial (pick one)

| Code | Meaning |
|---|---|
| `OK` | Correct object into the correct basket (all steps, in order, for S4/S5) |
| `WRONG_OBJ` | Picked a different object than instructed (incl. the distractor) |
| `WRONG_BASKET` | Correct object, wrong basket |
| `WRONG_ORDER` | S4/S5: both done, but in the wrong order |
| `GRASP_FAIL` | Reached the object but failed to grasp (missed, or closed the hand too early, dropped) |
| `NO_ACTION` | Did not start, or froze |
| `TIMEOUT` | Still not finished at the time limit |
| `UNSAFE` | Stopped by the operator (collision, wild motion) |

For **S7** the correct outcome is `NO_ACTION`; record a pick of the distractor as `WRONG_OBJ`.

### Results sheet: one block per checkpoint

```
Checkpoint: ______________________   Date: ________   Operator: ________
Steps executed per chunk: ____   Time limit: ____ s

Scene | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | OK count | Notes
S1    |         |         |         |         |         |   /5     |
S2    |         |         |         |         |         |   /5     |
S3    |         |         |         |         |         |   /5     |
S4    |         |         |         |         |         |   /5     |
S5    |         |         |         |         |         |   /5     |
S6    |         |         |         |         |         |   /5     |
S7    |         |         |         |         |         |   /5     |  (OK = NO_ACTION)

Offline (Part A): right-arm skill ____  right-hand skill ____  no-cameras ____%  no-state ____%  object lift ____
```

### Reading the results

| Pattern | Meaning |
|---|---|
| S1 OK, S3 fails | Memorised path; vision is not used for the object's position |
| S2 fails | The basket is not found visually |
| S4 or S5 order wrong | The instruction is not followed |
| S6 picks the distractor | Picks by colour, not object identity |
| S7 picks anything | No "target missing" behaviour. Expected without target-absent training data; use a VLM presence check before running the policy. |
| Many `GRASP_FAIL` | Grasp timing (hand closing early); compare with the offline right-hand skill |

Run each scene 5 times and compare **counts**. With 5 trials per checkpoint, only a large gap is clearly real (about 4 out of 5, e.g. 5/5 vs 1/5). For smaller gaps, run more trials before deciding which checkpoint is better.
