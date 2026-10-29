#!/usr/bin/env bash
# Downloads the G1 Dex3 dataset from the Hugging Face Hub (if needed) and trains a DINO expert-stream config.
#
# Usage (from anywhere; needs `huggingface-cli login` once, for the dataset, the DINOv3 weights and the upload):
#   bash scripts/train_dino_expert.sh                      # all cameras
#   CONFIG=pi05_g1_dex3_dino_expert_wrist EXP_NAME=dino_expert_wrist \
#       HUB_REPO_ID=rajat-vajpayee/pi05-g1-dex3-dino-expert-wrist bash scripts/train_dino_expert.sh
#   bash scripts/train_dino_expert.sh --batch-size=16      # extra args go to scripts/train.py
#
# Settings (environment variables):
#   CONFIG            train config                      (default: pi05_g1_dex3_dino_expert)
#   EXP_NAME          experiment name                   (default: dino_expert)
#   DATA_ROOT         HF_LEROBOT_HOME for the dataset   (default: $HOME/hf_lerobot)
#   DATASET_REPO      HF dataset repo                   (default: rajat-vajpayee/schoool_plus_830_dataset)
#   HUB_REPO_ID       HF model repo for checkpoints; empty = keep them local only
#                                                       (default: rajat-vajpayee/pi05-g1-dex3-dino-expert)
#   SAVE_TRAIN_STATE  1 = full checkpoints (can --resume), 0 = weights only (less disk / host RAM) (default: 0)
#   RESUME            1 = continue the run EXP_NAME (needs SAVE_TRAIN_STATE=1), 0 = start over (default: 0)
#   NUM_WORKERS       data loader workers               (default: 4)
set -euo pipefail

CONFIG="${CONFIG:-pi05_g1_dex3_dino_expert}"
EXP_NAME="${EXP_NAME:-dino_expert}"
DATA_ROOT="${DATA_ROOT:-$HOME/hf_lerobot}"
DATASET_REPO="${DATASET_REPO:-rajat-vajpayee/schoool_plus_830_dataset}"
HUB_REPO_ID="${HUB_REPO_ID-rajat-vajpayee/pi05-g1-dex3-dino-expert}"
SAVE_TRAIN_STATE="${SAVE_TRAIN_STATE:-0}"
RESUME="${RESUME:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# The train configs read "splitted_lerobot_and_school/{train,val}" relative to HF_LEROBOT_HOME.
DATASET_DIR="$DATA_ROOT/splitted_lerobot_and_school"

cd "$(dirname "$0")/.."
log() { echo -e "\n=== $* ==="; }

log "Python environment"
GIT_LFS_SKIP_SMUDGE=1 uv sync

log "Hugging Face login"
if ! uv run huggingface-cli whoami >/dev/null 2>&1; then
    echo "Not logged in to Hugging Face. Run: uv run huggingface-cli login" >&2
    exit 1
fi
uv run huggingface-cli whoami

log "FFmpeg (needed by torchcodec to decode the videos)"
if [ -d "$HOME/ffmpeg/lib" ]; then
    export LD_LIBRARY_PATH="$HOME/ffmpeg/lib:${LD_LIBRARY_PATH:-}"
fi
if ! uv run python -c "import torchcodec" >/dev/null 2>&1; then
    echo "torchcodec can't load FFmpeg; installing FFmpeg 6 into ~/ffmpeg with micromamba (no sudo)."
    if [ ! -x "$HOME/bin/micromamba" ]; then
        arch="$(uname -m)"; [ "$arch" = "x86_64" ] && arch=64
        (cd "$HOME" && curl -Ls "https://micro.mamba.pm/api/micromamba/linux-$arch/latest" | tar -xj bin/micromamba)
    fi
    "$HOME/bin/micromamba" create -y -p "$HOME/ffmpeg" -c conda-forge "ffmpeg=6"
    export LD_LIBRARY_PATH="$HOME/ffmpeg/lib:${LD_LIBRARY_PATH:-}"
    uv run python -c "import torchcodec" || { echo "torchcodec still can't load FFmpeg" >&2; exit 1; }
fi
echo "ok"

log "Dataset $DATASET_REPO -> $DATASET_DIR"
# Skips files that are already there, so re-running only fetches what is missing.
uv run huggingface-cli download "$DATASET_REPO" --repo-type dataset --local-dir "$DATASET_DIR"
export HF_LEROBOT_HOME="$DATA_ROOT"
# LeRobot falls back to the Hub (and fails) if any episode file is missing, so check before training.
uv run python - <<'EOF'
from lerobot.common.datasets import lerobot_dataset

for split in ("train", "val"):
    meta = lerobot_dataset.LeRobotDatasetMetadata(f"splitted_lerobot_and_school/{split}")
    paths = [meta.get_data_file_path(ep) for ep in range(meta.total_episodes)]
    paths += [meta.get_video_file_path(ep, key) for ep in range(meta.total_episodes) for key in meta.video_keys]
    missing = [str(p) for p in paths if not (meta.root / p).is_file()]
    print(f"{split}: {meta.total_episodes} episodes, {meta.total_frames} frames, {len(missing)} missing files")
    if missing:
        raise SystemExit(f"Missing files in {meta.root}, e.g. {missing[:3]}")
EOF

log "Training $CONFIG ($EXP_NAME)"
args=(--exp-name="$EXP_NAME" --num-workers="$NUM_WORKERS" --log-interval=50)
if [ "$RESUME" = "1" ]; then args+=(--resume); else args+=(--overwrite); fi
if [ "$SAVE_TRAIN_STATE" != "1" ]; then args+=(--no-save-train-state); fi
if [ -n "$HUB_REPO_ID" ]; then args+=(--hub-repo-id="$HUB_REPO_ID" --hub-tensorboard-interval=50); fi
echo "uv run scripts/train.py $CONFIG ${args[*]} $*"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "$CONFIG" "${args[@]}" "$@" 2>&1 | tee "train_${EXP_NAME}.log"
