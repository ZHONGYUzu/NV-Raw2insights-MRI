#!/bin/bash

# FT-00 pipeline smoke test:
#   - official NV-Raw2Insights-MRI Small checkpoint
#   - Sub0001 training / Sub0002 validation
#   - fixed VISTA acc8 mask
#   - one epoch, two sampled training windows

#SBATCH --job-name=ft00_small_cine_a8
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ft00_small_cine_a8_%j.out
#SBATCH --error=logs/ft00_small_cine_a8_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export HF_HOME=/home/students/studxuzho1/hf_cache

REPO_ROOT=/home/students/studxuzho1/NV-Raw2insights-MRI
PYTHON_BIN=/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python
SOURCE_JSON_ROOT=/home/students/studxuzho1/dataset_v1/json_input
SPLIT_ROOT=dataset/finetune/FT00_CINE_acc8
CONFIG=configs/nv_raw2insights_mri_small_cine_acc8_ft00.json
EXP_ROOT=exp/ft00_small_cine_acc8_smoke

cd "$REPO_ROOT"
mkdir -p logs "$SPLIT_ROOT/train" "$SPLIT_ROOT/val"

install -m 0644 \
  "$SOURCE_JSON_ROOT/Sub0001.json" \
  "$SPLIT_ROOT/train/Sub0001_cine.json"
install -m 0644 \
  "$SOURCE_JSON_ROOT/Sub0002.json" \
  "$SPLIT_ROOT/val/Sub0002_cine.json"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG"
echo "Train JSON: $SPLIT_ROOT/train/Sub0001_cine.json"
echo "Val JSON: $SPLIT_ROOT/val/Sub0002_cine.json"

if [[ -e "$EXP_ROOT/ft00_small_cine_acc8_smoke.pt" ]]; then
  echo "Refusing to auto-resume an existing FT-00 run: $EXP_ROOT" >&2
  exit 2
fi

nvidia-smi
"$PYTHON_BIN" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
"$PYTHON_BIN" scripts/validate_cine_inference_data.py "$SPLIT_ROOT/train"
"$PYTHON_BIN" scripts/validate_cine_inference_data.py "$SPLIT_ROOT/val"
"$PYTHON_BIN" scripts/train.py --config "$CONFIG"

echo "Finish: $(date --iso-8601=seconds)"
