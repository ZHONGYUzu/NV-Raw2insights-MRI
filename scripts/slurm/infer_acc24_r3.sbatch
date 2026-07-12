#!/bin/bash

#SBATCH --job-name=cine_test_acc24
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/cine_test_acc24_%j.out
#SBATCH --error=logs/cine_test_acc24_%j.err

set -euo pipefail

REPO_ROOT=/home/students/studxuzho1/NV-Raw2insights-MRI
CONFIG=configs/nv_raw2insights_mri_base.json
INPUT_JSON=dataset/CINE_test_acc24/json_input
OUTPUT_ROOT=output/CINE_test_acc24

mkdir -p logs

cd "$REPO_ROOT"

source ~/.bashrc
conda activate nv-raw2insights-mri

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Repository: $REPO_ROOT"
echo "Config: $CONFIG"
echo "Input JSON: $INPUT_JSON"
echo "Output root: $OUTPUT_ROOT"

nvidia-smi

python scripts/validate_cine_inference_data.py "$INPUT_JSON"

python scripts/inference.py \
  -c "$CONFIG" \
  -i "$INPUT_JSON" \
  -o "$OUTPUT_ROOT" \
  --profile-timing

echo "Finished: $(date)"
