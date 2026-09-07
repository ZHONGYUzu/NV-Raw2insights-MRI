#!/bin/bash

#SBATCH --job-name=sdum010-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-010/debug-%j.out
#SBATCH --error=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-010/debug-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/home/students/studxuzho1/hf_cache

REPO_ROOT=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon
PYTHON_BIN=/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python
RUN_DIR="$REPO_ROOT/runs/sdum-010"

cd "$REPO_ROOT"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Run: sdum-010 debug"
echo "Physical mask: NV training generator ktRadial8, 20x20 ACS"
echo "Conditioning: ktRadial8 / acceleration 8"

"$PYTHON_BIN" scripts/validate_cine_inference_data.py "$RUN_DIR/json_input"

"$PYTHON_BIN" scripts/inference.py \
    -c configs/nv_raw2insights_mri_base.json \
    -i "$RUN_DIR/json_input" \
    -o "$RUN_DIR/debug_output" \
    --debug \
    --profile-timing

"$PYTHON_BIN" -c "from pathlib import Path; p=Path('$RUN_DIR/debug_output/val_img4ranking/P006_cine_lax.mat'); assert p.is_file(), p; print('Debug output:', p)"

echo "sdum-010 debug completed successfully."
