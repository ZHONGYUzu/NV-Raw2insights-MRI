#!/bin/bash

#SBATCH --job-name=sdum16-debug
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-acc16-debug-%A_%a.out
#SBATCH --error=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-acc16-debug-%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/home/students/studxuzho1/hf_cache

REPO_ROOT=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon
PYTHON_BIN=/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python
RUNS=(sdum-011 sdum-012 sdum-013)
MASK_TYPES=(Uniform16 ktGaussian16 ktRadial16)
INDEX=${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array with indices 0-2}
RUN=${RUNS[$INDEX]}
MASK_TYPE=${MASK_TYPES[$INDEX]}
RUN_DIR="$REPO_ROOT/runs/$RUN"

cd "$REPO_ROOT"

echo "Job ID: ${SLURM_JOB_ID:-manual}_${INDEX}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Run: $RUN debug"
echo "Physical mask and conditioning: $MASK_TYPE / acceleration 16 / 20 ACS"

"$PYTHON_BIN" scripts/validate_cine_inference_data.py "$RUN_DIR/json_input"

"$PYTHON_BIN" scripts/inference.py \
    -c configs/nv_raw2insights_mri_base.json \
    -i "$RUN_DIR/json_input" \
    -o "$RUN_DIR/debug_output" \
    --debug \
    --profile-timing

"$PYTHON_BIN" -c "from pathlib import Path; p=Path('$RUN_DIR/debug_output/val_img4ranking/P006_cine_lax.mat'); assert p.is_file(), p; print('Debug output:', p)"

echo "$RUN debug completed successfully."
