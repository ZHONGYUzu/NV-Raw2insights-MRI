#!/bin/bash

#SBATCH --job-name=sdum008-eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-008/eval-%j.out
#SBATCH --error=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-008/eval-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon
PYTHON_BIN=/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python
RUN_DIR="$REPO_ROOT/runs/sdum-008"
EVAL_DIR="$RUN_DIR/output/evaluation_full_direct"

cd "$REPO_ROOT"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Protocol: full-size direct-scale; frame-level PSNR, SSIM, NMSE"

if [[ -e "$EVAL_DIR/summary_metrics.json" ]]; then
    echo "Refusing to overwrite existing evaluation: $EVAL_DIR" >&2
    exit 1
fi

"$PYTHON_BIN" scripts/evaluate_cmrxrecon2023_cohort.py \
    --manifest "$RUN_DIR/cohort_manifest.json" \
    --prediction-dir "$RUN_DIR/output/val_img_full" \
    -o "$EVAL_DIR"

"$PYTHON_BIN" -c "import json; p='$EVAL_DIR/summary_metrics.json'; d=json.load(open(p)); assert d['num_cases']==15, d['num_cases']; assert d['num_frames']==528, d['num_frames']; print(json.dumps(d['aggregate_frame_metrics'], indent=2))"

echo "sdum-008 evaluation completed successfully."
