#!/bin/bash

# LEGACY ranking-crop protocol, not the active full-image CINE benchmark.

#SBATCH --job-name=sdum003-eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-003/logs/eval-%j.out
#SBATCH --error=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-003/logs/eval-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/students/studxuzho1/envs/nv-raw2insights-mri

PREP_REPO=/home/students/studxuzho1/NV-Raw2insights-MRI
SDUM_REPO=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon
PYTHON_BIN=/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python

cd "$PREP_REPO"

"$PYTHON_BIN" scripts/evaluate_sdum_paired_ranking.py \
    --manifest "$SDUM_REPO/runs/sdum-003/cohort_manifest.json" \
    --ground-truth-dir "$SDUM_REPO/runs/sdum-002/ground_truth" \
    --run sdum-002="$SDUM_REPO/runs/sdum-002/output/val_img4ranking" \
    --run sdum-003="$SDUM_REPO/runs/sdum-003/output/val_img4ranking" \
    --filetype cine_lax \
    -o "$SDUM_REPO/runs/sdum-003/output/evaluation_ranking_paired"
