#!/bin/bash

#SBATCH --job-name=sdum-acc16-prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-acc16-prepare-%j.out
#SBATCH --error=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-acc16-prepare-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon
PYTHON_BIN=/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python

cd "$REPO_ROOT"

for run in sdum-011 sdum-012 sdum-013; do
    if [[ -e "runs/$run" ]]; then
        echo "Refusing to replace existing run: runs/$run" >&2
        exit 1
    fi
done

"$PYTHON_BIN" scripts/create_cmrxrecon2023_training_masks.py \
    --output-root runs/sdum-011 \
    --family Uniform \
    --acceleration 16 \
    --acs-lines 20

"$PYTHON_BIN" scripts/create_cmrxrecon2023_training_masks.py \
    --output-root runs/sdum-012 \
    --family ktGaussian \
    --acceleration 16 \
    --acs-lines 20 \
    --seed 20260823

"$PYTHON_BIN" scripts/create_cmrxrecon2023_training_masks.py \
    --output-root runs/sdum-013 \
    --family ktRadial \
    --acceleration 16 \
    --acs-lines 20

for run in sdum-011 sdum-012 sdum-013; do
    "$PYTHON_BIN" scripts/validate_cine_inference_data.py "runs/$run/json_input"
done

echo "Prepared and validated sdum-011, sdum-012, and sdum-013."
