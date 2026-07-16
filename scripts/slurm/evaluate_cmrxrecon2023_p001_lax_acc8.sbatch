#!/bin/bash

#SBATCH --job-name=eval_cmrx23_p001_a8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/eval_cmrx23_p001_a8_%j.out
#SBATCH --error=logs/eval_cmrx23_p001_a8_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=${REPO_ROOT:-/home/students/studxuzho1/NV-Raw2insights-MRI}
PYTHON_BIN=${PYTHON_BIN:-/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python}

FULL_KSPACE=/mnt/qdata/rawdata/CMRxRecon/CMRxRecon_2023_training_init/ChallengeData/MultiCoil/CINE/TrainingSet/FullSample/P001/cine_lax.mat
RECON=output/CMRxRecon2023CINELAXAcc8/val_img4ranking/P001_cine_lax.mat
EVALUATION_DIR=output/CMRxRecon2023CINELAXAcc8/evaluation

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$FULL_KSPACE" ]]; then
  echo "Full-sampled k-space not found: $FULL_KSPACE" >&2
  exit 2
fi
if [[ ! -f "$RECON" ]]; then
  echo "Reconstruction not found: $RECON" >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Full k-space: $FULL_KSPACE"
echo "Reconstruction: $RECON"
echo "Evaluation output: $EVALUATION_DIR"

"$PYTHON_BIN" scripts/evaluate_cmrxrecon2023_fullsample.py \
  --full-kspace "$FULL_KSPACE" \
  --recon "$RECON" \
  -o "$EVALUATION_DIR" \
  --slice 1 \
  --time 6

echo "Finished: $(date --iso-8601=seconds)"
