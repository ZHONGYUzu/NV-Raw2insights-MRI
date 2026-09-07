#!/bin/bash

#SBATCH --job-name=eval_vista_s8_all10
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_vista_s8_all10_%j.out
#SBATCH --error=logs/eval_vista_s8_all10_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=${REPO_ROOT:-/home/students/studxuzho1/NV-Raw2insights-MRI}
PYTHON_BIN=${PYTHON_BIN:-/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python}
H5_ROOT=${H5_ROOT:-/mnt/qdata/rawdata/CINE/2D_h5_compressed}
RUN_ROOT=${RUN_ROOT:-Results/raw_vista_seed8_acc8_16_24_all10_v1}

cd "$REPO_ROOT"
mkdir -p logs "$RUN_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Dataset: fixed 10-case in-house CINE H5 benchmark"
echo "Subjects: docs/test_subjects.txt"
echo "K-space/CSM: full H5 kSpace plus external H5 dMap"
echo "Masks: raw VISTA seed 8, no forced ACS, acc8/16/24"
echo "Ground truth: H5 dImgC"
echo "Run root: $RUN_ROOT"
echo "Git commit: $(git rev-parse HEAD)"

for root in dataset/h5_converted dataset/h5_converted_acc16 dataset/h5_converted_acc24; do
  "$PYTHON_BIN" scripts/validate_cine_inference_data.py \
    "$root/json_input" \
    --skip-acs-check \
    --require-smap
  "$PYTHON_BIN" scripts/calculate_mask_acceleration.py "$root/json_input"
done

for root in output/h5_converted output/h5_converted_acc16 output/h5_converted_acc24; do
  count=$(find "$root/val_img4ranking" -maxdepth 1 -name 'Sub*.mat' -type f | wc -l)
  if [[ "$count" -ne 10 ]]; then
    echo "Expected 10 reconstructions under $root, found $count" >&2
    exit 2
  fi
done

"$PYTHON_BIN" scripts/evaluate_and_plot_cine_dimgc.py \
  --acc acc8=output/h5_converted/val_img4ranking \
  --acc acc16=output/h5_converted_acc16/val_img4ranking \
  --acc acc24=output/h5_converted_acc24/val_img4ranking \
  --input-json acc8=dataset/h5_converted/json_input \
  --input-json acc16=dataset/h5_converted_acc16/json_input \
  --input-json acc24=dataset/h5_converted_acc24/json_input \
  --h5-root "$H5_ROOT" \
  --case-glob 'Sub*.mat' \
  --normalize p99.5 \
  --combine-accelerations \
  -o "$RUN_ROOT"

echo "Finish: $(date --iso-8601=seconds)"
