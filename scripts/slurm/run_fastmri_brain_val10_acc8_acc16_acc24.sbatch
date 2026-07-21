#!/bin/bash

#SBATCH --job-name=fastmri_b10_aall
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fastmri_b10_aall_%j.out
#SBATCH --error=logs/fastmri_b10_aall_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=${REPO_ROOT:-/home/students/studxuzho1/NV-Raw2insights-MRI}
PYTHON_BIN=${PYTHON_BIN:-/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python}

RAW_SOURCE=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
CONFIG=configs/nv_raw2insights_mri_base_fastmri_brain_acc8.json
DATASET_ROOT=dataset/FastMRIBrainMulticoilVal10
INPUT_H5=$DATASET_ROOT/h5_input
MANIFEST=$DATASET_ROOT/cohort_manifest.json

finish () {
  status=$?
  echo "Exit status: $status"
  echo "Finished: $(date --iso-8601=seconds)"
  exit "$status"
}
trap finish EXIT

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$RAW_SOURCE" ]]; then
  echo "Raw fastMRI validation directory not found: $RAW_SOURCE" >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Raw source: $RAW_SOURCE"
echo "Shared derived input: $DATASET_ROOT"
echo "Accelerations: 8 16 24"

nvidia-smi
"$PYTHON_BIN" --version

# Select the cohort once. Every acceleration therefore uses exactly the same
# AXT1 volumes and ground truth. H5 inputs are read-only source symlinks.
"$PYTHON_BIN" scripts/create_fastmri_brain_cohort.py \
  --source-dir "$RAW_SOURCE" \
  --output-root "$DATASET_ROOT" \
  --num-cases 10 \
  --seed 20260719 \
  --acquisition AXT1 \
  --overwrite

"$PYTHON_BIN" scripts/validate_fastmri_inference_data.py \
  "$INPUT_H5" \
  --manifest "$MANIFEST"

"$PYTHON_BIN" scripts/validate_fastmri_reader_geometry.py \
  "$INPUT_H5" \
  --max-cases 1

expected_cases=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_selected_cases"])' "$MANIFEST")

for acc in 8 16 24; do
  output_root="output/FastMRIBrainMulticoilVal10GeomFixAxisAdapterAcc${acc}"
  prediction_dir="$output_root/val_img4ranking"
  results_root="Results/FastMRIBrainMulticoilVal10GeomFixAxisAdapterAcc${acc}"

  echo "========== Starting nominal acceleration ${acc} =========="
  echo "Predictions: $prediction_dir"
  echo "Evaluation: $results_root"

  "$PYTHON_BIN" scripts/inference.py \
    -c "$CONFIG" \
    -i "$INPUT_H5" \
    -o "$output_root" \
    --num-workers 0 \
    --accelerations "$acc" \
    --profile-timing \
    --profile-interval 10

  actual_cases=$(find "$prediction_dir" -maxdepth 1 -type f -name '*.mat' | wc -l)
  if [[ "$actual_cases" -ne "$expected_cases" ]]; then
    echo "Acceleration $acc: expected $expected_cases predictions but found $actual_cases" >&2
    exit 1
  fi

  "$PYTHON_BIN" scripts/evaluate_fastmri_brain_cohort.py \
    --manifest "$MANIFEST" \
    --prediction-dir "$prediction_dir" \
    --label "fastMRI Brain AXT1 Acc${acc}" \
    -o "$results_root"

  echo "Acceleration $acc completed: $actual_cases cases"
done

echo "Shared manifest: $MANIFEST"
echo "All accelerations completed: 8 16 24"
