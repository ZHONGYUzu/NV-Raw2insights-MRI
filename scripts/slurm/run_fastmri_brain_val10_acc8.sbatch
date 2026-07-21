#!/bin/bash

#SBATCH --job-name=fastmri_b10_a8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fastmri_b10_a8_%j.out
#SBATCH --error=logs/fastmri_b10_a8_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=${REPO_ROOT:-/home/students/studxuzho1/NV-Raw2insights-MRI}
PYTHON_BIN=${PYTHON_BIN:-/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python}

RAW_SOURCE=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
CONFIG=configs/nv_raw2insights_mri_base_fastmri_brain_acc8.json
DATASET_ROOT=dataset/FastMRIBrainMulticoilVal10Acc8
INPUT_H5=$DATASET_ROOT/h5_input
MANIFEST=$DATASET_ROOT/cohort_manifest.json
OUTPUT_ROOT=output/FastMRIBrainMulticoilVal10GeomFixAxisAdapterAcc8
PREDICTION_DIR=$OUTPUT_ROOT/val_img4ranking
RESULTS_ROOT=Results/FastMRIBrainMulticoilVal10GeomFixAxisAdapterAcc8

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
echo "Derived input: $DATASET_ROOT"
echo "Inference output: $OUTPUT_ROOT"
echo "Evaluation results: $RESULTS_ROOT"

nvidia-smi
"$PYTHON_BIN" --version

# Select ten AXT1 validation volumes reproducibly. The derived H5 files are
# symlinks; the raw fastMRI dataset remains read-only and is not copied.
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

"$PYTHON_BIN" scripts/inference.py \
  -c "$CONFIG" \
  -i "$INPUT_H5" \
  -o "$OUTPUT_ROOT" \
  --num-workers 0 \
  --accelerations 8 \
  --profile-timing \
  --profile-interval 10

expected_cases=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_selected_cases"])' "$MANIFEST")
actual_cases=$(find "$PREDICTION_DIR" -maxdepth 1 -type f -name '*.mat' | wc -l)
if [[ "$actual_cases" -ne "$expected_cases" ]]; then
  echo "Expected $expected_cases predictions but found $actual_cases under $PREDICTION_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/evaluate_fastmri_brain_cohort.py \
  --manifest "$MANIFEST" \
  --prediction-dir "$PREDICTION_DIR" \
  -o "$RESULTS_ROOT"

echo "Completed cases: $actual_cases"
echo "Manifest: $MANIFEST"
echo "Predictions: $PREDICTION_DIR"
echo "Evaluation: $RESULTS_ROOT"
