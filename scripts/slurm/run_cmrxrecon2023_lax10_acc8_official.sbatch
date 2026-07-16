#!/bin/bash

#SBATCH --job-name=cmrx23_lax10_a8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cmrx23_lax10_a8_%j.out
#SBATCH --error=logs/cmrx23_lax10_a8_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=${REPO_ROOT:-/home/students/studxuzho1/NV-Raw2insights-MRI}
PYTHON_BIN=${PYTHON_BIN:-/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python}

RAW_TRAINING_ROOT=/mnt/qdata/rawdata/CMRxRecon/CMRxRecon_2023_training_init/ChallengeData/MultiCoil/CINE/TrainingSet
CONFIG=configs/nv_raw2insights_mri_base.json
DATASET_ROOT=dataset/CMRxRecon2023CINELAX10Acc8Official
INPUT_JSON=$DATASET_ROOT/json_input
MANIFEST=$DATASET_ROOT/cohort_manifest.json
OUTPUT_ROOT=output/CMRxRecon2023CINELAX10Acc8Official
PREDICTION_DIR=$OUTPUT_ROOT/val_img4ranking
RESULTS_ROOT=Results/CMRxRecon2023CINELAX10Acc8Official

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
if [[ ! -d "$RAW_TRAINING_ROOT/FullSample" ]]; then
  echo "Raw FullSample directory not found: $RAW_TRAINING_ROOT/FullSample" >&2
  exit 2
fi
if [[ ! -d "$RAW_TRAINING_ROOT/AccFactor08" ]]; then
  echo "Raw AccFactor08 directory not found: $RAW_TRAINING_ROOT/AccFactor08" >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Raw inputs: $RAW_TRAINING_ROOT"
echo "Derived inputs: $DATASET_ROOT"
echo "Inference output: $OUTPUT_ROOT"
echo "Evaluation results: $RESULTS_ROOT"

nvidia-smi
"$PYTHON_BIN" --version

# Reproducibly select ten paired LAX cases. Official masks are copied into the
# model interface unchanged; no additional ACS lines are forced.
"$PYTHON_BIN" scripts/create_cmrxrecon2023_cine_cohort.py \
  --raw-root "$RAW_TRAINING_ROOT" \
  --output-root "$DATASET_ROOT" \
  --view cine_lax \
  --acceleration-folder AccFactor08 \
  --mask-key mask08 \
  --mask-type ktRadial8 \
  --num-cases 10 \
  --seed 20260716 \
  --force-acs-lines 0 \
  --overwrite

"$PYTHON_BIN" scripts/validate_cine_inference_data.py "$INPUT_JSON"

"$PYTHON_BIN" scripts/inference.py \
  -c "$CONFIG" \
  -i "$INPUT_JSON" \
  -o "$OUTPUT_ROOT" \
  --num-workers 0 \
  --fixed-mask-types mask_ktRadial8 \
  --accelerations 8 \
  --profile-timing

expected_cases=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_selected_cases"])' "$MANIFEST")
actual_cases=$(find "$PREDICTION_DIR" -maxdepth 1 -type f -name 'P*_cine_lax.mat' | wc -l)
if [[ "$actual_cases" -ne "$expected_cases" ]]; then
  echo "Expected $expected_cases predictions but found $actual_cases under $PREDICTION_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/evaluate_cmrxrecon2023_cohort.py \
  --manifest "$MANIFEST" \
  --prediction-dir "$PREDICTION_DIR" \
  -o "$RESULTS_ROOT"

echo "Completed cases: $actual_cases"
echo "Manifest: $MANIFEST"
echo "Predictions: $PREDICTION_DIR"
echo "Evaluation: $RESULTS_ROOT"
