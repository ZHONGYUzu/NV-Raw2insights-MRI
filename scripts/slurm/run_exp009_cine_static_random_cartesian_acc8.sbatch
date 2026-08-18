#!/bin/bash

# First submit a one-case debug run (the default):
#   sbatch scripts/slurm/run_exp009_cine_static_random_cartesian_acc8.sbatch
# Then complete the same output directory:
#   sbatch --export=ALL,DEBUG=0 scripts/slurm/run_exp009_cine_static_random_cartesian_acc8.sbatch

#SBATCH --job-name=exp009_cine_ood_r8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp009_cine_ood_r8_%j.out
#SBATCH --error=logs/exp009_cine_ood_r8_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=${REPO_ROOT:-/home/students/studxuzho1/NV-Raw2insights-MRI}
PYTHON_BIN=${PYTHON_BIN:-/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-dataset/h5_converted}
DATASET_ROOT=${DATASET_ROOT:-dataset/EXP009_CINE_static_random_cartesian_acc8_seed9009}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/EXP009_CINE_static_random_cartesian_acc8_seed9009}
CONFIG=${CONFIG:-configs/nv_raw2insights_mri_base.json}
SEED=${SEED:-9009}
DEBUG=${DEBUG:-1}

cd "$REPO_ROOT"
mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Experiment: EXP009"
echo "Dataset: fixed 10-case in-house CINE H5 benchmark"
echo "Physical mask: static random Cartesian PE lines, identical across all 25 frames"
echo "Mask seed: $SEED"
echo "Nominal acceleration: 8"
echo "Forced ACS lines: 0"
echo "Sensitivity maps: external H5 dMap inherited from $SOURCE_ROOT"
echo "Model conditioning alias: Uniform8"
echo "Debug: $DEBUG"
echo "Dataset root: $DATASET_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Git commit: $(git rev-parse HEAD)"

create_args=(
  --source-root "$SOURCE_ROOT"
  --output-root "$DATASET_ROOT"
  --case-glob 'Sub*.json'
  --acceleration 8
  --seed "$SEED"
  --acs-lines 0
  --conditioning-family Uniform
)
if [[ -d "$DATASET_ROOT/json_input" ]]; then
  create_args+=(--overwrite)
fi
"$PYTHON_BIN" scripts/create_static_random_cartesian_cine_dataset.py "${create_args[@]}"

"$PYTHON_BIN" scripts/validate_cine_inference_data.py \
  "$DATASET_ROOT/json_input" \
  --skip-acs-check \
  --require-smap
"$PYTHON_BIN" scripts/calculate_mask_acceleration.py "$DATASET_ROOT/json_input"

inference_args=(
  -c "$CONFIG"
  -i "$DATASET_ROOT/json_input"
  -o "$OUTPUT_ROOT"
  --fixed-mask-types mask_Uniform8
  --disable-acs-region
  --num-workers 0
  --profile-timing
)
if [[ "$DEBUG" == "1" ]]; then
  inference_args+=(--debug)
elif [[ "$DEBUG" != "0" ]]; then
  echo "DEBUG must be 0 or 1, got: $DEBUG" >&2
  exit 2
fi

nvidia-smi
"$PYTHON_BIN" scripts/inference.py "${inference_args[@]}"

echo "Finish: $(date --iso-8601=seconds)"
