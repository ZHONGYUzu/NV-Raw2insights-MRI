#!/bin/bash

# Submit one or more new seeds across all three accelerations, for example:
#   sbatch --array=0-5 --export=ALL,MASK_SEEDS=15,23 \
#     scripts/slurm/run_h5_dmap_raw_vista_multiseed.sbatch

#SBATCH --job-name=vista_multiseed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vista_multiseed_%A_%a.out
#SBATCH --error=logs/vista_multiseed_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=${REPO_ROOT:-/home/students/studxuzho1/NV-Raw2insights-MRI}
PYTHON_BIN=${PYTHON_BIN:-/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python}
MASK_SOURCE=${MASK_SOURCE:-/home/students/studxusiy1/mr_recon/masks}
SOURCE_ROOT=${SOURCE_ROOT:-dataset/h5_converted}
CONFIG=${CONFIG:-configs/nv_raw2insights_mri_base.json}
MASK_SEEDS=${MASK_SEEDS:-}

if [[ -z "$MASK_SEEDS" ]]; then
  echo "MASK_SEEDS is required, for example MASK_SEEDS=15,23" >&2
  exit 2
fi

IFS=',' read -r -a seeds <<< "$MASK_SEEDS"
accs=(8 16 24)
task_id=${SLURM_ARRAY_TASK_ID:-0}
seed_index=$((task_id / 3))
acc_index=$((task_id % 3))

if (( seed_index >= ${#seeds[@]} )); then
  echo "Array task $task_id exceeds ${#seeds[@]} seed(s) x 3 accelerations" >&2
  exit 2
fi

SEED=${seeds[$seed_index]}
ACC=${accs[$acc_index]}
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Invalid seed: $SEED" >&2
  exit 2
fi

DATASET_ROOT="dataset/h5_converted_vista_seed${SEED}_acc${ACC}"
OUTPUT_ROOT="output/h5_converted_vista_seed${SEED}_acc${ACC}"
MASK_SUBDIR="Mask_VISTA_seed${SEED}_acc${ACC}"

cd "$REPO_ROOT"
mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Array task: $task_id"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Dataset: fixed 10-case in-house CINE H5 benchmark"
echo "Mask protocol: raw VISTA seed=$SEED acc=$ACC, no forced ACS"
echo "Sensitivity maps: external H5 dMap inherited from $SOURCE_ROOT"
echo "Dataset root: $DATASET_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Git commit: $(git rev-parse HEAD)"

"$PYTHON_BIN" scripts/find_common_vista_mask_seeds.py "$MASK_SOURCE" \
  --pe 156,162,174,180 \
  --acc 8,16,24 | tee "logs/vista_seed_inventory_${SLURM_ARRAY_JOB_ID:-manual}.txt"

create_case () {
  local sub=$1
  local pe=$2

  "$PYTHON_BIN" scripts/create_custom_cine_acc_dataset.py \
    --source-root "$SOURCE_ROOT" \
    --source-kspace-subdir h5_kspace \
    --output-root "$DATASET_ROOT" \
    --target-kspace-subdir h5_kspace \
    --mask-subdir "$MASK_SUBDIR" \
    --input-mask-dir "$MASK_SOURCE" \
    --mask-glob "mask_VISTA_${pe}x25_acc${ACC}_${SEED}.txt" \
    --mask-type "ktRadial${ACC}" \
    --case-glob "${sub}_kspace_full.mat" \
    --frequency-size 192 \
    --no-force-acs
}

create_case Sub0014 180
create_case Sub0026 162
create_case Sub0030 174
create_case Sub0047 156
create_case Sub0049 156
create_case Sub0051 156
create_case Sub0096 156
create_case Sub0105 156
create_case Sub0112 162
create_case Sub0122 156

"$PYTHON_BIN" scripts/validate_cine_inference_data.py \
  "$DATASET_ROOT/json_input" \
  --skip-acs-check \
  --require-smap
"$PYTHON_BIN" scripts/calculate_mask_acceleration.py "$DATASET_ROOT/json_input"

nvidia-smi
"$PYTHON_BIN" scripts/inference.py \
  -c "$CONFIG" \
  -i "$DATASET_ROOT/json_input" \
  -o "$OUTPUT_ROOT" \
  --num-workers 0 \
  --profile-timing

echo "Finish: $(date --iso-8601=seconds)"
