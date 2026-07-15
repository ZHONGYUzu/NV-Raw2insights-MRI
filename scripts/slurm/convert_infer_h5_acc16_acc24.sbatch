#!/bin/bash

#SBATCH --job-name=cine_h5_r2r3
#SBATCH --array=16,24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cine_h5_r2r3_%A_%a.out
#SBATCH --error=logs/cine_h5_r2r3_%A_%a.err

set -euo pipefail

REPO_ROOT=/home/students/studxuzho1/NV-Raw2insights-MRI
SOURCE_ROOT=dataset/h5_converted
CONFIG=configs/nv_raw2insights_mri_base.json
MASK_SOURCE=/home/students/studxusiy1/mr_recon/masks
ACC=${SLURM_ARRAY_TASK_ID}

case "$ACC" in
  16)
    RUN_LABEL=R2
    MASK_SUBDIR=Mask_TaskR2
    DATASET_ROOT=dataset/h5_converted_acc16
    OUTPUT_ROOT=output/h5_converted_acc16
    ;;
  24)
    RUN_LABEL=R3
    MASK_SUBDIR=Mask_TaskR3
    DATASET_ROOT=dataset/h5_converted_acc24
    OUTPUT_ROOT=output/h5_converted_acc24
    ;;
  *)
    echo "Unsupported acceleration: $ACC" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"
source "$HOME/.bashrc"
conda activate "$HOME/envs/nv-raw2insights-mri"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-manual}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-manual}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Run label: $RUN_LABEL"
echo "Acceleration: $ACC"
echo "Source dataset: $SOURCE_ROOT"
echo "Target dataset: $DATASET_ROOT"
echo "Mask subdirectory: $MASK_SUBDIR"
echo "Output root: $OUTPUT_ROOT"

nvidia-smi

create_case () {
  local sub=$1
  local pe=$2

  python scripts/create_custom_cine_acc_dataset.py \
    --source-root "$SOURCE_ROOT" \
    --source-kspace-subdir h5_kspace \
    --output-root "$DATASET_ROOT" \
    --target-kspace-subdir h5_kspace \
    --mask-subdir "$MASK_SUBDIR" \
    --input-mask-dir "$MASK_SOURCE" \
    --mask-glob "mask_VISTA_${pe}x25_acc${ACC}_8.txt" \
    --mask-type "ktRadial${ACC}" \
    --case-glob "${sub}_kspace_full.mat" \
    --frequency-size 192 \
    --no-force-acs \
    --overwrite
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

python scripts/validate_cine_inference_data.py \
  "$DATASET_ROOT/json_input" \
  --skip-acs-check \
  --require-smap

python scripts/calculate_mask_acceleration.py \
  "$DATASET_ROOT/json_input"

python scripts/inference.py \
  -c "$CONFIG" \
  -i "$DATASET_ROOT/json_input" \
  -o "$OUTPUT_ROOT" \
  --profile-timing

echo "Finished: $(date --iso-8601=seconds)"
