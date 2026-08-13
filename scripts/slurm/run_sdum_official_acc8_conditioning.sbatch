#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/home/students/studxuzho1/hf_cache

: "${RUN_ID:?Submit with RUN_ID=sdum-004 style identifier}"
: "${MASK_ALIAS:?Submit with MASK_ALIAS=Uniform8, Uniform16, or Uniform24}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/students/studxuzho1/envs/nv-raw2insights-mri

REPO_ROOT=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon
PREP_REPO=/home/students/studxuzho1/NV-Raw2insights-MRI
PYTHON_BIN=/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python
RAW_TRAINING_ROOT=/mnt/qdata/rawdata/CMRxRecon/CMRxRecon_2023_training_init/ChallengeData/MultiCoil/CINE/TrainingSet
CONFIG=configs/nv_raw2insights_mri_base.json
RUN_DIR="runs/$RUN_ID"
INPUT_DIR="$RUN_DIR/json_input"
STAGE_OUTPUT="$RUN_DIR/full_size_inference"
OUTPUT_DIR="$RUN_DIR/output"
FULL_DIR="$OUTPUT_DIR/val_img_full"
RANKING_DIR="$OUTPUT_DIR/val_img4ranking"
EXPECTED_CASES=15
SUBJECTS=P006,P012,P016,P024,P042,P047,P048,P052,P054,P067,P086,P089,P094,P098,P110

cd "$REPO_ROOT"
mkdir -p "$RUN_DIR/logs" "$OUTPUT_DIR"
if [[ ! -e "$RUN_DIR/ground_truth" ]]; then
    ln -s ../sdum-002/ground_truth "$RUN_DIR/ground_truth"
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Run: $RUN_ID"
echo "Physical mask: unchanged CMRxRecon2023 official AccFactor08/mask08"
echo "Runtime/model conditioning alias: $MASK_ALIAS"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"

"$PYTHON_BIN" "$PREP_REPO/scripts/create_cmrxrecon2023_cine_cohort.py" \
    --raw-root "$RAW_TRAINING_ROOT" \
    --output-root "$RUN_DIR" \
    --view cine_lax \
    --acceleration-folder AccFactor08 \
    --mask-key mask08 \
    --mask-type "$MASK_ALIAS" \
    --subjects "$SUBJECTS" \
    --num-cases "$EXPECTED_CASES" \
    --force-acs-lines 0 \
    --overwrite

"$PYTHON_BIN" "$PREP_REPO/scripts/validate_cine_inference_data.py" "$INPUT_DIR"

if [[ -d "$FULL_DIR" ]]; then
    full_count=$(find "$FULL_DIR" -maxdepth 1 -type f -name '*.mat' | wc -l)
else
    full_count=0
fi
if [[ "$full_count" -ne "$EXPECTED_CASES" ]]; then
    "$PYTHON_BIN" scripts/inference.py \
        -c "$CONFIG" \
        -i "$INPUT_DIR" \
        -o "$STAGE_OUTPUT" \
        --profile-timing

    stage_count=$(find "$STAGE_OUTPUT/val_img4ranking" -maxdepth 1 -type f -name '*.mat' | wc -l)
    if [[ "$stage_count" -ne "$EXPECTED_CASES" ]]; then
        echo "Expected $EXPECTED_CASES full-size staged predictions, found $stage_count" >&2
        exit 1
    fi
    if [[ -e "$FULL_DIR" ]]; then
        echo "Refusing to replace existing incomplete full-size directory: $FULL_DIR" >&2
        exit 1
    fi
    mv "$STAGE_OUTPUT/val_img4ranking" "$FULL_DIR"
    cp "$STAGE_OUTPUT/config.json" "$OUTPUT_DIR/config.json"
fi

"$PYTHON_BIN" scripts/export_sdum_ranking_crops.py \
    "$FULL_DIR" \
    "$RANKING_DIR" \
    --filetype cine_lax

full_count=$(find "$FULL_DIR" -maxdepth 1 -type f -name '*.mat' | wc -l)
ranking_count=$(find "$RANKING_DIR" -maxdepth 1 -type f -name '*.mat' | wc -l)
echo "Full-size predictions: $full_count"
echo "Ranking-crop predictions: $ranking_count"
if [[ "$full_count" -ne "$EXPECTED_CASES" || "$ranking_count" -ne "$EXPECTED_CASES" ]]; then
    echo "Expected $EXPECTED_CASES outputs in both directories" >&2
    exit 1
fi

echo "$RUN_ID completed successfully."
