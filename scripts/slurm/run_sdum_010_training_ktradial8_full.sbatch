#!/bin/bash

#SBATCH --job-name=sdum010-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-010/full-%j.out
#SBATCH --error=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon/runs/sdum-010/full-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/home/students/studxuzho1/hf_cache

REPO_ROOT=/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon
PYTHON_BIN=/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python
RUN_DIR="$REPO_ROOT/runs/sdum-010"
INPUT_DIR="$RUN_DIR/json_input"
STAGE_OUTPUT="$RUN_DIR/full_size_inference"
OUTPUT_DIR="$RUN_DIR/output"
FULL_DIR="$OUTPUT_DIR/val_img_full"
RANKING_DIR="$OUTPUT_DIR/val_img4ranking"
EXPECTED_CASES=15

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"
if [[ ! -e "$RUN_DIR/ground_truth" ]]; then
    ln -s ../sdum-002/ground_truth "$RUN_DIR/ground_truth"
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Run: sdum-010 full"
echo "Physical mask: NV training generator ktRadial8, 20x20 ACS"
echo "Conditioning: ktRadial8 / acceleration 8"

"$PYTHON_BIN" scripts/validate_cine_inference_data.py "$INPUT_DIR"

if [[ -d "$FULL_DIR" ]]; then
    full_count=$(find "$FULL_DIR" -maxdepth 1 -type f -name '*.mat' | wc -l)
else
    full_count=0
fi

if [[ "$full_count" -ne "$EXPECTED_CASES" ]]; then
    if [[ "$full_count" -ne 0 ]]; then
        echo "Refusing to replace incomplete formal output: $FULL_DIR contains $full_count files" >&2
        exit 1
    fi

    "$PYTHON_BIN" scripts/inference.py \
        -c configs/nv_raw2insights_mri_base.json \
        -i "$INPUT_DIR" \
        -o "$STAGE_OUTPUT" \
        --profile-timing

    stage_count=$(find "$STAGE_OUTPUT/val_img4ranking" -maxdepth 1 -type f -name '*.mat' | wc -l)
    if [[ "$stage_count" -ne "$EXPECTED_CASES" ]]; then
        echo "Expected $EXPECTED_CASES staged predictions, found $stage_count" >&2
        exit 1
    fi
    if [[ -e "$FULL_DIR" ]]; then
        echo "Refusing to replace existing path: $FULL_DIR" >&2
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

echo "sdum-010 full inference completed successfully."
