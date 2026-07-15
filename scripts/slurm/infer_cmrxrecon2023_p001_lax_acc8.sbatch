#!/bin/bash

#SBATCH --job-name=cmrx23_p001_lax_a8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cmrx23_p001_lax_a8_%j.out
#SBATCH --error=logs/cmrx23_p001_lax_a8_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Override these at submission time with, for example:
# sbatch --export=ALL,REPO_ROOT=/path/to/repo,PYTHON_BIN=/path/to/python <this-file>
REPO_ROOT=${REPO_ROOT:-/home/students/studxuzho1/NV-Raw2insights-MRI}
PYTHON_BIN=${PYTHON_BIN:-/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python}

RAW_CASE_ROOT=/mnt/qdata/rawdata/CMRxRecon/CMRxRecon_2023_training_init/ChallengeData/MultiCoil/CINE/TrainingSet
RAW_KSPACE=$RAW_CASE_ROOT/FullSample/P001/cine_lax.mat
RAW_MASK=$RAW_CASE_ROOT/AccFactor08/P001/cine_lax_mask.mat

CONFIG=configs/nv_raw2insights_mri_base.json
DATASET_ROOT=dataset/CMRxRecon2023CINELAXAcc8
MASK_DIR=$DATASET_ROOT/MultiCoil/CINE/Mask_TaskR1
JSON_DIR=$DATASET_ROOT/json_input
DERIVED_MASK=$MASK_DIR/P001_cine_lax_mask_ktRadial8.mat
DESCRIPTOR=$JSON_DIR/P001_cine_lax.json

OUTPUT_ROOT=output/CMRxRecon2023CINELAXAcc8
OUTPUT_MAT=$OUTPUT_ROOT/val_img4ranking/P001_cine_lax.mat
FIG_DIR=$OUTPUT_ROOT/figs
INSPECTION=$OUTPUT_ROOT/output_inspection.txt

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
if [[ ! -f "$RAW_KSPACE" ]]; then
  echo "Raw full-sampled k-space not found: $RAW_KSPACE" >&2
  exit 2
fi
if [[ ! -f "$RAW_MASK" ]]; then
  echo "Raw acceleration-8 mask not found: $RAW_MASK" >&2
  exit 2
fi

mkdir -p "$MASK_DIR" "$JSON_DIR" "$OUTPUT_ROOT" "$FIG_DIR"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Python: $PYTHON_BIN"
echo "Raw k-space: $RAW_KSPACE"
echo "Raw mask: $RAW_MASK"
echo "Derived dataset: $DATASET_ROOT"
echo "Output root: $OUTPUT_ROOT"

nvidia-smi
"$PYTHON_BIN" --version

# Keep the raw dataset read-only. The full-sampled k-space is referenced in
# place; only the small mask and JSON descriptor are written under dataset/.
RAW_KSPACE="$RAW_KSPACE" \
RAW_MASK="$RAW_MASK" \
DERIVED_MASK="$DERIVED_MASK" \
DESCRIPTOR="$DESCRIPTOR" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.io

raw_kspace = Path(os.environ["RAW_KSPACE"])
raw_mask = Path(os.environ["RAW_MASK"])
derived_mask = Path(os.environ["DERIVED_MASK"])
descriptor_path = Path(os.environ["DESCRIPTOR"])

with h5py.File(raw_kspace, "r") as mat_file:
    if "kspace_full" not in mat_file:
        raise KeyError(f"{raw_kspace}: missing kspace_full")
    source = mat_file["kspace_full"]
    if source.shape != (12, 3, 10, 204, 448):
        raise ValueError(f"{raw_kspace}: unexpected k-space shape {source.shape}")
    if not source.dtype.fields or not {"real", "imag"}.issubset(source.dtype.fields):
        raise TypeError(f"{raw_kspace}: expected compound complex dtype, got {source.dtype}")

with h5py.File(raw_mask, "r") as mat_file:
    if "mask08" not in mat_file:
        raise KeyError(f"{raw_mask}: missing mask08")
    mask = np.asarray(mat_file["mask08"], dtype=np.float32)

if mask.shape != (204, 448):
    raise ValueError(f"{raw_mask}: unexpected mask shape {mask.shape}")
if not np.isfinite(mask).all():
    raise ValueError(f"{raw_mask}: mask contains non-finite values")
if not np.all(np.isin(mask, (0, 1))):
    raise ValueError(f"{raw_mask}: mask is not binary")
if mask[mask.shape[0] // 2, mask.shape[1] // 2] != 1:
    raise ValueError(f"{raw_mask}: mask center is not sampled")

derived_mask.parent.mkdir(parents=True, exist_ok=True)
scipy.io.savemat(derived_mask, {"mask": mask}, do_compression=True)

descriptor = {
    "kspace": str(raw_kspace),
    "mask": [str(derived_mask.resolve())],
}
descriptor_path.parent.mkdir(parents=True, exist_ok=True)
descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n")

print(f"Derived mask: {derived_mask}")
print(f"Descriptor: {descriptor_path}")
print(f"Sampled PE lines: {int(mask.sum() / mask.shape[1])}/{mask.shape[0]}")
print(f"Effective acceleration including ACS: {mask.size / mask.sum():.6f}")
PY

"$PYTHON_BIN" scripts/validate_cine_inference_data.py "$JSON_DIR"

"$PYTHON_BIN" scripts/inference.py \
  -c "$CONFIG" \
  -i "$JSON_DIR" \
  -o "$OUTPUT_ROOT" \
  --debug \
  --num-workers 0 \
  --fixed-mask-types mask_ktRadial8 \
  --accelerations 8 \
  --profile-timing

if [[ ! -f "$OUTPUT_MAT" ]]; then
  echo "Expected reconstruction was not created: $OUTPUT_MAT" >&2
  exit 1
fi

OUTPUT_MAT="$OUTPUT_MAT" "$PYTHON_BIN" - <<'PY' | tee "$INSPECTION"
import os

import numpy as np
import scipy.io

path = os.environ["OUTPUT_MAT"]
image = scipy.io.loadmat(path)["img4ranking"]

print(f"file: {path}")
print(f"shape: {image.shape}")
print(f"dtype: {image.dtype}")
print(f"finite: {np.isfinite(image).all()}")
print(f"min/max/mean: {image.min()} / {image.max()} / {image.mean()}")
print(f"nonzero: {np.count_nonzero(image)} / {image.size}")

if image.shape != (448, 204, 3, 12):
    raise ValueError(f"Unexpected reconstruction shape: {image.shape}")
if image.dtype != np.float32:
    raise TypeError(f"Unexpected reconstruction dtype: {image.dtype}")
if not np.isfinite(image).all():
    raise ValueError("Reconstruction contains non-finite values")
PY

"$PYTHON_BIN" scripts/visualize_mat.py \
  "$OUTPUT_MAT" \
  -o "$FIG_DIR" \
  --figsize 2 \
  --dpi 100 \
  --no-colorbar

echo "Reconstruction: $OUTPUT_MAT"
echo "Inspection: $INSPECTION"
echo "Figure: $FIG_DIR/P001_cine_lax.png"
