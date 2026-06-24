# Daily Log: External Smap, Flexible Acceleration, And Optional ACS

Date: 2026-06-23

## Context

Today we checked the current inference pipeline around four implementation questions:

- how the ACS region is used,
- whether existing sensitivity maps can be used directly,
- whether already-undersampled k-space can be used as input,
- whether acceleration rates are truly limited to 8, 16, and 24.

The previous conclusion was:

- ACS is mainly used to estimate coil sensitivity maps in the current Restormer inference path.
- The model has an internal `sensitivity_maps` argument, but the data pipeline did not expose external smap input.
- The current pipeline expects fully sampled k-space plus a mask and applies the mask internally.
- Acceleration rates are not universally hardcoded, but the default config and pretrained inference path are built around 8, 16, and 24.

## Repo State Before Code Changes

Before this implementation round, Git showed a clean worktree. There were no pending modifications.

The FYI documentation note about ACS, smap, undersampled k-space, and acceleration behavior was already present from Git's point of view.

## Goal

Modify the repo so custom CINE inference can move toward:

- accepting external sensitivity maps from the existing H5 `dMap`,
- trying additional acceleration rates beyond the default 8, 16, and 24,
- avoiding forced central ACS filling when intentionally using raw mask structure,
- avoiding ACS-region extraction when external smaps are provided or when masks do not have a filled ACS center.

## Code Changes Made

### External Sensitivity Map Support

Updated `scripts/readers.py`.

Changes:

- Added `CMRxReconKeys.SMAP = "sensitivity_maps"`.
- `CMRxReconReader` now accepts optional JSON fields:
  - `sensitivity_maps`
  - `smap`
  - `dMap`
- The reader loads the referenced MAT/H5-like file and accepts keys:
  - `sensitivity_maps`
  - `smap`
  - `dMap`
- Loaded smaps are reshaped into logical order:

```text
(time, slice, coil, phase, frequency)
```

For documented CINE `dMap`, this means:

```text
source H5 dMap:        (slice, coil, 1, phase, frequency)
reader logical smap:   (1, slice, coil, phase, frequency)
```

Updated `scripts/transforms.py`.

Changes:

- Added `ExtractOptionalDataKeyFromMetaKeyd`.
- Added `PrepareSensitivityMapd`.
- `PrepareSensitivityMapd` converts smaps to complex tensor layout and expands a singleton time dimension across all cardiac frames when needed.
- Prepared smaps are rearranged to match the flattened inference sample layout used by `kspace_masked_ifft`.

Updated `scripts/inference.py`.

Changes:

- Inference now extracts optional `sensitivity_maps` from metadata.
- During each model window, smaps are windowed using the same `window_idx` as the input.
- The model call now passes:

```python
output = model(inp, mas.bool(), mask_type, acc_factor, acq_type, sensitivity_maps=smap)
```

Updated `scripts/models/latent_recon.py`.

Changes:

- `Flow_SkipConnected_MRI_Recon.forward()` now accepts `sensitivity_maps`.
- External smaps are passed through to the wrapped cascaded reconstruction model.

Updated `scripts/models/restormer/restormer.py`.

Changes:

- If `sensitivity_maps` is provided, Restormer now uses it instead of recomputing maps from ACS.
- Provided smaps with shape `(B, T, C, H, W, 2)` are rearranged to `(B*T, C, H, W, 2)`.
- Added explicit assertions for unknown `mask_type` and `acc_factor`, so unsupported config combinations fail clearly instead of silently using index `-1`.

## H5 `dMap` Conversion Support

Updated `scripts/convert_cine_h5_kspace_to_mat.py`.

Changes:

- Added optional argument:

```bash
--smap-key dMap
```

- When supplied, the converter writes:

```text
dataset/<root>/MultiCoil/Cine/SensitivityMap_TaskR1/<case>_sensitivity_maps.mat
```

- The case JSON is updated with:

```json
{
  "sensitivity_maps": "/absolute/path/to/<case>_sensitivity_maps.mat"
}
```

Example command:

```bash
python scripts/convert_cine_h5_kspace_to_mat.py \
  --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
  --output-root dataset/CustomCINEDataR1 \
  --glob 'Sub000[1-5].h5' \
  --smap-key dMap \
  --overwrite
```

## More Flexible Acceleration Rates

Updated `scripts/inference.py`.

Added CLI overrides:

```bash
--fixed-mask-types
--accelerations
```

Example for R=4:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR4/json_input \
  -o output/CustomCINEOutputR4 \
  --fixed-mask-types mask_ktRadial4 \
  --accelerations 4
```

Important caution:

- This makes the code path configurable for new acceleration values.
- It does not guarantee pretrained-model quality for unseen acceleration rates.
- R=2, R=3, or R=4 may still be out-of-distribution unless the model was trained or fine-tuned for those conditions.

## Optional ACS Forcing

Updated `scripts/convert_vista_txt_mask_to_mat.py`.

Added:

```bash
--no-force-acs
```

This is equivalent to using `--acs-lines 0`; it keeps the original mask center structure instead of forcing central phase lines to 1.

Updated `scripts/create_custom_cine_acc_dataset.py`.

Added:

```bash
--no-force-acs
```

Also preserves any source JSON `sensitivity_maps` path when creating a new acceleration dataset variant.

Example:

```bash
python scripts/create_custom_cine_acc_dataset.py \
  --source-root dataset/CustomCINEDataR1 \
  --output-root dataset/CustomCINEDataR4 \
  --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
  --mask-glob 'mask_VISTA_132x25_acc4_8.txt' \
  --mask-type ktRadial4 \
  --frequency-size 176 \
  --no-force-acs
```

## Validation Changes

Updated `scripts/validate_cine_inference_data.py`.

Changes:

- Validator now checks optional external smap files.
- Smap time dimension may be either:
  - `1`, for static/time-averaged maps,
  - or equal to k-space time size.
- Smap spatial, slice, and coil dimensions must match k-space.
- Added:

```bash
--skip-acs-check
```

Use this when masks intentionally do not have a filled ACS center and inference will use external smaps or `--disable-acs-region`.

Example:

```bash
python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR4/json_input \
  --skip-acs-check
```

## Inference Without ACS Region Extraction

Updated `scripts/inference.py`.

Added:

```bash
--disable-acs-region
```

Use this when:

- the mask center is not fully positive,
- external smaps are provided,
- or ACS-region estimation should be skipped intentionally.

Example:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR4/json_input \
  -o output/CustomCINEOutputR4 \
  --fixed-mask-types mask_ktRadial4 \
  --accelerations 4 \
  --disable-acs-region
```

## Verification

Syntax and whitespace checks passed.

The first compile attempt with `python` failed because `python` was not on the local PATH.

The second compile attempt with `python3` initially failed because Python tried to write bytecode under the macOS user cache directory:

```text
/Users/zhongyu/Library/Caches/com.apple.python
```

Rerunning with a workspace-safe bytecode cache succeeded:

```bash
PYTHONPYCACHEPREFIX=/tmp/nv_raw_compile_cache python3 -m compileall scripts
```

Whitespace check:

```bash
git diff --check
```

No whitespace errors were reported.

## Current Pending Files

The following files are modified and unstaged:

```text
scripts/convert_cine_h5_kspace_to_mat.py
scripts/convert_vista_txt_mask_to_mat.py
scripts/create_custom_cine_acc_dataset.py
scripts/inference.py
scripts/models/latent_recon.py
scripts/models/restormer/restormer.py
scripts/readers.py
scripts/transforms.py
scripts/validate_cine_inference_data.py
docs/daily-log-2026-06-23-smap-acc-acs.md
```

## Remaining Cautions

- The external smap path is implemented for inference, but it has not been tested on the server with real H5-derived `dMap` files yet.
- New acceleration rates can now be configured, but pretrained-model behavior may be unreliable for acceleration classes not seen during training.
- If external smaps are used, verify their scaling, phase convention, and orientation carefully.
- If raw masks are used without forced ACS, use `--skip-acs-check` during validation and `--disable-acs-region` during inference unless external smaps are definitely being passed and used.

## 2026-06-24 Ground Truth Generation Note

Added `scripts/generate_cine_h5_ground_truth.py` to generate ground truth directly from source CINE H5 files.

The comparison-oriented default now writes MAT files, not NPY files, because the inference output is also MAT. This keeps manual comparison straightforward:

```text
GT:     dataset/GT_from_kspace_dMap/Sub0001.mat, key gt
Recon:  output/<experiment>/val_img4ranking/Sub0001.mat, key img4ranking
Shape:  (176, 132, 12, 25) = (frequency, phase, slice, time)
Dtype:  float32 magnitude
```

The default GT source is `kSpace` + `dMap`:

- `kSpace` is inverse-FFT'd into coil images.
- `dMap` is used for sensitivity-map coil combination.
- The combined complex image is converted to magnitude and transposed to match inference output orientation.

Command for the first ten sorted H5 files:

```bash
mkdir -p dataset/GT_from_kspace_dMap

for h5 in $(find /mnt/qdata/rawdata/CINE/2D_h5_compressed -maxdepth 1 -name '*.h5' | sort | head -n 10); do
  base=$(basename "$h5" .h5)
  python scripts/generate_cine_h5_ground_truth.py \
    --input-h5-dir "$(dirname "$h5")" \
    --output-dir dataset/GT_from_kspace_dMap \
    --glob "${base}.h5"
done
```

The script still supports the older NPY layout for existing evaluation scripts via `--format npy`, but the MAT output is the clearer format for comparing against current inference reconstructions.

## 2026-06-24 CINE1 Ten-Case Multi-Acceleration Plan

Updated the planned server workflow for a 10-case experiment using four mask accelerations:

```text
R = 2, 4, 8, 16
```

Naming decisions:

```text
Shared base dataset: dataset/CINE1

Acceleration datasets:
dataset/CINE1acr2
dataset/CINE1acr4
dataset/CINE1acr8
dataset/CINE1acr16

Inference outputs:
output/CINE1acr2
output/CINE1acr4
output/CINE1acr8
output/CINE1acr16
```

Ground-truth root:

```text
/home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap
```

### Generate MAT Ground Truth

Skip this step if GT already exists at the path above.

```bash
python scripts/generate_cine_h5_ground_truth.py \
  --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
  --output-dir /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap \
  --glob 'Sub000[1-9].h5' \
  --source kspace-dmap \
  --format mat \
  --mat-key gt \
  --normalize max

python scripts/generate_cine_h5_ground_truth.py \
  --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
  --output-dir /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap \
  --glob 'Sub0010.h5' \
  --source kspace-dmap \
  --format mat \
  --mat-key gt \
  --normalize max
```

Expected GT files:

```text
Sub0001.mat
...
Sub0010.mat
```

Each GT MAT contains:

```text
key: gt
shape: (176, 132, 12, 25)
layout: (frequency, phase, slice, time)
```

### Convert Shared K-Space And Smap Dataset

Convert full k-space and external H5 `dMap` smaps once into `dataset/CINE1`.

```bash
python scripts/convert_cine_h5_kspace_to_mat.py \
  --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
  --output-root dataset/CINE1 \
  --glob 'Sub000[1-9].h5' \
  --smap-key dMap

python scripts/convert_cine_h5_kspace_to_mat.py \
  --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
  --output-root dataset/CINE1 \
  --glob 'Sub0010.h5' \
  --smap-key dMap
```

Expected JSON descriptors under:

```text
dataset/CINE1/json_input
```

Each JSON should contain both:

```json
{
  "kspace": ".../Sub0001_kspace_full.mat",
  "sensitivity_maps": ".../Sub0001_sensitivity_maps.mat",
  "mask": []
}
```

### Create R2/R4/R8/R16 Datasets Without ACS Filling

The masks should be converted without forcing central ACS lines:

```bash
for acc in 2 4 8 16; do
  python scripts/create_custom_cine_acc_dataset.py \
    --source-root dataset/CINE1 \
    --output-root dataset/CINE1acr${acc} \
    --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
    --mask-glob "mask_VISTA_132x25_acc${acc}_8.txt" \
    --mask-type ktRadial${acc} \
    --case-glob 'Sub00*_kspace_full.mat' \
    --frequency-size 176 \
    --no-force-acs
done
```

This keeps the VISTA mask center unmodified. Because the ACS center is not forced to one, later validation should use `--skip-acs-check`, and inference should use `--disable-acs-region`.

### Validate All Four Datasets

```bash
for acc in 2 4 8 16; do
  python scripts/validate_cine_inference_data.py \
    dataset/CINE1acr${acc}/json_input \
    --skip-acs-check
done
```

Expected pattern:

```text
OK: Sub0001.json: k-space (25, 12, 15, 132, 176), masks 1, smap yes
...
Validated 10 case(s).
```

### Debug Inference

Run one case per acceleration first:

```bash
for acc in 2 4 8 16; do
  python scripts/inference.py \
    -c configs/nv_raw2insights_mri_base.json \
    -i dataset/CINE1acr${acc}/json_input \
    -o output/CINE1acr${acc}_debug \
    --fixed-mask-types mask_ktRadial${acc} \
    --accelerations ${acc} \
    --disable-acs-region \
    --debug \
    --profile-timing
done
```

Expected debug output per acceleration:

```text
output/CINE1acr2_debug/val_img4ranking/Sub0001.mat
output/CINE1acr4_debug/val_img4ranking/Sub0001.mat
output/CINE1acr8_debug/val_img4ranking/Sub0001.mat
output/CINE1acr16_debug/val_img4ranking/Sub0001.mat
```

### Full Inference

```bash
for acc in 2 4 8 16; do
  python scripts/inference.py \
    -c configs/nv_raw2insights_mri_base.json \
    -i dataset/CINE1acr${acc}/json_input \
    -o output/CINE1acr${acc} \
    --fixed-mask-types mask_ktRadial${acc} \
    --accelerations ${acc} \
    --disable-acs-region \
    --profile-timing
done
```

Expected output folders:

```text
output/CINE1acr2/val_img4ranking
output/CINE1acr4/val_img4ranking
output/CINE1acr8/val_img4ranking
output/CINE1acr16/val_img4ranking
```

Each output MAT should contain:

```text
key: img4ranking
shape: (176, 132, 12, 25)
```

### Quick MAT-To-MAT Shape Check

```bash
python - <<'PY'
from pathlib import Path
import scipy.io

gt_root = Path("/home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap")

for acc in (2, 4, 8, 16):
    pred_root = Path(f"output/CINE1acr{acc}/val_img4ranking")
    print(f"\nacc{acc}")
    for pred_path in sorted(pred_root.glob("Sub*.mat")):
        case = pred_path.stem
        pred = scipy.io.loadmat(pred_path)["img4ranking"]
        gt = scipy.io.loadmat(gt_root / f"{case}.mat")["gt"]
        print(case, pred.shape, gt.shape, pred.dtype, gt.dtype, pred.shape == gt.shape)
PY
```

### Visualize Outputs

```bash
for acc in 2 4 8 16; do
  python scripts/visualize_mat.py \
    output/CINE1acr${acc}/val_img4ranking \
    -o output/CINE1acr${acc}/figs
done
```

### Experimental Cautions

- R8 and R16 are closer to the pretrained model's default configured rates.
- R2 and R4 are exploratory unless validated or fine-tuned.
- No-ACS-filled masks require `--disable-acs-region` unless the path is known to use external smaps correctly.
- The current inference still applies the mask internally through `KspaceMaskd`; input k-space should remain fully sampled.
