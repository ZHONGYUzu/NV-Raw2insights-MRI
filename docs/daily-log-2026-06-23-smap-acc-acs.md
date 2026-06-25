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
    --mask-type ktRadialVISTA${acc} \
    --case-glob 'Sub00*_kspace_full.mat' \
    --frequency-size 176 \
    --no-force-acs
done
```

This keeps the VISTA mask center unmodified. Because the ACS center is not forced to one, later validation should use `--skip-acs-check`, and inference should use `--disable-acs-region`.

The saved mask filename uses a repo/model-facing alias:

```text
Sub0001_mask_ktRadialVISTA2.mat
Sub0001_mask_ktRadialVISTA4.mat
Sub0001_mask_ktRadialVISTA8.mat
Sub0001_mask_ktRadialVISTA16.mat
```

The actual mask values still come from the VISTA TXT files. The `ktRadialVISTA*` string is only a compatibility label so the model still maps the mask into its radial-like conditioning bucket while preserving the VISTA identity in the filename.

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
    --fixed-mask-types mask_ktRadialVISTA${acc} \
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
    --fixed-mask-types mask_ktRadialVISTA${acc} \
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

### Inference Resume And Progress Behavior

The inference script skips only complete case-level output files.

Completed outputs are detected under:

```text
output/CINE1acr<acc>/val_img4ranking/SubXXXX.mat
```

If inference is interrupted in the middle of a case, that case is not resumed from the last forward pass. It is rerun from the beginning the next time. Already completed case MAT files in the same output folder are skipped.

To resume the R16 run, for example:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CINE1acr16/json_input \
  -o output/CINE1acr16 \
  --fixed-mask-types mask_ktRadialVISTA16 \
  --accelerations 16 \
  --disable-acs-region \
  --profile-timing
```

Current server run example observed:

```text
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CINE1acr16/json_input \
  -o output/CINE1acr16 \
  --fixed-mask-types mask_ktRadialVISTA16 \
  --accelerations 16 \
  --disable-acs-region \
  --profile-timing
```

The progress bar `0/10 ... 10/10` refers to the current acceleration dataset only. The input folder `dataset/CINE1acr16/json_input` containing 10 JSON files does not mean inference is complete; completion is determined by the 10 MAT files under `output/CINE1acr16/val_img4ranking`.

Check completed output count:

```bash
find output/CINE1acr16/val_img4ranking -maxdepth 1 -name 'Sub*.mat' | wc -l
```

## 2026-06-24 Evaluation Workflow After Full Inference

After all four full inference runs finish, expected prediction roots are:

```text
output/CINE1acr2/val_img4ranking
output/CINE1acr4/val_img4ranking
output/CINE1acr8/val_img4ranking
output/CINE1acr16/val_img4ranking
```

Prediction MAT format:

```text
key: img4ranking
shape: (176, 132, 12, 25)
layout: (frequency, phase, slice, time)
```

Ground-truth MAT root:

```text
/home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap
```

Ground-truth MAT format:

```text
key: gt
shape: (176, 132, 12, 25)
layout: (frequency, phase, slice, time)
```

### Evaluation Step 1: Shape And Finite Check

```bash
python - <<'PY'
from pathlib import Path
import scipy.io
import numpy as np

gt_root = Path("/home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap")

for acc in (2, 4, 8, 16):
    pred_root = Path(f"output/CINE1acr{acc}/val_img4ranking")
    print(f"\n===== acc{acc} =====")
    for pred_path in sorted(pred_root.glob("Sub*.mat")):
        case = pred_path.stem
        pred = scipy.io.loadmat(pred_path)["img4ranking"]
        gt = scipy.io.loadmat(gt_root / f"{case}.mat")["gt"]
        print(
            case,
            "pred", pred.shape, pred.dtype,
            "gt", gt.shape, gt.dtype,
            "finite", np.isfinite(pred).all(),
            "match", pred.shape == gt.shape,
        )
PY
```

Expected result for every case:

```text
finite True
match True
```

Observed server result:

```text
acc2:  Sub0001-Sub0010 all pred (176, 132, 12, 25) float32; gt (176, 132, 12, 25) float32; finite True; match True
acc4:  Sub0001-Sub0010 all pred (176, 132, 12, 25) float32; gt (176, 132, 12, 25) float32; finite True; match True
acc8:  Sub0001-Sub0010 all pred (176, 132, 12, 25) float32; gt (176, 132, 12, 25) float32; finite True; match True
acc16: Sub0001-Sub0009 observed all pred (176, 132, 12, 25) float32; gt (176, 132, 12, 25) float32; finite True; match True
```

Note:

- The pasted terminal output for acc16 shows Sub0001 through Sub0009. If Sub0010 was also checked afterward, record it as matching too.
- This confirms the MAT-to-MAT orientation is correct for the completed/observed outputs.

### Evaluation Step 2: Per-Acceleration Reconstruction Grids

These images show each reconstruction independently.

```bash
for acc in 2 4 8 16; do
  python scripts/visualize_mat.py \
    output/CINE1acr${acc}/val_img4ranking \
    -o output/CINE1acr${acc}/figs
done
```

Output example:

```text
output/CINE1acr8/figs/Sub0001.png
```

Image format:

```text
one PNG per case per acceleration
rows    = 12 slices
columns = 25 time frames
image   = reconstruction magnitude
```

This view is best for quickly checking blank frames, slice/time ordering, or severe artifacts within one acceleration setting.

### Evaluation Step 3: MAT-To-MAT Quantitative Metrics

Use the MAT-to-MAT evaluator for the current GT format. It reads prediction key `img4ranking` and GT key `gt`; both arrays are expected to be full-size `(frequency, phase, slice, time)`.

```bash
python scripts/evaluate_cine_mat_gt_metrics.py \
  --acc acc2=output/CINE1acr2/val_img4ranking \
  --acc acc4=output/CINE1acr4/val_img4ranking \
  --acc acc8=output/CINE1acr8/val_img4ranking \
  --acc acc16=output/CINE1acr16/val_img4ranking \
  --gt-root /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap \
  -o output/CINE1_metrics_mat_gt
```

Expected metric count:

```text
10 cases x 12 slices x 25 frames x 4 accelerations = 12000 frame rows
```

Metric outputs:

```text
output/CINE1_metrics_mat_gt/frame_metrics.csv
output/CINE1_metrics_mat_gt/summary_metrics.csv
output/CINE1_metrics_mat_gt/metrics_boxplot.png
output/CINE1_metrics_mat_gt/metrics_violin.png
```

CSV formats:

```text
frame_metrics.csv:
acceleration, case, slice, time, shape, psnr, nrmse, nmse, mse, mae

summary_metrics.csv:
acceleration, num_frames, num_cases, psnr_mean, psnr_std, ...
```

### Evaluation Step 4: Cross-Acceleration Visual Comparison

This makes one comparison PNG per case at one selected slice/time using the MAT GT files.

```bash
python scripts/plot_cine_mat_gt_acc_comparison.py \
  --acc acc2=output/CINE1acr2/val_img4ranking \
  --acc acc4=output/CINE1acr4/val_img4ranking \
  --acc acc8=output/CINE1acr8/val_img4ranking \
  --acc acc16=output/CINE1acr16/val_img4ranking \
  --gt-root /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap \
  -o output/CINE1_acc_comparison_mat_gt \
  --slice-index 6 \
  --time-index 12
```

Output examples:

```text
output/CINE1_acc_comparison_mat_gt/Sub0001_slice06_time12.png
...
output/CINE1_acc_comparison_mat_gt/Sub0010_slice06_time12.png
```

Image format:

```text
columns: GT | acc2 | acc4 | acc8 | acc16
row 1:   GT image and reconstruction images
row 2:   absolute error maps
row 3:   high-contrast absolute error maps, not signed difference
```

This is the preferred visual format for comparing acceleration settings on the same slice/time frame.

### Evaluation Review Order

Recommended inspection order:

```text
1. Shape/finite check output
2. output/CINE1acr*/figs/Sub0001.png
3. output/CINE1_acc_comparison_mat_gt/Sub0001_slice06_time12.png
4. output/CINE1_metrics_mat_gt/summary_metrics.csv
5. output/CINE1_metrics_mat_gt/frame_metrics.csv for detailed outliers
6. output/CINE1_gifs/acc*/Sub*_gt_recon_error.gif for temporal behavior
```

Expected qualitative trend if the model behaves normally:

```text
acc2  should usually look best or close to best
acc4  should be good
acc8  should be familiar/default-ish
acc16 may show more artifacts
```

R2 and R4 remain exploratory because they are outside the original confirmed pretrained setup.

### Evaluation Step 5: Per-Case GIFs For Each Acceleration

Added `scripts/make_cine_acc_gifs.py`.

Purpose:

- Generate animated CINE GIFs after quantitative metrics and cross-acceleration PNG comparisons.
- Produce one GIF per case per acceleration.
- Animate all time frames for one selected slice only.
- For the current review, use the sixth slice for every case. In script arguments, prefer `--slice-number 6` because it is one-based and directly means the sixth slice.
- If GT is provided, each GIF frame is a three-panel comparison:

```text
GT | Recon | Abs error
```

Default output root:

```text
output/CINE1_gifs
```

Recommended command for the current CINE1 experiment:

```bash
python scripts/make_cine_acc_gifs.py \
  --acc acc2=output/CINE1acr2/val_img4ranking \
  --acc acc4=output/CINE1acr4/val_img4ranking \
  --acc acc8=output/CINE1acr8/val_img4ranking \
  --acc acc16=output/CINE1acr16/val_img4ranking \
  --gt-root /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap \
  -o output/CINE1_gifs \
  --slice-number 6 \
  --fps 5
```

Expected output layout:

```text
output/CINE1_gifs/
  acc2/
    Sub0001_acc2_slice06_gt_recon_error.gif
    ...
    Sub0010_acc2_slice06_gt_recon_error.gif
  acc4/
    Sub0001_acc4_slice06_gt_recon_error.gif
    ...
  acc8/
    ...
  acc16/
    ...
```

Each GIF:

```text
case: one subject, e.g. Sub0001
acceleration: one setting, e.g. acc8
slice: sixth slice, which is one-based slice number 6 and zero-based index 5
frames: 25 cardiac time frames
panels per frame: GT | Recon | Abs error
```

For a quick smoke test before generating all GIFs:

```bash
python scripts/make_cine_acc_gifs.py \
  --acc acc2=output/CINE1acr2/val_img4ranking \
  --acc acc4=output/CINE1acr4/val_img4ranking \
  --acc acc8=output/CINE1acr8/val_img4ranking \
  --acc acc16=output/CINE1acr16/val_img4ranking \
  --gt-root /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap \
  -o output/CINE1_gifs_smoke \
  --slice-number 6 \
  --fps 5 \
  --max-cases 1
```

If GT panels are not needed and only reconstruction GIFs are desired, omit `--gt-root`:

```bash
python scripts/make_cine_acc_gifs.py \
  --acc acc8=output/CINE1acr8/val_img4ranking \
  -o output/CINE1_gifs_recon_only \
  --slice-number 6
```

GIF interpretation:

- The `GT` panel is the MAT ground truth from `dataset/GT_from_kspace_dMap`, key `gt`.
- The `Recon` panel is the model output from `val_img4ranking`, key `img4ranking`.
- The `Abs error` panel is `abs(recon - gt)` for the same frame.
- All panels use the current orientation used in comparison PNGs.
