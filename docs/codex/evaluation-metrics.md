# Evaluation And Metrics

This note documents how to inspect custom CINE inference outputs and compare
them against normalized fully sampled ground truth.

## Prediction Layout

Current inference writes:

```text
<output_root>/val_img4ranking/Sub0001.mat
```

MAT key:

```text
img4ranking
```

Expected shape:

```text
(frequency, phase, slice, time) = (176, 132, 12, 25)
```

The output is `float32` magnitude data. The folder/key names are historical
compatibility names; the current array is not the old ranking crop.

## Normalized Ground Truth Layout

Known normalized GT files:

```text
/home/students/studxuzho1/dataset_v0/norm_img/norm_img_Sub0001.npy
```

Shape:

```text
(slice, time, phase, frequency, channel) = (12, 25, 132, 176, 1)
```

To align with current prediction orientation:

```python
import numpy as np
import scipy.io

pred = scipy.io.loadmat(
    "output/CustomCINEOutputR1/val_img4ranking/Sub0001.mat"
)["img4ranking"]                                           # (176, 132, 12, 25)

gt = np.load(
    "/home/students/studxuzho1/dataset_v0/norm_img/norm_img_Sub0001.npy"
)                                                          # (12, 25, 132, 176, 1)
gt = np.abs(gt[..., 0]).transpose(3, 2, 0, 1)              # (176, 132, 12, 25)

assert pred.shape == gt.shape, (pred.shape, gt.shape)
```

Shape alignment does not prove intensity normalization is identical. Confirm the
GT normalization method before interpreting PSNR, SSIM, or NMSE.

## Quick Quality Inspection

Use `scripts/inspect_cine_recon_quality.py` for per-case metrics and comparison
PNGs against normalized NPY GT:

```bash
python scripts/inspect_cine_recon_quality.py \
  output/CustomCINEOutputR1/val_img4ranking \
  --gt-root /home/students/studxuzho1/dataset_v0/norm_img \
  -o output/CustomCINEOutputR1/quality_checks
```

Outputs include:

```text
output/CustomCINEOutputR1/quality_checks/metrics.csv
output/CustomCINEOutputR1/quality_checks/Sub0001_slice##_time##_compare.png
```

The script reports NMSE, PSNR, SSIM summary values, prediction stats, GT stats,
and comparison PNG paths.

## Multi-Acceleration Metrics

Use `scripts/evaluate_cine_acc_metrics.py` to compare R1/R2/R3 outputs across
shared cases:

```bash
python scripts/evaluate_cine_acc_metrics.py \
  --acc acc8=output/CustomCINEOutputR1/val_img4ranking \
  --acc acc16=output/CustomCINEOutputR2/val_img4ranking \
  --acc acc24=output/CustomCINEOutputR3/val_img4ranking \
  --gt-root /home/students/studxuzho1/dataset_v0/norm_img \
  -o output/acc_metrics
```

Outputs:

```text
output/acc_metrics/frame_metrics.csv
output/acc_metrics/summary_metrics.csv
output/acc_metrics/metrics_boxplot.png
output/acc_metrics/metrics_violin.png
```

Frame-level rows are computed for every shared case, slice, and time frame.
Default plotted metrics are:

```text
psnr, nrmse, nmse
```

Optional metrics include:

```text
mse, mae
```

## Qualitative Acceleration Comparison

Use `scripts/plot_cine_acc_comparison.py` to make input/recon/GT/error grids
across acceleration settings:

```bash
python scripts/plot_cine_acc_comparison.py \
  --acc acc8=output/CustomCINEOutputR1/val_img4ranking \
  --acc acc16=output/CustomCINEOutputR2/val_img4ranking \
  --acc acc24=output/CustomCINEOutputR3/val_img4ranking \
  --gt-root /home/students/studxuzho1/dataset_v0/norm_img \
  -o output/acc_comparison_pngs
```

If the output roots follow the established `CustomCINEOutputR*` naming, the
script can infer matching JSON input directories. Otherwise pass explicit
inputs:

```bash
python scripts/plot_cine_acc_comparison.py \
  --acc acc8=/path/to/acc8/val_img4ranking \
  --input-json acc8=/path/to/acc8/json_input \
  --gt-root /home/students/studxuzho1/dataset_v0/norm_img \
  -o output/acc_comparison_pngs
```

## H5-Derived GT Comparison

Use `scripts/compare_cine_h5_gt_to_recon.py` when you want to compare a saved
reconstruction against a reference derived directly from the source H5.

`kSpace+dMap` reference:

```bash
python scripts/compare_cine_h5_gt_to_recon.py \
  --h5 /mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0001.h5 \
  --recon output/CustomCINEOutputR1/val_img4ranking/Sub0001.mat \
  --gt-source kspace-dmap \
  --png output/quality_checks/Sub0001_h5_kspace_dmap_compare.png
```

`dImgC` sanity-check reference:

```bash
python scripts/compare_cine_h5_gt_to_recon.py \
  --h5 /mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0001.h5 \
  --recon output/CustomCINEOutputR1/val_img4ranking/Sub0001.mat \
  --gt-source dimgc \
  --png output/quality_checks/Sub0001_h5_dimgc_compare.png
```

Existing normalized NPY reference:

```bash
python scripts/compare_cine_h5_gt_to_recon.py \
  --npy /home/students/studxuzho1/dataset_v0/norm_img/norm_img_Sub0001.npy \
  --recon output/CustomCINEOutputR1/val_img4ranking/Sub0001.mat \
  --gt-source norm-npy \
  --png output/quality_checks/Sub0001_norm_npy_compare.png
```

Current project decision:

- Use `kSpace+dMap` as the more explicit generated GT method.
- Use `dImgC` as a sanity-check reference unless the source data owner documents
  `dImgC` as the official evaluation target.
- Use normalized NPY GT for the established metric scripts.

## Old Ranking Crop

Do not apply `run4Ranking`, central two-slice selection, first-three-frame
selection, or the old spatial crop unless intentionally reproducing the old
ranking protocol.

Current full-size evaluation should compare arrays shaped:

```text
(176, 132, 12, 25)
```
