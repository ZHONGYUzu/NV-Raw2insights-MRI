# NV-Raw2Insights-MRI In-House CINE Benchmark Report

Report date: 2026-07-12

Source SOP: [`NV-Raw2insights-MRI_SOP.md`](NV-Raw2insights-MRI_SOP.md)

Documentation scope: this report covers the active variable-PE, FE=192,
10-subject benchmark. It does not replace the historical 132×176
`CustomCINEDataR1/R2/R3` baseline records.

## 1. Benchmark Objective

This benchmark establishes a reproducible baseline for NV-Raw2Insights-MRI on
selected in-house 2D CINE MRI cases. The goal is to compare reconstruction
quality across VISTA acceleration settings using a fixed test cohort, a fixed
ground-truth generation method, and consistent full-frame evaluation.

## 2. Experimental Setup

| Item | Setting |
| --- | --- |
| Model | NV-Raw2Insights-MRI Base |
| Checkpoint | `nv_raw2insights_mri_base.pt` from `nvidia/NV-Raw2insights-MRI` |
| Parameters | 758.91M |
| Fine-tuning | No |
| Dataset | In-house CINE MRI H5 data |
| Test cases | 10 fixed cases |
| Test subject file | `docs/test_subjects.txt` |
| Coil sensitivity maps | H5 `dMap` exists but is not used in Experiment 1 |
| GT source | H5 `dImgC` |
| GT output | `dataset/GT_from_dImgC/<case>.mat` |
| GT MAT key | `gt` |
| GT / prediction layout | `(frequency, phase, slice, time)` |
| GT normalization | Max magnitude per case |
| Acceleration settings | acc8, acc16, acc24 |
| Mask seed | 8 |
| ACS handling | Force 20 central phase lines with `--acs-lines 20` |
| Main metrics | PSNR, SSIM, NMSE |
| Hardware recorded in SOP | NVIDIA Tesla V100-SXM2-32GB; driver 535.309.01; CUDA 12.2 |

Selected cases:

```text
Sub0014 Sub0026 Sub0030 Sub0047 Sub0049 Sub0051 Sub0096 Sub0105 Sub0112 Sub0122
```

## 3. Data And Shape Summary

All selected cases have 25 cardiac time frames, 15 compressed coils in
`kSpace`, and frequency size `FE = 192`. Phase encoding size and slice count
vary by case, so masks must be selected per PE group.

| PE | Cases |
| ---: | --- |
| 180 | Sub0014 |
| 174 | Sub0030 |
| 162 | Sub0026, Sub0112 |
| 156 | Sub0047, Sub0049, Sub0051, Sub0096, Sub0105, Sub0122 |

Slice counts recorded in the SOP:

| Case | Slices | Frames |
| --- | ---: | ---: |
| Sub0014 | 14 | 25 |
| Sub0026 | 15 | 25 |
| Sub0030 | 12 | 25 |
| Sub0047 | 3 | 25 |
| Sub0049 | 3 | 25 |
| Sub0051 | 14 | 25 |
| Sub0096 | 17 | 25 |
| Sub0105 | 11 | 25 |
| Sub0112 | 18 | 25 |
| Sub0122 | 12 | 25 |

Total evaluation volume:

```text
119 slices x 25 frames = 2975 frame-level samples per acceleration
2975 x 3 accelerations = 8925 expected metric rows
```

## 4. Workflow Status

Completed according to the SOP:

- Fixed 10-case benchmark cohort recorded in `docs/test_subjects.txt`.
- Generated GT from H5 `dImgC` using max-magnitude normalization per case.
- Saved GT as MAT files with key `gt` and layout `(frequency, phase, slice, time)`.
- Generated GT thumbnail grids and temporal GIFs.
- Inspected H5 shapes and confirmed PE-specific mask requirements.
- Confirmed required VISTA masks for PE `156`, `162`, `174`, and `180` at acc8, acc16, and acc24.
- Defined dataset/output roots for acc8, acc16, and acc24.
- Recorded full acc16 inference timing.
- Recorded acc8 resume timing for the final two remaining cases.

Pending or not yet recorded in the SOP:

- Fresh full 10-case acc8 timing from an empty output folder.
- Full acc24 inference timing.
- Final PSNR, SSIM, and NMSE summary metrics.
- Reconstruction/error-map visualizations.
- Final qualitative comparison across acc8, acc16, and acc24.

## 5. Dataset And Output Roots

| Acceleration | Dataset root | Output root |
| --- | --- | --- |
| acc8 | `dataset/CINE_test_acc8` | `output/CINE_test_acc8` |
| acc16 | `dataset/CINE_test_acc16` | `output/CINE_test_acc16` |
| acc24 | `dataset/CINE_test_acc24` | `output/CINE_test_acc24` |

Mask files are selected by case PE and acceleration, for example:

```text
mask_VISTA_156x25_acc8_8.txt
mask_VISTA_162x25_acc16_8.txt
mask_VISTA_180x25_acc24_8.txt
```

Converted mask filenames use model-compatible acceleration aliases:

```text
ktRadial8, ktRadial16, ktRadial24
```

## 6. Inference Timing Results

### acc8

The SOP records an acc8 resume run, not a fresh 10-case run. Existing MAT files
were skipped and only the remaining two cases were processed.

| Item | Recorded value |
| --- | --- |
| Node | `node-gpu-07` |
| Test files before filtering | 10 |
| Test files after filtering | 2 |
| Processed cases | 2/10 |
| Resume elapsed time | 46.78 min |

Per-case timing:

| Case | Slices | Forwards | Total time (s) | Total time (min) | Model time/forward (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sub0112 | 18 | 450 | 1678.18 | 27.97 | 3.696 |
| Sub0122 | 12 | 300 | 1070.48 | 17.84 | 3.535 |

Because this was a resume run, it should not be used as the full acc8 wall-time
benchmark unless the previously completed eight-case timing log is also added.

### acc16

The SOP records a complete 10-case acc16 inference run.

| Item | Recorded value |
| --- | --- |
| Node | `node-gpu-01` |
| Test files before filtering | 10 |
| Test files after filtering | 10 |
| Completed cases | 10/10 |
| Total elapsed time | 321.46 min |

Per-case timing:

| Case | Slices | Forwards | Total time (s) | Total time (min) | Model time/forward (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sub0014 | 14 | 350 | 2442.74 | 40.71 | 6.936 |
| Sub0026 | 15 | 375 | 2444.92 | 40.75 | 6.481 |
| Sub0030 | 12 | 300 | 1991.88 | 33.20 | 6.596 |
| Sub0047 | 3 | 75 | 444.16 | 7.40 | 5.879 |
| Sub0049 | 3 | 75 | 469.40 | 7.82 | 6.215 |
| Sub0051 | 14 | 350 | 2169.36 | 36.16 | 6.159 |
| Sub0096 | 17 | 425 | 2628.56 | 43.81 | 6.150 |
| Sub0105 | 11 | 275 | 1648.53 | 27.48 | 5.956 |
| Sub0112 | 18 | 450 | 2863.77 | 47.73 | 6.321 |
| Sub0122 | 12 | 300 | 1882.28 | 31.37 | 6.230 |

Summary from the table:

| Metric | Value |
| --- | ---: |
| Total forwards | 2975 |
| Sum of per-case times | 315.20 min |
| Recorded total elapsed time | 321.46 min |
| Mean case time | 31.52 min |
| Median case time | 34.68 min |
| Fastest case | Sub0047, 7.40 min |
| Slowest case | Sub0112, 47.73 min |

### acc24

Full acc24 timing is not recorded yet in the SOP. Expected command:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CINE_test_acc24/json_input \
  -o output/CINE_test_acc24 \
  --profile-timing
```

## 7. Quantitative Results

Quantitative metrics are available for acc8 and acc16. The SOP defines
evaluation against MAT GT generated from `dImgC`:

```bash
python scripts/evaluate_cine_mat_gt_metrics.py \
  --acc acc8=output/CINE_test_acc8/val_img4ranking \
  --acc acc16=output/CINE_test_acc16/val_img4ranking \
  --acc acc24=output/CINE_test_acc24/val_img4ranking \
  --gt-root dataset/GT_from_dImgC \
  -o output/CINE_test_metrics_dImgC
```

Expected outputs:

```text
output/CINE_test_metrics_dImgC/frame_metrics.csv
output/CINE_test_metrics_dImgC/summary_metrics.csv
output/CINE_test_metrics_dImgC/metrics_boxplot.png
output/CINE_test_metrics_dImgC/metrics_violin.png
```

Current metric summary table:

| Acceleration | Cases | Frame rows | PSNR mean ± std | PSNR median | NRMSE mean ± std | NMSE mean ± std | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| acc8 | 10 | 2975 | 23.981 ± 4.399 | 24.374 | 0.436 ± 0.142 | 0.211 ± 0.138 | Complete |
| acc16 | 10 | 2975 | 22.058 ± 4.364 | 22.376 | 0.541 ± 0.162 | 0.319 ± 0.186 | Complete |
| acc24 | 10 | 2975 | TBD | TBD | TBD | TBD | Pending |

Detailed metric summary:

| Acceleration | PSNR Q25 | PSNR Q75 | NRMSE median | NRMSE Q25 | NRMSE Q75 | NMSE median | NMSE Q25 | NMSE Q75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| acc8 | 21.187 | 26.947 | 0.427 | 0.342 | 0.512 | 0.182 | 0.117 | 0.262 |
| acc16 | 19.223 | 24.993 | 0.537 | 0.441 | 0.643 | 0.288 | 0.194 | 0.413 |

Additional error metrics:

| Acceleration | MSE mean ± std | MSE median | MSE Q25 | MSE Q75 | MAE mean ± std | MAE median | MAE Q25 | MAE Q75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| acc8 | 0.0121 ± 0.0099 | 0.0091 | 0.0059 | 0.0147 | 0.0414 ± 0.0102 | 0.0395 | 0.0329 | 0.0474 |
| acc16 | 0.0181 ± 0.0125 | 0.0154 | 0.0092 | 0.0229 | 0.0494 ± 0.0120 | 0.0471 | 0.0400 | 0.0565 |

Current observation: acc8 outperforms acc16 on the recorded metrics, with
higher PSNR and lower NRMSE, NMSE, MSE, and MAE. acc24 remains pending.

## 8. Visualization Status

Completed GT visualization outputs:

```text
output/GT_from_dImgC_figs
output/GT_from_dImgC_gifs
```

Qualitative reconstruction visualization should include:

```text
Undersampled input | Reconstruction | Ground Truth | Error map
```

Recommended qualitative outputs still pending:

- Per-case reconstruction PNG grids for acc8, acc16, and acc24.
- GT/reconstruction/error comparison figures.
- Optional reconstruction GIFs using the center slice per case.
- Boxplot and violin plot from final metric CSVs.

## 9. Current Findings And Risks

- The selected benchmark cases are not shape-identical. PE-specific masks are
  required for correct conversion and validation.
- The source VISTA masks are phase/time masks and must be expanded to
  `(time, phase, frequency)` with `frequency-size = 192`.
- Central ACS lines must be positive for all frames when `use_acs_region=true`;
  the SOP uses `--acs-lines 20`.
- Inference skips cases with existing MAT outputs. Timing logs must state
  whether a run is fresh or resumed.
- GT is derived from `dImgC` and normalized per case by max magnitude. This must
  be reported when interpreting PSNR, SSIM, and NMSE.
- The model is computationally expensive. Forwards per case equal
  `slices x 25`, so slice count is the main driver of inference time.

## 10. Next Steps

1. Record whether acc8 has a complete fresh 10-case timing log or only resumed
   logs.
2. Complete acc24 full inference and add timing to the SOP.
3. Run `scripts/evaluate_cine_mat_gt_metrics.py` after all three acceleration
   outputs are present.
4. Verify `frame_metrics.csv` contains 8925 rows for the three acceleration
   settings.
5. Fill the final PSNR, SSIM, and NMSE table with mean, standard deviation,
   median, best case, and worst case.
6. Generate reconstruction/error-map figures for the weekly summary.
