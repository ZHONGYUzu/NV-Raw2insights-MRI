# Server Run Audit

Audit date: 2026-08-05 (Europe/Berlin)

## Scope And Method

This is a read-only audit of the shared project state under
`/home/students/studxuzho1/NV-Raw2insights-MRI`. The audit was performed from
`login-01` with the environment
`/home/students/studxuzho1/envs/nv-raw2insights-mri` activated. No model was
loaded and no inference or evaluation job was started. Evidence came from
existing output files, saved `config.json` files, result summaries, and Slurm
logs.

Server repository state at audit time:

```text
host: login-01
Python: 3.12.13
Git commit: 3e381f2
Git worktree: clean
```

## Executive Summary

The server contains ten completed formal inference runs: three in-house CINE
forced-ACS runs, three in-house CINE external-`dMap` raw-VISTA/no-forced-ACS
runs, one CMRxRecon2023 CINE LAX run, and three fastMRI Brain runs. Every formal
output directory contains 10 reconstruction MAT files.

All saved formal-run configurations point to the same model:

```text
model variant: nv_raw2insights_mri_base
checkpoint: nv_raw2insights_mri_base.pt
Hugging Face repository: nvidia/NV-Raw2insights-MRI
snapshot: 38860e9a6153d857b55439b3d3fd55f850eccdcd
resolved size: 3,038,170,078 bytes
HF cache blob ID: 667189bac5f159e52b2f1640b504b4efd549717c37720c2397e749d16ec056b8
fine-tuning: none recorded
```

The checkpoint content was not independently re-hashed during this audit,
because the checksum read was stopped to avoid sustained I/O on the login node.
The snapshot path and Hugging Face cache blob ID nevertheless match across all
saved run configurations and inspected logs. The acceleration, mask protocol,
dataset adapter, sensitivity-map source, and optional hard data consistency
changed between experiments; the checkpoint did not.

## Formal Run Inventory

| Family | Protocol | Nominal acceleration | Output | Completion | Recorded inference time | Metrics |
| --- | --- | ---: | --- | ---: | ---: | --- |
| In-house CINE | Internal CSM estimate, 20 forced ACS lines | 8 | `output/CINE_test_acc8` | 10/10 | Full fresh timing unavailable; final 2-case resume was 46.78 min | Complete, 2,975 frames |
| In-house CINE | Internal CSM estimate, 20 forced ACS lines | 16 | `output/CINE_test_acc16` | 10/10 | 321.46 min | Complete, 2,975 frames |
| In-house CINE | Internal CSM estimate, 20 forced ACS lines | 24 | `output/CINE_test_acc24` | 10/10 | Formal total not found | Complete, 2,975 frames |
| In-house CINE | External H5 `dMap`, raw VISTA, no forced ACS | 8 | `output/h5_converted` | 10/10 | 177.73 min | Partial: Sub0014 only, 350 frames |
| In-house CINE | External H5 `dMap`, raw VISTA, no forced ACS | 16 | `output/h5_converted_acc16` | 10/10 | 173.40 min | Partial: Sub0014 only, 350 frames |
| In-house CINE | External H5 `dMap`, raw VISTA, no forced ACS | 24 | `output/h5_converted_acc24` | 10/10 | 172.97 min | Partial: Sub0014 only, 350 frames |
| CMRxRecon2023 | Official CINE LAX cohort | 8 | `output/CMRxRecon2023CINELAX10Acc8Official` | 10/10 | 60.52 min | Complete, 360 frames |
| fastMRI Brain | AXT1 cohort, geometry-aligned adapter | 8 | `output/FastMRIBrainMulticoilVal10Acc8` | 10/10 | 54.45 min | Complete, 158 slices |
| fastMRI Brain | AXT1 cohort, geometry-aligned adapter | 16 | `output/FastMRIBrainMulticoilVal10Acc16` | 10/10 | 49.58 min | Complete, 158 slices |
| fastMRI Brain | AXT1 cohort, geometry-aligned adapter | 24 | `output/FastMRIBrainMulticoilVal10Acc24` | 10/10 | 49.62 min | Complete, 158 slices |

The raw-VISTA acc16 and acc24 jobs ran on `node-gpu-03` on 2026-07-15. The
CMRxRecon2023 job ran on `node-gpu-03` on 2026-07-16. The fastMRI three-rate job
ran on `node-gpu-01` on 2026-07-19. The retained SOP records the forced-ACS
acc8 resume on `node-gpu-07` and the complete forced-ACS acc16 run on
`node-gpu-01`.

## In-House CINE Forced-ACS Results

These runs use the fixed 10-case, variable-PE, FE=192 cohort, internally
estimated sensitivity maps, and 20 forced central ACS lines. Evaluation uses
2,975 slice-frame samples per acceleration.

| Acceleration | PSNR mean ± std (dB) | SSIM mean ± std | NRMSE mean ± std |
| ---: | ---: | ---: | ---: |
| 8 | 23.981 ± 4.399 | 0.778 ± 0.053 | 0.436 ± 0.142 |
| 16 | 22.058 ± 4.364 | 0.747 ± 0.055 | 0.541 ± 0.162 |
| 24 | 21.250 ± 4.298 | 0.729 ± 0.057 | 0.592 ± 0.171 |

Quality decreases monotonically as nominal acceleration rises: acc8 has the
highest PSNR and SSIM and the lowest NRMSE; acc24 has the weakest values. These
three rows are directly comparable within this protocol.

Evidence:

```text
Results/forcefill_masks/frame_metrics.csv
Results/forcefill_masks/summary_metrics.csv
Results/forcefill_masks/metrics_boxplot.png
Results/forcefill_masks/metrics_violin.png
```

## External-dMap Raw-VISTA / No-Forced-ACS Results

All three inference directories contain 10/10 outputs. The converted JSON
descriptors contain the `sensitivity_maps` key, whereas the forced-ACS dataset
descriptors contain only k-space and mask paths. The acc16 and acc24 Slurm logs
end normally with complete 10/10 progress and recorded elapsed times.

The available quantitative summary is not a full-cohort comparison: it covers
only `Sub0014` (14 slices × 25 frames = 350 frames per acceleration).

| Acceleration | PSNR mean (dB) | SSIM mean | NRMSE mean | NMSE mean |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 17.779 | 0.391 | 0.637 | 0.407 |
| 16 | 16.681 | 0.323 | 0.723 | 0.523 |
| 24 | 16.348 | 0.315 | 0.751 | 0.566 |

The ordering again favors acc8, but these values must not be compared directly
with the forced-ACS table as a pure sensitivity-map ablation: the sampling
protocol and evaluation normalization also differ. A full 10-case raw-VISTA
evaluation remains outstanding.

Evidence:

```text
Results/raw_vista/frame_metrics.csv
Results/raw_vista/summary_metrics.csv
logs/cine_h5_r2r3_41705_16.out
logs/cine_h5_r2r3_41705_24.out
```

## fastMRI Brain Results

The formal AXT1 cohort contains 10 cases and 158 evaluated slices per
acceleration. Metrics use the original direct intensity scale after geometry
alignment and center cropping to the reference shape.

| Acceleration | PSNR mean (dB) | SSIM mean | NRMSE mean | NMSE mean |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 15.549 | 0.484 | 0.822 | 0.697 |
| 16 | 15.505 | 0.479 | 0.826 | 0.702 |
| 24 | 15.480 | 0.462 | 0.827 | 0.701 |

Differences across acceleration are small relative to the overall error level.
The unusually weak and nearly flat acceleration response should be interpreted
with the documented fastMRI geometry/conditioning caveats rather than as proof
of acceleration robustness.

Evidence:

```text
Results/FastMRIBrainMulticoilVal10Acc8/summary_metrics.json
Results/FastMRIBrainMulticoilVal10Acc16/summary_metrics.json
Results/FastMRIBrainMulticoilVal10Acc24/summary_metrics.json
logs/fastmri_b10_aall_41798.out
```

## CMRxRecon2023 CINE LAX

The official ten-case acc8 output contains 10/10 reconstruction files and the
Slurm log records a successful 60.52-minute inference on `node-gpu-03`. Its
separate `ResultsCMRxRecon2023CINELAX10Acc8Official/` directory contains 360
frame-level metric rows, a boxplot, 10 reconstruction/reference comparison
PNGs, and a summary JSON. Aggregate results are PSNR 30.334 dB, SSIM 0.728,
NRMSE 0.336, and NMSE 0.119. The cohort mean effective acceleration is 4.136
(case range 4.000–4.340) despite its nominal acc8 class. The earlier one-case
debug artifact is retained under
`output/archiv/2026-07-17_140755/CMRxRecon2023CINELAXAcc8` and should not be
confused with the official ten-case output.

## Debug And Ablation Outputs

The following fastMRI outputs are development artifacts, not full formal
cohorts:

| Output | MAT files | Purpose indicated by name/config |
| --- | ---: | --- |
| `output/FastMRIBrainMulticoilValGeomFixDebugAcc8` | 1 | Geometry-fix debug |
| `output/FastMRIBrainMulticoilValGeomFixAxisAdapterDebugAcc8` | 1 | Axis-adapter debug |
| `output/FastMRIBrainMulticoilValGeomFixSliceWindowDebugAcc8` | 2 | Slice-window debug |
| `output/FastMRIBrainMulticoilValGeomFixSliceWindowHardDCDebugAcc8` | 1 | Hard-data-consistency debug |

Their result folders exist under `Results/`, but their small and differing
sample counts make them unsuitable for the main cross-dataset table.

## Reproducibility Findings And Gaps

1. No inspected formal run changed model weights. Every saved configuration
   resolves to the same Base checkpoint snapshot and cache blob.
2. The forced-ACS CINE acc8/16/24 comparison is the most complete current
   three-rate benchmark: all outputs and all metrics are present.
3. The external-`dMap` raw-VISTA runs have complete reconstructions but only a
   one-case quantitative evaluation.
4. The CMRxRecon2023 official cohort has complete metrics and comparison PNGs,
   but no violin plot was found in its separate results directory.
5. Fresh full-run timing is missing for forced-ACS acc8, and formal timing was
   not found for forced-ACS acc24.
6. Current server commit `3e381f2` is clean, but the Slurm logs do not record a
   Git commit for every historical job. The current fastMRI batch script has
   evolved since job 41798, so the log is the authoritative source for the
   output paths actually used by that job.
7. The legacy five-case 132×176 `CustomCINEDataR1/R2/R3` experiment remains in
   documentation, but its `CustomCINEOutputR*` directories were not present in
   the active server `output/` root during this audit.

## Recommended Next Actions

1. Run evaluation only (not inference) for all 10 external-`dMap` raw-VISTA
   cases at acc8/16/24 and save it in a new clearly named result directory.
2. Add a violin plot for the already evaluated official 10-case CMRxRecon2023
   output if uniform visualization across experiments is required.
3. Preserve job IDs, Git commit, checkpoint snapshot/blob ID, node, start/end
   time, and exact command in every future run log.
4. If strict byte identity is required, compute the checkpoint SHA-256 on a
   compute node or during a low-I/O period and add it to this record.
