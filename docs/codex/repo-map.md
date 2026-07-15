# Legacy Baseline Repo Map

This map is scoped to the original five-case 132×176 custom-CINE baseline. See
[`../README.md`](../README.md) before applying these paths to another cohort.
The workflow is server-oriented; local Codex work should prefer static
inspection, focused code edits, and lightweight checks.

## Top-Level Docs

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Authoritative agent instructions, project constraints, server paths, and custom CINE runbook. |
| `FYI.md` | Human project log and detailed explanatory notes. |
| `README.md` | General project overview. |
| `docs/setup.md` | General environment setup notes. |
| `docs/inference.md` | General inference guide, including custom CINE snippets. |
| `docs/training.md` | General training guide. |
| `docs/README.md` | Documentation hierarchy and experiment-generation map. |
| `docs/codex/` | Focused Codex/project-memory docs for custom CINE work. |
| `docs/history/` | Dated logs and superseded analyses. |

## Main Configs

| Path | Notes |
| --- | --- |
| `configs/nv_raw2insights_mri_small.json` | Small model config. |
| `configs/nv_raw2insights_mri_base.json` | Active config for confirmed custom CINE five-case runs. |
| `configs/nv_raw2insights_mri_large.json` | Large model config. |

Current custom CINE notes usually assume `configs/nv_raw2insights_mri_base.json`.
The base config includes fixed mask aliases for accelerations `8`, `16`, and
`24`, and sets `use_acs_region=true`.

## Core Inference Code

| Path | Purpose |
| --- | --- |
| `scripts/inference.py` | Main inference entry point. Defaults to `dataset/CustomCINEDataR1/json_input` and `output/CustomCINEOutputR1`. |
| `scripts/readers.py` | Contains `CMRxReconReader`, which reads JSON descriptors and MAT k-space/mask files. |
| `scripts/transforms.py` | Contains masking and MRI transforms, including fixed-mask acceleration parsing. |
| `scripts/utils.py` | Utility functions, including output saving helpers. |
| `scripts/run4ranking.py` | Historical ranking crop implementation. Current custom CINE inference no longer applies it in postprocessing. |

## Custom CINE Conversion And Validation

| Path | Purpose |
| --- | --- |
| `scripts/convert_cine_h5_kspace_to_mat.py` | Converts source H5 `kSpace` into CMRxRecon-style MAT files and JSON descriptors. |
| `scripts/convert_vista_txt_mask_to_mat.py` | Converts one VISTA TXT mask into case-specific MAT masks and updates JSON descriptors. |
| `scripts/create_custom_cine_acc_dataset.py` | Creates R2/R3-style dataset variants with a different fixed mask acceleration while reusing R1 k-space. |
| `scripts/validate_cine_inference_data.py` | Checks converted JSON, k-space, mask shape, mask values, acceleration naming, and ACS center positivity. |

## Visualization And Evaluation

| Path | Purpose |
| --- | --- |
| `scripts/visualize_mat.py` | Creates PNG grids from saved MAT reconstructions. |
| `scripts/inspect_cine_recon_quality.py` | Compares predictions against normalized GT, writes metrics CSV and comparison PNGs. |
| `scripts/evaluate_cine_acc_metrics.py` | Computes frame-level metrics across acceleration outputs and writes summary CSV/plots. |
| `scripts/plot_cine_acc_comparison.py` | Plots input/reconstruction/GT/error grids across acceleration settings. |
| `scripts/compare_cine_h5_gt_to_recon.py` | Compares a saved reconstruction against GT derived from H5 `kSpace+dMap`, H5 `dImgC`, or normalized NPY. |
| `scripts/plot_cine_qualitative_results.py` | Qualitative plotting helper. |
| `scripts/make_cine_gif.py` | Creates GIFs for CINE visualization. |
| `scripts/calculate_mask_acceleration.py` | Mask acceleration inspection helper. |

## Established Custom CINE Layout

The established R1 input dataset is:

```text
dataset/CustomCINEDataR1/
  MultiCoil/
    Cine/
      UnderSample_TaskR1/
        Sub0001_kspace_full.mat
        ...
      Mask_TaskR1/
        Sub0001_mask_ktRadial8.mat
        ...
  json_input/
    Sub0001.json
    ...
    Sub0005.json
```

Matching output roots:

| Label | Dataset root | Output root | Mask acceleration |
| --- | --- | --- | --- |
| `R1` | `dataset/CustomCINEDataR1` | `output/CustomCINEOutputR1` | acc8 |
| `R2` | `dataset/CustomCINEDataR2` | `output/CustomCINEOutputR2` | acc16 |
| `R3` | `dataset/CustomCINEDataR3` | `output/CustomCINEOutputR3` | acc24 |

`UnderSample_TaskR1` and `Mask_TaskR1` are CMRxRecon-style folder names used for
reader compatibility. They stay the same across R1/R2/R3 custom experiment
roots.
