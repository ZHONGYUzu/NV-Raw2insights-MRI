# Documentation Guide

Use this page to choose the right document before copying paths or commands.
The repository contains upstream usage guides, an active in-house benchmark,
an older custom-CINE baseline, and dated experiment records. Those scopes are
intentionally separate.

## Authority And Scope

| Scope | Primary document | Notes |
| --- | --- | --- |
| Repository overview | [`../README.md`](../README.md) | Public project summary and quick start. |
| Agent safety and project constraints | [`../AGENTS.md`](../AGENTS.md) | Operating rules and retained legacy-baseline runbook. |
| General setup, inference, and training | [`setup.md`](setup.md), [`inference.md`](inference.md), [`training.md`](training.md) | General user guides. |
| CMRxRecon 2023 one-case compatibility run | [`CMRxRecon2023_P001_LAX_acc8.md`](CMRxRecon2023_P001_LAX_acc8.md) | Native P001 LAX full-sampled k-space with the official nominal acc8 mask. |
| fastMRI brain multi-coil inference | [`FastMRI_brain_inference.md`](FastMRI_brain_inference.md) | Native H5, reproducible AXT2 validation cohort, generated acc8 mask, and RSS evaluation. |
| Active 10-case benchmark procedure | [`NV-Raw2insights-MRI_SOP.md`](NV-Raw2insights-MRI_SOP.md) | Variable PE, FE 192, `dImgC` GT, and `CINE_test_acc*` roots. |
| Active benchmark status/results | [`CINE_benchmark_report.md`](CINE_benchmark_report.md) | Completed, pending, timing, and risk record for the SOP cohort. |
| Legacy 5-case custom-CINE baseline | [`codex/README.md`](codex/README.md) | Fixed 132×176 cases using `CustomCINEDataR1/R2/R3`. |
| Dated investigations and run logs | [`history/README.md`](history/README.md) | Evidence and historical context; not the default source for current commands. |
| Human project memory | [`../FYI.md`](../FYI.md) | Dated explanations; sections may describe older code states. |

## Experiment Generations

| Generation | Cohort/layout | Sensitivity maps and ACS | Main paths | Status |
| --- | --- | --- | --- | --- |
| Legacy R1/R2/R3 baseline | 5 cases, PE 132, FE 176 | Internal maps; 20 forced ACS lines | `dataset/CustomCINEDataR*`, `output/CustomCINEOutputR*` | Retained reproducible baseline. |
| CINE1 external-map study | 10 cases, PE 132, FE 176 | External H5 `dMap`; no forced ACS | `dataset/CINE1*`, `output/CINE1*` | Historical experiment. |
| H5-converted acc8 run | 10 variable-PE cases, FE 192 | External H5 `dMap`; no forced ACS | `dataset/h5_converted`, `output/h5_converted` | Completed dated run. |
| Current benchmark | 10 variable-PE cases, FE 192 | SOP experiment 1 uses internal maps and 20 forced ACS lines | `dataset/CINE_test_acc*`, `output/CINE_test_acc*` | Active benchmark workflow. |

Do not combine shape assumptions, mask commands, or output roots across these
generations. In particular, PE 132 and FE 176 are properties of the legacy
Sub0001–Sub0005 baseline, not universal properties of the full source dataset.

## Current Code Facts

- Custom H5 k-space remains fully sampled during conversion; inference applies
  the selected mask through `KspaceMaskd`.
- External sensitivity maps are now supported for inference. JSON descriptors
  may use `sensitivity_maps`, `smap`, or `dMap`; current converters write
  `sensitivity_maps`.
- `--disable-acs-region` is available for intentional no-forced-ACS runs.
- Saved reconstructions use MAT key `img4ranking` under `val_img4ranking/`, but
  current postprocessing preserves the full spatial, slice, and time extent.
- `output/` is used by custom in-house experiments. The public example uses
  `outputs/`; both directories are intentional and ignored as generated data.

## File Placement Rules

- Put stable user procedures in `docs/`.
- Put focused legacy-baseline notes in `docs/codex/` only when they remain useful
  for that baseline.
- Put date-stamped logs, completed run transcripts, and superseded analyses in
  `docs/history/`.
- Keep generated results in `output/` or `outputs/`, never in `docs/`.
- Update this index when adding a new experiment generation or changing which
  SOP/report is active.
