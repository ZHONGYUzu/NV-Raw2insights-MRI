# Legacy Custom-CINE Baseline Notes

This folder documents the original five-case, fixed-shape custom CINE baseline:
PE 132, FE 176, and `CustomCINEDataR1/R2/R3` roots. It remains useful for
reproducing that baseline, but it is not the active variable-shape 10-case
benchmark. Start at [`../README.md`](../README.md) for the full document map.

Use these files as the first stop for code navigation and project-specific
details:

| File | Purpose |
| --- | --- |
| [repo-map.md](repo-map.md) | Map of important scripts, configs, docs, and generated folders. |
| [data-conversion.md](data-conversion.md) | How custom CINE H5 files are converted into repo-readable MAT/JSON inputs. |
| [mask-format.md](mask-format.md) | VISTA TXT mask shape, MAT mask shape, ACS handling, and mask naming rules. |
| [inference-workflow.md](inference-workflow.md) | Debug and production inference workflow for R1/R2/R3 custom CINE experiments. |
| [evaluation-metrics.md](evaluation-metrics.md) | Output inspection, GT alignment, metrics, and comparison plotting. |

`AGENTS.md` remains the authoritative agent operating guide. The active 10-case
benchmark procedure is `docs/NV-Raw2insights-MRI_SOP.md`; dated and superseded
material is indexed under `docs/history/`.
