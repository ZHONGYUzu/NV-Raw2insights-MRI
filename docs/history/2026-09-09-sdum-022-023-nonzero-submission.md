# SDUM-022/023 nonzero pilot submission — 2026-09-09

Submitted with explicit user authorization. Snapshot at submission: preparation
jobs PENDING (Priority); inference jobs PENDING (Dependency). Submission is not
completion. Each inference requires successful completion of its preparation.

Code: `db120e63cd43c2161260fa3cc892ea58cd05d6d7`, clean server worktree
`/home/students/studxuzho1/NV-Raw2insights-MRI-cine`, branch `codex/cine-smallangle`.
This documentation update does not change the submitted implementation.

| Experiment | Condition | CPU preparation | GPU inference |
| --- | --- | --- | --- |
| 022 | fixed 1 degree | 43923 | 43924 |
| 022 | fixed 3 degrees | 43925 | 43926 |
| 022 | fixed 5 degrees | 43927 | 43928 |
| 023 | step 0 to 1 degree | 43929 | 43930 |
| 023 | step 0 to 3 degrees | 43931 | 43932 |
| 023 | step 0 to 5 degrees | 43933 | 43934 |

Each job processes Sub0014 and Sub0047, all slices/frames, normalization none,
VISTA nominal acc8 seed8 plus 20 ACS lines. Each inference requests one GPU.
Checkpoint: `nv_raw2insights_mri_base.pt`, NVIDIA Hugging Face snapshot
`38860e9a6153d857b55439b3d3fd55f850eccdcd`, same file as zero controls.

Results remain under `/home/students/studxuzho1/NV-Raw2insights-MRI-experiments/`
in the existing `sdum-022/run_001/pilot` and `sdum-023/run_001/pilot` roots.
Server submission manifest:
`sdum-022/run_001/pilot/provenance/nonzero_submission_01.tsv`.
Logs: each experiment's `pilot/logs/prepare-<job>.out/.err` and
`pilot/logs/infer-<job>.out/.err`.

Completed zero controls 43908/43909 are reused, not resubmitted. Their predictions
were elementwise identical for both cases. RSS reference/output intensity
suitability remains unresolved; no formal RSS metric acceptance is implied.
No automatic full-cohort expansion or new quantitative evaluation was submitted.

See [protocol](../SDUM_022_023_smallangle.md) for input, mask, reference and display rules.
