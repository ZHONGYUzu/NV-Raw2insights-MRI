# Phase 1 pre-consolidation backup

Date: 2026-08-24

This record freezes the Git and uncommitted-code state immediately before
server-only code is consolidated into the local repository. It is a manifest,
not a copy of datasets, outputs, caches, or nested Git metadata.

## Safety boundary

- Local is the only code-editing source during consolidation.
- No server file was modified, moved, or deleted.
- No `dataset/`, `output/`, `Results/`, `runs/`, `logs/`, cache directory, or
  `external/.git` content is in scope.
- Existing local core files are not overwritten with server versions.
- Nothing in this phase has been staged or committed yet.

## Local repository

```text
path:   /Users/zhongyu/Desktop/NV-Raw2insights-MRI
branch: codex/server-code-consolidation
HEAD:   eeb6273e873dc2ee0f858b22e7c73b68fe0ea76e
base:   CINE-MRI at the same HEAD
```

Tracked modifications present when the branch was created:

```text
M README.md
M docs/README.md
M scripts/evaluate_cmrxrecon2023_cohort.py
```

Untracked paths present when the branch was created:

```text
2512.17137v2.pdf
SERVER_WORKFLOW.md
docs/.~lock.Benchmark_v1_with_CINE_model_benchmark_matrix.xlsx#
docs/.~lock.CINE_model_benchmark_matrix.xlsx#
docs/CODE_AND_EXPERIMENT_MIGRATION_PLAN.md
docs/history/2026-08-05-server-run-audit.md
docs/history/2026-08-11-vista-mask-generalization-run-plan.md
docs/history/2026-08-24-migration-phase0-asset-classification.md
docs/history/2026-08-24-migration-phase0-code-difference-inventory.md
docs/history/2026-08-24-migration-phase0-data-and-experiment-inventory.md
docs/history/2026-08-24-migration-phase0-git-state-snapshot.md
downloads/
experiments/
scripts/create_benchmark_v3.py
scripts/create_cine_benchmark_matrix.py
scripts/create_cmrxrecon2023_training_masks.py
scripts/evaluate_sdum_paired_ranking.py
scripts/export_sdum_ranking_crops.py
scripts/find_common_vista_mask_seeds.py
scripts/merge_cine_benchmark_workbooks.py
scripts/plot_sdum_previews.py
scripts/slurm/evaluate_h5_dmap_raw_vista_seed8_all10.sbatch
scripts/slurm/evaluate_sdum_002_003_paired.sbatch
scripts/slurm/evaluate_sdum_008_full_direct.sbatch
scripts/slurm/evaluate_sdum_009_full_direct.sbatch
scripts/slurm/prepare_sdum_011_012_013_acc16.sbatch
scripts/slurm/run_h5_dmap_raw_vista_multiseed.sbatch
scripts/slurm/run_sdum_008_training_uniform8_debug.sbatch
scripts/slurm/run_sdum_008_training_uniform8_full.sbatch
scripts/slurm/run_sdum_009_training_ktgaussian8_debug.sbatch
scripts/slurm/run_sdum_009_training_ktgaussian8_full.sbatch
scripts/slurm/run_sdum_010_training_ktradial8_debug.sbatch
scripts/slurm/run_sdum_010_training_ktradial8_full.sbatch
scripts/slurm/run_sdum_011_012_013_acc16_debug.sbatch
scripts/slurm/run_sdum_011_012_013_acc16_full.sbatch
scripts/slurm/run_sdum_official_acc8_conditioning.sbatch
tmp/
```

The two server-only scripts listed below were not present in this initial local
status. They were transferred only after an `rsync --dry-run` showed each as a
new file (`>f+++++++`).

## Server CINE worktree

```text
path:   /home/students/studxuzho1/NV-Raw2insights-MRI
branch: CINE-MRI
HEAD:   6f0c8b6ec831e62639f3762bf5009f5390a99ab3
status:
  ?? scripts/evaluate_sdum_paired_ranking.py
  ?? scripts/prepare_sdum_007_static_random_cartesian.py
```

`scripts/evaluate_sdum_paired_ranking.py` already existed locally and was not
copied. The only missing script copied from this worktree was:

```text
scripts/prepare_sdum_007_static_random_cartesian.py
SHA-256: 02514e4bbebbd40627c1e8cb8760520afd7d939dcdd8e04af9853ff4b7c6e5a5
```

## Server CMRxRecon worktree

```text
path:   /home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon
branch: feature/cmrxrecon-adapter
HEAD:   940d08f1581eab34b20faa6cbf0a6e0ff755ba42
status:
   M scripts/inference.py
  ?? external/
  ?? scripts/create_cmrxrecon2023_training_masks.py
  ?? scripts/evaluate_cmrxrecon2023_cohort.py
  ?? scripts/evaluate_cmrxrecon2023_fullsample.py
  ?? scripts/plot_sdum_previews.py
  ?? scripts/prepare_cmrxrecon2024_split_zip_subset.py
  ?? scripts/slurm/evaluate_sdum_008_full_direct.sbatch
  ?? scripts/slurm/evaluate_sdum_009_full_direct.sbatch
  ?? scripts/slurm/prepare_sdum_011_012_013_acc16.sbatch
  ?? scripts/slurm/run_sdum_008_training_uniform8_debug.sbatch
  ?? scripts/slurm/run_sdum_008_training_uniform8_full.sbatch
  ?? scripts/slurm/run_sdum_009_training_ktgaussian8_debug.sbatch
  ?? scripts/slurm/run_sdum_009_training_ktgaussian8_full.sbatch
  ?? scripts/slurm/run_sdum_010_training_ktradial8_debug.sbatch
  ?? scripts/slurm/run_sdum_010_training_ktradial8_full.sbatch
  ?? scripts/slurm/run_sdum_011_012_013_acc16_debug.sbatch
  ?? scripts/slurm/run_sdum_011_012_013_acc16_full.sbatch
  ?? scripts/validate_cine_inference_data.py
```

Every listed project file except the following script was already present
locally or represented by equivalent local history:

```text
scripts/prepare_cmrxrecon2024_split_zip_subset.py
SHA-256: 17b696b3bd320e12585bc88c36cb696a2ffea5eb9f6b42d052ae451c13464c36
```

The server `scripts/inference.py` was deliberately not copied. Its dirty timing
change is already represented locally by commit `e3b281b`, so replacing the
local core file would add risk without adding code.

## Cohort metadata fallback preservation

The local modified file is the authoritative consolidation copy:

```text
scripts/evaluate_cmrxrecon2023_cohort.py
local SHA-256:  3405cfdbb9c96ad9da419c29b1dda43cbfc31e9d230112a9beb88d2e9412c9d0
server SHA-256: 3405cfdbb9c96ad9da419c29b1dda43cbfc31e9d230112a9beb88d2e9412c9d0
```

The matching checksums prove that the newer subject/view cohort metadata
fallback logic is preserved locally. No server-to-local overwrite was needed.

## Transfer verification

The two transfers targeted explicit files, not directories. Local checksums
after transfer match the server checksums exactly:

```text
02514e4bbebbd40627c1e8cb8760520afd7d939dcdd8e04af9853ff4b7c6e5a5  scripts/prepare_sdum_007_static_random_cartesian.py
17b696b3bd320e12585bc88c36cb696a2ffea5eb9f6b42d052ae451c13464c36  scripts/prepare_cmrxrecon2024_split_zip_subset.py
```

Read-only Python AST parsing succeeded for both transferred scripts and for
`scripts/evaluate_cmrxrecon2023_cohort.py`.

## Recovery references

- The original uncommitted files remain at the two server paths above.
- The full Phase 0 Git snapshot is in
  `docs/history/2026-08-24-migration-phase0-git-state-snapshot.md`.
- The cross-worktree code comparison is in
  `docs/history/2026-08-24-migration-phase0-code-difference-inventory.md`.
- The local dirty state remains unstaged on
  `codex/server-code-consolidation` until explicit functional commit groups are
  reviewed.
