# Migration Phase 0: Git State Snapshot

> Snapshot date: 2026-08-24
> Scope: local repository, GitHub `origin`, and the two server worktrees
> Operation type: read-only inspection
> This record completes Phase 0, item 2 of `docs/CODE_AND_EXPERIMENT_MIGRATION_PLAN.md`. No fetch, pull, commit, push, merge, checkout, file move, or deletion was performed.

## Summary

| Location | Worktree path | Branch | HEAD | Remote relationship | Dirty state |
| --- | --- | --- | --- | --- | --- |
| Local | `/Users/zhongyu/Desktop/NV-Raw2insights-MRI` | `CINE-MRI` | `eeb6273e873dc2ee0f858b22e7c73b68fe0ea76e` | 1 commit ahead of GitHub `origin/CINE-MRI` | 3 tracked modifications, 32 untracked status entries |
| GitHub | `ZHONGYUzu/NV-Raw2insights-MRI` | `CINE-MRI` | `6f0c8b6ec831e62639f3762bf5009f5390a99ab3` | Authoritative remote observation | Not applicable |
| Server main | `/home/students/studxuzho1/NV-Raw2insights-MRI` | `CINE-MRI` | `6f0c8b6ec831e62639f3762bf5009f5390a99ab3` | Matches GitHub `origin/CINE-MRI` | 0 tracked modifications, 2 untracked entries |
| GitHub | `ZHONGYUzu/NV-Raw2insights-MRI` | `feature/cmrxrecon-adapter` | `940d08f1581eab34b20faa6cbf0a6e0ff755ba42` | Authoritative remote observation | Not applicable |
| Server CMRx | `/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon` | `feature/cmrxrecon-adapter` | `940d08f1581eab34b20faa6cbf0a6e0ff755ba42` | Matches GitHub `origin/feature/cmrxrecon-adapter` | 1 tracked modification, 18 untracked entries |
| GitHub | `ZHONGYUzu/NV-Raw2insights-MRI` | `main` | `99be712b2e9fd34bf6a2f709373d311354fbda96` | Authoritative remote observation | Not applicable |

The local `CINE-MRI` commit not yet on GitHub is:

```text
eeb6273 Add FT00 CINE smoke-test Slurm job
```

## Local Repository

Repository identity:

```text
worktree:   /Users/zhongyu/Desktop/NV-Raw2insights-MRI
git common: .git
branch:     CINE-MRI
HEAD:       eeb6273e873dc2ee0f858b22e7c73b68fe0ea76e
tracking:   origin/CINE-MRI, ahead 1
```

Only one local Git worktree is currently registered. `/Users/zhongyu/Documents/NV-Raw2insights-MRI` is a symbolic link to the Desktop path and is not a second Git worktree.

### Tracked modifications

```text
M README.md
M docs/README.md
M scripts/evaluate_cmrxrecon2023_cohort.py
```

`docs/README.md` includes the migration-plan index entry created during this migration discussion. The other tracked modifications predate this snapshot and must not be overwritten during consolidation.

### Untracked status entries

Git reports 32 untracked status entries. A directory entry may contain multiple files, so this count is not a recursive file count.

```text
2512.17137v2.pdf
SERVER_WORKFLOW.md
docs/.~lock.Benchmark_v1_with_CINE_model_benchmark_matrix.xlsx#
docs/.~lock.CINE_model_benchmark_matrix.xlsx#
docs/CODE_AND_EXPERIMENT_MIGRATION_PLAN.md
docs/history/2026-08-05-server-run-audit.md
docs/history/2026-08-11-vista-mask-generalization-run-plan.md
docs/history/2026-08-24-migration-phase0-git-state-snapshot.md
downloads/
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

Note: the snapshot document itself was created after the initial count, so it is listed above but was not part of the original 32-entry count. A later status command will therefore report at least one additional untracked entry unless another entry has changed.

## GitHub Origin

Configured origin:

```text
git@github.com:ZHONGYUzu/NV-Raw2insights-MRI.git
```

Live branch HEADs were read from the repository's public HTTPS endpoint because neither the server nor the current local SSH environment could authenticate to GitHub using the configured SSH URL. The read-only HTTPS query succeeded:

```text
6f0c8b6ec831e62639f3762bf5009f5390a99ab3  refs/heads/CINE-MRI
940d08f1581eab34b20faa6cbf0a6e0ff755ba42  refs/heads/feature/cmrxrecon-adapter
99be712b2e9fd34bf6a2f709373d311354fbda96  refs/heads/main
```

At snapshot time, GitHub has exactly these three branch heads. The planned `research/multidataset` and dataset-specific feature branches have not been created.

## Server Worktrees

Both server paths are Git worktrees sharing the same Git common directory. Their canonical Ceph-backed paths are:

```text
/mnt/ceph/vol_02_home_students/studxuzho1/NV-Raw2insights-MRI
/mnt/ceph/vol_02_home_students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon
```

### Main server worktree

```text
user path:  /home/students/studxuzho1/NV-Raw2insights-MRI
branch:     CINE-MRI
HEAD:       6f0c8b6ec831e62639f3762bf5009f5390a99ab3
tracking:   origin/CINE-MRI
```

Tracked modifications: none.

Untracked entries:

```text
scripts/evaluate_sdum_paired_ranking.py
scripts/prepare_sdum_007_static_random_cartesian.py
```

### CMRx server worktree

```text
user path:  /home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon
branch:     feature/cmrxrecon-adapter
HEAD:       940d08f1581eab34b20faa6cbf0a6e0ff755ba42
tracking:   origin/feature/cmrxrecon-adapter
```

Tracked modification:

```text
M scripts/inference.py
```

Untracked entries:

```text
external/
scripts/create_cmrxrecon2023_training_masks.py
scripts/evaluate_cmrxrecon2023_cohort.py
scripts/evaluate_cmrxrecon2023_fullsample.py
scripts/plot_sdum_previews.py
scripts/prepare_cmrxrecon2024_split_zip_subset.py
scripts/slurm/evaluate_sdum_008_full_direct.sbatch
scripts/slurm/evaluate_sdum_009_full_direct.sbatch
scripts/slurm/prepare_sdum_011_012_013_acc16.sbatch
scripts/slurm/run_sdum_008_training_uniform8_debug.sbatch
scripts/slurm/run_sdum_008_training_uniform8_full.sbatch
scripts/slurm/run_sdum_009_training_ktgaussian8_debug.sbatch
scripts/slurm/run_sdum_009_training_ktgaussian8_full.sbatch
scripts/slurm/run_sdum_010_training_ktradial8_debug.sbatch
scripts/slurm/run_sdum_010_training_ktradial8_full.sbatch
scripts/slurm/run_sdum_011_012_013_acc16_debug.sbatch
scripts/slurm/run_sdum_011_012_013_acc16_full.sbatch
scripts/validate_cine_inference_data.py
```

The `external/` directory is one Git status entry but contains multiple nested files. It must be inspected separately during the code-difference phase and must not be committed recursively without checking for nested Git metadata.

## Interpretation And Safety Decisions

1. The server main worktree matches GitHub, but its two untracked scripts exist only outside committed history and must be preserved before any server synchronization.
2. The server CMRx worktree matches its GitHub branch at the commit level, but its tracked `inference.py` modification and 18 untracked entries are not represented by that remote HEAD.
3. The local repository is the only location containing commit `eeb6273` and also contains substantial dirty work. It must not be reset, overwritten, or replaced by a clean clone.
4. The GitHub branch list confirms that the planned integration branch and new adapter branches do not yet exist. This matches the migration plan; they should not be created until code consolidation is reviewed.
5. GitHub SSH authentication is not currently usable from the checked environments. This does not block read-only inventory, but push/pull authentication must be resolved or switched deliberately before the later synchronization phase.

## Phase Status

- Phase 0, item 1 — confirm related server tasks have ended: **complete**.
- Phase 0, item 2 — record local, GitHub, and server branch/HEAD/dirty state: **complete**.
- Phase 0, item 3 — generate tracked/untracked file differences and branch-unique commit inventory: **not started by this snapshot**.

No dirty entry listed here has yet been classified as keep, merge, archive, duplicate, or discard. Classification belongs to the following inventory steps.
