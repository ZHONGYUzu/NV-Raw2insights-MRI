# VISTA Mask Generalization Run Plan

Historical status on 2026-08-11: prepared locally; server submission pending.
This is a frozen plan, not a current job-status report. The later seed-8
evaluation record is retained in ../../SERVER_WORKFLOW.md (2026-08-12).

## Dataset

This run uses the active fixed 10-case, variable-shape in-house CINE benchmark,
not the legacy five-case 132x176 dataset. Subjects are defined in
`docs/test_subjects.txt`. Source H5 files are read-only under:

```text
/mnt/qdata/rawdata/CINE/2D_h5_compressed
```

Each inference case uses full coil-resolved H5 `kSpace`, external H5 `dMap`
sensitivity maps, and a raw VISTA mask without forced ACS. H5 `dImgC` is the
evaluation ground truth. All cases have 25 frames, 15 coils, and FE 192; PE is
156, 162, 174, or 180 depending on the case.

## Run 1: Existing Seed-8 Full Evaluation

The acc8, acc16, and acc24 seed-8 reconstructions already exist for all ten
cases. Submit the evaluation-only run with:

```bash
sbatch scripts/slurm/evaluate_h5_dmap_raw_vista_seed8_all10.sbatch
```

Default result root:

```text
Results/raw_vista_seed8_acc8_16_24_all10_v1
```

## Run 2: Two Additional Common Seeds

First discover seeds available for every required PE and acceleration:

```bash
python scripts/find_common_vista_mask_seeds.py \
  /home/students/studxusiy1/mr_recon/masks
```

Choose two values from `COMMON_SEEDS`, excluding the existing seed 8. If the
chosen seeds are 15 and 23, submit six GPU tasks with:

```bash
sbatch --array=0-5 --export=ALL,MASK_SEEDS=15,23 \
  scripts/slurm/run_h5_dmap_raw_vista_multiseed.sbatch
```

Array mapping is seed-major, acceleration-minor: tasks 0-2 run the first seed
at acc8/16/24 and tasks 3-5 run the second seed at acc8/16/24.

Do not select seed numbers until the server discovery command confirms that
all 12 required PE/acceleration combinations exist.
