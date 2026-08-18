# EXP009: CINE Static-Random Cartesian Mask

Status: prepared locally; server debug submission pending.

## Purpose

EXP009 tests mask-distribution shift while holding the CINE cohort, pretrained
NV-Raw2Insights Base checkpoint, nominal acceleration, and external sensitivity
maps fixed. The physical mask selects one random set of phase-encoding lines
and uses that identical set for every cardiac frame.

## ACS Decision

The experiment forces zero ACS lines. At nominal R=8, the selected cases have
only about 20--22 sampled PE lines per frame. Reserving 20 central ACS lines
would consume essentially the entire sampling budget and eliminate the random
mask variable that EXP009 is intended to test.

Sensitivity maps therefore come from the source H5 `dMap`, and inference uses
`--disable-acs-region`. Validation deliberately uses `--skip-acs-check` and
requires every descriptor to contain external sensitivity maps. A randomly
selected center line is allowed, but no central line or block is forced.

## Fixed Design

- Cohort: the active ten-case variable-PE CINE benchmark.
- Physical mask: static random Cartesian PE sampling, repeated over 25 frames
  and over the full FE dimension.
- Nominal acceleration: 8.
- Sampling budget: `round(PE / 8)` lines per frame.
- Random seed: 9009; cases with the same PE/FE use the same mask.
- Forced ACS lines: 0.
- Sensitivity maps: external H5 `dMap` inherited from `dataset/h5_converted`.
- Model-conditioning alias: `Uniform8`. This is an implementation label for
  the pretrained model, not a claim that the physical mask is in-distribution.

## Server Commands

Run the one-case debug job first:

```bash
sbatch scripts/slurm/run_exp009_cine_static_random_cartesian_acc8.sbatch
```

After checking the first reconstruction, finish the same output directory:

```bash
sbatch --export=ALL,DEBUG=0 \
  scripts/slurm/run_exp009_cine_static_random_cartesian_acc8.sbatch
```

Default derived input and output roots:

```text
dataset/EXP009_CINE_static_random_cartesian_acc8_seed9009
output/EXP009_CINE_static_random_cartesian_acc8_seed9009
```

The full job recreates the derived masks deterministically, validates shapes
and external maps, reports effective acceleration, and then runs inference.
Source H5 data and existing derived k-space/sensitivity-map MAT files remain
read-only.
