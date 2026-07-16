# CMRxRecon 2023 P001 LAX Acceleration-8 Debug Run

This procedure runs the base NV-Raw2Insights-MRI model on one small native
CMRxRecon 2023 multi-coil CINE case. It is intended as the first compatibility
test before processing more subjects or the larger SAX acquisition.

The SLURM job performs the complete workflow: validate the raw MAT structure,
convert the official mask into the repository interface, write one JSON
descriptor, validate the derived input, run inference, inspect the saved MAT,
and create a PNG grid.

## Submit The Job

Run these commands from the repository root on the GPU server:

```bash
mkdir -p logs
sbatch scripts/slurm/infer_cmrxrecon2023_p001_lax_acc8.sbatch
```

The `logs/` directory must exist before `sbatch` because SLURM opens the log
files before the job script begins.

The script defaults to these established server paths:

```text
repository: /home/students/studxuzho1/NV-Raw2insights-MRI
Python:     /home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python
```

Override either path at submission if needed:

```bash
sbatch \
  --export=ALL,REPO_ROOT=/path/to/NV-Raw2insights-MRI,PYTHON_BIN=/path/to/python \
  scripts/slurm/infer_cmrxrecon2023_p001_lax_acc8.sbatch
```

Monitor the job with:

```bash
squeue -j <job-id>
tail -f logs/cmrx23_p001_lax_a8_<job-id>.out
```

## Read-Only Source Inputs

The raw CMRxRecon tree is never modified. The job reads:

```text
/mnt/qdata/rawdata/CMRxRecon/
  CMRxRecon_2023_training_init/
    ChallengeData/MultiCoil/CINE/TrainingSet/
      FullSample/P001/cine_lax.mat
        kspace_full: (time=12, slice=3, coil=10, PE=204, FE=448)
      AccFactor08/P001/cine_lax_mask.mat
        mask08: (PE=204, FE=448)
```

The full-sampled `kspace_full` array is used as input. The official mask is
applied once inside `scripts/inference.py`; the already undersampled
`AccFactor08/P001/cine_lax.mat` is intentionally not used as model input.

The official mask samples 47 of 204 PE lines. Its effective acceleration is
approximately 4.34 when the densely sampled ACS region is included, while its
CMRxRecon challenge class remains nominal acceleration 8.

## Derived Input Under The Repository

The job creates only a small mask copy and descriptor:

```text
dataset/CMRxRecon2023CINELAXAcc8/
  MultiCoil/CINE/Mask_TaskR1/
    P001_cine_lax_mask_ktRadial8.mat
      key: mask
      shape: (204, 448)
      dtype: float32
  json_input/
    P001_cine_lax.json
```

The JSON points to the raw full-sampled k-space in place and to the derived
mask. The `_mask_ktRadial8.mat` suffix is required because the reader uses the
mask filename for mask filtering and acceleration-class conditioning.

## Model Output Under The Repository

The job writes:

```text
output/CMRxRecon2023CINELAXAcc8/
  config.json
  output_inspection.txt
  val_img4ranking/
    P001_cine_lax.mat
      key: img4ranking
      expected shape: (448, 204, 3, 12)
      expected dtype: float32
      layout: (frequency, phase, slice, time)
  figs/
    P001_cine_lax.png
```

SLURM standard output and error are stored separately:

```text
logs/cmrx23_p001_lax_a8_<job-id>.out
logs/cmrx23_p001_lax_a8_<job-id>.err
```

Inference skips an existing reconstruction. To repeat model inference from
scratch without deleting a prior result, change `OUTPUT_ROOT` in a copied job
script or move to a new, clearly named output root.

## Evaluate Against Full-Sampled RSS

After inference succeeds, submit the CPU-only evaluation job:

```bash
mkdir -p logs
sbatch scripts/slurm/evaluate_cmrxrecon2023_p001_lax_acc8.sbatch
```

It derives the reference by applying a centered orthonormal 2D inverse FFT to
`kspace_full`, combining its ten coil images with root-sum-of-squares, and
matching the prediction layout. Direct-scale PSNR, SSIM, NRMSE, NMSE, MSE, and
MAE are computed for all 36 slice/time frames. A global least-squares scale-fit
is included only as a diagnostic and is not the primary result.

Evaluation outputs remain beside the model result:

```text
output/CMRxRecon2023CINELAXAcc8/evaluation/
  P001_cine_lax_fullsample_rss.mat
  frame_metrics.csv
  summary_metrics.json
  frame_metrics_boxplot.png
  comparison.png
```

The evaluation log is written to:

```text
logs/eval_cmrx23_p001_a8_<job-id>.out
logs/eval_cmrx23_p001_a8_<job-id>.err
```
