# fastMRI Brain Multi-Coil Inference

This workflow runs the released base model at nominal accelerations 8, 16, and
24 on ten reproducibly selected AXT1 volumes from the fastMRI brain multi-coil
validation set. Native H5 files are read directly; they are not converted into
the CMRxRecon JSON/MAT interface.

## Source Data

The source remains read-only:

```text
/mnt/qdata/rawdata/fastMRI/brain/multicoil_val/
  *.h5
    kspace: fully sampled multi-coil k-space
    reconstruction_rss: fastMRI validation ground truth
    acquisition: raw protocol label such as AXT2
```

No external mask files are required. Inference generates a fastMRI-style
equispaced mask with nominal acceleration 8, 16, or 24 and center fraction
0.04. It is conditioned as the pretrained model's corresponding uniform
Cartesian mask class. Keeping the same center fraction controls the ACS width
while changing the nominal acceleration.

AXT1 is mapped to the checkpoint's existing `T1w` conditioning class. The
inspected validation folder contains 34 AXT1, 77 AXT1PRE, 287 AXT1POST, and 106
AXFLAIR volumes, but no AXT2 volumes. AXT1 is used for the first controlled
cohort so pre/post-contrast acquisitions are not mixed. Other supported reader
mappings are AXT1PRE/AXT1POST to `T1w`, and AXFLAIR/AXT2 to `T2w`. The
checkpoint's class dimensions are not changed.

## Submit The Complete Job

After synchronizing the code to the server, run from the repository root:

```bash
cd /home/students/studxuzho1/NV-Raw2insights-MRI
mkdir -p logs
sbatch scripts/slurm/run_fastmri_brain_val10_acc8_acc16_acc24.sbatch
```

Monitor it with:

```bash
squeue -j <job-id>
tail -F logs/fastmri_b10_aall_<job-id>.out
```

The job performs cohort creation, validation, inference, output counting, and
evaluation. Selection uses the lowest SHA-256 ranks of `seed:filename` with
seed `20260719`, which is stable across Python versions. Only AXT1 volumes are
eligible for this first controlled run.

## Generated Paths

Derived input metadata and H5 symlinks:

```text
dataset/FastMRIBrainMulticoilVal10/
  cohort_manifest.json
  h5_input/
    <10 symlinks to source H5 files>
```

Model output:

```text
output/
  FastMRIBrainMulticoilVal10Acc8/
  FastMRIBrainMulticoilVal10Acc16/
  FastMRIBrainMulticoilVal10Acc24/
    config.json
    val_img4ranking/
      <10 MAT reconstructions with key img4ranking in each experiment root>
```

Evaluation output:

```text
Results/
  FastMRIBrainMulticoilVal10Acc8/
  FastMRIBrainMulticoilVal10Acc16/
  FastMRIBrainMulticoilVal10Acc24/
    ground_truth/
    comparisons/
    frame_metrics.csv
    summary_metrics.json
    frame_metrics_boxplot.png
    frame_metrics_violin.png
```

The reader standardizes input k-space to 384×384 with centered cropping or
zero-padding before masking. Saved model output is converted back to
slice/height/width order and center-cropped to the spatial shape of
`reconstruction_rss` for evaluation. Metrics are computed per slice using the
original direct intensity scale; prediction and reference are not independently
renormalized.

## One-Case Debug Run

After the cohort has been created, use a separate output root:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base_fastmri_brain_acc8.json \
  -i dataset/FastMRIBrainMulticoilVal10/h5_input \
  -o output/FastMRIBrainMulticoilValDebugAcc8 \
  --debug \
  --num-workers 0 \
  --accelerations 8 \
  --profile-timing
```

`--debug` processes only the first sorted H5 symlink. Do not include it in the
ten-case production run.
