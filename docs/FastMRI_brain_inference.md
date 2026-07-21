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

Before submitting the full job, validate one prepared volume on the server:

```bash
python scripts/validate_fastmri_reader_geometry.py \
  dataset/FastMRIBrainMulticoilVal10/h5_input \
  --max-cases 1
```

The validator reads only the middle slice rather than loading and Fourier
transforming the approximately 500 MB volume. The processed slice k-space must
have spatial shape 320×320 and the reported zero-filled NMSE against
`reconstruction_rss` must be at most `1e-6`.

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
  FastMRIBrainMulticoilVal10GeomFixFixedMaskAcc8/
  FastMRIBrainMulticoilVal10GeomFixFixedMaskAcc16/
  FastMRIBrainMulticoilVal10GeomFixFixedMaskAcc24/
    config.json
    val_img4ranking/
      <10 MAT reconstructions with key img4ranking in each experiment root>
```

Evaluation output:

```text
Results/
  FastMRIBrainMulticoilVal10GeomFixFixedMaskAcc8/
  FastMRIBrainMulticoilVal10GeomFixFixedMaskAcc16/
  FastMRIBrainMulticoilVal10GeomFixFixedMaskAcc24/
    ground_truth/
    comparisons/
    frame_metrics.csv
    summary_metrics.json
    frame_metrics_boxplot.png
    frame_metrics_violin.png
```

The older roots without `GeomFix` were produced by directly converting
640×320 k-space to 384×384 and are geometrically invalid. They are retained as
an audit record and must not be used for quantitative reporting.

The raw brain k-space has spatial shape 640×320 while `reconstruction_rss` is
320×320. To remove readout oversampling without changing the field of view, the
reader applies a centered inverse FFT, center-crops each complex coil image to
320×320, and applies a centered FFT before masking. It never directly crops one
k-space axis while padding the other. Before GPU inference, the SLURM job checks
that fully sampled RSS from the processed k-space numerically reproduces the H5
`reconstruction_rss` geometry.

An investigated model-axis adapter can swap the two spatial axes around model
inference, but the one-case experiment degraded the direct-scale metrics. It is
therefore retained only as an ablation and disabled by
`fastmri_model_axis_adapter: false` in the active fastMRI config.

Because fastMRI volumes have one frame per slice while the released model uses
five-view windows, the active controlled test repeats the same static slice
across the five model views. Neighboring-slice windows remain available
as an explicit ablation via `fastmri_adjacent_slice_window`, but are disabled
by default so spatial averaging is not mixed into the axis experiment.

Saved model output is converted back to slice/height/width order for evaluation.
Metrics are computed per slice using the original direct intensity scale;
prediction and reference are not independently renormalized.

## One-Case Debug Run

After the cohort has been created, use a separate output root:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base_fastmri_brain_acc8.json \
  -i dataset/FastMRIBrainMulticoilVal10/h5_input \
  -o output/FastMRIBrainMulticoilValGeomFixFixedMaskDebugAcc8 \
  --debug \
  --num-workers 0 \
  --accelerations 8 \
  --profile-timing
```

`--debug` processes only the first sorted H5 symlink. Do not include it in the
ten-case production run.

Evaluate and visually inspect that one corrected case before submitting the
full job:

```bash
python scripts/evaluate_fastmri_brain_cohort.py \
  --manifest dataset/FastMRIBrainMulticoilVal10/cohort_manifest.json \
  --prediction-dir output/FastMRIBrainMulticoilValGeomFixFixedMaskDebugAcc8/val_img4ranking \
  --label "fastMRI Brain AXT1 GeomFix FixedMask Debug Acc8" \
  --max-cases 1 \
  -o Results/FastMRIBrainMulticoilValGeomFixFixedMaskDebugAcc8
```

Confirm the corresponding PNG under `comparisons/` has matching orientation
and anatomy before running all ten cases and all three accelerations.

## Zero-Filled Baseline

Before accepting a model result, create a zero-filled RSS baseline with the
same official-style equispaced mask parameters and fixed offset used by model
inference:

```bash
python scripts/create_fastmri_zero_filled_baseline.py \
  --manifest dataset/FastMRIBrainMulticoilVal10/cohort_manifest.json \
  -o output/FastMRIBrainMulticoilValZeroFilledFixedMaskAcc8 \
  --acceleration 8 \
  --center-fraction 0.04 \
  --offset 0 \
  --max-cases 1
```

Evaluate it through the same direct-scale evaluator:

```bash
python scripts/evaluate_fastmri_brain_cohort.py \
  --manifest dataset/FastMRIBrainMulticoilVal10/cohort_manifest.json \
  --prediction-dir output/FastMRIBrainMulticoilValZeroFilledFixedMaskAcc8 \
  --label "fastMRI AXT1 Zero-Filled FixedMask Acc8" \
  --reconstruction-title "Zero-filled RSS" \
  --max-cases 1 \
  -o Results/FastMRIBrainMulticoilValZeroFilledFixedMaskAcc8
```

The active inference config sets `fastmri_equispaced_offset: 0`, so a new model
debug run and this baseline use exactly the same sampled columns. Do not compare
the fixed-mask baseline against an older model output generated with a random
offset as though the masks were identical.

If the model output is substantially blurrier than the reference, run a
one-case hard-data-consistency diagnostic into another new output root:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base_fastmri_brain_acc8.json \
  -i dataset/FastMRIBrainMulticoilVal10/h5_input \
  -o output/FastMRIBrainMulticoilValGeomFixFixedMaskHardDCDebugAcc8 \
  --debug \
  --num-workers 0 \
  --accelerations 8 \
  --hard-data-consistency \
  --profile-timing
```

This replaces the final prediction at acquired k-space locations with the
measured values. It is an ablation for diagnosing checkpoint soft-DC and mask
conditioning compatibility; do not silently mix its results with the default
model outputs.

## Preprocessing References

- The official fastMRI data documentation defines brain multi-coil k-space as
  `(slice, coil, height, width)` and `reconstruction_rss` as the central 320×320
  RSS reconstruction: <https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/README.md>.
- The official transform implementation removes oversampling by applying an
  inverse FFT, complex image-domain center crop, and FFT:
  <https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/transforms.py>.
- Zbontar et al. explain that ground-truth images are center-cropped to 320×320
  to compensate for readout-direction oversampling:
  <https://arxiv.org/abs/1811.08839>.
