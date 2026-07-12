# NV-Raw2insights-MRI SOP

This SOP records the benchmark workflow for the selected in-house CINE test
subjects. Run server commands from the NV-Raw2insights-MRI repository root.

Fixed test subjects are recorded in:

```text
docs/test_subjects.txt
```

Current selected cases:

```text
Sub0014 Sub0026 Sub0030 Sub0047 Sub0049 Sub0051 Sub0096 Sub0105 Sub0112 Sub0122
```

## Step 1: Generate GT From H5 dImgC

Goal: convert the H5 `dImgC` coil-combined fully sampled image into the same
layout as inference outputs.

Input H5 key:

```text
dImgC
```

Source layout:

```text
(slice, 1, time, phase, frequency)
```

Output:

```text
dataset/GT_from_dImgC/<case>.mat
```

Output MAT key and layout:

```text
key: gt
shape: (frequency, phase, slice, time)
```

Do not assume all selected subjects have identical spatial size or slice count.
Record the actual shape for every case during verification.

Normalization:

```text
--normalize max
```

This divides the complex `dImgC` image by the maximum magnitude per case before
saving the magnitude GT.

### Generate GT

```bash
SUBJECTS="Sub0014 Sub0026 Sub0030 Sub0047 Sub0049 Sub0051 Sub0096 Sub0105 Sub0112 Sub0122"

mkdir -p dataset/GT_from_dImgC

for sub in $SUBJECTS; do
  python scripts/generate_cine_h5_ground_truth.py \
    --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
    --output-dir dataset/GT_from_dImgC \
    --glob "${sub}.h5" \
    --source dimgc \
    --format mat \
    --mat-key gt \
    --normalize max
done
```

### Verify GT

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import scipy.io

root = Path("dataset/GT_from_dImgC")
for path in sorted(root.glob("Sub*.mat")):
    gt = scipy.io.loadmat(path)["gt"]
    print(path.name, gt.shape, gt.dtype, np.isfinite(gt).all(), gt.min(), gt.max())
PY
```

Expected pattern:

```text
Sub0014.mat (<frequency>, <phase>, <slice>, 25) float32 True <min> <max>
```

There should be 10 MAT files, one for each selected test subject.

### Generate GT Thumbnails

```bash
python scripts/visualize_mat.py \
  dataset/GT_from_dImgC \
  -o output/GT_from_dImgC_figs \
  -k gt \
  --cmap gray \
  --figsize 2 \
  --dpi 120
```

Expected output:

```text
output/GT_from_dImgC_figs/Sub0014.png
...
output/GT_from_dImgC_figs/Sub0122.png
```

### Generate GT GIFs

Create one temporal GIF per selected subject for a representative slice. Leave
`--slice-index` unset so `scripts/make_cine_gif.py` automatically selects the
center slice for each case. This matters because the selected subjects may have
different slice counts.

```bash
SUBJECTS="Sub0014 Sub0026 Sub0030 Sub0047 Sub0049 Sub0051 Sub0096 Sub0105 Sub0112 Sub0122"

for sub in $SUBJECTS; do
  python scripts/make_cine_gif.py \
    "dataset/GT_from_dImgC/${sub}.mat" \
    --key gt \
    -o output/GT_from_dImgC_gifs \
    --fps 5 \
    --percentile 99.5 \
    --transpose-display
done
```

Expected output:

```text
output/GT_from_dImgC_gifs/Sub0014_slice##_allframes.gif
...
output/GT_from_dImgC_gifs/Sub0122_slice##_allframes.gif
```

Use these GIFs to check temporal motion, intensity consistency, and obvious
slice/time orientation problems before running model evaluation.

If a fixed slice is required for a specific figure, first inspect each GT shape
and only use a slice index that exists for every selected case.

### Record After Completion

Record these details in the experiment notes or weekly report:

```text
GT source: H5 dImgC
GT normalization: max magnitude per case
GT output root: dataset/GT_from_dImgC
GT MAT key: gt
GT layout: frequency, phase, slice, time
GT thumbnail output: output/GT_from_dImgC_figs
GT GIF output: output/GT_from_dImgC_gifs
GT GIF slice index: center slice selected per case
Selected subjects file: docs/test_subjects.txt
```

## Step 2: Inspect H5 Shapes Before Inference Dataset Conversion

Goal: confirm each selected subject's H5 dimensions before creating converted
k-space and mask files. The selected in-house cases may have different
frequency size, phase size, and slice count, so do not hardcode
`--frequency-size 176` unless this inspection confirms it for the cases being
converted.

```bash
python scripts/inspect_cine_h5_shapes.py \
  --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
  --subjects-file docs/test_subjects.txt
```

The script prints one CSV-style row per selected case:

```text
case,dImgC(slice,cha,time,PE,FE),dMap(slice,coil,time,PE,FE),kSpace(slice,coil,time,PE,FE),logical kspace(time,slice,coil,PE,FE),PE,FE,time,slices,coils
```

It also prints a summary of unique k-space shapes and mask spatial
requirements:

```text
mask spatial requirements (PE, FE):
  <PE>x<FE>: <count>
```

Use the `FE` value from this table as `--frequency-size` when converting VISTA
masks for each case or group of cases. If multiple `(PE, FE)` groups appear,
convert masks separately per group and validate each generated JSON before
inference.

Record these details before inference:

```text
H5 shape inspection command:
H5 shape inspection output/log:
Unique kSpace shapes:
Unique mask spatial requirements:
Cases per frequency size:
```

Current selected test-set inspection result:

```text
FE: 192 for all selected cases
time frames: 25 for all selected cases
coil count: 15 for all selected cases

PE 180: Sub0014
PE 162: Sub0026, Sub0112
PE 174: Sub0030
PE 156: Sub0047, Sub0049, Sub0051, Sub0096, Sub0105, Sub0122
```

Required VISTA source masks:

```text
mask_VISTA_156x25_acc8_8.txt
mask_VISTA_156x25_acc16_8.txt
mask_VISTA_156x25_acc24_8.txt
mask_VISTA_162x25_acc8_8.txt
mask_VISTA_162x25_acc16_8.txt
mask_VISTA_162x25_acc24_8.txt
mask_VISTA_174x25_acc8_8.txt
mask_VISTA_174x25_acc16_8.txt
mask_VISTA_174x25_acc24_8.txt
mask_VISTA_180x25_acc8_8.txt
mask_VISTA_180x25_acc16_8.txt
mask_VISTA_180x25_acc24_8.txt
```

Check mask existence:

```bash
for pe in 156 162 174 180; do
  for acc in 8 16 24; do
    ls /home/students/studxusiy1/mr_recon/masks/mask_VISTA_${pe}x25_acc${acc}_8.txt
  done
done
```

## Step 3: Create Inference Dataset For acc8

Use explicit acceleration names in dataset and output paths:

```text
dataset/CINE_test_acc8
dataset/CINE_test_acc16
dataset/CINE_test_acc24

output/CINE_test_acc8
output/CINE_test_acc16
output/CINE_test_acc24
```

Convert full k-space from H5 into repo-compatible MAT/JSON descriptors. This
does not apply a mask to k-space.

```bash
SUBJECTS="Sub0014 Sub0026 Sub0030 Sub0047 Sub0049 Sub0051 Sub0096 Sub0105 Sub0112 Sub0122"

for sub in $SUBJECTS; do
  python scripts/convert_cine_h5_kspace_to_mat.py \
    --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
    --output-root dataset/CINE_test_acc8 \
    --glob "${sub}.h5"
done
```

Convert and register the acc8 VISTA masks. This creates MAT mask files and
updates each JSON descriptor with the mask path. It does not pre-mask the
k-space; inference still applies the mask internally.

```bash
convert_mask () {
  sub=$1
  pe=$2
  acc=$3
  root=$4

  python scripts/convert_vista_txt_mask_to_mat.py \
    --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
    --output-root "$root" \
    --frequency-size 192 \
    --glob "mask_VISTA_${pe}x25_acc${acc}_8.txt" \
    --acs-lines 20 \
    --case-id "$sub"
}

convert_mask Sub0014 180 8 dataset/CINE_test_acc8
convert_mask Sub0026 162 8 dataset/CINE_test_acc8
convert_mask Sub0030 174 8 dataset/CINE_test_acc8
convert_mask Sub0047 156 8 dataset/CINE_test_acc8
convert_mask Sub0049 156 8 dataset/CINE_test_acc8
convert_mask Sub0051 156 8 dataset/CINE_test_acc8
convert_mask Sub0096 156 8 dataset/CINE_test_acc8
convert_mask Sub0105 156 8 dataset/CINE_test_acc8
convert_mask Sub0112 162 8 dataset/CINE_test_acc8
convert_mask Sub0122 156 8 dataset/CINE_test_acc8
```

Validate acc8 before preparing other accelerations:

```bash
python scripts/validate_cine_inference_data.py dataset/CINE_test_acc8/json_input
```

Expected final line:

```text
Validated 10 case(s).
```

## Step 4: Create Inference Datasets For acc16 And acc24

Create acc16 and acc24 datasets from the acc8 full k-space files, replacing only
the mask and JSON descriptors. Use the same PE-specific mask mapping.

```bash
create_acc_case () {
  sub=$1
  pe=$2
  acc=$3
  out=$4
  mask_type=$5

  python scripts/create_custom_cine_acc_dataset.py \
    --source-root dataset/CINE_test_acc8 \
    --output-root "$out" \
    --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
    --mask-glob "mask_VISTA_${pe}x25_acc${acc}_8.txt" \
    --mask-type "$mask_type" \
    --case-glob "${sub}_kspace_full.mat" \
    --frequency-size 192 \
    --acs-lines 20
}
```

Prepare acc16:

```bash
create_acc_case Sub0014 180 16 dataset/CINE_test_acc16 ktRadial16
create_acc_case Sub0026 162 16 dataset/CINE_test_acc16 ktRadial16
create_acc_case Sub0030 174 16 dataset/CINE_test_acc16 ktRadial16
create_acc_case Sub0047 156 16 dataset/CINE_test_acc16 ktRadial16
create_acc_case Sub0049 156 16 dataset/CINE_test_acc16 ktRadial16
create_acc_case Sub0051 156 16 dataset/CINE_test_acc16 ktRadial16
create_acc_case Sub0096 156 16 dataset/CINE_test_acc16 ktRadial16
create_acc_case Sub0105 156 16 dataset/CINE_test_acc16 ktRadial16
create_acc_case Sub0112 162 16 dataset/CINE_test_acc16 ktRadial16
create_acc_case Sub0122 156 16 dataset/CINE_test_acc16 ktRadial16
```

Prepare acc24:

```bash
create_acc_case Sub0014 180 24 dataset/CINE_test_acc24 ktRadial24
create_acc_case Sub0026 162 24 dataset/CINE_test_acc24 ktRadial24
create_acc_case Sub0030 174 24 dataset/CINE_test_acc24 ktRadial24
create_acc_case Sub0047 156 24 dataset/CINE_test_acc24 ktRadial24
create_acc_case Sub0049 156 24 dataset/CINE_test_acc24 ktRadial24
create_acc_case Sub0051 156 24 dataset/CINE_test_acc24 ktRadial24
create_acc_case Sub0096 156 24 dataset/CINE_test_acc24 ktRadial24
create_acc_case Sub0105 156 24 dataset/CINE_test_acc24 ktRadial24
create_acc_case Sub0112 162 24 dataset/CINE_test_acc24 ktRadial24
create_acc_case Sub0122 156 24 dataset/CINE_test_acc24 ktRadial24
```

Optional consistency step: convert acc16 and acc24 JSON paths to absolute paths.
This is not required if inference is always run from the repository root, but it
keeps descriptors consistent with acc8.

```bash
python - <<'PY'
import json
from pathlib import Path

for root in [Path("dataset/CINE_test_acc16"), Path("dataset/CINE_test_acc24")]:
    for path in sorted((root / "json_input").glob("Sub*.json")):
        data = json.loads(path.read_text())
        data["kspace"] = str(Path(data["kspace"]).resolve())
        data["mask"] = [str(Path(item).resolve()) for item in data["mask"]]
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(path)
PY
```

Validate all acceleration datasets:

```bash
python scripts/validate_cine_inference_data.py dataset/CINE_test_acc8/json_input
python scripts/validate_cine_inference_data.py dataset/CINE_test_acc16/json_input
python scripts/validate_cine_inference_data.py dataset/CINE_test_acc24/json_input
```

Expected final line for each:

```text
Validated 10 case(s).
```

## Step 5: Debug Inference

Run one-case debug inference for each acceleration before full inference.
`--debug` processes the first sorted JSON descriptor only.

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CINE_test_acc8/json_input \
  -o output/CINE_test_acc8_debug \
  --debug --profile-timing

python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CINE_test_acc16/json_input \
  -o output/CINE_test_acc16_debug \
  --debug --profile-timing

python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CINE_test_acc24/json_input \
  -o output/CINE_test_acc24_debug \
  --debug --profile-timing
```

Inspect debug outputs:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import scipy.io

for root in [
    Path("output/CINE_test_acc8_debug/val_img4ranking"),
    Path("output/CINE_test_acc16_debug/val_img4ranking"),
    Path("output/CINE_test_acc24_debug/val_img4ranking"),
]:
    for path in sorted(root.glob("Sub*.mat")):
        image = scipy.io.loadmat(path)["img4ranking"]
        print(path, image.shape, image.dtype, np.isfinite(image).all(), image.min(), image.max())
PY
```

The prediction shape should match the corresponding GT shape for that case:

```text
(frequency, phase, slice, time)
```

## Step 6: Full Inference

Run full inference only after validation and debug inference pass.

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CINE_test_acc8/json_input \
  -o output/CINE_test_acc8 \
  --profile-timing

python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CINE_test_acc16/json_input \
  -o output/CINE_test_acc16 \
  --profile-timing

python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CINE_test_acc24/json_input \
  -o output/CINE_test_acc24 \
  --profile-timing
```

### Recorded Full Inference Timing

Confirmed acc16 full inference command:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CINE_test_acc16/json_input \
  -o output/CINE_test_acc16 \
  --profile-timing
```

Observed run:

```text
node: node-gpu-01
model: NV-Raw2Insights-MRI Base
checkpoint: nv_raw2insights_mri_base.pt from nvidia/NV-Raw2insights-MRI
model parameters: 758.91M
test files before filtering: 10
test files after filtering: 10
completed cases: 10/10
total elapsed time: 321.46 mins
```

Per-case acc16 timing:

| Case | Slices | Frames | Forwards | Total time (s) | Total time (min) | Model time/forward (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sub0014 | 14 | 25 | 350 | 2442.74 | 40.71 | 6.936 |
| Sub0026 | 15 | 25 | 375 | 2444.92 | 40.75 | 6.481 |
| Sub0030 | 12 | 25 | 300 | 1991.88 | 33.20 | 6.596 |
| Sub0047 | 3 | 25 | 75 | 444.16 | 7.40 | 5.879 |
| Sub0049 | 3 | 25 | 75 | 469.40 | 7.82 | 6.215 |
| Sub0051 | 14 | 25 | 350 | 2169.36 | 36.16 | 6.159 |
| Sub0096 | 17 | 25 | 425 | 2628.56 | 43.81 | 6.150 |
| Sub0105 | 11 | 25 | 275 | 1648.53 | 27.48 | 5.956 |
| Sub0112 | 18 | 25 | 450 | 2863.77 | 47.73 | 6.321 |
| Sub0122 | 12 | 25 | 300 | 1882.28 | 31.37 | 6.230 |

Notes:

```text
Forwards per case = number of slices x 25 cardiac frames.
The first case included checkpoint/model setup overhead in the progress bar.
Use the final "inference completed" line for total run time.
```

Inference skips a case if its output MAT already exists under:

```text
<output_root>/val_img4ranking/<case>.mat
```

To force a clean rerun without deleting existing outputs, choose a new output
root such as:

```text
output/CINE_test_acc8_rerun01
```

## Step 7: Evaluate Against dImgC GT

Use the MAT-to-MAT evaluator because both prediction and dImgC GT are stored as
MAT files in `(frequency, phase, slice, time)` layout.

```bash
python scripts/evaluate_cine_mat_gt_metrics.py \
  --acc acc8=output/CINE_test_acc8/val_img4ranking \
  --acc acc16=output/CINE_test_acc16/val_img4ranking \
  --acc acc24=output/CINE_test_acc24/val_img4ranking \
  --gt-root dataset/GT_from_dImgC \
  -o output/CINE_test_metrics_dImgC
```

Expected outputs:

```text
output/CINE_test_metrics_dImgC/frame_metrics.csv
output/CINE_test_metrics_dImgC/summary_metrics.csv
output/CINE_test_metrics_dImgC/metrics_boxplot.png
output/CINE_test_metrics_dImgC/metrics_violin.png
```

Frame-level row count should equal:

```text
sum(case slice counts) x 25 time frames x 3 accelerations
```

For the current selected cases, slice counts are:

```text
14 + 15 + 12 + 3 + 3 + 14 + 17 + 11 + 18 + 12 = 119
```

Expected frame rows:

```text
119 x 25 x 3 = 8925
```
