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
