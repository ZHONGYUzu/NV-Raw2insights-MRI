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
expected shape: (176, 132, 12, 25)
```

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
Sub0014.mat (176, 132, 12, 25) float32 True <min> <max>
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

Create one temporal GIF per selected subject for a representative slice. The
example below uses zero-based slice index `6`, near the middle of the 12-slice
CINE volume.

```bash
SUBJECTS="Sub0014 Sub0026 Sub0030 Sub0047 Sub0049 Sub0051 Sub0096 Sub0105 Sub0112 Sub0122"

for sub in $SUBJECTS; do
  python scripts/make_cine_gif.py \
    "dataset/GT_from_dImgC/${sub}.mat" \
    --key gt \
    --slice-index 6 \
    -o output/GT_from_dImgC_gifs \
    --fps 5 \
    --percentile 99.5 \
    --transpose-display
done
```

Expected output:

```text
output/GT_from_dImgC_gifs/Sub0014_slice06_allframes.gif
...
output/GT_from_dImgC_gifs/Sub0122_slice06_allframes.gif
```

Use these GIFs to check temporal motion, intensity consistency, and obvious
slice/time orientation problems before running model evaluation.

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
GT GIF slice index: 6
Selected subjects file: docs/test_subjects.txt
```
