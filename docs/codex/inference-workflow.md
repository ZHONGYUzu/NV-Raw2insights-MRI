# Inference Workflow

This note documents the custom CINE inference workflow. Run these commands from
the NV-Raw2Insights-MRI repository root on the server, not locally, unless you
are only doing static inspection.

## Inputs And Outputs

Established input roots:

| Label | Input JSON directory | Output root |
| --- | --- | --- |
| `R1` | `dataset/CustomCINEDataR1/json_input` | `output/CustomCINEOutputR1` |
| `R2` | `dataset/CustomCINEDataR2/json_input` | `output/CustomCINEOutputR2` |
| `R3` | `dataset/CustomCINEDataR3/json_input` | `output/CustomCINEOutputR3` |

`scripts/inference.py` defaults to:

```text
input:  dataset/CustomCINEDataR1/json_input
output: output/CustomCINEOutputR1
```

Keeping `-i` and `-o` explicit is still recommended in documented experiment
commands.

## Preflight Validation

Before inference:

```bash
python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR1/json_input
```

Expected pattern:

```text
OK: Sub0001.json: k-space (25, 12, 15, 132, 176), masks 1
Validated 5 case(s).
```

## One-Case Debug Run

Run debug inference first. `--debug` processes only the first sorted JSON
descriptor.

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR1/json_input \
  -o output/CustomCINEOutputDebugR1 \
  --debug --profile-timing
```

Expected debug output:

```text
output/CustomCINEOutputDebugR1/config.json
output/CustomCINEOutputDebugR1/val_img4ranking/Sub0001.mat
```

The confirmed `Sub0001` debug output shape was:

```text
(176, 132, 12, 25)
```

## Production R1 Run

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR1/json_input \
  -o output/CustomCINEOutputR1 \
  --profile-timing
```

Expected tree:

```text
output/CustomCINEOutputR1/
  config.json
  val_img4ranking/
    Sub0001.mat
    Sub0002.mat
    Sub0003.mat
    Sub0004.mat
    Sub0005.mat
```

Each MAT file contains key:

```text
img4ranking
```

Despite the legacy name, the current array is the full magnitude reconstruction:

```text
(frequency, phase, slice, time) = (176, 132, 12, 25)
```

## R2 And R3 Runs

Validate before running:

```bash
python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR2/json_input

python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR3/json_input
```

Debug runs:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR2/json_input \
  -o output/CustomCINEOutputDebugR2 \
  --debug --profile-timing

python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR3/json_input \
  -o output/CustomCINEOutputDebugR3 \
  --debug --profile-timing
```

Production runs:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR2/json_input \
  -o output/CustomCINEOutputR2 \
  --profile-timing

python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR3/json_input \
  -o output/CustomCINEOutputR3 \
  --profile-timing
```

## Output Inspection

Inspect shapes and numeric ranges:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import scipy.io

root = Path('output/CustomCINEOutputR1/val_img4ranking')
for path in sorted(root.glob('Sub000*.mat')):
    image = scipy.io.loadmat(path)['img4ranking']
    print(path.name, image.shape, image.dtype, np.isfinite(image).all(), image.min(), image.max())
PY
```

Expected pattern:

```text
Sub0001.mat (176, 132, 12, 25) float32 True <min> <max>
```

Create PNG grids:

```bash
python scripts/visualize_mat.py \
  output/CustomCINEOutputR1/val_img4ranking \
  -o output/CustomCINEOutputR1/figs
```

For a 4D output `(176, 132, 12, 25)`, the visualization grid has 12 rows
and 25 columns: rows are slices, columns are time frames.

## Skip Behavior And Reruns

Inference skips a JSON when the corresponding output MAT already exists:

```text
<output_root>/val_img4ranking/<case>.mat
```

To force a clean rerun without deleting results, use a new output directory, for
example:

```text
output/CustomCINEOutputRerun01R1
```

Conversion scripts similarly refuse to replace existing MAT files unless
`--overwrite` is supplied.

## Timing Notes

For the documented shape, each case has:

```text
25 time frames * 12 slices = 300 forwards
```

A previous one-case `Sub0001` debug run took about 29 minutes on one GPU. A
five-case run took about 2.4 hours on the same setup. Runtime depends on server
GPU and workload.
