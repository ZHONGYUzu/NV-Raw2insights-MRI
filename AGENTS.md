# AGENTS.md

## Project Overview

This repo works with CINE 2D MRI H5 data for model training, inference, or evaluation.

Documentation scope: use `docs/README.md` to distinguish the active 10-case
variable-shape benchmark from the retained five-case 132×176 baseline below.
`docs/NV-Raw2insights-MRI_SOP.md` is the active benchmark procedure;
date-stamped investigations and completed run records belong in `docs/history/`.

## Data Format Notes

Known H5 example:

- File inspected on server: `/mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0001.h5`
- H5 keys: `dImgC`, `dMap`, `kSpace`
- File attributes: none observed, `{}`

Each H5 file mainly contains:

| Key | Shape convention | Example shape | Dtype | Meaning |
| --- | --- | --- | --- | --- |
| `dImgC` | `(nSlice, nCha, nPha, nPE, nFE)` | `(12, 1, 25, 132, 176)` | `complex128` | Coil-combined fully sampled image. `nCha = 1`, so this is already coil-combined and not coil-resolved. |
| `dMap` | `(nSlice, nCha, nPha, nPE, nFE)` | `(12, 15, 1, 132, 176)` | `complex64` | Coil sensitivity maps. These are time-averaged sensitivity maps. `nCha = 15`, meaning coil compression keeps 15 coils. |
| `kSpace` | `(nSlice, nCha, nPha, nPE, nFE)` | `(12, 15, 25, 132, 176)` | `complex128` | Coil-resolved fully sampled k-space data. |

Dimension meaning:

- `nSlice`: number of slices, example size `12`.
- `nCha`: number of channels/coils. For `dImgC`, example size `1` because it is coil-combined. For `dMap` and `kSpace`, example size `15` after coil compression.
- `nPha`: number of cardiac phases/time frames. For `dImgC` and `kSpace`, example size `25`. For `dMap`, example size `1` because maps are time-averaged.
- `nPE`: phase-encoding dimension, `132` for the documented Sub0001 legacy example; other benchmark cases use different PE sizes.
- `nFE`: frequency-encoding/readout dimension, `176` for the documented Sub0001 legacy example; the newer 10-case benchmark uses `192`.

Important assumptions or cautions:

- Complex values are stored natively as H5 complex dtypes.
- Arrays use `(nSlice, nCha, nPha, nPE, nFE)` ordering.
- `dImgC` is coil-combined fully sampled image data.
- `dMap` contains coil sensitivity maps.
- `kSpace` contains coil-resolved fully sampled k-space data.
- Whether the model expects normalized input remains to be documented.

### Normalized Fully Sampled Ground Truth

Confirmed server file:

```text
/home/students/studxuzho1/dataset_v0/norm_img/norm_img_Sub0001.npy
```

- Shape: `(12, 25, 132, 176, 1)`.
- Dtype: `complex64`.
- Dimension order: `(slice, time, phase, frequency, channel)`.
- This corresponds to evaluator layout `ztpfc`, not `ztfpc`.
- The final channel dimension is `1` because the image is coil-combined.
- The file is a normalized fully sampled image derived from full `kSpace` and `dMap` and can be used as ground truth after matching the saved inference orientation. The current inference path saves all slices and all time frames; it no longer applies central-slice selection, time-frame selection, or the `run4Ranking` spatial crop.
- To match a current inference output shaped `(frequency, phase, slice, time)`, squeeze the final singleton channel and transpose the ground truth from `(slice, time, phase, frequency)` with `gt = gt[..., 0].transpose(3, 2, 0, 1)`.

### Current Inference Crop Behavior

- Commit `c842626` removed the evaluation-time call to `run4Ranking` from `postprocess_mri_recon`.
- Current CMRxRecon inference postprocessing applies `recon.transpose()`, magnitude with `np.abs`, and conversion to `float32`; it preserves the complete spatial image, every slice, and every cardiac frame.
- `crop_k_space(output, (final_shape[-2], final_shape[-1]))` is still used immediately after each model forward. This only restores the model output to the original PE/FE spatial size through a centered k-space crop when needed. It is not the historical ranking crop and does not select slices or time frames.
- The output directory name `val_img4ranking` and MAT key `img4ranking` are retained for compatibility, but the stored array is now the full reconstruction.
- For the documented CINE shape, the expected saved reconstruction shape is `(176, 132, 12, 25)`, corresponding to `(frequency, phase, slice, time)`.
- Evaluation against `norm_img_Sub0001.npy` should compare the full arrays after orientation matching. Do not apply `run4Ranking`, central two-slice selection, first-three-frame selection, or the one-third by one-half spatial crop unless intentionally reproducing the old ranking protocol.

### Custom CINE Inference Conversion Notes

Goal: run NV-Raw2Insights-MRI inference on the CINE H5 data without modifying raw source data.

- Source H5 files under `/mnt/qdata/rawdata/CINE/2D_h5_compressed` are treated as read-only.
- Source VISTA txt masks under `/home/students/studxusiy1/mr_recon/masks` are treated as read-only.
- Conversion scripts should write new derived files into a separate working/output dataset directory, never back into the raw data or mask folders.
- For the unmodified repo inference pipeline, provide full coil-resolved k-space plus a mask. Do not pre-apply the mask to k-space, because `scripts/inference.py` uses `KspaceMaskd` to create `kspace_masked` internally.
- Convert H5 `kSpace` from `(nSlice, nCha, nPha, nPE, nFE)` to repo logical order `(time, slice, coil, phase, frequency)` with:

```python
k = np.transpose(kspace_h5, (2, 0, 1, 3, 4))
```

- Example: `Sub0001.h5` `kSpace` shape `(12, 15, 25, 132, 176)` becomes `(25, 12, 15, 132, 176)`.
- Save converted k-space with one of the repo-supported keys: `kspace`, `kspace_full`, or `kus`.
- VISTA mask files are named like `mask_VISTA_<nPE>x<nT>_acc<R>_<sd>.txt`.
  - Example: `mask_VISTA_132x25_acc8_15.txt`.
  - `nPE` is phase encoding, `nT` is temporal frames, `R` is acceleration, and `sd` is random seed.
  - Confirmed shape example: `(132, 25)` with binary values `0` and `1`.
  - The mask acts on phase encoding and temporal dimensions only, not frequency/readout or coil.
- For repo fixed-mask inference, expand VISTA `(phase, time)` to `(time, phase, frequency)`:

```python
mask_raw = np.loadtxt(mask_txt, delimiter=",").astype(np.float32)  # (132, 25)
mask = mask_raw.T                                                  # (25, 132)
acs_lines = 20
start = (mask.shape[1] - acs_lines) // 2
mask[:, start:start + acs_lines] = 1.0
mask = np.repeat(mask[:, :, None], k.shape[-1], axis=2)            # (25, 132, 176)
```

- Save converted mask with key `mask`.
- The model uses `get_acs_region(mask)` during inference when `use_acs_region=true`. If the center of the mask is not positive for all selected frames, inference fails with `ValueError: The center of the mask is not positive.` The mask converter therefore forces central ACS phase lines by default with `--acs-lines 20`.
- Inference mask pairing recommendation:
  - First sanity test: use one fixed VISTA mask for 1-3 subjects.
  - Full inference: use the same fixed mask for all compatible H5 cases, e.g. all cases with `nPha = 25` and phase dimension `132`.
  - Robustness experiment: repeat inference in separate output folders with different mask seeds.
  - Do not mix random mask seeds/accelerations in the first full run unless the experiment intentionally tests robustness to sampling patterns.
- For a controlled acc8 experiment, use `mask_VISTA_132x25_acc8_8.txt` as the fixed source mask for every compatible case.
- The current bundled converter reuses that same source TXT but writes one case-named MAT copy per subject (`Sub0001_mask_ktRadial8.mat`, etc.). Each JSON points to its corresponding case-named mask. The mask arrays are identical when the same TXT source and options are used.
- Use CMRxRecon-like paths so the reader can infer acquisition from `MultiCoil/Cine`.
- Recommended derived layout:

```text
<derived_dataset_root>/
  MultiCoil/
    Cine/
      UnderSample_TaskR1/
        Sub0001_kspace_full.mat
      Mask_TaskR1/
        Sub0001_mask_ktRadial8.mat
  json_input/
    Sub0001.json
```

- Even if the physical sampling is VISTA, prefer a model-known alias such as `ktRadial8` in the converted mask filename for first inference tests. The repo parses digits from `mask_type` to get `acc_factor`, so avoid names like `mask_VISTA_132x25_acc8_15.mat` because the parsed acceleration can become invalid.
- Example JSON descriptor:

```json
{
  "kspace": "<derived_dataset_root>/MultiCoil/Cine/UnderSample_TaskR1/Sub0001_kspace_full.mat",
  "mask": [
    "<derived_dataset_root>/MultiCoil/Cine/Mask_TaskR1/Sub0001_mask_ktRadial8.mat"
  ]
}
```

#### Authoritative Legacy Five-Case Custom Data Process

Run every command below from the NV-Raw2Insights-MRI repository root on the server. The server repository path is not yet documented, so first `cd` to the directory that contains `scripts/`, `configs/`, and this `AGENTS.md`.

Code used by this process:

```text
scripts/convert_cine_h5_kspace_to_mat.py
scripts/convert_vista_txt_mask_to_mat.py
scripts/validate_cine_inference_data.py
scripts/inference.py
scripts/visualize_mat.py
scripts/plot_cine_acc_comparison.py
scripts/evaluate_cine_acc_metrics.py
configs/nv_raw2insights_mri_base.json
```

Source paths, which must remain read-only:

```text
H5 directory:       /mnt/qdata/rawdata/CINE/2D_h5_compressed
Example H5:         /mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0001.h5
VISTA mask folder:  /home/students/studxusiy1/mr_recon/masks
Selected mask TXT:  /home/students/studxusiy1/mr_recon/masks/mask_VISTA_132x25_acc8_8.txt
Ground truth root:  /home/students/studxuzho1/dataset_v0/norm_img
Example ground truth: /home/students/studxuzho1/dataset_v0/norm_img/norm_img_Sub0001.npy
```

#### Historical Short-Name Proposal (Not Active)

This `cine1/cine2/cine3` proposal was not adopted by the established legacy
datasets or the newer benchmark. Do not rename existing roots to these names.
Use `CustomCINEDataR1/R2/R3` only when reproducing the legacy five-case run, and
use the `CINE_test_acc*` roots defined by `docs/NV-Raw2insights-MRI_SOP.md` for
the active benchmark.

The historical proposal preferred short experiment roots and keeping details in notes, metadata, script headers, or adjacent documentation instead of encoding every detail in path names.

Use:

```text
dataset/cine1
dataset/cine2
dataset/cine3

output/cine1
output/cine2
output/cine3
```

The proposed mapping was:

| New run | Meaning | Dataset root | Output root |
| --- | --- | --- | --- |
| `cine1` | acc8 custom CINE experiment | `dataset/cine1` | `output/cine1` |
| `cine2` | acc16 custom CINE experiment | `dataset/cine2` | `output/cine2` |
| `cine3` | acc24 custom CINE experiment | `dataset/cine3` | `output/cine3` |

Do not rename legacy folders until the new process has completed successfully and the old paths can be cleaned systematically.

The converted custom dataset is already established at:

```text
dataset/CustomCINEDataR1
```

Do not rename or move this existing input dataset. Its task-specific data folders are already named `UnderSample_TaskR1` and `Mask_TaskR1`, while inference reads the case descriptors from its existing `json_input/` folder.

Established input tree:

```text
dataset/CustomCINEDataR1/
  MultiCoil/
    Cine/
      UnderSample_TaskR1/
        Sub0001_kspace_full.mat
        ...
      Mask_TaskR1/
        Sub0001_mask_ktRadial8.mat
        ...
  json_input/
    Sub0001.json
    ...
    Sub0005.json
```

Create a new first-level `output/` folder beside `dataset/`, not inside it. The result subfolder uses the same custom experiment suffix as the input dataset:

```text
output/CustomCINEOutputR1
```

These are the default custom-CINE paths in `scripts/inference.py`, so the production run can omit `-i` and `-o`. Keeping them explicit in documented commands makes the selected experiment paths clear.

#### Custom CINE Experiment Naming Map

Do not confuse the two naming layers:

- `UnderSample_TaskR1` and `Mask_TaskR1` are CMRxRecon-style task folder names kept for reader compatibility. These stay the same for all custom CINE variants.
- `CustomCINEDataR1`, `CustomCINEDataR2`, and `CustomCINEDataR3` are this repo's custom experiment dataset roots. Here, `R1/R2/R3` are experiment labels, not acceleration factors.
- `CustomCINEOutputR1`, `CustomCINEOutputR2`, and `CustomCINEOutputR3` are the matching inference output roots.

Current mapping:

| Custom label | Mask acceleration | Source TXT mask | Dataset root | Output root |
| --- | --- | --- | --- | --- |
| `R1` | acc8 | `mask_VISTA_132x25_acc8_8.txt` | `dataset/CustomCINEDataR1` | `output/CustomCINEOutputR1` |
| `R2` | acc16 | `mask_VISTA_132x25_acc16_8.txt` | `dataset/CustomCINEDataR2` | `output/CustomCINEOutputR2` |
| `R3` | acc24 | `mask_VISTA_132x25_acc24_8.txt` | `dataset/CustomCINEDataR3` | `output/CustomCINEOutputR3` |

All three use the same full k-space source data. R2 and R3 symlink to R1 k-space by default and only replace the fixed mask and JSON descriptors.

1. Confirm the repository files and source inputs:

```bash
pwd
ls -l scripts/convert_cine_h5_kspace_to_mat.py \
      scripts/convert_vista_txt_mask_to_mat.py \
      scripts/validate_cine_inference_data.py \
      scripts/inference.py \
      configs/nv_raw2insights_mri_base.json

ls -l /mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub000{1,2,3,4,5}.h5
ls -l /home/students/studxusiy1/mr_recon/masks/mask_VISTA_132x25_acc8_8.txt
```

2. Convert the five full, unmasked H5 k-space arrays:

```bash
python scripts/convert_cine_h5_kspace_to_mat.py \
  --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
  --output-root dataset/CustomCINEDataR1 \
  --glob 'Sub000[1-5].h5'
```

For each subject, this command reads H5 key `kSpace`, converts logical shape `(slice, coil, time, PE, FE)` to `(time, slice, coil, PE, FE)`, and writes MAT key `kspace_full`. The MAT array is stored with reversed axes because `CMRxReconReader` reverses scipy-loaded complex MAT axes when reading it back. Do not replace the converter with a plain `savemat(k)` call without accounting for that reader behavior.

This step creates:

```text
dataset/CustomCINEDataR1/MultiCoil/Cine/UnderSample_TaskR1/Sub0001_kspace_full.mat
dataset/CustomCINEDataR1/MultiCoil/Cine/UnderSample_TaskR1/Sub0002_kspace_full.mat
dataset/CustomCINEDataR1/MultiCoil/Cine/UnderSample_TaskR1/Sub0003_kspace_full.mat
dataset/CustomCINEDataR1/MultiCoil/Cine/UnderSample_TaskR1/Sub0004_kspace_full.mat
dataset/CustomCINEDataR1/MultiCoil/Cine/UnderSample_TaskR1/Sub0005_kspace_full.mat
dataset/CustomCINEDataR1/json_input/Sub0001.json
...
dataset/CustomCINEDataR1/json_input/Sub0005.json
```

At this point each JSON has the k-space path and an empty mask list. Do not run inference yet.

3. Convert and attach the fixed acceleration-8 mask to every case:

```bash
for sub in Sub0001 Sub0002 Sub0003 Sub0004 Sub0005; do
  python scripts/convert_vista_txt_mask_to_mat.py \
    --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
    --output-root dataset/CustomCINEDataR1 \
    --frequency-size 176 \
    --glob 'mask_VISTA_132x25_acc8_8.txt' \
    --acs-lines 20 \
    --case-id "$sub"
done
```

The converter reads source shape `(phase=132, time=25)`, transposes it to `(time=25, phase=132)`, forces 20 central phase lines to one for every frame, expands frequency to shape `(25, 132, 176)`, and saves MAT key `mask`. The source k-space remains fully sampled; `KspaceMaskd` applies this mask inside inference.

This step creates:

```text
dataset/CustomCINEDataR1/MultiCoil/Cine/Mask_TaskR1/Sub0001_mask_ktRadial8.mat
...
dataset/CustomCINEDataR1/MultiCoil/Cine/Mask_TaskR1/Sub0005_mask_ktRadial8.mat
```

It also updates each JSON under `json_input/`. For example, `dataset/CustomCINEDataR1/json_input/Sub0001.json` must contain absolute server paths equivalent to:

```json
{
  "kspace": "dataset/CustomCINEDataR1/MultiCoil/Cine/UnderSample_TaskR1/Sub0001_kspace_full.mat",
  "mask": [
    "dataset/CustomCINEDataR1/MultiCoil/Cine/Mask_TaskR1/Sub0001_mask_ktRadial8.mat"
  ]
}
```

4. Double-check the complete derived input before inference:

```bash
find dataset/CustomCINEDataR1/MultiCoil/Cine \
  -maxdepth 2 -type f | sort

find dataset/CustomCINEDataR1/json_input \
  -maxdepth 1 -name 'Sub000*.json' -type f | sort

cat dataset/CustomCINEDataR1/json_input/Sub0001.json

python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR1/json_input
```

Expected validator summary for every case:

```text
OK: Sub0001.json: k-space (25, 12, 15, 132, 176), masks 1
...
Validated 5 case(s).
```

The validator checks that paths exist, the acquisition path contains `MultiCoil/Cine`, k-space is logical 5D complex data, mask shape matches time/PE/FE, mask values are binary, the filename yields acceleration 8, and the central ACS location is positive for all frames.

5. Run a one-case debug inference into a separate output folder:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR1/json_input \
  -o output/CustomCINEOutputDebugR1 \
  --debug --profile-timing
```

`--debug` processes only the first sorted JSON descriptor. Confirm these files afterward:

```text
output/CustomCINEOutputDebugR1/config.json
output/CustomCINEOutputDebugR1/val_img4ranking/Sub0001.mat
```

6. Run all five cases into the production output folder:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR1/json_input \
  -o output/CustomCINEOutputR1 \
  --profile-timing
```

The exact production output tree is:

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

A previous five-case server run completed successfully in approximately 142.85 minutes and reported five input files before and after filtering. Runtime depends on the server GPU and current workload.

Each MAT file contains key `img4ranking`. Despite the legacy folder/key name, current code saves the full magnitude reconstruction. Expected shape for these cases is `(176, 132, 12, 25)` = `(frequency, phase, slice, time)` and dtype is `float32`.

Inference skips a JSON when its corresponding MAT already exists under the selected output folder. To force a clean re-run without deleting results, use another `R1`-named output directory such as `output/CustomCINEOutputRerun01R1`. The conversion scripts similarly refuse to replace existing MAT files unless `--overwrite` is supplied.

7. Inspect output keys, shapes, and numeric ranges:

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

Expected shape line pattern:

```text
Sub0001.mat (176, 132, 12, 25) float32 True <minimum> <maximum>
```

8. Generate PNG grids in the output visualization folder:

```bash
python scripts/visualize_mat.py \
  output/CustomCINEOutputR1/val_img4ranking \
  -o output/CustomCINEOutputR1/figs
```

This creates:

```text
output/CustomCINEOutputR1/figs/Sub0001.png
...
output/CustomCINEOutputR1/figs/Sub0005.png
```

9. Align normalized ground truth for evaluation:

```python
import numpy as np
import scipy.io

pred = scipy.io.loadmat(
    "output/CustomCINEOutputR1/val_img4ranking/Sub0001.mat"
)["img4ranking"]                                           # (176, 132, 12, 25)

gt = np.load(
    "/home/students/studxuzho1/dataset_v0/norm_img/norm_img_Sub0001.npy"
)                                                          # (12, 25, 132, 176, 1)
gt = np.abs(gt[..., 0]).transpose(3, 2, 0, 1)              # (176, 132, 12, 25)

assert pred.shape == gt.shape, (pred.shape, gt.shape)
```

This shape/orientation alignment does not guarantee that prediction and ground-truth intensity normalization are identical. Confirm the ground-truth normalization method before interpreting PSNR, SSIM, or NMSE values.

10. Compare acceleration settings visually for a selected slice/time frame:

```bash
python scripts/plot_cine_acc_comparison.py \
  --acc acc8=output/CustomCINEOutputR1/val_img4ranking \
  --acc acc16=output/CustomCINEOutputR2/val_img4ranking \
  --acc acc24=output/CustomCINEOutputR3/val_img4ranking \
  --gt-root /home/students/studxuzho1/dataset_v0/norm_img \
  --case-glob 'Sub000[1-5].mat' \
  -o output/acc_comparison_pngs
```

This script creates GT/reconstruction/error PNG grids with colorbars. It is intended for visual inspection only: it computes metrics only for the selected displayed slice/time frame and does not satisfy full quantitative evaluation requirements.

11. Compute full-size quantitative metrics across acceleration settings:

```bash
python scripts/evaluate_cine_acc_metrics.py \
  --acc acc8=output/CustomCINEOutputR1/val_img4ranking \
  --acc acc16=output/CustomCINEOutputR2/val_img4ranking \
  --acc acc24=output/CustomCINEOutputR3/val_img4ranking \
  --gt-root /home/students/studxuzho1/dataset_v0/norm_img \
  --case-glob 'Sub000[1-5].mat' \
  -o output/acc_metrics_5subjects
```

This evaluation script uses the full prediction and reference arrays after orientation matching; it does not apply `run4Ranking`, spatial crop, central-slice selection, or first-three-frame selection. It computes per-frame metrics for every shared subject, every slice, and every temporal frame. With five cases and the documented `12` slices and `25` frames, each acceleration produces `5 x 12 x 25 = 1500` frame-level metric rows. For ten subjects, the expected count is `10 x nSlices x 25` rows per acceleration.

Outputs:

```text
output/acc_metrics_5subjects/frame_metrics.csv
output/acc_metrics_5subjects/summary_metrics.csv
output/acc_metrics_5subjects/metrics_boxplot.png
output/acc_metrics_5subjects/metrics_violin.png
```

Default plotted metrics are PSNR, NRMSE, and NMSE. Additional supported metrics are MSE and MAE via `--metrics`.

#### Acc16 / R2 Dataset Variant

For the second experiment, keep `dataset/CustomCINEDataR1` as the established acc8 dataset and create a separate acc16 dataset root:

```text
dataset/CustomCINEDataR2/
  MultiCoil/
    Cine/
      UnderSample_TaskR1/
        Sub0001_kspace_full.mat  # symlink to R1 k-space by default
        ...
      Mask_TaskR1/
        Sub0001_mask_ktRadial16.mat
        ...
  json_input/
    Sub0001.json
    ...
```

Use seed `8` for acc16 to match the acc8 sanity run seed while changing only the acceleration:

```bash
python scripts/create_custom_cine_acc_dataset.py \
  --source-root dataset/CustomCINEDataR1 \
  --output-root dataset/CustomCINEDataR2 \
  --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
  --mask-glob 'mask_VISTA_132x25_acc16_8.txt' \
  --mask-type ktRadial16 \
  --case-glob 'Sub000[1-5]_kspace_full.mat' \
  --frequency-size 176 \
  --acs-lines 20
```

By default, this script creates symlinks to the R1 k-space MAT files instead of duplicating large full k-space data. Use `--kspace-mode copy` only if the server environment cannot follow symlinks.

Validate R2 before inference:

```bash
python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR2/json_input
```

Run a one-case R2 debug inference:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR2/json_input \
  -o output/CustomCINEOutputDebugR2 \
  --debug --profile-timing
```

Run all five R2 cases:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR2/json_input \
  -o output/CustomCINEOutputR2 \
  --profile-timing
```

Visualize R2 outputs:

```bash
python scripts/visualize_mat.py \
  output/CustomCINEOutputR2/val_img4ranking \
  -o output/CustomCINEOutputR2/figs
```

#### Acc24 / R3 Dataset Variant

For the third experiment, create a separate acc24 dataset root and output root while reusing the same full k-space from R1:

```text
dataset/CustomCINEDataR3/
  MultiCoil/
    Cine/
      UnderSample_TaskR1/
        Sub0001_kspace_full.mat  # symlink to R1 k-space by default
        ...
      Mask_TaskR1/
        Sub0001_mask_ktRadial24.mat
        ...
  json_input/
    Sub0001.json
    ...
```

Use seed `8` for acc24 to match the acc8 and acc16 experiments while changing only acceleration:

```bash
python scripts/create_custom_cine_acc_dataset.py \
  --source-root dataset/CustomCINEDataR1 \
  --output-root dataset/CustomCINEDataR3 \
  --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
  --mask-glob 'mask_VISTA_132x25_acc24_8.txt' \
  --mask-type ktRadial24 \
  --case-glob 'Sub000[1-5]_kspace_full.mat' \
  --frequency-size 176 \
  --acs-lines 20
```

Validate R3 before inference:

```bash
python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR3/json_input
```

Run a one-case R3 debug inference:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR3/json_input \
  -o output/CustomCINEOutputDebugR3 \
  --debug --profile-timing
```

Run all five R3 cases:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR3/json_input \
  -o output/CustomCINEOutputR3 \
  --profile-timing
```

Visualize R3 outputs:

```bash
python scripts/visualize_mat.py \
  output/CustomCINEOutputR3/val_img4ranking \
  -o output/CustomCINEOutputR3/figs
```

## How This Repo Is Usually Run

This repo is usually run on a server, not fully locally.

- Server name:
- Server OS:
- GPU type/count:
- CUDA version:
- Python version:
- Environment manager: conda / venv / module / Docker / other
- Main environment name:

## Server Setup Commands

Paste the commands normally typed after SSH/login.

```bash
# Example:
# cd /path/to/repo
# module load cuda/...
# conda activate ...
# export DATA_ROOT=...
# export CHECKPOINT_DIR=...
```

## Important Paths

Use server paths here. Do not include secrets.

- Server repo path:
- Local repo path: `/Users/zhongyu/Documents/NV-Raw2insights-MRI`
- Data root: `/mnt/qdata/rawdata/CINE/2D_h5_compressed`
- Example H5 file: `/mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0001.h5`
- Training data:
- Validation data:
- Test data:
- Checkpoints:
- Pretrained model:
- Configs: `configs/`
- Logs:
- Outputs/results:

## Common Commands

### Training

```bash
# Paste training command here.
```

### Inference

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i /path/to/json_input \
  -o /path/to/output
```

### Evaluation

```bash
# Paste evaluation command here.
```

### Debug / Small Test Run

```bash
# Paste a cheap quick command here, if available.
```

## Environment Variables

List variable names and what they mean. Do not store real secrets.

- `DATA_ROOT`: likely `/mnt/qdata/rawdata/CINE/2D_h5_compressed`
- `CHECKPOINT_DIR`:
- `CUDA_VISIBLE_DEVICES`:
- `WANDB_API_KEY`: required? yes/no. Do not paste value.
- `HF_TOKEN`: required? yes/no. Do not paste value.

## Dependencies

- Install command: `pip install -r requirements.txt`
- Requirements file: `requirements.txt`
- Conda environment file:
- Private packages or custom installs:
- Any version constraints that matter:

## Code Entry Points

- Training script: `scripts/train.py`
- Inference script: `scripts/inference.py`
- Evaluation metrics: `scripts/evaluation.py`
- Main config files: `configs/*.json`
- Dataset/loading code: `scripts/readers.py`, `scripts/mri_data/`
- Model definitions: `scripts/models/`
- Losses/metrics: `scripts/evaluation.py`
- Output writing code: `scripts/utils.py`, `scripts/run4ranking.py`

## Config Notes

Describe which config files are normally edited and which fields matter.

- Batch size:
- Learning rate:
- Dataset path:
- Checkpoint path:
- Output directory:
- Model variant:
- Number of coils:
- Number of time frames:
- Image size:

## Local Codex Guidance

Codex is usually used locally for code reading/editing, not full server execution.

- Do not run full training locally unless explicitly asked.
- Do not assume server-only data/checkpoints exist locally.
- Prefer static inspection, small tests, or dry runs.
- Ask before running GPU-heavy or long-running jobs.
- Ask before changing dataset paths, checkpoint paths, or experiment configs.
- Keep server-specific paths documented here rather than hardcoding new ones.
- Unknown code files may be ignored unless they are needed for the requested task.
- Ask for confirmation when project documentation contains logically inconsistent information that could affect implementation.

## Testing / Verification

Commands that are safe to run locally:

```bash
python -m compileall scripts
```

Commands that should only run on the server:

```bash
python scripts/train.py ...
python scripts/inference.py ...
```

## Known Gotchas

- The final two H5 dimensions are PE then FE/readout. They are `132` and `176` for the legacy Sub0001–Sub0005 baseline; inspect each case because the newer benchmark has variable PE and FE `192`.
- Raw H5 data and source masks are read-only and must not be copied into or modified by this repository.
- Mask filenames influence acceleration parsing in the current reader pipeline.
- Fixed masks need a positive central ACS region when `use_acs_region=true`.
- The custom CINE k-space and VISTA mask converters are available locally under `scripts/`, together with a pre-inference validator.

## Legacy Five-Case Research / Experiment Notes

- Goal for this retained baseline: reconstruct five custom CINE 2D MRI cases and evaluate them against normalized fully sampled ground truth.
- Config: `configs/nv_raw2insights_mri_base.json` for the confirmed five-case run.
- Important checkpoint: automatically resolved base-model checkpoint unless explicitly overridden.
- Expected output: one `.mat` reconstruction per case under `val_img4ranking/`.

### 2026-06-15 Debug Inference Notes

- A one-case debug inference for `Sub0001.json` completed successfully after adding the missing NumPy import used by `postprocess_mri_recon`:

```python
import numpy as np
```

- The completed debug output was:

```text
output/CustomCINEOutputDebugR1/val_img4ranking/Sub0001.mat
```

- Confirmed output inspection on the server:

```text
shape: (176, 132, 12, 25)
dtype: float32
finite: True
min/max/mean: 0.0008832483 / 1.0092976 / 0.09585182
nonzero: 6969600 / 6969600
```

- This matches the expected full reconstruction layout `(frequency, phase, slice, time)`.
- Runtime for `Sub0001` was about 29 minutes on one GPU:

```text
samples=300
forwards=300
model_per_forward=5.647s
model=1694.05s
total=1707.62s
```

- The runtime is expected because each CINE case has `25` frames and `12` slices, so inference runs `25 * 12 = 300` forwards with `batch_size=1` through the `758.91M` parameter base model.
- A timing-only bug produced a negative `data_load` value because `time.time()` and `time.perf_counter()` were mixed in `scripts/inference.py`. This does not affect reconstruction outputs; it only affects profiling text.
- Estimated inference time on the same setup:

```text
1 case:   about 29 minutes
5 cases:  about 2.4 hours
15 cases: about 7.25 hours
130 cases: about 63 hours
```

### Fine-Tuning Split Recommendation For 130 Cases

- Split by case/subject only. Do not split by slice or cardiac frame.
- Recommended first split:

```text
Train: 100 cases
Val:    15 cases
Test:   15 cases
Total: 130 cases
```

- Alternative training-heavy split:

```text
Train: 105 cases
Val:    10 cases
Test:   15 cases
Total: 130 cases
```

- Use the training set for fine-tuning, the validation set for checkpoint selection and hyperparameter decisions, and the test set only for final untouched evaluation.
- For routine development, run inference on 1 case first, then 5 cases, then the validation set. Avoid repeatedly running all 130 cases unless the pipeline and checkpoint choice are already settled.
