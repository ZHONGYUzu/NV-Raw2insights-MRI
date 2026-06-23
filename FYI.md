# FYI.md

Personal project log and detail records for the custom CINE MRI reconstruction
work. `AGENTS.md` is the operating guide for Codex/agents; this file is for
human memory, project progress, explanations, and decisions worth keeping.

## 2026-06-23 Documentation Cleanup

Decision:

- Keep `AGENTS.md` as the authoritative agent/runbook file.
- Keep `FYI.md` as a daily log plus detailed project record.
- Allow a small amount of intentional overlap for expensive-to-forget facts:
  output shape, axis order, crop behavior, and supported acceleration rates.

Current documentation split:

| File | Role |
| --- | --- |
| `AGENTS.md` | Agent instructions, safe operating rules, conversion/inference commands, paths, and gotchas. |
| `FYI.md` | Human-readable project memory, explanations, progress notes, and interpretation details. |

## 2026-06-23 Output Visualization Record

Topic: why current PNG outputs show many slices and frames.

For the custom CINE inference output, each saved MAT file contains:

```text
img4ranking shape = (176, 132, 12, 25)
```

This means:

```text
(frequency, phase, slice, time)
```

For example, for `Sub0004.mat`:

```text
frequency/readout = 176
phase             = 132
slices            = 12
time frames       = 25
```

When running:

```bash
python scripts/visualize_mat.py \
  output/CustomCINEOutputR3/val_img4ranking/Sub0004.mat \
  -o output/CustomCINEOutputR3/figs
```

the script plots all slices and all time frames as a grid:

```text
rows    = slice index
columns = time/frame index
```

Therefore the output PNG is:

```text
12 rows x 25 columns
```

Example index mapping:

```text
slice=6, time=10 -> row 7, column 11
```

The displayed row and column are one-based visual positions, while the array
indices are zero-based.

In code, `visualize_mat.py` treats a 4D array like this:

```python
n_slices = img.shape[2]
n_times = img.shape[3]
```

and plots:

```python
img_4d[:, :, slice_index, time_index]
```

So `slice=6, time=10` corresponds to:

```python
img4ranking[:, :, 6, 10]
```

## 2026-06-23 Current Inference Behavior Record

Current inference still uses:

```python
output = complex_abs(crop_k_space(output, (final_shape[-2], final_shape[-1])))
```

This line is in `scripts/inference.py`.

Important:

```text
crop_k_space != run4Ranking crop
```

`crop_k_space()` only restores or keeps the model output at the original spatial
PE/FE size. For the documented CINE data:

```text
PE = 132
FE = 176
```

After model inference and spatial-size correction, current postprocessing in
`postprocess_mri_recon()` does:

```python
recon = recon.transpose()
recon = np.abs(recon).astype(np.float32)
```

So the saved reconstruction preserves:

```text
full spatial size
all slices
all time frames
```

For the documented five-case custom CINE data, the expected saved shape is:

```text
(176, 132, 12, 25)
```

The legacy output folder and MAT key are still:

```text
val_img4ranking
img4ranking
```

These names are retained for compatibility, but the array is no longer the old
cropped ranking array.

## 2026-06-23 Reader And Axis Record

`CMRxReconReader` is the CMRxRecon-style loader used by the current inference
pipeline. It reads each case JSON, loads the referenced k-space and mask MAT
files, adds metadata such as `filename`, `mask_type`, `acquisition`, `shape`,
`num_frames`, `num_slices`, and `num_coils`, and returns k-space in the repo's
canonical logical order:

```text
(time, slice, coil, PE, FE)
```

For the custom CINE source data:

```text
raw H5 kSpace:          (slice, coil, time, PE, FE) = (12, 15, 25, 132, 176)
reader logical k-space: (time, slice, coil, PE, FE) = (25, 12, 15, 132, 176)
```

The reader intentionally reverses scipy-loaded complex MAT axes:

```python
data_shape = dat[kspace_key].shape[::-1]
data = dat[kspace_key].transpose()
```

This means the converted MAT file is stored on disk with reversed axes:

```text
(FE, PE, coil, slice, time) = (176, 132, 15, 12, 25)
```

After `CMRxReconReader` reads it back and transposes it, the in-memory k-space
becomes the expected logical shape:

```text
(time, slice, coil, PE, FE) = (25, 12, 15, 132, 176)
```

Do not replace the custom converter with a plain `savemat()` of
`(time, slice, coil, PE, FE)`. If saved directly in that order,
`CMRxReconReader` would transpose it again and produce the wrong logical shape:

```text
(FE, PE, coil, slice, time) = (176, 132, 15, 12, 25)
```

## 2026-06-23 Source H5 Key Record

The custom CINE source H5 files use a dataset-specific schema with three main
keys:

| H5 key | Meaning | Shape in documented data |
| --- | --- | --- |
| `kSpace` | Fully sampled coil-resolved k-space. This is the raw multi-coil reconstruction input before any VISTA undersampling mask is applied. | `(slice, coil, time, PE, FE)` = `(12, 15, 25, 132, 176)` |
| `dMap` | Coil sensitivity maps, also called `smap`, sensitivity maps, or CSM. These describe how each coil sees the image spatially and are used for sensitivity-map coil combination. | `(slice, coil, 1, PE, FE)` = `(12, 15, 1, 132, 176)` |
| `dImgC` | Coil-combined fully sampled image data. The channel dimension is `1` because the coil dimension has already been combined. | `(slice, 1, time, PE, FE)` = `(12, 1, 25, 132, 176)` |

`dMap` has a singleton time dimension because the sensitivity maps are
time-averaged/static and reused across the cardiac frames.

These key names are not universal H5 rules. HDF5 is a flexible container format,
so another MRI H5 dataset can use different key names, groups, attributes, and
array layouts. For this project, treat `kSpace`, `dMap`, and `dImgC` as the
expected schema for the documented custom CINE source files.

## 2026-06-23 Ground Truth Reference Record

There are two reasonable fully sampled reference images that can be derived
from the same source H5 file.

`dImgC` GT uses the image that is already reconstructed and coil-combined inside
the H5 file:

```text
dImgC -> take channel 0 -> magnitude -> normalize -> evaluation orientation
```

This is simple and useful as a sanity-check reference, but it depends on the
coil-combination and preprocessing method used when the H5 was originally
created.

`kSpace+dMap` GT reconstructs the reference explicitly from the full multi-coil
k-space and sensitivity maps:

```text
kSpace -> centered inverse FFT -> coil images
       -> combine coils with dMap sensitivity maps
       -> magnitude -> normalize -> evaluation orientation
```

The sensitivity-map combination is:

```python
coil_img = ifft2c(kSpace)
combined = np.sum(coil_img * np.conj(dMap), axis=1) / (
    np.sum(np.abs(dMap) ** 2, axis=1) + 1e-8
)
```

Both references should show the same anatomy, but they may not be numerically
identical because of differences in coil-combination method, FFT centering,
sensitivity-map normalization, phase handling, intensity normalization, or
vendor/export preprocessing.

Current decision:

- Use `kSpace+dMap` as the more explicit generated GT method.
- Use `dImgC` as a sanity-check reference unless the source data owner documents
  `dImgC` as the official evaluation target.

## 2026-06-23 Acceleration Support Record

For the current `nv_raw2insights_mri_small`, `nv_raw2insights_mri_base`, and
`nv_raw2insights_mri_large` configs, the confirmed supported acceleration rates
are:

```text
8, 16, 24
```

The configs list these fixed mask names:

```text
mask_Uniform8
mask_Uniform16
mask_Uniform24
mask_ktRadial8
mask_ktRadial16
mask_ktRadial24
mask_ktGaussian8
mask_ktGaussian16
mask_ktGaussian24
```

They also define:

```json
"accelerations": [8.0, 16.0, 24.0]
```

There are two separate behaviors to keep in mind.

First, fixed-mask inference filters mask paths through `fixed_mask_types` in
`CMRxReconReader`. A mask named only like `Sub0001_mask_ktRadial6.mat` is not
matched by the default fixed-mask list, so it is not a supported default
inference mask.

Second, the transform code can parse arbitrary digits from a fixed mask name:

```python
acc_factor = int("".join(ch for ch in mask_type if ch.isdigit()))
```

So `ktRadial6` can be parsed as acceleration `6` if the config is modified to
let that mask through. However, the Restormer model builds its acceleration
conditioning labels from the configured `accelerations` list. With the default
configs, an acceleration such as `6` is not found and becomes an out-of-config
conditioning case.

Current decision:

- Treat only `8`, `16`, and `24` as confirmed supported rates for the current
  pretrained foundation-model inference path.
- Treat other acceleration rates as out-of-distribution unless there is an
  explicit config change and separate validation.

## Historical Record: Old Ranking Crop Behavior

The old behavior used `run4Ranking()` from `scripts/run4ranking.py`.

For CINE data, the old ranking crop did three separate reductions:

1. Selected only the central 2 slices.
2. Selected only the first 3 time frames.
3. Cropped the spatial image to the middle one-third by one-half region.

The relevant old logic was:

```python
reconImg = img[:, :, round(sz / 2 + 0.00001) - 2 : round(sz / 2 + 0.00001), :3]
img4ranking = crop(
    np.abs(reconImg),
    (round(sx / 3), round(sy / 2 + 0.00001), 2, 3),
).astype(np.float32)
```

For a full custom CINE output with:

```text
sx = 176
sy = 132
sz = 12
st = 25
```

the old ranking crop would produce approximately:

```text
(59, 66, 2, 3)
```

instead of:

```text
(176, 132, 12, 25)
```

So old visualization would show a much smaller grid:

```text
2 rows x 3 columns
```

not the current:

```text
12 rows x 25 columns
```

## Historical Record: Why An Older PNG May Have Looked Like One Slice And One Frame

If a previous PNG looked like only one slice and one frame, likely reasons
include:

```text
the saved MAT was already 2D
the saved MAT had shape (H, W, 1, 1)
singleton dimensions were squeezed
the file came from a selected/cropped debug result
the old ranking protocol had already reduced the output before visualization
```

The current output is intentionally full-size.

## Short Version

Current behavior:

```text
model output
-> crop_k_space only to match PE/FE size
-> abs + transpose
-> save full reconstruction
-> visualize as 12 slices x 25 time frames
```

Old behavior:

```text
model output
-> run4Ranking
-> central spatial crop
-> central 2 slices
-> first 3 time frames
-> save small ranking array
```

Key distinction:

```text
crop_k_space != run4Ranking crop
```

`crop_k_space()` is a spatial-size correction around the model output.

`run4Ranking()` was an evaluation/ranking-specific reduction that intentionally
discarded most slices, most time frames, and much of the spatial field of view.
