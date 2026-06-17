# FYI: Current Inference Output vs Old Ranking Crop

## Why the Current PNG Shows Many Slices and Frames

For the custom CINE inference output, each saved MAT file contains:

```text
img4ranking shape = (176, 132, 12, 25)
```

This means:

```text
(frequency, phase, slice, time)
```

So for `Sub0004.mat`:

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

For example, `slice=6, time=10` is shown at:

```text
row 7, column 11
```

because both indices are zero-based.

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

## Current Inference Behavior

Current inference still uses:

```python
output = complex_abs(crop_k_space(output, (final_shape[-2], final_shape[-1])))
```

This line is in `scripts/inference.py`.

Important: this `crop_k_space()` call is not the old ranking crop.

Its purpose is only to restore or keep the model output at the original spatial PE/FE size. For the documented CINE data, that means:

```text
PE = 132
FE = 176
```

After model inference and spatial-size correction, current postprocessing in `postprocess_mri_recon()` does:

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

## Old Ranking Crop Behavior

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

## Why an Older PNG May Have Looked Like One Slice and One Frame

If a previous PNG looked like only one slice and one frame, likely reasons include:

```text
the saved MAT was already 2D
the saved MAT had shape (H, W, 1, 1)
singleton dimensions were squeezed
the file came from a selected/cropped debug result
the old ranking protocol had already reduced the output before visualization
```

The current output is intentionally full-size. The legacy folder name:

```text
val_img4ranking
```

and MAT key:

```text
img4ranking
```

are retained for compatibility, but the array is no longer the old cropped ranking array.

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

The key distinction:

```text
crop_k_space != run4Ranking crop
```

`crop_k_space()` is a spatial-size correction around the model output.

`run4Ranking()` was an evaluation/ranking-specific reduction that intentionally discarded most slices, most time frames, and much of the spatial field of view.
