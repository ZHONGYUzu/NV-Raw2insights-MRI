# Mask Format

This note documents VISTA TXT masks and the fixed-mask MAT files consumed by
the custom CINE inference pipeline.

## Source VISTA Masks

Source masks are read-only and live under:

```text
/home/students/studxusiy1/mr_recon/masks
```

Filename pattern:

```text
mask_VISTA_<nPE>x<nT>_acc<R>_<sd>.txt
```

Example:

```text
mask_VISTA_132x25_acc8_8.txt
```

Meaning:

| Field | Meaning |
| --- | --- |
| `132` | Phase-encoding size `nPE`. |
| `25` | Number of cardiac time frames `nT`. |
| `acc8` | Nominal acceleration. |
| final `8` | Random seed identifier. |

The documented TXT mask shape is:

```text
(phase, time) = (132, 25)
```

Values are binary `0` and `1`. The mask acts on phase encoding and temporal
dimensions, not on coil or frequency/readout.

## Converted MAT Mask Shape

For repo fixed-mask inference, the converter changes VISTA `(phase, time)` to:

```text
(time, phase, frequency) = (25, 132, 176)
```

Conversion logic:

```python
mask_raw = np.loadtxt(mask_txt, delimiter=",").astype(np.float32)  # (132, 25)
mask = mask_raw.T                                                  # (25, 132)
mask[:, center_start:center_stop] = 1.0                            # ACS lines
mask = np.repeat(mask[:, :, None], 176, axis=2)                    # (25, 132, 176)
```

Converted MAT files use key:

```text
mask
```

## ACS Lines

The active base config has:

```json
"use_acs_region": true,
"acs_lines": 20
```

During inference, the model uses `get_acs_region(mask)`. If the center of the
mask is not positive for all selected frames, inference can fail with:

```text
ValueError: The center of the mask is not positive.
```

For custom CINE conversion, force central ACS phase lines by default:

```bash
--acs-lines 20
```

## Mask Naming

Converted fixed masks should use model-known aliases:

```text
Sub0001_mask_ktRadial8.mat
Sub0001_mask_ktRadial16.mat
Sub0001_mask_ktRadial24.mat
```

Avoid names like:

```text
Sub0001_mask_VISTA_132x25_acc8_8.mat
```

Reason: the reader and transforms infer mask type and acceleration from the
filename. The transform parses digits from `mask_type`:

```python
acc_factor = int("".join(ch for ch in mask_type if ch.isdigit()))
```

Names with extra digits can produce the wrong acceleration.

## Confirmed Supported Rates

The current small/base/large configs support these fixed-mask accelerations:

```text
8, 16, 24
```

Default fixed mask aliases include:

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

The configs also define:

```json
"accelerations": [8.0, 16.0, 24.0]
```

Other acceleration rates require explicit config changes and validation. Treat
them as out-of-distribution for the current pretrained checkpoint path.

## R1/R2/R3 Mapping

| Custom label | Mask acceleration | Source TXT mask | Converted alias |
| --- | --- | --- | --- |
| `R1` | acc8 | `mask_VISTA_132x25_acc8_8.txt` | `ktRadial8` |
| `R2` | acc16 | `mask_VISTA_132x25_acc16_8.txt` | `ktRadial16` |
| `R3` | acc24 | `mask_VISTA_132x25_acc24_8.txt` | `ktRadial24` |

R1/R2/R3 are experiment labels for dataset/output roots. They are not the
acceleration factors themselves.
