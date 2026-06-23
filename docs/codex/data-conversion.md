# Data Conversion

This note documents how custom CINE H5 source files are converted into
CMRxRecon-style MAT/JSON inputs for `scripts/inference.py`.

## Source Data

Known source H5 files are under:

```text
/mnt/qdata/rawdata/CINE/2D_h5_compressed
```

The documented H5 schema uses:

| Key | Shape | Meaning |
| --- | --- | --- |
| `kSpace` | `(slice, coil, time, PE, FE)` | Fully sampled coil-resolved k-space. |
| `dMap` | `(slice, coil, 1, PE, FE)` | Time-averaged coil sensitivity maps. |
| `dImgC` | `(slice, 1, time, PE, FE)` | Coil-combined fully sampled image. |

For `Sub0001.h5`, the important dimensions are:

```text
kSpace = (12, 15, 25, 132, 176)
dMap   = (12, 15, 1, 132, 176)
dImgC  = (12, 1, 25, 132, 176)
```

The source H5 directory is read-only. Conversion scripts must write derived
files under a separate dataset root.

## K-Space Conversion

Use:

```bash
python scripts/convert_cine_h5_kspace_to_mat.py \
  --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
  --output-root dataset/CustomCINEDataR1 \
  --glob 'Sub000[1-5].h5'
```

The converter:

- reads H5 key `kSpace`;
- casts to `complex64`;
- converts logical shape from `(slice, coil, time, PE, FE)` to
  `(time, slice, coil, PE, FE)`;
- stores the MAT array with reversed axes because `CMRxReconReader` transposes
  scipy-loaded complex MAT arrays when reading them back;
- writes MAT key `kspace_full`;
- writes or updates one JSON descriptor per case.

Example:

```text
source H5 kSpace:     (12, 15, 25, 132, 176)
reader logical order: (25, 12, 15, 132, 176)
MAT storage order:    (176, 132, 15, 12, 25)
```

Do not replace the converter with a plain `scipy.io.savemat()` call on
`(time, slice, coil, PE, FE)`. That would be transposed again by the reader and
produce the wrong in-memory shape.

## Output Tree

For R1, conversion creates:

```text
dataset/CustomCINEDataR1/
  MultiCoil/
    Cine/
      UnderSample_TaskR1/
        Sub0001_kspace_full.mat
        ...
  json_input/
    Sub0001.json
    ...
```

After k-space conversion, each JSON has a k-space path and an empty mask list.
Do not run inference until masks have been attached.

Example JSON after k-space conversion:

```json
{
  "kspace": "/abs/path/to/dataset/CustomCINEDataR1/MultiCoil/Cine/UnderSample_TaskR1/Sub0001_kspace_full.mat",
  "mask": []
}
```

## Mask Attachment

Attach a fixed VISTA mask after k-space conversion. For R1 acc8:

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

This creates one case-named MAT mask per subject and updates each JSON:

```json
{
  "kspace": "/abs/path/to/dataset/CustomCINEDataR1/MultiCoil/Cine/UnderSample_TaskR1/Sub0001_kspace_full.mat",
  "mask": [
    "/abs/path/to/dataset/CustomCINEDataR1/MultiCoil/Cine/Mask_TaskR1/Sub0001_mask_ktRadial8.mat"
  ]
}
```

## R2 And R3 Dataset Variants

Use `scripts/create_custom_cine_acc_dataset.py` to build additional fixed-mask
variants while reusing R1 k-space:

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

For R3, use `CustomCINEDataR3`, `mask_VISTA_132x25_acc24_8.txt`, and
`ktRadial24`.

The helper symlinks to R1 k-space by default. Use `--kspace-mode copy` only if
the server environment cannot follow symlinks.

## Validation

Always validate converted input before expensive inference:

```bash
python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR1/json_input
```

Expected line pattern:

```text
OK: Sub0001.json: k-space (25, 12, 15, 132, 176), masks 1
Validated 5 case(s).
```

The validator checks path existence, `MultiCoil/Cine` acquisition parsing,
logical 5D complex k-space, mask shape, binary mask values, acceleration naming,
and ACS center positivity.
