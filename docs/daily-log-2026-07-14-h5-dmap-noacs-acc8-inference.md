# Daily Log: H5 `dMap`, No-ACS acc8 Inference

Run date: 2026-07-14 to 2026-07-15  
Transcript timestamp at model load: 2026-07-14 22:01:06  
Status: completed successfully

## Experiment Purpose

This run evaluates the fixed 10-subject CINE benchmark cohort at nominal
acceleration 8 using the converted full H5 k-space, external H5 `dMap` coil
sensitivity maps, and VISTA masks without forced central ACS filling. The same
full k-space and sensitivity maps are intended to be reused for the acc16 and
acc24 experiments; only the sampling masks and model-conditioning acceleration
change.

## Server Environment

```text
User: studxuzho1
Node: node-gpu-07
Conda environment: envs/nv-raw2insights-mri/
Repository: /home/students/studxuzho1/NV-Raw2insights-MRI
```

The run emitted an unauthenticated Hugging Face Hub warning. The checkpoint
download still completed successfully.

## Model And Checkpoint

```text
Config: configs/nv_raw2insights_mri_base.json
Model variant: NV-Raw2Insights-MRI Base
Parameters: 758.91M
Checkpoint repository: nvidia/NV-Raw2insights-MRI
Checkpoint file: nv_raw2insights_mri_base.pt
Checkpoint snapshot: 38860e9a6153d857b55439b3d3fd55f850eccdcd
Loaded variables: 5004/5004
Unchanged variables: 0
```

Server checkpoint path:

```text
/home/students/studxuzho1/hf_cache/hub/models--nvidia--NV-Raw2insights-MRI/snapshots/38860e9a6153d857b55439b3d3fd55f850eccdcd/nv_raw2insights_mri_base.pt
```

## Dataset Setup

```text
Input descriptors: dataset/h5_converted/json_input
Output root: output/h5_converted
Nominal acceleration: acc8
VISTA mask seed: 8
Forced ACS filling: disabled
K-space source: H5 kSpace, converted as full unmasked k-space
Sensitivity-map source: H5 dMap, supplied externally
Frequency size: 192
Cardiac frames: 25
Compressed coils: 15
Input cases before filtering: 10
Input cases after filtering: 10
```

### Input And Output Paths

Source inputs, which remain read-only:

```text
H5 source root: /mnt/qdata/rawdata/CINE/2D_h5_compressed
H5 case pattern: /mnt/qdata/rawdata/CINE/2D_h5_compressed/<case>.h5
H5 k-space key: kSpace
H5 sensitivity-map key: dMap
VISTA source root: /home/students/studxusiy1/mr_recon/masks
VISTA mask pattern: mask_VISTA_<PE>x25_acc8_8.txt
```

Converted inference inputs under the server repository:

```text
Dataset root: /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/h5_converted
K-space: /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/h5_converted/MultiCoil/Cine/h5_kspace/<case>_kspace_full.mat
Sensitivity maps: /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/h5_converted/MultiCoil/Cine/h5_dmap/<case>_sensitivity_maps.mat
Masks: /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/h5_converted/MultiCoil/Cine/Mask_TaskR1/<case>_mask_ktRadial8.mat
Descriptors: /home/students/studxuzho1/NV-Raw2insights-MRI/dataset/h5_converted/json_input/<case>.json
```

Inference outputs:

```text
Output root: /home/students/studxuzho1/NV-Raw2insights-MRI/output/h5_converted
Saved config: /home/students/studxuzho1/NV-Raw2insights-MRI/output/h5_converted/config.json
Reconstructions: /home/students/studxuzho1/NV-Raw2insights-MRI/output/h5_converted/val_img4ranking/<case>.mat
MAT reconstruction key: img4ranking
```

Fixed cases from `docs/test_subjects.txt`:

```text
Sub0014 Sub0026 Sub0030 Sub0047 Sub0049
Sub0051 Sub0096 Sub0105 Sub0112 Sub0122
```

PE grouping:

| PE | Cases |
| ---: | --- |
| 180 | Sub0014 |
| 174 | Sub0030 |
| 162 | Sub0026, Sub0112 |
| 156 | Sub0047, Sub0049, Sub0051, Sub0096, Sub0105, Sub0122 |

The mask filenames provide `ktRadial8` model conditioning. Because the masks
were generated without forced ACS filling, the measured effective acceleration
should be recorded separately from the nominal acceleration if quantitative
sampling comparisons are reported.

## Inference Command

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/h5_converted/json_input \
  -o output/h5_converted \
  --profile-timing
```

## Timing Results

| Case | Slices | Forwards | Model/forward (s) | Model time (s) | Case total (s) | Case total (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sub0014 | 14 | 350 | 3.754 | 1314.05 | 1330.67 | 22.18 |
| Sub0026 | 15 | 375 | 3.424 | 1284.03 | 1299.81 | 21.66 |
| Sub0030 | 12 | 300 | 3.647 | 1093.95 | 1108.68 | 18.48 |
| Sub0047 | 3 | 75 | 3.261 | 244.56 | 248.46 | 4.14 |
| Sub0049 | 3 | 75 | 3.263 | 244.75 | 248.15 | 4.14 |
| Sub0051 | 14 | 350 | 3.258 | 1140.37 | 1155.16 | 19.25 |
| Sub0096 | 17 | 425 | 3.258 | 1384.52 | 1404.28 | 23.40 |
| Sub0105 | 11 | 275 | 3.261 | 896.89 | 909.65 | 15.16 |
| Sub0112 | 18 | 450 | 3.422 | 1539.89 | 1561.25 | 26.02 |
| Sub0122 | 12 | 300 | 3.258 | 977.33 | 993.45 | 16.56 |

Aggregate timing:

```text
Total cases: 10
Total slices: 119
Total forwards: 2975
Summed per-case model time: 10120.34 s (168.67 min)
Summed reported per-case total: 10259.56 s (170.99 min)
Weighted model time per forward: 3.402 s
End-to-end elapsed time: 177.73 min (2:57:43)
```

The first case reported `data_load=401.55s`, while subsequent cases reported
approximately `0.08-0.46s`. This is consistent with one-time startup, initial
data access, and checkpoint/model warm-up costs being concentrated in the
first case. The progress bar recorded 28:52 for the first iteration, compared
with the case timing total of 1330.67 seconds (22.18 minutes).

## Completion Record

The transcript ended with:

```text
100% 10/10 [2:57:43<00:00, 1066.39s/it]
inference completed! test elapsed time: 177.73 mins
```

Expected reconstruction location:

```text
output/h5_converted/val_img4ranking/<case>.mat
```

Before quantitative evaluation, verify that all 10 MAT files exist, contain
the `img4ranking` key, are finite `float32` arrays, and match each case's GT
shape after orientation alignment.
