# AGENTS.md

## Project Overview

This repo works with CINE 2D MRI H5 data for model training, inference, or evaluation.

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
- `nPE`: phase-encoding dimension, always `132` for the documented data.
- `nFE`: frequency-encoding/readout dimension, always `176` for the documented data.

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
- The file is a normalized fully sampled image derived from full `kSpace` and `dMap`, and can be used as ground truth after applying the same orientation, central-slice selection, time-frame selection, and ranking crop as the saved inference output.

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
- For a controlled acc8 experiment, a suitable fixed mask is `mask_VISTA_132x25_acc8_8.txt`; convert it once and let every compatible case JSON point to the same converted `.mat` mask.
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

#### Five-Case Test Conversion Commands

For a small test run using only `Sub0001.h5` through `Sub0005.h5`, write the derived dataset to:

```text
/home/students/studxuzho1/dataset_v1
```

K-space conversion command:

```bash
python scripts/convert_cine_h5_kspace_to_mat.py \
  --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
  --output-root /home/students/studxuzho1/dataset_v1 \
  --glob 'Sub000[1-5].h5'
```

This writes derived k-space files to:

```text
/home/students/studxuzho1/dataset_v1/MultiCoil/Cine/UnderSample_TaskR1
```

Mask conversion command for acceleration 8, seed 8:

```bash
for sub in Sub0001 Sub0002 Sub0003 Sub0004 Sub0005; do
  python scripts/convert_vista_txt_mask_to_mat.py \
    --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
    --output-root /home/students/studxuzho1/dataset_v1 \
    --frequency-size 176 \
    --glob 'mask_VISTA_132x25_acc8_8.txt' \
    --case-id "$sub"
done
```

This writes derived mask files to:

```text
/home/students/studxuzho1/dataset_v1/MultiCoil/Cine/Mask_TaskR1
```

Important mask-selection note: using only `--acc 8 --seed 8` selected 9 masks because the mask folder contains multiple sizes with the same acceleration and seed. Use the exact `--glob 'mask_VISTA_132x25_acc8_8.txt'` when converting this CINE data shape.

#### Successful Five-Case Inference Run

Confirmed server command:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i /home/students/studxuzho1/dataset_v1/json_input \
  -o /home/students/studxuzho1/dataset_v1/output
```

Observed successful log summary:

```text
#Total test files before filtering: 5
#Total test files after filtering: 5
#Test files: 5
#model_params: 758.91M
100%|...| 5/5 [2:22:50<00:00, 1714.13s/it]
inference completed! test elapsed time: 142.85 mins
```

Output reconstructions are written to:

```text
/home/students/studxuzho1/dataset_v1/output/val_img4ranking
```

Visualization command:

```bash
python scripts/visualize_mat.py \
  /home/students/studxuzho1/dataset_v1/output/val_img4ranking \
  -o /home/students/studxuzho1/dataset_v1/output/figs
```

Confirmed generated PNGs:

```text
/home/students/studxuzho1/dataset_v1/output/figs/Sub0001.png
/home/students/studxuzho1/dataset_v1/output/figs/Sub0002.png
/home/students/studxuzho1/dataset_v1/output/figs/Sub0003.png
/home/students/studxuzho1/dataset_v1/output/figs/Sub0004.png
/home/students/studxuzho1/dataset_v1/output/figs/Sub0005.png
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

- The final two H5 dimensions are always PE `132`, then FE/readout `176` for the documented dataset.
- Raw H5 data and source masks are read-only and must not be copied into or modified by this repository.
- Mask filenames influence acceleration parsing in the current reader pipeline.
- Fixed masks need a positive central ACS region when `use_acs_region=true`.
- The conversion scripts documented above may exist only on the server and are not currently present in this local repository.

## Current Research / Experiment Notes

- Current goal: reconstruct custom CINE 2D MRI data with NV-Raw2Insights-MRI and evaluate it against normalized fully sampled ground truth.
- Active config: `configs/nv_raw2insights_mri_base.json` for the confirmed five-case run.
- Important checkpoint: automatically resolved base-model checkpoint unless explicitly overridden.
- Expected output: one `.mat` reconstruction per case under `val_img4ranking/`.
