# Inference Guide

This guide describes how to run inference with **NV-Raw2insights-MRI** (SDUM) for universal MRI reconstruction.

## Prerequisites

1. Complete the [Setup guide](setup.md) (environment, dependencies, optional Hugging Face login).
2. Ensure input data matches the expected format (see [Input format](#input-format) below).

## Hardware Requirements

Inference can be run on a single GPU or scaled with multiple GPUs via `torchrun`. GPU memory requirements depend on the model variant and input size; the base model typically runs on GPUs with 16 GB+ VRAM for common slices. Reduce batch size or use a smaller variant if you encounter OOM errors.

## Input Format

The inference script expects an **input directory** containing one or more case descriptors. Each case is represented by a **JSON file** that points to the corresponding `.mat` file(s) and metadata (e.g. mask type, acquisition). The layout should follow the same structure as the CMRxRecon challenge (e.g. one JSON per case, with keys indicating paths to k-space and acquisition mask used by the reader).

Example layout:

```
input_dir/
├── Center001_UIH_30T_umr780_P007_cine_lax_4ch_kus_ktRadial16.json
├── ...
```

**Example case file** (`example/Center001_UIH_30T_umr780_P007_cine_lax_4ch_kus_ktRadial16.json`):

This JSON is a case descriptor for one reconstruction case. It uses paths relative to the input directory (or repo root when using `example` as input).

| Field   | Type     | Description |
|--------|----------|-------------|
| `kspace` | string | Path to the undersampled k-space `.mat` file (e.g. `example/MultiCoil/Cine/UnderSample_TaskR1/cine_lax_4ch_kus_ktRadial16.mat`). |
| `mask`   | array  | List of paths to mask `.mat` file(s) used for this acquisition (e.g. `["example/MultiCoil/Cine/Mask_TaskR1/cine_lax_4ch_mask_ktRadial16.mat"]`). |

Example content:

```json
{
    "kspace": "example/MultiCoil/Cine/UnderSample_TaskR1/cine_lax_4ch_kus_ktRadial16.mat",
    "mask": [
        "example/MultiCoil/Cine/Mask_TaskR1/cine_lax_4ch_mask_ktRadial16.mat"
    ]
}
```

The reader loads these `.mat` files according to the schema in `scripts/readers.py` and `scripts/mri_data/data_utils.py`. You can add more cases by placing additional JSON files (with unique names) in the same input directory, each referencing their own `kspace` and `mask` paths.

The repository includes an `example` folder with a sample JSON so you can run a quick test. See `scripts/readers.py` and `scripts/mri_data/data_utils.py` for the exact schema and how `.mat` files are loaded.

## Custom CINE H5 and VISTA masks

The custom CINE source H5 files use `(slice, coil, time, PE, FE)`, while the
inference reader expects logical `(time, slice, coil, PE, FE)`. Use the bundled
converter instead of transposing and saving manually; it accounts for the
additional axis reversal performed by `CMRxReconReader` for scipy MAT files.

The converters only write beneath `--output-root`. They do not modify the raw
H5 files or source VISTA masks.

```bash
python scripts/convert_cine_h5_kspace_to_mat.py \
  --input-h5-dir /mnt/qdata/rawdata/CINE/2D_h5_compressed \
  --output-root dataset/CustomCINEDataR1 \
  --glob 'Sub000[1-5].h5'

for sub in Sub0001 Sub0002 Sub0003 Sub0004 Sub0005; do
  python scripts/convert_vista_txt_mask_to_mat.py \
    --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
    --output-root dataset/CustomCINEDataR1 \
    --frequency-size 176 \
    --glob 'mask_VISTA_132x25_acc8_8.txt' \
    --case-id "$sub"
done
```

The first command creates k-space MAT files and initial JSON descriptors. The
second creates fixed masks with 20 central ACS lines by default and adds each
mask to its case JSON. Mask filenames use the model-known `ktRadial8` alias so
the reader extracts acceleration factor 8 correctly.

Before a full run, inspect one descriptor and use debug mode, which limits the
input list to one case:

```bash
cat dataset/CustomCINEDataR1/json_input/Sub0001.json

python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR1/json_input

python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR1/json_input \
  -o output/CustomCINEOutputDebugR1 \
  --debug --profile-timing
```

Then run all cases without `--debug`:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR1/json_input \
  -o output/CustomCINEOutputR1
```

For this repository layout, `scripts/inference.py` defaults to input
`dataset/CustomCINEDataR1/json_input` and output
`output/CustomCINEOutputR1`. Explicit `-i` and `-o` arguments still override
these defaults.

### Acc16 / R2 Variant

To run the same five cases with the acc16 seed-8 VISTA mask, create a separate
dataset root and output root:

```bash
python scripts/create_custom_cine_acc_dataset.py \
  --source-root dataset/CustomCINEDataR1 \
  --output-root dataset/CustomCINEDataR2 \
  --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
  --mask-glob 'mask_VISTA_132x25_acc16_8.txt' \
  --mask-type ktRadial16

python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR2/json_input

python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR2/json_input \
  -o output/CustomCINEOutputR2 \
  --profile-timing
```

The R2 builder symlinks to the R1 k-space MAT files by default and writes new
`ktRadial16` mask MATs and JSON descriptors under `dataset/CustomCINEDataR2`.

For acc24, use the same helper with the R3 dataset/output roots:

```bash
python scripts/create_custom_cine_acc_dataset.py \
  --source-root dataset/CustomCINEDataR1 \
  --output-root dataset/CustomCINEDataR3 \
  --input-mask-dir /home/students/studxusiy1/mr_recon/masks \
  --mask-glob 'mask_VISTA_132x25_acc24_8.txt' \
  --mask-type ktRadial24

python scripts/validate_cine_inference_data.py \
  dataset/CustomCINEDataR3/json_input

python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i dataset/CustomCINEDataR3/json_input \
  -o output/CustomCINEOutputR3 \
  --profile-timing
```

## Running Inference

### Single-GPU

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i /path/to/input_dir \
  -o /path/to/output_dir
```

- `-c` / `--config`: Path to the model config JSON (e.g. `nv_raw2insights_mri_base`, `nv_raw2insights_mri_small`, `nv_raw2insights_mri_large`).
- `-i` / `--input_path`: Directory containing case JSON files (and referenced `.mat` data).
- `-o` / `--output_path`: Directory where reconstructions and logs will be written.
- `-d` / `--debug`: Optional; enables debug behavior (e.g. deterministic seeds, extra logging).

Checkpoints are resolved automatically from the config’s `model_variant` (download from Hugging Face if not present). To use a **local checkpoint** instead:

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -m /path/to/nv_raw2insights_mri_base.pt \
  -i /path/to/input_dir \
  -o /path/to/output_dir
```

### Multi-GPU (torchrun)

For faster inference on multiple GPUs:

```bash
torchrun --nproc_per_node=8 scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i /path/to/input_dir \
  -o /path/to/output_dir
```

Adjust `--nproc_per_node` to the number of GPUs on the node. If `MASTER_PORT` is not set, the script will try to pick a free port automatically.

## Config and Checkpoint Overrides

- **Config**: All inference-time settings (model architecture, data paths, etc.) are read from the JSON specified by `-c`. Edit the config file or use a copy to change behavior.
- **Checkpoint**: By default, the checkpoint is determined by `model_variant` in the config and the Hugging Face registry (see [Setup](setup.md)). Override with `-m` to force a local `.pt` file.

## Outputs

The script writes reconstructions and run metadata under the given output directory. Reconstructed images are saved as `.mat` files under `val_img4ranking/` (one file per case, containing the `img4ranking` array). The run also writes a copy of the config (e.g. `config.json`) used for the run.

## Visualizing output

Use `scripts/visualize_mat.py` to visualize `.mat` files produced by inference. It reads `.mat` files (with `h5py` or `scipy`), extracts the `img4ranking` array, normalizes for display, and shows all slices and time frames in a grid. Supported array shapes: 2D (single slice), 3D (H, W, slices), or 4D (H, W, slices, time).

**Example** — after running inference as below, visualize all reconstructions in the output folder and save figures:

```bash
# Run inferenc
python scripts/inference.py -c configs/nv_raw2insights_mri_base.json -i example -o outputs/example_output_base

# Visualize all .mat files in the reconstruction folder and save PNGs
python scripts/visualize_mat.py outputs/example_output_base/val_img4ranking -o outputs/example_output_base/figs
```

<table>
  <tr>
    <th>Example Output</th>
  </tr>
  <tr>
    <td valign="middle" width="100%">
      <img src="https://github.com/user-attachments/assets/fa259cd4-5038-478f-b16f-85ac115d2b31" width="100%" alt="Reference image">
    </td>
  </tr>
</table>
