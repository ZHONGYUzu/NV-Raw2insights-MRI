# NV-Raw2Insights-MRI

[![License](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/Weights-NVIDIA%20Open%20Model-green.svg)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/nvidia/NV-Raw2Insights-MRI)
[![Paper](https://img.shields.io/badge/arXiv-2512.17137-red.svg)](https://arxiv.org/abs/2512.17137)

Universal MRI reconstruction from undersampled k-space. A single model handles diverse protocols, anatomies, contrasts, and acceleration factors without task-specific fine-tuning.

<p align="center">
<img width="900" alt="NV-Raw2Insights-MRI" src="https://github.com/user-attachments/assets/27ce9dea-c592-4dd5-b984-38542e1cf8e6" />
</p>

## Overview

NV-Raw2Insights-MRI is built on the Scalable Deep Unrolled Model (SDUM) framework. It combines a Restormer-based cascaded unrolled architecture with learned coil sensitivity estimation, sampling-aware weighted data consistency, and universal conditioning on protocol metadata. Trained on heterogeneous data from CMRxRecon2024, CMRxRecon2025, and fastMRI brain datasets, a single model achieves state-of-the-art results across cardiac, brain, and knee MRI reconstruction.

This project was conducted by NVIDIA in collaboration with the [CMRxRecon Team](https://github.com/CmrxRecon), [Fudan University](https://hupi.fudan.edu.cn/en/), and [Johns Hopkins University](https://profiles.hopkinsmedicine.org/provider/shanshan-jiang/2777746).

## News

- **[March 2026]** — Released NV-Raw2Insights-MRI as part of the NVIDIA MedTech Open Models
- **[February 2026]** — Achieved 1st place across all four tracks in the [CMRxRecon2025 Challenge](https://www.synapse.org/Synapse:syn59814210/wiki/634966) without task-specific fine-tuning

## Model Variants

| Model | Cascades | Parameters | HuggingFace | License |
|-------|:--------:|:----------:|-------------|---------|
| [NV-Raw2Insights-MRI-Small](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | 6 | 230M | [Download](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| [NV-Raw2Insights-MRI-Base](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | 18 | 760M | [Download](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| [NV-Raw2Insights-MRI-Large](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | 34 | 1.4B | [Download](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

Checkpoints are automatically downloaded from [HuggingFace](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) when not provided locally.

## Acceleration Scope

Do not mix the acceleration settings of the source benchmarks, paper tables,
and released checkpoint configs:

| Scope | Acceleration factors |
|-------|----------------------|
| CMRxRecon2024 Task 1 | `4x`, `8x`, `10x` |
| CMRxRecon2024 Task 2 | `4x`, `8x`, `12x`, `16x`, `20x`, `24x` |
| SDUM paper experiments | CMRxRecon2024 uses the Task 1/2 ranges above; a separately trained fastMRI brain model reports `4x` and `6x` |
| Current released Small/Base/Large configs | `8x`, `16x`, `24x` |
| Formal checkpoint reproduction in this repository | **`8x`, `16x`, `24x`** |

Sources: [CMRxRecon2024 official task definitions](https://github.com/CmrxRecon/CMRxRecon2024#challenge-tasks),
[SDUM paper](https://arxiv.org/abs/2512.17137), and the released
[`small`](configs/nv_raw2insights_mri_small.json),
[`base`](configs/nv_raw2insights_mri_base.json), and
[`large`](configs/nv_raw2insights_mri_large.json) configs.

Accordingly, the active in-house checkpoint benchmark must use nominal
acceleration labels `8`, `16`, and `24` unless a separate out-of-distribution
experiment is explicitly declared. These are model-conditioning and experiment
labels. When ACS lines are added or forced, also report the measured effective
acceleration; it may differ from the nominal label.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Inference

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i example \
  -o outputs/example_output_base
```

For multi-GPU inference:

```bash
torchrun --nproc_per_node=8 scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i /path/to/input \
  -o /path/to/output
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Documentation map](docs/README.md) | Guide hierarchy, experiment scopes, and historical records |
| [Setup](docs/setup.md) | Full installation guide |
| [Inference](docs/inference.md) | Inference options, configs, multi-GPU |
| [Training](docs/training.md) | Training and fine-tuning guide |

### Server-only data and hidden Git records

Large datasets, generated reconstructions, evaluation results, logs, and the
contents of `dataset/archiv/` and `output/archiv/` remain on the compute server
and must not be committed to Git. A server checkout may hide these generated
paths with `.git/info/exclude`; unlike `.gitignore`, that file applies only to
the individual checkout and is neither committed nor synchronized to other
clones.

After server-side dataset reorganization, missing tracked `.gitkeep` files may
also be marked `skip-worktree`. This keeps the server working tree clean without
restoring the old placeholder directories or recording their deletion in a
commit. These records are also local to that checkout.

Inspect the server-only configuration from the repository root with:

```bash
# Exclusion rules for generated server paths
cat "$(git rev-parse --git-path info/exclude)"

# Tracked paths hidden with skip-worktree
git ls-files -v | grep '^S'

# Explain why a generated path is ignored
git check-ignore -v <path>

# Show ignored paths temporarily
git status --short --ignored
```

The optional server-local explanation is stored at
`.git/info/SERVER_HIDDEN_FILES.md` when created. Because anything under `.git/`
is repository metadata, that note is not transferred by `git pull`, push,
clone, or normal file synchronization. Do not manually create a `.git` folder
if it is absent; use `git rev-parse --git-dir` to verify that the current
directory is a Git checkout.

## Performance

### Model Scaling (PSNR vs cascade depth)

| Cascades (T) | PSNR (dB) | Parameters |
|:------------:|:---------:|:----------:|
| 1 | 28.73 | 42M |
| 3 | 30.21 | 126M |
| 6 | 32.09 | 253M |
| 10 | 32.54 | 422M |
| 18 | 33.18 | 759M |

### Inference Compute (per slice, NVIDIA H100, T=18)

| Input Size | Time (s) | Memory (GB) |
|:----------:|:--------:|:-----------:|
| 128x128 | 0.32 | 4.78 |
| 256x256 | 1.03 | 6.07 |
| 256x512 | 2.06 | 7.98 |
| 328x512 | 2.67 | 9.26 |
| 328x640 | 3.30 | 9.62 |
| 328x768 | 3.97 | 10.83 |

## License

| Component | License |
|-----------|---------|
| Source code | [Apache 2.0](LICENSE) |
| Model weights | [NVIDIA Open Model License](LICENSE.weights) |

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Citation

```bibtex
@article{wang2025sdum,
  title={SDUM: A Scalable Deep Unrolled Model for Universal MRI Reconstruction},
  author={Wang, Puyang and Guo, Pengfei and Chai, Keyi and Zhou, Jinyuan and Xu, Daguang and Jiang, Shanshan},
  journal={arXiv preprint arXiv:2512.17137},
  year={2025}
}
```

Please also cite the [CMRxRecon dataset](https://www.synapse.org/Synapse:syn59814210/wiki/) papers.

## Resources

- [SDUM Paper](https://arxiv.org/abs/2512.17137) — arXiv
- [HuggingFace Model](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) — Weights and model card
- [CMRxRecon2025 Challenge](https://www.synapse.org/Synapse:syn59814210/wiki/634966) — Benchmark
