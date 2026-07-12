# NV-Raw2Insights-MRI In-House CINE Benchmark Report

## 1. 本周目标

在自有 CINE MRI 数据集上建立 NV-Raw2Insights-MRI 的初始性能基线，并形成可复用的 benchmark 流程。

## 2. 实验设置

| Item | Setting |
| --- | --- |
| Model | NV-Raw2Insights-MRI Base |
| Dataset | 自有 CINE MRI 数据 |
| Test cases | 10 selected cases |
| Selected case IDs | Sub0014, Sub0026, Sub0030, Sub0047, Sub0049, Sub0051, Sub0096, Sub0105, Sub0112, Sub0122 |
| Selected case record | `docs/test_subjects.txt` |
| GT source | H5 `dImgC` |
| GT output format | MAT, key `gt`, layout `(frequency, phase, slice, time)` |
| GT normalization | max magnitude per case |
| Metrics | PSNR, SSIM, NMSE |
| Pretrained weights | Yes |
| Fine-tuning | No |
| Hardware | NVIDIA Tesla V100-SXM2-32GB |
| Driver / CUDA | NVIDIA driver 535.309.01 / CUDA 12.2 |
| Acceleration factors | acc8, acc16, acc24 |

## 3. 本周完成内容

### 已完成

- Randomly selected and recorded 10 fixed test cases for fair model/scenario comparison.
- Generated GT from H5 `dImgC` for the selected cases.
- Converted GT to inference-compatible MAT format.
- Generated GT thumbnail grids.
- Generated GT temporal GIFs using the center slice per case.
- Inspected selected H5 shapes before inference dataset conversion.
- Confirmed selected cases have variable PE size and slice count.
- Confirmed FE is `192` and time frames are `25` for all selected cases.
- Confirmed required VISTA masks exist for PE `156`, `162`, `174`, and `180` at acc8/acc16/acc24.
- Started SOP documentation for the full benchmark workflow.

### 已确认的 H5 shape 分组

| PE | Cases |
| --- | --- |
| 180 | Sub0014 |
| 162 | Sub0026, Sub0112 |
| 174 | Sub0030 |
| 156 | Sub0047, Sub0049, Sub0051, Sub0096, Sub0105, Sub0122 |

Shared dimensions:

```text
FE = 192
time frames = 25
coil count = 15
```

### 待完成

- Convert selected H5 `kSpace` into inference dataset format.
- Convert and register acceleration-specific VISTA masks.
- Validate JSON/MAT inference inputs for all acceleration settings.
- Run debug inference.
- Run full inference.
- Generate reconstruction outputs.
- Compute PSNR, SSIM, and NMSE.
- Generate reconstruction/error-map figures.
- Generate inference GIFs if needed.
- Summarize quantitative results.

## 4. 初步结果

Quantitative results are not filled yet because full inference and metric evaluation have not been confirmed complete.

| Acceleration | Cases | PSNR | SSIM | NMSE | Time/case |
| --- | ---: | --- | --- | --- | --- |
| acc8 | 10 | TBD | TBD | TBD | TBD |
| acc16 | 10 | TBD | TBD | TBD | TBD |
| acc24 | 10 | TBD | TBD | TBD | TBD |

For the final report, include:

- mean ± standard deviation
- median
- best and worst case
- per-case or per-slice distribution
- box plot or violin plot

Expected frame-level metric count depends on the final acceleration set. For the current selected 10 cases, total slice count is:

```text
14 + 15 + 12 + 3 + 3 + 14 + 17 + 11 + 18 + 12 = 119
```

For three accelerations, expected frame rows would be:

```text
119 slices x 25 frames x 3 accelerations = 8925 rows
```

## 5. 可视化结果

Required qualitative visualization:

```text
Undersampled input | Reconstruction | Ground Truth | Error map
```

CINE temporal visualization:

```text
25 cardiac phases GIF, center slice per case
```

Current generated GT visualization outputs:

```text
output/GT_from_dImgC_figs
output/GT_from_dImgC_gifs
```

Reconstruction and error-map visualizations are pending inference completion.

## 6. 当前发现的问题

- VISTA mask 需要按照每个 case 的 PE 选择对应文件，例如 `mask_VISTA_156x25_acc8_8.txt`。
- mask 中心 ACS 需要 fully sampled；当前流程使用 `--acs-lines 20`。
- GT 与 reconstruction 的归一化方法需要明确记录。当前 GT 使用 H5 `dImgC` 并按每 case max magnitude normalize。
- 不同加速倍数的 mask 命名需要与模型解析匹配，例如 `ktRadial8`, `ktRadial16`, `ktRadial24`。
- Base 模型计算量大；每个 case 包含多个 slices 和 25 cardiac frames，推理时间可能较长。
- 当前结果仍需在 full inference 后确认 GT、方向和 metric pipeline 是否完全正确。

## 7. 待确认项


