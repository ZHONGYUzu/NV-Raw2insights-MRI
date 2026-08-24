# 迁移阶段 0：数据与实验资产清单

> 快照日期：2026-08-24
> 范围：服务器两个 worktree、仓库外派生数据、实验输出、评价结果、模型缓存和已知外部只读数据源
> 操作类型：只读统计
> 本文完成 `docs/CODE_AND_EXPERIMENT_MIGRATION_PLAN.md` 阶段 0 的第 4 项。统计期间没有复制、移动、覆盖、重命名、压缩、计算全量大文件哈希或删除任何文件。

对应的初始机器可读清单：[`../../experiments/registry.csv`](../../experiments/registry.csv)。

## 1. 统计口径与总量

核心资产根目录的精确总量为 `105,185,883,342` bytes，约 `105.2 GB`（十进制）或 `98.0 GiB`。这个总量：

- 包含两个服务器 worktree 内的数据、输出和 archive；
- 包含 `dataset_v0`、`dataset_v1`、CMRxRecon2024 派生子集、两处 Hugging Face cache；
- 不包含 `/mnt/qdata`、`/mnt/ceph/vol_06_rawdata_01` 和 fastMRI rawdata 等只读原始数据；
- 不包含 Python/conda 环境；环境目录遍历没有及时返回，且环境应由依赖重建，不属于实验资产；
- 不包含相邻的 RAM、SNRAware 等其他项目。

| 根目录 | 精确大小 | 约合 | 说明 |
| --- | ---: | ---: | --- |
| `/home/students/studxuzho1/NV-Raw2insights-MRI` | 41,230,212,668 B | 38.4 GiB | 主 worktree，含 21 GB dataset、2.6 GB output、15 GB archive |
| `/home/students/studxuzho1/NV-Raw2insights-MRI-cmrxrecon` | 3,262,006,648 B | 3.0 GiB | CMRx worktree，主要为 3.1 GB SDUM runs |
| `/home/students/studxuzho1/dataset_v0` | 42,806,918,105 B | 39.9 GiB | 早期转换输入、134 个 normalized GT NPY 和多加速率输入 |
| `/home/students/studxuzho1/dataset_v1` | 3,836,214,156 B | 3.6 GiB | 5-case 后续数据和输出 |
| `/home/students/studxuzho1/datasets` | 1,038,029,806 B | 0.97 GiB | CMRxRecon2024 `sdum-001` 派生子集 |
| `/home/students/studxuzho1/failure_runs` | 110,733 B | 108 KiB | 早期失败/转换脚本和日志 |
| `/home/students/studxuzho1/hf_cache` | 4,092,503,415 B | 3.8 GiB | 当前多数 Slurm 配置引用的显式 HF cache |
| `/home/students/studxuzho1/.cache/huggingface` | 8,919,887,811 B | 8.3 GiB | 默认 HF cache，含 Large checkpoint |

服务器在本次盘点开始和结束前没有运行或排队的相关 Slurm 作业。

## 2. 时间字段说明

Ceph/Linux 目录不一定提供可靠的 birth/creation time，因此本清单使用三种时间：

1. `first_file_mtime`：目录内最早文件修改时间；
2. `last_file_mtime`：目录内最新文件修改时间；
3. `dir_mtime`：目录条目本身最后修改时间。

它们是可核对的时间范围，不应被解释为绝对创建时间。`experiments/registry.csv` 记录前两项。

## 3. 外部只读数据源

这些数据不在上述 98.0 GiB 内，不得由整理过程修改：

| 数据源 | 路径 | 用途 | 策略 |
| --- | --- | --- | --- |
| CINE 2D H5 | `/mnt/qdata/rawdata/CINE/2D_h5_compressed` | in-house CINE 原始 fully sampled H5 | 只读；记录选择病例、size、mtime，派生文件另存 |
| VISTA masks | `/home/students/studxusiy1/mr_recon/masks` | CINE 固定/多 seed mask 来源 | 只读；记录具体 TXT 文件名和 SHA-256 |
| CMRxRecon2023 training | `/mnt/ceph/vol_06_rawdata_01/CMRxRecon/CMRxRecon_2023_training_init/ChallengeData/MultiCoil/CINE/TrainingSet` | SDUM-002 至 013 的 full sample 与官方 mask | 只读；每个 manifest 记录 case、view 和 source path |
| fastMRI brain multicoil validation | `/mnt/ceph/vol_12_rawdata_fastMRI/fastMRI/brain/multicoil_val` | fastMRI 10-case cohort | 只读；本仓库 dataset 仅保存 manifest 和 10 个 symlink |

## 4. 主 worktree：现役数据集

| 路径（相对主 worktree） | 大小 | 文件/链接 | 病例 | 时间范围 | 用途 |
| --- | ---: | ---: | ---: | --- | --- |
| `dataset/CINE_test_acc8` | 10,402,711,468 B | 30 files | 10 | 2026-07-12 08:06–08:32 | 当前 10-case CINE benchmark acc8，含 k-space、mask、JSON |
| `dataset/CINE_test_acc16` | 614,425 B | 20 files + 10 links | 10 | 2026-07-12 08:50–08:51 | acc16 variant，k-space 通过 symlink 复用 |
| `dataset/CINE_test_acc24` | 489,091 B | 20 files + 10 links | 10 | 2026-07-12 08:57–08:58 | acc24 variant，k-space 通过 symlink 复用 |
| `dataset/h5_converted` | 10,809,353,028 B | 40 files | 10 | 2026-07-14 20:31–21:09 | external H5 `dMap`/raw VISTA 转换输入 |
| `dataset/h5_converted_acc16` | 707,149 B | 20 files + 10 links | 10 | 2026-07-15 09:16 | acc16 H5-derived variant |
| `dataset/h5_converted_acc24` | 544,124 B | 20 files + 10 links | 10 | 2026-07-15 12:33 | acc24 H5-derived variant |
| `dataset/GT_from_dImgC` | 336,119,629 B | 10 MAT | 10 | 2026-07-12 06:41–06:42 | active CINE benchmark ground truth |
| `dataset/CMRxRecon2023CINELAX10Acc8Official` | 41,176 B | 21 files | 10 | 2026-07-16 09:23 | CMRxRecon2023 LAX official acc8 descriptors/masks |
| `dataset/FastMRIBrainMulticoilVal10` | 12,648 B | 1 JSON + 10 links | 10 | 2026-07-19 20:08 | fastMRI AXT1 multicoil validation cohort manifest |

`dataset/archiv` 只有 11 个小型历史 descriptor/MAT 文件（4.4 KB），后续并入 legacy registry，不与顶层 `archive/` 混淆。

## 5. 主 worktree：模型输出

正式 10-case 输出均包含 10 个 reconstruction MAT 和 1 个 config JSON：

| 输出路径 | 大小 | 病例 | 时间范围 | 对应输入 |
| --- | ---: | ---: | --- | --- |
| `output/CINE_test_acc8` | 370,867,943 B | 10 | 2026-07-12 09:21–18:46 | `dataset/CINE_test_acc8` |
| `output/CINE_test_acc16` | 370,867,945 B | 10 | 2026-07-12 10:55–16:17 | `dataset/CINE_test_acc16` |
| `output/CINE_test_acc24` | 370,867,945 B | 10 | 2026-07-12 19:40–23:11 | `dataset/CINE_test_acc24` |
| `output/h5_converted` | 370,867,939 B | 10 | 2026-07-14 22:01–2026-07-15 00:58 | external dMap/raw VISTA acc8 |
| `output/h5_converted_acc16` | 370,867,951 B | 10 | 2026-07-15 09:38–12:32 | external dMap/raw VISTA acc16 |
| `output/h5_converted_acc24` | 370,867,951 B | 10 | 2026-07-15 12:57–15:50 | external dMap/raw VISTA acc24 |
| `output/CMRxRecon2023CINELAX10Acc8Official` | 117,708,860 B | 10 | 2026-07-16 09:33–10:34 | CMRxRecon2023 official acc8 |
| `output/FastMRIBrainMulticoilVal10Acc8` | 93,235,001 B | 10 | 2026-07-19 20:16–21:11 | fastMRI acc8 |
| `output/FastMRIBrainMulticoilVal10Acc16` | 93,235,003 B | 10 | 2026-07-19 21:13–22:03 | fastMRI acc16 |
| `output/FastMRIBrainMulticoilVal10Acc24` | 93,235,003 B | 10 | 2026-07-19 22:07–22:56 | fastMRI acc24 |

另外存在 4 个 fastMRI geometry/debug 输出，共 4 个目录、5 个 reconstruction MAT，单目录约 6.6–13.1 MB；它们是开发诊断资产，不与正式 10-case cohort 混用。

其他生成资产：

- `output/GT_from_dImgC_figs`：10 PNG，73,981,499 B；
- `output/GT_from_dImgC_gifs`：10 GIF，7,354,524 B；
- `output/archiv`：37 个混合历史文件，32,333,810 B；
- `outputs/example_output_base`：公开 example 的 1 MAT、1 PNG、1 config，共 4,963,296 B；
- `result/`：80 张 force-fill qualitative PNG，共 18,543,964 B。

## 6. 主 worktree：评价结果

| 结果路径 | 大小 | 组成 | 时间范围 | 说明 |
| --- | ---: | --- | --- | --- |
| `Results/CINE_vista_mask_generalization_v1` | 35,233,753 B | 1 JSON, 2 CSV, 162 PNG | 2026-08-12 10:08–10:56 | 10-case seed8 acc8/16/24 完整评价 |
| `Results/raw_vista` | 7,186,892 B | 2 CSV, 16 PNG | 2026-07-15 15:56 | raw VISTA review |
| `Results/forcefill_masks` | 7,696,953 B | 2 CSV, 18 PNG | 2026-07-16 08:47–21:53 | forced-ACS review |
| `Results/FastMRIBrainMulticoilVal10Acc8` | 64,941,556 B | 10 GT MAT, 1 CSV, 12 PNG | 2026-07-19 21:11–21:12 | fastMRI acc8 evaluation |
| `Results/FastMRIBrainMulticoilVal10Acc16` | 64,702,699 B | 10 GT MAT, 1 CSV, 12 PNG | 2026-07-19 22:03–22:04 | fastMRI acc16 evaluation |
| `Results/FastMRIBrainMulticoilVal10Acc24` | 64,510,679 B | 10 GT MAT, 1 CSV, 12 PNG | 2026-07-19 22:56–22:57 | fastMRI acc24 evaluation |
| 4 个 fastMRI debug Results | 28,036,474 B 合计 | 各 1 GT MAT、1 CSV、3 PNG | 2026-07-20–21 | geometry/axis/hard-DC diagnostics |
| `ResultsCMRxRecon2023CINELAX10Acc8Official` | 111,844,362 B | 10 GT MAT, 1 CSV, 11 PNG, 1 JSON | 2026-07-16 10:34–10:35 | 正式 CMRxRecon2023 LAX10 评价 |
| `ResultsCMRxRecon2023CINELAXAcc8` | 12,510,610 B | 1 GT MAT, 1 CSV, 2 PNG, 1 JSON | 2026-07-16 08:43 | 1-case CMRx debug/compatibility 评价 |

## 7. SDUM run 清单

`sdum-007` 位于主 worktree；其他 SDUM run 位于 CMRx worktree。表中 MAT 数包含 mask、GT、full-size、ranking 或 debug MAT，不等同于病例数。

| Run | 数据/采样含义 | 病例 | 大小 | files / MAT / JSON / CSV / PNG | 时间范围 | 当前可观察完整性 |
| --- | --- | ---: | ---: | --- | --- | --- |
| `sdum-001` | CMRxRecon2024 15-case mixed acquisition Uniform8 subset | 15 | 4,286,975 B | 40 / 15 / 16 / 0 / 2 | 2026-08-10–13 | 15 recon；只有 preview，无正式 metrics summary |
| `sdum-002` | CMRxRecon2023 LAX15 official AccFactor08/ktRadial8 | 15 | 177,662,755 B | 115 / 60 / 17 / 0 / 31 | 2026-08-11–13 | 输出/GT 存在；无 summary |
| `sdum-003` | 同 cohort，official AccFactor04 但 runtime alias ktRadial8 | 15 | 14,351,122 B | 100 / 46 / 19 / 4 / 17 | 2026-08-12–13 | paired ranking summary 存在 |
| `sdum-004` | 同 cohort，official acc8 / `Uniform8` conditioning | 15 | 347,025,652 B | 105 / 60 / 19 / 1 / 17 | 2026-08-13 | full direct summary 存在 |
| `sdum-005` | 同 cohort，official acc8 / `Uniform16` conditioning mismatch | 15 | 347,025,039 B | 105 / 60 / 19 / 1 / 17 | 2026-08-13 | full direct summary 存在 |
| `sdum-006` | 同 cohort，official acc8 / `Uniform24` conditioning mismatch | 15 | 179,830,954 B | 69 / 45 / 18 / 0 / 0 | 2026-08-13 | full/ranking 输出存在；缺 metrics summary |
| `sdum-007` | in-house CINE 10-case static random Cartesian acc8，无 forced ACS | 10 | 419,473,834 B | 41 / 21 / 13 / 0 / 0 | 2026-08-18 | 10 full recon + debug + masks；缺 metrics summary |
| `sdum-008` | CMRx LAX15 generated `Uniform8`, 20 ACS | 15 | 357,723,496 B | 111 / 61 / 20 / 1 / 16 | 2026-08-23 | full direct summary 存在 |
| `sdum-008-debug` | SDUM-008 单例准备 debug | 1 | 23,702 B | 3 / 1 / 2 / 0 / 0 | 2026-08-23 | 独立 debug preparation，未评价 |
| `sdum-009` | CMRx LAX15 generated `ktGaussian8`, 20 ACS | 15 | 359,308,041 B | 105 / 61 / 20 / 1 / 16 | 2026-08-23 | full direct summary 存在 |
| `sdum-010` | CMRx LAX15 generated `ktRadial8`, 20 ACS | 15 | 375,423,411 B | 135 / 61 / 20 / 1 / 46 | 2026-08-23–24 | full direct summary 存在 |
| `sdum-011` | CMRx LAX15 generated `Uniform16`, 20 ACS | 15 | 357,182,924 B | 101 / 61 / 20 / 1 / 16 | 2026-08-23–24 | full direct summary 存在 |
| `sdum-012` | CMRx LAX15 generated `ktGaussian16`, 20 ACS | 15 | 358,433,465 B | 101 / 61 / 20 / 1 / 16 | 2026-08-23–24 | full direct summary 存在 |
| `sdum-013` | CMRx LAX15 generated `ktRadial16`, 20 ACS | 15 | 358,126,181 B | 101 / 61 / 20 / 1 / 16 | 2026-08-23–24 | full direct summary 存在 |

相较早期审计，`sdum-010` 至 `sdum-013` 已经补齐 `evaluation_full_direct/summary_metrics.json`。当前明确缺 summary 的正式 run 是 `sdum-001`、`002`、`006` 和 `007`；是否需要补评价由阶段 0 第 5 项分类决定。

## 8. CMRxRecon2024 `sdum-001` 派生数据

路径：`/home/students/studxuzho1/datasets/CMRxRecon2024/sdum-001`

- 大小：1,038,029,741 B；
- 文件：30 MAT；
- 病例：15 个，每例一个 undersampled k-space 和一个 mask；
- acquisition：BlackBlood 6、Cine 6、Flow2d 3；
- 时间：2026-08-10 21:34–21:38；
- 对应 run：CMRx worktree `runs/sdum-001`。

这是派生数据，不是 CMRxRecon2024 原始 archive。整理时必须与 run manifest 一起保留或一起归档。

## 9. 早期仓库外数据

### 9.1 `dataset_v0`（42,806,918,105 B）

| 子目录 | 大小 | 文件特征 | 时间范围 | 说明 |
| --- | ---: | --- | --- | --- |
| `data_converted` | 5,018,130,508 B | 3 large files | 2026-05-20 | 早期 converted data |
| `data_json` | 301 B | 3 JSON | 2026-05-20 | 早期 descriptors |
| `model_input_test_multi_acc` | 20,072,571,548 B | 9 MAT, 6 JSON | 2026-06-03 | 早期 multi-acc 输入 |
| `model_input_test_multi_acc_repaired` | 1,677,514,668 B | 5 MAT, 2 JSON, 1 link | 2026-06-03–08 | repaired subset |
| `model_input_test_v1` | 5,018,152,411 B | 2 MAT, 3 JSON, 6 ancillary files | 2026-06-03 | 早期 v1 input |
| `model_input_test_v2` | 2,511,386,696 B | 1 MAT, 3 JSON | 2026-06-03 | 早期 v2 input |
| `model_input_test_v3` | 36,808 B | 1 JSON | 2026-06-08 | descriptor-only attempt |
| `norm_img` | 8,507,149,056 B | 134 NPY | 2026-06-02–03 | normalized fully sampled GT；必须保留 |
| `h5_previews` | 1,910,913 B | 8 PNG | 2026-05-28 | inspection previews |
| `output_test_multi_acc` | 36,810 B | 1 JSON | 2026-06-08 | output config/descriptor，非重建主体 |

### 9.2 `dataset_v1`（3,836,214,156 B）

- `MultiCoil`：63 MAT，3,833,909,000 B；
- `json_input`：5 JSON；
- `output`：5 MAT、1 CSV、9 PNG、1 config，共 1,398,925 B；
- `raw_rec_image`：15 PNG；
- 时间范围：2026-06-10 至 2026-06-11；
- 推定 5-case 后续实验，阶段 0 第 5 项前保持 `pending`。

### 9.3 `failure_runs`

10 个早期转换/JSON/GT 生成 Python 脚本与日志，共 110,733 B，时间 2026-05-20 至 2026-06-03。它们可能解释 `dataset_v0` 的生成历史，应先归为 provenance，不应直接删除。

## 10. 顶层 archive

### 10.1 `archive/datasets`

总计 13,117,004,407 B，167 个普通文件和 50 个 symlink。

| 子目录 | 大小 | 文件/链接 | 病例或用途 |
| --- | ---: | ---: | --- |
| `CINE1` | 8,150,773,197 B | 30 / 0 | 10-case external-map CINE1 base |
| `CINE1acr2` | 721,464 B | 20 / 10 | 10-case acceleration variant |
| `CINE1acr4` | 857,344 B | 20 / 10 | 10-case acceleration variant |
| `CINE1acr8` | 786,114 B | 20 / 10 | 10-case acceleration variant |
| `CINE1acr16` | 496,194 B | 20 / 10 | 10-case acceleration variant |
| `CustomCINEDataR1` | 3,923,476,119 B | 18 / 0 | legacy 5-case acc8 baseline |
| `CustomCINEDataR2` | 209,657 B | 13 / 5 | legacy 5-case acc16 variant |
| `CustomCINEDataR3` | 181,239 B | 13 / 5 | legacy 5-case acc24 variant |
| `CustomCINEDataDirectVista` | 785,372,095 B | 3 / 0 | 1-case direct VISTA debug input |
| `GT_from_kspace_dMap` | 254,130,720 B | 10 / 0 | 10-case historical k-space+dMap GT |

### 10.2 `archive/outputs`

总计 2,904,749,816 B、253 个文件。主要内容：

- CINE1 acc2/4/8/16 正式输出各约 503–524 MB，每组 10 MAT；
- CINE1 四个 debug 输出各约 27.9 MB，每组 1 MAT；
- CustomCINE R1/R2/R3 输出分别约 142/140/250 MB，每组 5 MAT；
- CustomCINE R1 debug 约 48.9 MB；R2/R3 debug 只有小型占位文件；
- CINE1 GIF、qualitative、comparison 和 metrics；
- legacy 5-case acceleration comparison、metrics 与 qualitative PNG。

archive 中的每个直接子目录都已在本次原始统计中记录。初始 CSV 先以 `archive/datasets` 和 `archive/outputs` 父级登记；任何子目录如果以后要移动或删除，必须先成为独立 CSV 行并补 checksum。

## 11. Hugging Face 模型缓存

两个 cache 都指向同一 Hub snapshot：

```text
38860e9a6153d857b55439b3d3fd55f850eccdcd
```

| Cache | Checkpoint | Blob ID | 大小 | 关系 |
| --- | --- | --- | ---: | --- |
| `hf_cache` | Base | `667189bac5f159e52b2f1640b504b4efd549717c37720c2397e749d16ec056b8` | 3,038,170,078 B | 与默认 cache 完全相同的 blob ID 和大小 |
| `hf_cache` | Small | `8ba66defef7c66738c1eac25f2ea6a5d164fb9e547c36d90b469d8d5287ed624` | 911,518,382 B | 仅此 cache 存在 |
| default cache | Base | `667189bac5f159e52b2f1640b504b4efd549717c37720c2397e749d16ec056b8` | 3,038,170,078 B | 重复候选，尚未删除 |
| default cache | Large | `0a1d06d74dcd3dbd0c3692edc203527370a796fbb6f61be4c58bdcbcb817c54b` | 5,738,819,266 B | 仅默认 cache 存在 |

Base checkpoint 的潜在可回收空间约 3.04 GB，但在阶段 0 第 5 项前不得删除。建议最终统一 `HF_HOME=/home/students/studxuzho1/hf_cache`，把 Large 以 Hub 支持的方式纳入同一 cache，再验证所有 config/Slurm 引用和 blob ID。

## 12. 其他相关目录排查

- 对 home 下仓库、cache 和 env 之外的 Python、shell、Slurm、Markdown、JSON、YAML 文件执行了模型名称内容搜索；没有发现额外 `NV-Raw2insights` 引用。
- `/home/students/studxuzho1/ram-fastmri-brain-adapter`、`ram-results`、`ram/` 和 SNRAware 目录属于相邻 RAM/SNRAware 项目；路径名可能包含 CMRx/fastMRI，但没有纳入本项目核心资产总量。
- CMRx worktree 的 `external/CMRxRecon2025` 是 3.9 MB 的干净嵌套 clone，固定在 commit `12a37ac76d8e97a21eb414dc1f5ade86aec16b63`。只需要记录来源或转为正式 submodule，不提交其 `.git`。
- 模型专用环境 `/home/students/studxuzho1/envs/nv-raw2insights-mri` 存在，但递归统计未及时返回；它应通过 environment/lockfile 重建，不作为不可替代实验资产。

## 13. Checksum 策略

本阶段没有对约 98 GiB 资产执行全量哈希，避免给共享存储造成不必要 I/O。后续采用分层策略：

| 资产类型 | 策略 |
| --- | --- |
| JSON/YAML/config/manifest/CSV/脚本 | 每个文件计算 SHA-256，写入 `artifacts.json` 或 registry |
| 小型 mask MAT | 每个文件计算 SHA-256；同时记录 shape、dtype、nominal/effective acceleration |
| 大型 k-space/GT/reconstruction MAT、NPY、H5 | 仅在准备移动、归档或判断重复时逐文件 SHA-256；平时记录 size+mtime+shape |
| symlink 复用的数据集 | 记录 link target；只对真实源文件计算一次 SHA-256 |
| Hugging Face LFS blobs | 使用完整 blob ID、size 和 snapshot commit；移动 cache 后抽样或全量验证 blob hash |
| 外部只读 raw source | 记录绝对路径、选择病例、size、mtime；只对正式 cohort 的实际输入做一次 checksum manifest |
| PNG/GIF/PDF previews | 归档时按目录生成 SHA-256 manifest；不作为重建数值结果的唯一证据 |

## 14. 阶段状态

- 阶段 0，第 1 项——确认相关服务器任务结束：**完成**。
- 阶段 0，第 2 项——记录 branch、HEAD、dirty 状态：**完成**。
- 阶段 0，第 3 项——生成代码差异清单：**完成**。
- 阶段 0，第 4 项——生成数据与实验资产清单：**完成**。
- 阶段 0，第 5 项——将资产标为 `active`、`completed`、`incomplete`、`legacy`、`duplicate-candidate` 或 `unknown`：**尚未执行**。

当前 CSV 中的正式分类统一为 `pending-item5`。在第 5 项完成前，不移动或删除任何数据、run、output、archive、cache 或环境目录。

## 15. 后续分类更新

阶段 0 第 5 项已于同日完成。分类结果见
[`2026-08-24-migration-phase0-asset-classification.md`](2026-08-24-migration-phase0-asset-classification.md)。
`experiments/registry.csv` 已更新为正式分类；本节上方的 `pending-item5` 描述仅记录第 4 项完成时的历史状态。
