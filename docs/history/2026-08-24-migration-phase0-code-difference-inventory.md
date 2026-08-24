# 迁移阶段 0：代码差异清单

> 快照日期：2026-08-24
> 范围：本地工作树、GitHub 分支、服务器两个 worktree
> 操作类型：只读检查
> 本文完成 `docs/CODE_AND_EXPERIMENT_MIGRATION_PLAN.md` 阶段 0 的第 3 项。检查期间没有复制、覆盖、提交、合并、切换或删除文件。

## 1. 结论

本次逐文件和逐提交比较得到以下结论：

1. 服务器真正只有服务器存在、需要在后续收拢的项目代码只有两个 Python 脚本：
   - `scripts/prepare_sdum_007_static_random_cartesian.py`
   - `scripts/prepare_cmrxrecon2024_split_zip_subset.py`
2. 服务器其余未提交 Python/Slurm 文件均能在本地找到 SHA-256 完全一致的副本，或与本地已经跟踪的文件完全一致。
3. 服务器 CMRx worktree 对 `scripts/inference.py` 的未提交修改只是把 `previous_case_end` 改为 `time.perf_counter()`；该修复已经存在于 `CINE-MRI` 的提交 `e3b281b`，不能用旧 CMRx 文件覆盖较新的本地 `inference.py`。
4. `feature/cmrxrecon-adapter` 相对共同基线有 5 个提交。其中 2 个在 `CINE-MRI` 中有 patch-equivalent 提交；另外 3 个的有效文件或 ignore 规则也已经出现在本地工作树中。
5. 因此不应整体 merge `feature/cmrxrecon-adapter`。后续只需要迁移两个服务器独有脚本，并把本地已有但尚未提交的有效文件按功能分组提交。
6. `external/CMRxRecon2025` 是独立嵌套 Git clone，不是普通项目源码目录。后续只能记录 URL 与 commit，或明确转换成 submodule；不能直接递归提交其 `.git`。

## 2. 分支拓扑与独有提交

两个服务器分支的共同基线是：

```text
99be712b2e9fd34bf6a2f709373d311354fbda96  main
```

相对该共同基线：

| 分支/位置 | 基线后提交数 | 当前 HEAD |
| --- | ---: | --- |
| GitHub/服务器 `CINE-MRI` | 87 | `6f0c8b6` |
| 本地 `CINE-MRI` | 88 | `eeb6273` |
| GitHub/服务器 `feature/cmrxrecon-adapter` | 5 | `940d08f` |

本地比 GitHub `CINE-MRI` 多出的唯一提交：

```text
eeb6273 Add FT00 CINE smoke-test Slurm job
```

### 2.1 CMRx 分支的 5 个独有提交

```text
61eb9c0 remove crop function in inference
e36dd34 Save SDUM full-size and ranking-crop outputs
af7c6e1 Ignore server experiment runs
abb4b4a Add fixed SDUM conditioning runner
940d08f Fix NumPy import for full-size MRI output
```

`git cherry -v CINE-MRI feature/cmrxrecon-adapter` 的结果：

```text
- 61eb9c0 remove crop function in inference
+ e36dd34 Save SDUM full-size and ranking-crop outputs
+ af7c6e1 Ignore server experiment runs
+ abb4b4a Add fixed SDUM conditioning runner
- 940d08f Fix NumPy import for full-size MRI output
```

其中 `-` 表示在 `CINE-MRI` 中存在 patch-equivalent 改动：

| CMRx 提交 | CINE 等价提交 | 结论 |
| --- | --- | --- |
| `61eb9c0` | `c842626` | 已有等价的 full-size/no-ranking-crop 改动，不再迁移提交 |
| `940d08f` | `5828edd` | 已有等价 NumPy import 修复，不再迁移提交 |

其余三个提交的处置：

| CMRx 提交 | 内容 | 本地状态 | 处置 |
| --- | --- | --- | --- |
| `e36dd34` | 新增 `scripts/export_sdum_ranking_crops.py` | 本地未跟踪文件与服务器 tracked 文件 SHA-256 完全相同 | 保留本地文件，后续按 SDUM 功能提交；不 cherry-pick 整个分支 |
| `af7c6e1` | `.gitignore` 增加 `runs/` | 本地 `CINE-MRI` 的 `.gitignore` 已有 `runs/` | 功能已经存在，不需要迁移 |
| `abb4b4a` | 新增固定 conditioning Slurm runner | 本地未跟踪文件与服务器 tracked 文件 SHA-256 完全相同 | 保留本地文件，后续按 Slurm 功能提交 |

### 2.2 CMRx 分支从共同基线引入的文件

```text
M .gitignore
A scripts/export_sdum_ranking_crops.py
M scripts/inference.py
M scripts/mri_data/data_utils.py
A scripts/slurm/run_sdum_official_acc8_conditioning.sbatch
```

统计：5 个文件，221 行增加、6 行删除。这里的 `inference.py` 和 `data_utils.py` 是较旧分支上的实现；本地 CINE 分支此后又有大量修改，禁止以 CMRx 版本整体覆盖。

### 2.3 CINE 分支的独有提交清单

下面是 GitHub/服务器 `CINE-MRI` 相对 `feature/cmrxrecon-adapter` 的 87 个独有提交；本地另有前述 `eeb6273`：

```text
6f0c8b6 Add FT00 CINE acc8 smoke-test config
5059fb1 Add EXP009 static random Cartesian CINE mask experiment
72d10a2 Add CINE benchmark Excel workbooks
d4c7fb0 Add fixed-mask fastMRI zero-filled baseline
3e381f2 Align fastMRI sampling axis with model coordinates
769e235 Add hard data consistency diagnostic for fastMRI
f431433 modify infer script to adapt fastMRI dataset
65721e5 update validator again
8942685 upate validator
7fdcd07 Fix fastMRI readout oversampling geometry
37c55f4 update batch job
07cb756 update readme about git only in server part
689371d update fastMRI job
f845e9e change the order for 8 16 24
018cb1a plot script update
b9180f2 update sbatch for cmr dataset
5b2e2e1 cmr recon dataset process workflow update
edf158f update eval script
fa7418f update eval for forcefill dataset
857551c cmr dataset
c44ad10 info update
1acf890 reduce SLURM inference DataLoader memory
b26abaf scrum job update
554e040 update scrum script
d8cda47 slurm run r2 r3 mask
5ef8fa1 daily log 0714 update
f7a7d93 convert h5 data
04d89e7 convert h5 file. to desired data
171354b slurm test
7a1e843 record daily log or SOP update
6856c53 eval
36fe2b2 eval update
15ed526 eval metrix
e9f1fb1 eva for random datasets
0811bbf sop update
4ff1dc9 slurm job submit
555ef1a Add benchmark SOP and H5 shape inspection
204079b isnpect size before use
49189b9 Add center-slice GT GIF workflow
9e16228 Add center-slice GT GIF workflow
3d64c7f gif update
c983a54 Ignore local agent skill files
acf0c63 Stop tracking local agent skills
7ef709e test cased updated
a29ebb7 upupu
12a9b3a upup
9ac3c3b up
ee05c20 plot update
b7824e3 random modidy
302e389 eva update
2a0eb6e update eva three rows forma
4a66c10 eva script update
a642097 update
228c1ca daily log infer and eva
45db5ae Document ten-case CINE multi-acceleration workflow
eb72ccc Add CINE ground truth generation note
30e52c0 update convert from h5 to mat in gt generation purpose
7a263ae gt generation code
954452b daily 0623 update
1360814 repo modify to fit different acc rate, using dmap in h5 file
3740b2f update
fd06dba accrate trials
e76cdd0 clear structure for agent use
82ac856 update doc/
f5b5d23 add function compare existing GT with recon
df9b5c8 validate if GT correctly generated
52150c1 cal acc rate actual
47db2ba input recon gt error with title
ec078f4 agent md update
9deb425 qualitative results
79e2962 to visualize rec quality
c3d06a3 add colorbar
fefc4c1 update FYI about mask acc rate
e5d76b7 plot as example
9681295 add FYI and gif function
56a2bfb inspect recon quality file added
1c42c09 update agent md with r1 r2 r3 and the meanings
e71b3a3 mask 24
2452db0 transfer the acc 8 to acc 16 dataset
ab46b50 agent md update
e3b281b fix the data_load time negative value problem
5828edd add import np
28e5344 modify to push data folder but not data file
f41ea80 change data structure to fit the repo input request
c842626 remove crop function in inference
82611c2 for agent's info transferred fromthe first inference run
152960b Update gitignore for MRI project
```

## 3. Tracked 工作树修改

### 3.1 本地

| 文件 | Diff 规模 | 内容 | 状态/建议 |
| --- | ---: | --- | --- |
| `README.md` | +25/-0 | 增加 benchmark、论文和 released config 的 acceleration scope 说明 | 本地有效文档；保留，后续单独 review/提交 |
| `docs/README.md` | +1/-0 | 增加迁移路线文档索引 | 本次迁移产生；保留 |
| `scripts/evaluate_cmrxrecon2023_cohort.py` | +5/-3 | 当 manifest 缺少 `subject`/`view` 时，从 `case_id` fallback 推导 | 本地与服务器 CMRx 未跟踪副本 SHA-256 完全一致；保留本地版本 |

`evaluate_cmrxrecon2023_cohort.py` 的 fallback 是必要兼容逻辑：

```text
subject = case.get("subject") or case_id 的第一个下划线前缀
view    = case.get("view") or 去掉 subject 前缀后的剩余部分
```

### 3.2 服务器主 worktree

Tracked 修改：无。

### 3.3 服务器 CMRx worktree

| 文件 | Diff 规模 | 内容 | 状态/建议 |
| --- | ---: | --- | --- |
| `scripts/inference.py` | +1/-1 | `previous_case_end = time.time()` 语义改为 `time.perf_counter()` | 与 CINE 提交 `e3b281b` 的修复相同；不要覆盖本地较新的 inference 文件 |

本地当前 `scripts/inference.py` 已在相应位置使用 `time.perf_counter()`，因此服务器 dirty 修改属于重复修复，而不是待迁移的新逻辑。

## 4. 重名文件 SHA-256 对照

### 4.1 完全相同的本地/服务器文件

| 文件 | SHA-256 | 服务器状态 | 本地状态 |
| --- | --- | --- | --- |
| `scripts/evaluate_sdum_paired_ranking.py` | `fa1308f306522fb629b12b8a80b12e489a7642f43a867fab9c54b37e4187b8c7` | 主 worktree untracked | untracked |
| `scripts/create_cmrxrecon2023_training_masks.py` | `f23e198145f434862d4e8c33e484adb5b3922268425f7524aadb7b49ad64295c` | CMRx untracked | untracked |
| `scripts/evaluate_cmrxrecon2023_cohort.py` | `3405cfdbb9c96ad9da419c29b1dda43cbfc31e9d230112a9beb88d2e9412c9d0` | CMRx untracked | tracked modification |
| `scripts/evaluate_cmrxrecon2023_fullsample.py` | `3a82be371614bc1bbc4796e0b06f58bc96f95a4c530b5f66d6ad0e3fcce053db` | CMRx untracked | tracked |
| `scripts/plot_sdum_previews.py` | `4b09fa9fec646b99fdf0f3f77692f2a4b6307c5cddc3ca0e83a3cc633f011c2c` | CMRx untracked | untracked |
| `scripts/validate_cine_inference_data.py` | `446af33afe853f5f807af44227f41ff7093a50bedf3ca4e44b8d189250ec6391` | CMRx untracked | tracked |
| `scripts/export_sdum_ranking_crops.py` | `0de11d04d9f123dcef01e749dfe927e84935ce2a088fd39994f9f6243bbaac53` | CMRx tracked | untracked |
| `scripts/slurm/run_sdum_official_acc8_conditioning.sbatch` | `1226fc0088182fc373e6dbf2d096d64dbe2096c47ef96cd995d70e2d8341abdf` | CMRx tracked | untracked |
| `scripts/slurm/evaluate_sdum_008_full_direct.sbatch` | `f018bb24cd79ec917f45b85a6b93181c6316a9e65614ebe7fcd2590f8afb9147` | CMRx untracked | untracked |
| `scripts/slurm/evaluate_sdum_009_full_direct.sbatch` | `71e20d91cf6248cc5d95301db91e60888abd97556b72c26fb091f1ee23fa7944` | CMRx untracked | untracked |
| `scripts/slurm/prepare_sdum_011_012_013_acc16.sbatch` | `c8a6899ebeb260e6a0c8129f6476e6d0df502549a480f75afce8681ee8fe1b54` | CMRx untracked | untracked |
| `scripts/slurm/run_sdum_008_training_uniform8_debug.sbatch` | `ae9853057b65b8226e6d16c2171f1d45bb408993a0bad7f61c9cdf037670a5b3` | CMRx untracked | untracked |
| `scripts/slurm/run_sdum_008_training_uniform8_full.sbatch` | `06ac191694ba019a0386fbd056c9d6835a34ce8e91eeaebc5b7fb8a60bdcdf19` | CMRx untracked | untracked |
| `scripts/slurm/run_sdum_009_training_ktgaussian8_debug.sbatch` | `6fb0d297c0215537101457e97f55f3abfebbc2564d902d3d822c7a34bba97b75` | CMRx untracked | untracked |
| `scripts/slurm/run_sdum_009_training_ktgaussian8_full.sbatch` | `853b37f92ee0b3b12efbadb0d043aa23a674ccc45a20fd241d7b242836e2e4a4` | CMRx untracked | untracked |
| `scripts/slurm/run_sdum_010_training_ktradial8_debug.sbatch` | `474a06cfc2b980402c9849aa0c741132b77a908c153ae4a4a3d08074b04153fd` | CMRx untracked | untracked |
| `scripts/slurm/run_sdum_010_training_ktradial8_full.sbatch` | `6dc6bfdc8abada33ca554aed050936ac8215b23618e041883ab08272f14791eb` | CMRx untracked | untracked |
| `scripts/slurm/run_sdum_011_012_013_acc16_debug.sbatch` | `979eb038c64124b189c00108e1b750ef8140f9636438b6f5bc87777e32e1460a` | CMRx untracked | untracked |
| `scripts/slurm/run_sdum_011_012_013_acc16_full.sbatch` | `c54fa443d1d4f373cae47d8df3bb9702ad6707468fb0439dce9f6770e259048f` | CMRx untracked | untracked |

这些文件已经安全存在于本地，不需要从服务器重复复制。后续只需决定提交分组。

### 4.2 服务器独有代码

| 服务器文件 | SHA-256 | 用途 | 后续动作 |
| --- | --- | --- | --- |
| 主 worktree `scripts/prepare_sdum_007_static_random_cartesian.py` | `02514e4bbebbd40627c1e8cb8760520afd7d939dcdd8e04af9853ff4b7c6e5a5` | 为 10-case CINE OOD 实验生成静态随机 Cartesian PE mask、JSON 和 manifest | 阶段 1 使用 dry-run 后复制到本地，保留原始 SHA |
| CMRx worktree `scripts/prepare_cmrxrecon2024_split_zip_subset.py` | `17b696b3bd320e12585bc88c36cb696a2ffea5eb9f6b42d052ae451c13464c36` | 从 byte-split CMRxRecon2024 ZIP 中选择可用 subject、提取派生数据并写 run metadata | 阶段 1 使用 dry-run 后复制到本地，review 稀疏 ZIP/partial-file 行为后提交 |

阶段 1 前这两个文件都不能从服务器删除或覆盖。

## 5. 本地独有、尚未提交的项目代码

下列 7 个 project script 在两个服务器 dirty 清单中均没有对应项：

| 文件 | 用途 | 初步处置 |
| --- | --- | --- |
| `scripts/create_benchmark_v3.py` | 生成精简 benchmark v3 workbook | 保留；与 spreadsheet 产物分开提交/归档 |
| `scripts/create_cine_benchmark_matrix.py` | 生成可填写的 CINE benchmark matrix | 保留；与 workbook 来源关系需在文档中说明 |
| `scripts/find_common_vista_mask_seeds.py` | 查找各 PE/acceleration 共有的 VISTA seed | 保留；属于 CINE 数据准备工具 |
| `scripts/merge_cine_benchmark_workbooks.py` | 合并 CINE benchmark workbook | 保留；与 workbook 生成脚本同组 review |
| `scripts/slurm/evaluate_h5_dmap_raw_vista_seed8_all10.sbatch` | 评价 10-case H5 dMap/raw VISTA seed8 run | 保留；CINE benchmark Slurm 组 |
| `scripts/slurm/evaluate_sdum_002_003_paired.sbatch` | SDUM-002/003 paired ranking 评价 | 保留；SDUM evaluation 组 |
| `scripts/slurm/run_h5_dmap_raw_vista_multiseed.sbatch` | CINE raw VISTA multi-seed run | 保留；CINE benchmark Slurm 组 |

其他本地独有内容包括：

- `README.md` 的 acceleration scope 修改；
- `SERVER_WORKFLOW.md`；
- 两份既有 history 文档；
- 本次迁移路线、Git 状态快照和代码差异清单；
- 本地提交 `eeb6273` 及其 FT00 Slurm 文件。

## 6. 非项目代码与待下一阶段资产

这些 untracked 内容不属于本次代码合并，但必须在数据清单阶段登记：

| 路径 | 当前大小/性质 | 本阶段结论 |
| --- | --- | --- |
| `2512.17137v2.pdf` | 5.6 MB 论文 PDF | 文献资产；不与代码一起提交 |
| `downloads/` | 12 MB，18 个结果 CSV/PNG | 下载的评价结果；进入数据/结果 registry |
| `tmp/` | 15 MB，43 个 PDF 渲染与临时文件 | 临时资产；先登记，后续列为 cleanup candidate，不在本阶段删除 |
| `docs/.~lock.*.xlsx#` | Office lock 文件 | cleanup candidate；确认 workbook 未被打开后再删除 |
| 服务器 `external/CMRxRecon2025` | 3.9 MB 嵌套 Git clone | 外部源码引用，不递归提交 |

嵌套外部仓库元数据：

```text
URL:    https://github.com/CmrxRecon/CMRxRecon2025.git
branch: main
HEAD:   12a37ac76d8e97a21eb414dc1f5ade86aec16b63
dirty:  clean
```

当前实际用到的非 `.git` 文件只有：

```text
CMRxReconMaskGeneration/README.md
CMRxReconMaskGeneration/TaskAll/MaskGeneration_TrainingSet.m
```

## 7. 阶段 1 的精确收拢边界

后续收拢服务器代码时，只允许新增下面两个服务器独有文件：

```text
scripts/prepare_sdum_007_static_random_cartesian.py
scripts/prepare_cmrxrecon2024_split_zip_subset.py
```

不得从服务器覆盖：

```text
scripts/inference.py
scripts/mri_data/data_utils.py
scripts/evaluate_cmrxrecon2023_cohort.py
scripts/evaluate_cmrxrecon2023_fullsample.py
scripts/validate_cine_inference_data.py
```

不得从服务器复制：

```text
external/CMRxRecon2025/.git/
dataset/
output/
outputs/
Results/
runs/
logs/
archive/
Hugging Face cache
```

## 8. 阶段状态

- 阶段 0，第 1 项——确认相关服务器任务结束：**完成**。
- 阶段 0，第 2 项——记录 branch、HEAD、dirty 状态：**完成**。
- 阶段 0，第 3 项——生成 tracked/untracked/独有提交代码差异清单：**完成**。
- 阶段 0，第 4 项——生成数据和实验资产清单：**尚未开始**。

本清单的“保留/后续提交”是代码层面的初步判断，不代表已经复制、stage 或 commit。任何删除动作仍需等第 4、5 项完成并人工确认。
