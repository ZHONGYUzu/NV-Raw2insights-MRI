# 迁移阶段 0：资产分类决策

> 决策日期：2026-08-24
> 输入：代码差异清单、数据与实验资产清单、`experiments/registry.csv`
> 操作类型：只更新本地清单，不移动或删除服务器资产
> 本文完成 `docs/CODE_AND_EXPERIMENT_MIGRATION_PLAN.md` 阶段 0 的第 5 项。

## 1. 分类定义

| 分类 | 定义 | 是否允许删除 |
| --- | --- | --- |
| `active` | 当前 benchmark、统一入口迁移或仍需使用的 source/checkpoint/输入/输出 | 否 |
| `completed` | 已完成且主要输出或评价证据齐全，可进入只读归档流程 | 否；先生成 checksum manifest |
| `incomplete` | 正式实验缺少评价 summary、必要结果或闭环记录 | 否；先补齐或明确放弃理由 |
| `legacy` | 已被新流程替代的历史/debug/provenance 资产，仍用于解释旧结果 | 否；legacy 不等于垃圾文件 |
| `duplicate-candidate` | 已有强重复证据，但引用和恢复路径尚未完成最终验证 | 否；完成 checksum、引用迁移和人工确认后才可能清理 |
| `unknown` | 证据不足，无法判断用途或归属 | 否 |

分类只描述生命周期，不授予删除权限。本阶段没有删除、移动或重命名任何文件。

## 2. 分类结果

更新后的 registry 包含 71 个资产条目：

| 分类 | 数量 | 主要内容 |
| --- | ---: | --- |
| `active` | 17 | 当前 CINE benchmark、外部只读 source、两处仍需使用的 cache、Base/Small/Large checkpoint |
| `completed` | 30 | H5/dMap CINE、CMRx LAX10、fastMRI val10、已评价 SDUM run 和完成的 visualization |
| `incomplete` | 5 | SDUM-001、002、006、007 及 SDUM-001 CMRxRecon2024 派生数据 |
| `legacy` | 18 | dataset_v0/v1、archive、failure provenance、debug 输出/结果和单例兼容实验 |
| `duplicate-candidate` | 1 | 默认 cache 中重复的 Base checkpoint blob |
| `unknown` | 0 | 无 |

层级条目不能直接相加计算磁盘总量。例如两个 cache 父目录是 `active`，其中的 checkpoint blob 又各自作为子条目登记。

## 3. Active

### 3.1 当前 CINE benchmark

以下资产继续作为迁移和回归测试基线：

- `dataset/CINE_test_acc8`、`acc16`、`acc24`；
- `dataset/GT_from_dImgC`；
- 对应三个 `output/CINE_test_acc*`；
- `Results/CINE_vista_mask_generalization_v1`。

它们在统一入口建立前不得改路径。未来移动时需要先修改 manifest/config，并对 10-case 输入、GT 和 reconstruction 建 SHA-256 manifest。

### 3.2 外部只读 source

CINE H5、VISTA masks、CMRxRecon2023 training 和 fastMRI brain validation 被标为 `active`。这里的 active 只表示当前流程仍引用；原始数据继续只读，永远不因仓库整理而移动或清理。

### 3.3 Checkpoint/cache

- 显式 `hf_cache` 父目录：`active`，包含 Base 和 Small；
- 默认 Hugging Face cache 父目录：`active`，因为仍包含唯一 Large；
- 显式 cache 的 Base、Small blob：`active`；
- 默认 cache 的 Large blob：`active`。

因此禁止删除整个默认 cache。

## 4. Completed

下列实验已经形成可归档的完整结果链：

- CINE H5 `dMap`/raw VISTA acc8、16、24 的输入与输出；
- CMRxRecon2023 LAX10 official acc8 输入、输出与 Results；
- fastMRI brain val10 acc8、16、24 输入、输出与 Results；
- SDUM-003、004、005、008、009、010、011、012、013；
- CINE GT figures/GIF、raw VISTA review、forced-ACS review；
- public example output。

`completed` 的下一步是冻结 config/manifest/summary 和 checksum，不是立刻移动。只有 registry 中记录 archive target 后才允许归档。

## 5. Incomplete

| 资产 | 缺口 | 建议动作 |
| --- | --- | --- |
| `sdum-001` | 15 个 reconstruction 和 preview 存在，但没有正式 metrics summary | 明确 mixed-acquisition GT/evaluation protocol，再评价或记录无法评价原因 |
| `sdum-002` | official acc8 输出和 GT 存在，但没有 summary | 运行与后续 SDUM 一致的 full-direct/paired baseline 评价 |
| `sdum-006` | full/ranking 输出存在，没有 metrics summary | 使用现有 full-direct evaluator 补 summary |
| `sdum-007` | 10-case CINE reconstruction、mask 和 manifest 存在，没有 metrics summary | 使用 active CINE `dImgC` GT 补 full-size evaluation |
| CMRxRecon2024 `datasets/.../sdum-001` | 数据本身完整，但所属正式 run 尚未完成评价闭环 | 与 `sdum-001` 一起保留，不单独归档或删除 |

这些资产是下一阶段最优先补齐的对象。

## 6. Legacy

Legacy 资产包括：

- `dataset_v0`：其中 `norm_img` 的 134 个 NPY 是重要 provenance，必须保留；
- `dataset_v1`；
- `archive/datasets` 和 `archive/outputs`；
- `failure_runs` 的早期转换脚本与日志；
- `dataset/archiv`、`output/archiv`；
- 4 组 fastMRI debug output 和对应 4 组 debug Results；
- 旧 `result/` force-fill qualitative PNG；
- CMRxRecon2023 单例 Results；
- `sdum-008-debug`。

Legacy 的默认处置是“只读保留并补 provenance”。后续若需节省空间，必须先把具体子目录从父级 registry 拆出，验证它是否被正式结果取代，并生成 checksum/待删清单。

## 7. Duplicate Candidate

唯一的 `duplicate-candidate` 是：

```text
/home/students/studxuzho1/.cache/huggingface/hub/
  models--nvidia--NV-Raw2insights-MRI/blobs/
  667189bac5f159e52b2f1640b504b4efd549717c37720c2397e749d16ec056b8
```

证据：

- 与显式 `hf_cache` 中 Base blob 的完整 blob ID 相同；
- 两者大小均为 `3,038,170,078` bytes；
- 两个 cache 指向同一 snapshot `38860e9a6153d857b55439b3d3fd55f850eccdcd`。

它仍然不能删除。清理前必须：

1. 对两个 blob 重新计算 SHA-256；
2. 把 Large checkpoint 安全纳入 canonical cache，或确保默认 cache 仍完整；
3. 将所有 Slurm/config 统一到 canonical `HF_HOME`；
4. 做一次 Base 和 Large checkpoint resolution smoke test；
5. 生成精确待删路径并人工确认。

潜在回收空间约 3.04 GB。

## 8. Unknown 审计

本次没有保留 `unknown` 项：

- 小型不明目录已通过文件名和内容结构确认用途；
- RAM/SNRAware 路径确认为相邻项目，不纳入本 registry；
- Python 环境属于可重建运行环境，不作为实验资产分类。

这不意味着可以处理所有目录。任何新发现、未进入 registry 的路径自动视为 `unknown`，必须先登记。

## 9. 阶段 0 完成门槛

阶段 0 的五项均已完成：

1. 相关服务器任务已结束；
2. 本地、GitHub 和服务器 Git 状态已记录；
3. 代码差异和独有提交已记录；
4. 数据与实验资产已登记；
5. 资产已完成生命周期分类，`unknown=0`。

下一阶段可以开始“收拢服务器独有代码”，但仍禁止移动或删除大型资产。
