# 2026-09-07 代码整理步骤 1–3 交接

起点：codex/server-code-consolidation @ 7b30d82。
范围：预审的 11 份脚本（4 份移除，7 份保留）、配套 16 份 Slurm、旧工作簿。
阶段 4 留在新对话；本轮不合并、push、部署或运行服务器实验。

## 清理决定

用户自行维护 Excel。已移除 docs/BV.xlsx、docs/c1.xlsx、docs/benchmark_v3.xlsx、
RAM_stability_experiments.xlsx、SDUM_stability_experiments.xlsx。
前三份为 tracked，其余两份原来未跟踪。
移除的四份生成器（均原来未跟踪）：create_benchmark_v3.py、
create_cine_benchmark_matrix.py、create_sdum_stability_experiments_xlsx.py、
merge_cine_benchmark_workbooks.py（原位于 scripts/）。
Git 删除提交因此只显示三份工作簿。

可恢复副本位于：

- /Users/zhongyu/.Trash/NV-MRI-create_benchmark_v3-20260907.py
- /Users/zhongyu/.Trash/NV-MRI-old-workbooks-20260907/（其余八个对象）

未定位 v5 路径，未处理其他位置的工作簿。旧 history 快照的文件名保留为证据，
本记录取代其中继续保留生成器的建议。活动 README 文档未发现被删表格链接。

## 七份脚本销账（scripts/）

| 文件 | 用途与本轮处理 |
| --- | --- |
| create_cmrxrecon2023_training_masks.py | 完整 cohort 先分配 seed 再过滤调试病例；--seed-manifest 可固定旧映射；拒绝重复 ID、残留多余 descriptors |
| prepare_sdum_019_rotation_pilot.py | 参数/输入身份预检、逐病例完成记录、resume 检查；保持归一化系数等元数据；明确仅 GT 归一化 |
| evaluate_sdum_019_rotation_pilot.py | 非恒定 GT 零误差 PSNR=+inf，汇总保留；prediction-key 生效；不挤掉单 slice/time 维度 |
| evaluate_sdum_paired_ranking.py | 标注 LEGACY；保留单 slice 维度；图像索引限制在实际范围 |
| export_sdum_ranking_crops.py | 标注 LEGACY；禁止同目录、符号链接、硬链接覆盖原图；保留单 slice 维度 |
| plot_sdum_previews.py | auto/full/ranking 参数；按实际形状判别裁剪，不凭目录名；仅用于既有 sdum-001–005 预览 |
| find_common_vista_mask_seeds.py | 原样保留；固定 25 帧、只核对文件名，不替代内容验证 |

seed 兼容性：默认保持完整源 cohort 顺序对应的旧 RNG 算法。重排源 cohort 时
应提供旧 ktGaussian 输出 manifest；不带该参数时，顺序仍是实验协议的一部分。
本轮没有改写服务器旧 masks。

resume 兼容性：小 descriptor/mask 使用 SHA-256，大型 H5/派生输出使用
路径、大小、mtime_ns；不声称已做大型文件内容校验。缺少 preparation_config.json
的旧运行或中断于单病例写入的半成品会被拒绝；保留旧目录，使用新目录。
成功 resume 保留病例记录及 manifest 创建时间。参数与输出预检在写 cases.txt 前完成。

## Slurm 登记（scripts/slurm/）

| 组 | 文件 |
| --- | --- |
| CMRx masks（11份） | prepare_sdum_011_012_013_acc16.sbatch；run_sdum_008_training_uniform8_debug.sbatch；run_sdum_008_training_uniform8_full.sbatch；run_sdum_009_training_ktgaussian8_debug.sbatch；run_sdum_009_training_ktgaussian8_full.sbatch；run_sdum_010_training_ktradial8_debug.sbatch；run_sdum_010_training_ktradial8_full.sbatch；run_sdum_011_012_013_acc16_debug.sbatch；run_sdum_011_012_013_acc16_full.sbatch；evaluate_sdum_008_full_direct.sbatch；evaluate_sdum_009_full_direct.sbatch |
| 历史 ranking（2份） | evaluate_sdum_002_003_paired.sbatch；run_sdum_official_acc8_conditioning.sbatch |
| CINE VISTA（2份） | evaluate_h5_dmap_raw_vista_seed8_all10.sbatch；run_h5_dmap_raw_vista_multiseed.sbatch |
| 旧旋转（1份） | run_sdum_019_rotation_pilot.sbatch |

只修改三个 Slurm：ranking 用途标注；旧旋转协议/resume 限制说明；official
conditioning 取消自动 --overwrite，遇到已有 cohort/input/staged/final output 时停止。
其他配方参数和服务器路径保留。SBATCH 日志父目录需在提交前存在；脚本内部 mkdir
不能保证 Slurm 打开日志时目录已存在。full 配方会移动 staging 输出并产生 ranking
副本，本轮没有执行。文件数量检查不等于病例、数值和配置验证通过。
raw VISTA evaluator 使用历史 p99.5 归一化，不应直接与 direct-scale 指标混排。

## 验证与提交

全 scripts/tests compileall 通过，缓存写入临时目录；所有 .sbatch 通过 bash -n；
七份 CLI --help 通过；tests/test_code_consolidation.py 的 10 个合成数据回归测试通过。
覆盖 resume 拒绝/保持元数据、原始输入未变、FFT 恒等、旋转方向、单 slice、自定义 key、
PSNR、seed、裁剪保护及预览布局。未使用真实病例、模型或 GPU。

测试环境：/private/tmp/nv-consolidation-eVXton/venv，Python 3.12、NumPy 2.3.5、
SciPy 1.18.1、h5py 3.16.0、scikit-image 0.26.0、Matplotlib 3.11.1。
这些验证不代表完整服务器复现实验通过。

```bash
python -m unittest discover -s tests -p test_code_consolidation.py -v
git log --oneline 7b30d82..HEAD
git diff --stat 7b30d82..HEAD
git status --short
```

六个提交组：旧工作簿、CMRx mask/Slurm、历史 ranking/preview、CINE VISTA、旧旋转、
验证/交接文档。均按显式文件名暂存。

## 人工审核与下一对话

- 0–5°固定姿态与采集中时变运动需分开定义；本轮不开发新协议。
- sdum-019 默认实际为 0°/−10°/+10°/180°，未改成新实验；用户提及的90°实验需另查记录。
- 旧旋转仅 GT 除以 dImgC 最大值，预测尺度是否匹配仍需核对实际推理记录。
- 各角度 ROI、插值不同，指标变化不等于纯模型鲁棒性变化；未改变现有科学评价策略。
- 没有重算旧结果，旧 CSV 中的 PSNR NaN 不会自动更新。

仍留在本地、未纳入本轮：根 README.md、SERVER_WORKFLOW.md、2026-08-05 和
2026-08-11 两份 history、论文 PDF、downloads/、papers/、stability/、tmp/、两个锁文件。
本次七脚本收尾不等于整个仓库 clean，也不证明服务器在8月24日快照后没有新代码。
阶段4开始前需刷新现场状态。
