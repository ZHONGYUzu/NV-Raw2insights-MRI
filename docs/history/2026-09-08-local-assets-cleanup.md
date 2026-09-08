# 2026-09-08 剩余本地内容整理

本轮承接用户对剩余 dirty 内容的整理授权。没有删除或移动任何论文、下载、
实验图像或开发文件；通过文档提交和针对性忽略规则区分资产。

## 纳入版本控制

- README.md 的加速范围说明：对照官方 CMRx2024 task definitions、
  SDUM v2 论文和仓库 Small/Base/Large JSON 配置核对；明确 fastMRI 结果来自单独训练模型。
- SERVER_WORKFLOW.md：保留 2026-08-12 raw-VISTA/Job 42462 记录；加历史范围说明，
  避免把旧任务、结果路径误当作当前 SOP。
- docs/history/2026-08-05-server-run-audit.md：保留原日期、测量值及局限，不冒充新审计。
- docs/history/2026-08-11-vista-mask-generalization-run-plan.md：标记当时 pending，
  与后续 seed8 评价记录区分。未运行文档中的任何命令。

## 保留在本地、Git 忽略

| 路径 | 内容及策略 |
| --- | --- |
| 2512.17137v2.pdf、papers/*.pdf | 论文阅读副本，不上传 Git |
| downloads/ 下 PNG、CSV | 下载的 CINE/CMRx 比较图、指标副本，约 11 MB；非代码 |
| stability/ 下 PNG | 各实验预览，约 14 MB；不覆盖、不改实验内容 |
| tmp/pdfs/ 下 PDF、PNG、TXT、XML 与 rendered/PNG | 阅读时提取文本和渲染，保留原位 |
| tmp/pdfs/build_annotated_sdum.py | 个人 PDF 批注辅助脚本，明确忽略但保留本地；不作为模型代码 |
| .~lock.*# | LibreOffice 临时锁文件，保留原位并忽略 |

papers/ 约 10 MB，tmp/ 总计约 15 MB；统计为本机磁盘占用近似值。
忽略规则按用途/扩展名限定，未忽略整个 scripts/、stability/ 或 tmp/。
忽略不等于备份；这些本地文件不会因 push 被上传。

## 交由新对话继续的活跃开发

本轮发现的新文件 scripts/fullsample_noise_pilot.py，以及 tmp/ 下
sdum_smallangle.py、sdum_smallangle_preflight.py、sdum_preflight.sbatch、
sdum_detect_resources.py、sdum_probe_conventions.py、test_sdum_smallangle.py、
SDUM_022_023_smallangle.md 均保留，未修改、未提交、未新增忽略规则。
它们属于新实验工作，不能为了清空 git status 而隐藏。

本轮只收尾上述旧文档/本地资产。后续新对话应单独审核、提交活跃开发文件；
仓库仍可能显示这些文件为 untracked。未执行服务器同步、push 或分支合并。

README 核对来源：

- https://github.com/CmrxRecon/CMRxRecon2024#challenge-tasks
- https://arxiv.org/html/2512.17137v2 （fastMRI 单独训练及 Table 5）
- configs/nv_raw2insights_mri_small.json、configs/nv_raw2insights_mri_base.json、
  configs/nv_raw2insights_mri_large.json 的 accelerations 字段。
