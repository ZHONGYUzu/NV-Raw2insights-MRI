# NV-Raw2Insights-MRI 服务器工作流

> 历史范围：本文保留 2026-08-12 的 external-dMap/raw-VISTA 工作流及 Job 42462
> 记录，不代表今日任务状态，也不取代 docs/NV-Raw2insights-MRI_SOP.md。
> 下文“当前”“本次”均指该历史实验。重跑评价应选新结果目录。
> 本地唯一仓库实体为 /Users/zhongyu/Desktop/NV-Raw2insights-MRI；
> 下文 Documents 路径是其符号链接。迁移进度见 docs/history/2026-09-07-code-consolidation-closeout.md。

本机负责改代码，服务器负责数据处理、推理和评价。

## 当前环境

```text
SSH:          ssh login
服务器仓库:   /home/students/studxuzho1/NV-Raw2insights-MRI
Python:       /home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python
原始 CINE H5: /mnt/qdata/rawdata/CINE/2D_h5_compressed
本机仓库:     /Users/zhongyu/Documents/NV-Raw2insights-MRI
测试病例:     docs/test_subjects.txt（固定 10 例）
```

基本规则：

- 本机修改、测试并提交代码。
- login 节点只做 Git 同步、轻量检查和 Slurm 调度。
- Python、数据处理、推理、评价和绘图通过 Slurm 在计算节点运行。
- 原始 H5 和 VISTA mask 只读。
- `dataset/`、`output/`、`Results/`、`logs/` 保留在服务器，不提交 GitHub。

## 登录服务器

```bash
ssh login
cd /home/students/studxuzho1/NV-Raw2insights-MRI

git status
git rev-parse HEAD
```

## 当前 CINE seed8 运行

这三组使用 external H5 `dMap`、raw VISTA mask、无强制 ACS：

```text
acc8:  dataset/h5_converted        -> output/h5_converted
acc16: dataset/h5_converted_acc16  -> output/h5_converted_acc16
acc24: dataset/h5_converted_acc24  -> output/h5_converted_acc24
```

含义：

```text
dataset/  推理输入（k-space、dMap、mask、JSON）
output/   模型推理生成的 reconstruction
Results/  对一个或多个 output run 的评价和图片
```

一个 `output` run 可以使用不同 GT、归一化或画图方式重复 review，因此
`output/` 和 `Results/` 不要求一一对应。

## 当前全十例评价

评价脚本：

```text
scripts/slurm/evaluate_h5_dmap_raw_vista_seed8_all10.sbatch
```

提交命令：

```bash
cd /home/students/studxuzho1/NV-Raw2insights-MRI
mkdir -p logs

sbatch \
  --export=ALL,RUN_ROOT=Results/CINE_vista_mask_generalization_v1/seed8_acc8-16-24/dimgc_p995_full10_v1 \
  scripts/slurm/evaluate_h5_dmap_raw_vista_seed8_all10.sbatch
```

结果结构：

```text
Results/CINE_vista_mask_generalization_v1/
  seed8_acc8-16-24/
    config.json
    dimgc_p995_full10_v1/
      frame_metrics.csv
      summary_metrics.csv
      metrics_boxplot.png
      metrics_violin.png
      plots/
```

其中 `config.json` 记录三个来源 run、mask seed、GT、归一化和评价范围；
具体 review 文件放在 `dimgc_p995_full10_v1/` 中。

## 查看任务和结果

本次已完成评价的 Job ID 是 `42462`：

```bash
sacct -j 42462 \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList

tail -n 100 logs/eval_vista_s8_all10_42462.out
cat logs/eval_vista_s8_all10_42462.err
```

成功标准：

```text
State:    COMPLETED
ExitCode: 0:0
stderr:   空
```

查看配置和汇总指标：

```bash
cat Results/CINE_vista_mask_generalization_v1/seed8_acc8-16-24/config.json

cat Results/CINE_vista_mask_generalization_v1/seed8_acc8-16-24/\
dimgc_p995_full10_v1/summary_metrics.csv
```

本次输出应包含：

```text
10 个病例
2,975 帧/acceleration
8,925 行 frame metrics
3 行 summary metrics
162 张 PNG
```
