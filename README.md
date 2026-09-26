# RAL-MoE / CHANNEL 论文项目资料归档

归档日期：2026-09-26。这里统一保存 CHANNEL 工作区的论文源稿、代码、实验结果、结论、图表及集群中的必要复现资产。原始历史目录名保持不变；集群项目树位于 `cluster/`。

**代码、文稿与结果在 Git 中；大型数据、模型和交付包在 [同仓库 Release](https://github.com/panxy1019/RAL-MOE/releases/tag/channel-archive-20260926) 中。** 文件级路径、大小、SHA-256、附件位置及未纳入原因见 [`archive/`](archive/)。仅克隆仓库不会下载全部模型和数据。

## 阅读入口

| 内容 | 入口 |
|---|---|
| 统一结论与版本差异 | [RESULTS_INDEX.md](RESULTS_INDEX.md) |
| 论文主项目与历次审计 | [RAL_MOE_ROM_ICLR2027](RAL_MOE_ROM_ICLR2027/) |
| 第三批修订及对照 PDF | [round3_20260921](round3_20260921/) |
| 整合实验章节与 Overleaf 源码 | [merged_ICLR6_experiments_20260922](merged_ICLR6_experiments_20260922/)、[Overleaf](merged_ICLR6_experiments_20260922_Overleaf/) |
| 最新 Global MoE 同口径复评 | [global_moe_reeval_20260926](global_moe_reeval_20260926/) |
| POD–LSTM/GRU 结果及 FNO 历史报告 | [external_rnn_fno_20260923](external_rnn_fno_20260923/) |
| **FNO 最新完成记录** | [cluster/experiments/external_rnn_fno_20260923/fno_P/COMPLETED.json](cluster/experiments/external_rnn_fno_20260923/fno_P/COMPLETED.json) |
| ql-ROM-style 对照 | [qlrom_20260923](qlrom_20260923/) |
| E2 / T2-C 及固定权重对照 | [centeredsquare_fusion_v1](centeredsquare_fusion_v1/)、[experimental_strengthening_20260915](experimental_strengthening_20260915/) |
| 数据构建与迁移 | [DATA_CONSTRUCTION.md](DATA_CONSTRUCTION.md)、[POD 数据迁移指南](POD_DATASET_MIGRATION_TRAINING_EVALUATION_GUIDE.md) |
| 数据恢复与复核方法 | [REPRODUCING.md](REPRODUCING.md) |

各目录 README 描述的是对应批次。旧 README 中的“当前”“最终”“未完成”不能自动代表整个项目的最新状态。此次归档没有将分散的后续补充结果重新合并成一份新论文，也没有重训模型。

## 归档边界

本地资料保留历史版本。集群保留项目代码、配置、结果、日志、POD/ROM 资产、划分清单、正式或冻结检查点；训练中间检查点、候选筛选模型、巨型可重建缓存、运行环境和第三方依赖不作为完整集群备份上传。相同大型文件按 SHA-256 去重，恢复脚本会重建对应路径。

原有科学结论和历史报告保留。必要的凭据清理仅作用于上传副本，见清单中的 `redactions`。没有新增项目级开源许可证；原有引用、数据来源与第三方许可仍需遵守。
