# 结果、结论与版本索引

本索引依据现有报告和集群完成记录整理，不新增实验，不把不同参考场、窗口或训练预算的数字混成同一比较。

## 2026-09-26 Global MoE 同口径单检查点复评

[完整报告](global_moe_reeval_20260926/Global_MoE_同口径复评报告_20260926.md)及逐步 CSV、冻结窗口、聚合程序和独立验证均已保存。

| 流态 | K24 联合误差 % | K48 联合误差 % |
|---|---:|---:|
| Hopf | 0.1314 | 0.1299 |
| Periodic | 2.7135 | 3.8990 |

74 个窗口、3552 条逐步记录；只有一个检查点，不报告训练种子误差棒。该结果不支持 global 或 local 架构普遍更优的结论。模型身份、输入差异及公平性边界以原报告为准。

## 2026-09-23 POD–RNN、FNO 与 ql-ROM-style

- [POD–RNN 原报告](external_rnn_fno_20260923/POD_RNN_FNO_实验报告与当前状态.md)：12 个正式模型，444 个模型/种子/窗口均完成 K48。Periodic RNN 优于 original Proposed 的部分误差，但 active-matched Dense 仍更低，不能宣称 Proposed 全面领先。
- **版本纠正**：上述本地报告的 FNO“未完成”是历史阶段记录。[集群完成记录](cluster/experiments/external_rnn_fno_20260923/fno_P/COMPLETED.json)已于 2026-09-23 23:16（+0800）记录 Periodic FNO 三种子完成，K24/K48 均为 96/96 个种子-窗口。
- FNO 的 FV 原始 CFD 参考联合误差为 K24 **9.1854 ± 2.7778%**、K48 **14.0638 ± 5.4974%**。其中 `native_*` 是网格口径；不能与 FV 指标或共同 POD 重构参考混用。源代码见 [fno_experiment.py](cluster/experiments/external_rnn_fno_20260923/code/fno_experiment.py)，原始统计见 [summary.csv](cluster/experiments/external_rnn_fno_20260923/fno_P/summary.csv)。本次归档不把完成标志扩张为已重跑训练或全面验证所有计时流程。
- Hopf FNO 在冻结插值门槛下停止，不能将其填作完成的预测对比行。
- [ql-ROM-style 报告](qlrom_20260923/qlrom_comparison_report.md)：固定 32 维父 POD 空间内的受限独立实现，非官方或完整物理空间 ql-ROM。当前配置延迟较低而误差较大；POD 参考和原始 CFD 参考的表格必须分开。继承算子的限制不能被描述成完整 ql-ROM 方法的失败。

## E2、T2-C 与论文修订

[固定权重对照报告](experimental_strengthening_20260915/FINAL_REPORT_CN.md)中，Centered-square 两个重叠区的 T2-C 在指定 K24 末步指标下优于训练拟合/验证选择的固定权重。结论只覆盖对应冻结协议，不能扩张为所有时间步或分布上的保证。Oracle 使用未来参考，是不可部署诊断。

路由与融合实现见 [centeredsquare_fusion_v1](centeredsquare_fusion_v1/)，进一步的指标及计算成本审计分别保留在 `round3_20260921/`、`internal_routing_revised_20260923/`、`computational_cost_*_20260923/` 等目录。

论文历史基线为 `RAL_MOE_ROM_ICLR2027/`；9 月下旬还有多批方法、实验、图表及 Overleaf 修订。目录名和原报告保留其时间关系，本次不指定未经作者确认的一份 PDF 为唯一投稿终稿。
