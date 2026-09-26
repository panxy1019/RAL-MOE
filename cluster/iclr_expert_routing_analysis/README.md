# Circular expert routing analysis - 2026-09-05

先阅读 [详细分析报告](reports/EXPERT_ROUTING_ANALYSIS_REPORT.md)。本包完成了指定 Steady S4、Hopf H4、Periodic epoch85 checkpoints 的 post-hoc 路由分析；不包含重新训练或论文源文件改动。

关键限制：S4 是独立诊断，不对应当前论文 Steady 参考模型；Hopf 专家对固定，速度 routed mass 高度集中。不要将本包概括为“三个论文 specialists 均显示非塌缩专门化”。

## 内容

- `reports/`：完整中文报告、资产清单、英文论文段落、完整逐专家附表。
- `figures/`：三张矢量 PDF 及 PNG 预览。
- `raw/`：压缩 JSONL 路由记录、NPZ 预测/真值系数；不含 checkpoint。
- `stats/`：原始日志可独立重算的 CSV、汇总 JSON 与检查结果。
- `scripts/`：复用原运行器的适配器、旁路 hooks、统计和绘图工具。
- `logs/`：严格加载、误差复现、预测不变性、运行配置及哈希证据。
- `source/`：查验时保存的原代码与历史报告，作为来源副本，不是修改后的模型。

91 个评估窗口覆盖 11 个 held-out Re，保存 29376 行双通道路由。全部窗口 hooks on/off 预测逐元素一致；Hopf 和 Periodic 的历史逐 Re 误差完全复现。

主图与正文数据统一取宏步 stage 1；`used_forward_calls` 另存全部实际速度 RHS forward 的使用统计，压力只包含 stage 1。参见详细报告中的复现命令和限制。

本地根目录：`C:\Users\panxy1019\Documents\CHANNEL\iclr_expert_routing_analysis`。
远端根目录：`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/iclr_expert_routing_analysis`。
