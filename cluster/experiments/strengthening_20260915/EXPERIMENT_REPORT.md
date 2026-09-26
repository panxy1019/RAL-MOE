# 实验强化记录（执行中）

本轮依据 `ICLR_Experimental_Strengthening_and_Ablation_Plan.md`。此文件区分已验证结果与尚未完成的实验，不能视为全部任务完成证明。原模型、原缓存及论文未被覆盖。

## 1. Constant-alpha：已完成

### 协议

- 数据集：Centered-square 的 S–H、H–P overlap；复用原 strict-v3 K24 候选轨迹、开发集/测试集划分与原 T2-C checkpoint。
- 每个 overlap 仅有一个固定标量；无 Re 输入、无历史输入。S–H 的 alpha 乘 S，H–P 的 alpha 乘 P，不能误写为 H 的权重。
- 主对照：只在训练窗口上最小化 24 步平均归一化速度/压力平方误差，解析求解一维凸二次问题，并裁剪至 [0,1]。该确定性标量拟合不需要重复随机种子。
- 额外敏感性对照：只用验证集选择固定标量，网格 0:0.0001:1，沿用 worst-Re 优先、pooled mean 次级的选择原则。此对照与 train-fitted 主对照分开报告。
- 测试集不参与权重拟合或选择。报告的是测试窗口第 24 步相对物理场误差的均值，Ejoint=Eu+Ep，表中单位为百分比；不是 24 步累计平均误差。
- 原 T2-C 结果已重放核对；保留数据哈希、拟合系数、逐窗口和逐步结果。

### K24 结果

| 方法 | S–H Ejoint (%) | H–P Ejoint (%) |
|---|---:|---:|
| Equal blend | 3.0321 | 1.6146 |
| E2 probability blend | 2.1847 | 1.6618 |
| Constant-alpha，训练集拟合 | 2.5590 | 1.1549 |
| Constant-alpha，验证集敏感性对照 | 1.7340 | 1.1607 |
| 原 T2-C | 1.1958 | 0.8465 |
| Convex oracle，未来信息诊断 | 1.1842 | 0.8384 |

训练拟合 alpha：S–H = 0.35785055，H–P = 0.92868350。验证选择 alpha：S–H = 0.0349，H–P = 0.89。

### 解释与边界

T2-C 在两个已评估 overlap 的 K24 平均联合误差均低于两个固定权重对照，支持“现有收益不能由本实验中的一个固定混合比例完全解释”。这不是对所有固定权重拟合目标、所有分布或所有流态的普遍证明，也不能单独归因于历史输入；历史贡献应同时引用已有 mu-only 对照。固定权重的训练/验证选择差异也说明最优常数受开发窗口分布与优化目标影响。

逐步曲线并非单调增长，尤其压力误差存在振荡。不要据此宣称每一步都更优或长期误差单调受控。时间横轴为 rollout step，不能未经时间尺度换算标成统一物理时间。

Oracle 使用未来真值为每个窗口选择 alpha，是 future-information diagnostic，不可部署。原 oracle 按全 24 步平均目标选权重，并非逐步独立最优，因此不应称为每个单独时间步或 K24 指标的严格下界。

### 输出

- `constant_alpha_v1/results.json`：完整分 Re、分指标统计。
- `constant_alpha_v1/protocol.json`：数据、划分和复现协议。
- `constant_alpha_v1/fusion_error_curves.csv`：K1–K24 的 Eu/Ep/Ejoint。
- `constant_alpha_v1/router_fusion_windows.csv`：训练、验证、held-out 窗口权重。
- `constant_alpha_v1/figures/router_fusion_weights.pdf`、同名 PNG：实测 Re 点的 E2 与 T2-C 权重，不插值制造稠密扫描。
- `constant_alpha_v1/figures/fusion_error_growth.pdf`、同名 PNG：融合方法逐步误差。

## 2. Dense correction：正式训练中，未得最终结论

先执行 Circular Periodic 的原生训练协议；不能把这个单流态结果推广成三个算例、全部 S/H/P 的完整 Dense 消融。

单个三隐藏层 GELU 全连接网络替换所有 shared/routed experts 和 group/channel routers，联合输出速度、压力修正；保留原编码器、特征、Galerkin 骨架、压力基线与闭合置信度头。后者不是 sparse routing。保留原优化器、数据划分、rollout curriculum 和验证选择，移除仅适用于路由/专家多样性的惩罚。

| 参数统计 | 数值 |
|---|---:|
| 原 specialist 总可训练参数 | 48,617,547 |
| Dense 总可训练参数 | 48,609,532 |
| 相对差异 | -0.01649% |
| Dense 隐藏层宽度 | 4,820 |

这是 **总分配参数匹配**，不是 active-parameter 或 FLOP 匹配。相比稀疏网络，Dense 每步计算的参数更多；运行效率必须实测，不能预设。

一轮 smoke 已完成，检查覆盖真实训练、验证、保存及原生最终评估路径。首次 smoke 在“专家两两相似度统计”遇到空数组错误；单网络不适用该项，现显式记为 not applicable，没有绕过训练数值安全检查。

正式任务：远程 `experiments/strengthening_20260915/dense_periodic_v1`，seed=1600；训练完成后仅使用 `best_validation.pt`，不按 held-out 结果挑选 checkpoint。首轮训练损失很大但有限，后续下降；这不是通过了性能验证的证据，仍须审阅最终 rollout 结果。

最近检查：已记录 epoch 130，curriculum 已推进至 12 步；正式训练仍运行中。该数字是本次记录时的快照，不是最终轮数。

## 3. POD-Galerkin / Global MoE / RAL 逐步曲线

已发现旧 Global 与 local 缓存的起始时间不同，不能直接合并成公平比较。本轮以旧 Global 的四个 held-out Re、每个三个窗口为锚点，共 12 个 K48 窗口，重新定位相同物理起始时刻。Circular Periodic POD-Galerkin 与原 RAL Periodic specialist 对齐评估已完成，Global 原生接口的独立重跑也已完成。

不同 POD 坐标不能直接比较模态向量差。Global 预测通过跨基 Gram 矩阵映射到共同物理误差计算，包含均值场与 POD 基差异；网格点和积分权重已断言一致。共同参考为 local Periodic POD 重构真值，不是完整 FOM 快照。因此新增图必须标明 projected-truth evaluation，不能冒充完整物理真值精度。

Global 原协议使用 per-Re mean fields，此依赖保留并披露，不能把结果解释成未知工况下无需任何均值信息的部署测试。

压力规范审计发现：local 压力均值和基向量的面积加权均值接近零，而 Global 基并非如此。直接相减会把任意压力常数偏置误判成预测误差。本轮对 Global/local 重构压力都去除面积加权均值后再计算 Ep；local 原基已满足此规范，数值影响可忽略。不能使用去均值前的 Global 压力误差。

| 同窗口 K48；共同 projected truth | Eu (%) | Ep (%) | Ejoint (%) |
|---|---:|---:|---:|
| POD-Galerkin | 9.1217 | 36.2473 | 45.3691 |
| Global MoE | 1.6072 | 5.8646 | 7.4719 |
| RAL Periodic local specialist | 1.9352 | 7.4275 | 9.3627 |

**这一结果不支持“local 在所有长时预测中优于 Global”。** 两种学习修正均显著优于该 POD-Galerkin 对照，但 Global 在本组 K48 窗口的速度、压力和联合误差都更低。应分析局部模型优势的流态/时间尺度边界，而不是删除不利窗口或改变选择规则。该组不是 overlap fusion 测试，不能用其否定或替代上文 T2-C overlap 实验。

图表：`aligned_rollout_figures/three_method_error_growth.pdf`、同名 PNG，以及 `aggregate_curves.csv`。每个方法均保留相同的 12 个窗口；横轴为 step，各 Re 的物理时间步不同。纵轴为对数尺度。

Global 独立重放校验通过：重新运行的原生 Eu/Ep 与旧缓存按原生口径计算的 Eu/Ep，在全部 12×48 个位置一致，最大差异约 8.88e-16 个百分点（`GLOBAL_REPLAY_VERIFIED.json`）。此校验使用原压力规范仅验证缓存身份，正式跨图表比较仍使用统一去均值后的压力。

NaN/Inf 窗口不能删除后仅报告幸存者均值；本轮保留所有窗口并报告 finite count，发散后的聚合值记为 NaN。

## 4. Runtime：排队，尚无速度结论

已安排训练结束且 GPU 无其他计算进程后，顺序评估 Dense、Full local、Global、POD-Galerkin；每项 3 次 warm-up、10 次重复，单窗口 K48，GPU 同步后计时，记录 peak allocated/reserved memory。

计时包括原生模态 rollout、特征构建、CPU/GPU 传输和模态压力重构；不包括数据/模型加载、物理场解码及误差统计。POD-Galerkin 原生 NumPy 实现在 CPU 上运行，神经修正在 GPU 上运行：这是同一机器的原生实现成本，不是同为 GPU kernel 的严格效率比较。没有新增两专家融合端到端计时前，不应报告整个 RAL Top-2 pipeline 的 speedup。

远程执行状态：`post_training_status.json`。只有所有步骤成功才写入 COMPLETED；训练异常或评估失败会记录 FAILED，不能把排队当作完成。

## 5. Architecture ablation 与论文表述建议

- DataOnly vs Full：用于分析保留物理骨架的价值，但若 DataOnly 同时改变了离散推进合同，应承认存在联合改变，避免声称严格单因素因果归因。
- Vanilla-FNN-MoE vs Full：在路由保持的前提下分析 structured expert 的贡献，不能解读为 MoE 本身的贡献。
- Global MoE vs local specialists：分析局部表示/分解的整体价值；同时披露基、均值、模型与训练分解差异，不等同于单独“局部坐标”一个因素。
- Dense vs Full：待训练和评估完成后讨论；总参数匹配不足以自动保证计算量匹配。

建议范围表述：**evaluated regime transitions within parameterized incompressible flows**。不声称 unseen physics、任意时间流态切换或任意非定常跨流态泛化。

建议可用英文（只覆盖已完成的固定权重实验）：

> On the evaluated centered-square overlap windows, T2-C achieves lower K24 mean joint error than both a train-fitted constant blend and a validation-selected constant-weight sensitivity baseline. These comparisons support the utility of adaptive physical-space fusion beyond the tested fixed mixing ratios. The convex oracle uses future reference trajectories and is reported only as a diagnostic, not as a deployable method.

## 尚未完成

1. Dense 最终验证选择、held-out 结果与解释。
2. 已有三方法共同窗口误差曲线；仍需最终 Dense 叠加与完整独立重放校验。
3. 无并发负载的运行时间与内存实测。
4. 决定是否扩展 Dense 至其余流态/算例；当前不得声称完成全矩阵。
5. 最终汇总报告/交付包；本文件为执行中记录。
