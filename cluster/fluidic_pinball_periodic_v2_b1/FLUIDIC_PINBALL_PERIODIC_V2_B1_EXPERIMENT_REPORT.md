# Fluidic Pinball V2 Periodic：B1 Deep-FNN-H3 迁移训练与 Rollout 实验报告

> **已废止的评估版本（2026-08-06 勘误）：** 本报告最初使用 `Δt=0.25` 的 validation/final-test 相邻帧评估由 `Δt≈1.0` 训练的 H3 模型，造成时间采样契约不一致。其高 Re 误差及“高 Re 科学失效”结论不可用于模型性能判断。请改用 [修正版报告](C:/Users/panxy1019/Documents/Pinball/reports/FLUIDIC_PINBALL_PERIODIC_V2_B1_EXPERIMENT_REPORT_CORRECTED.md)。本文件仅为保留审计记录。

**实验日期：** 2026-08-06  
**实验对象：** B1（Deep-FNN-H3，无 KDA 投影）  
**随机种子：** 1248  
**状态：** 训练、验证 rollout、一次性 final-test rollout 均已完成  
**结论等级：** 数值稳定的基线，但未达到全 Reynolds 数范围内的高精度预测要求

## 1. 执行摘要

本实验将既有 B1 Deep-FNN-H3 迁移方案应用到最新的 Fluidic Pinball V2 periodic 数据集，并在冻结检查点上完成 validation 与一次性 final-test rollout。

主要结论如下：

- 训练完成 8,000 个优化步，模型包含 1,792,959 个可训练参数；冻结检查点为 `best_validation.pt`（step 8000）。
- validation 与 final-test 的全部 rollout 窗口均为有限值，`divergent = 0`，说明模型没有出现数值爆炸。
- final-test 上，K1 联合相对误差为 **6.50%**，K56 联合相对误差为 **41.20%**。这里“联合误差”定义为速度相对误差与压力相对误差之和，而非二者的欧氏合成。
- 模型在低 Re 区间表现非常好：final-test 的 Re=22.25、24.0、27.5 在 K56 上的联合误差分别为 **0.10%**、**0.10%**、**0.91%**。
- 从 Re≈31 开始误差突增；Re=31.0 的 K56 联合误差达到 **53.90%**。该现象是精度退化而不是数值发散，表明 B1 历史特征 FNN 对高 Re 周期流形、相位演化或压力闭合的表达不足。
- K56 final-test 中压力误差为 **27.45%**，速度误差为 **13.75%**，压力项是长时误差的主要来源。
- 因此，B1 可以作为后续 MoE/KDA 方法的稳定基线，但不能视为覆盖整个 periodic Reynolds 数区间的成功模型。

当前仍在运行的 Periodic MoE 训练不纳入本报告；待其训练结束后，应使用相同数据隔离与 rollout 协议进行独立比较。

## 2. 实验范围与参考材料

### 2.1 数据与方法来源

- 最新 periodic 数据集（集群）：`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/Pinball/fluidicPinball_v2/rom_assets_v2/periodic`
- 数据集构造报告：`C:/Users/panxy1019/Documents/Pinball/reports/FLUIDIC_PINBALL_DATASET_CONSTRUCTION_REPORT_V2.md`
- B1 迁移规范：`C:/Users/panxy1019/Documents/CHANNEL/cluster_hopf_transfer/kda_pr_fnn_rom_20260730/POD_DATASET_MIGRATION_TRAINING_EVALUATION_GUIDE.md`

### 2.2 本报告边界

本报告仅覆盖 B1 Deep-FNN-H3：训练结果、validation rollout 与解封后的一次性 final-test rollout。尚未完成的 Periodic MoE 不参与当前指标比较，也不据此作优劣判断。

## 3. 数据契约与隔离

### 3.1 数据划分

| 子集 | Re 数量 | 用途 |
|---|---:|---|
| Train | 47 | 参数拟合、归一化和 train-only ROM 接口构造 |
| Validation | 9 | 训练期模型选择与冻结后的正式验证 rollout |
| Final-test | 9 | 冻结检查点后的一次性最终评估 |

Validation Reynolds 数为：21.5、23.25、26.0、29.5、33.0、36.5、40.0、47.0、54.0。

Final-test Reynolds 数为：22.25、24.0、27.5、31.0、34.5、38.0、43.0、50.0、57.0。

### 3.2 POD 与特征维度

- 速度 POD 维度：`r_u = 17`
- 压力 POD 维度：`r_p = 16`
- 当前状态特征维度：63
- H3 历史特征维度：263
- rank999 资产的能量尾项：速度约 0.0902%，压力约 0.0848%

能量尾项不是物理场相对 L2 误差，不能直接与 rollout 相对误差相加。

### 3.3 防泄漏措施

- 训练与 validation 使用的系数视图不含任何 final-test Reynolds 数。
- Galerkin 与 Pressure–Poisson ROM 接口只由 train Reynolds 数构造。
- validation/final-test 所需 ROM 量通过 train-only 解析节点插值得到。
- 训练配置中 `heldout_loaded = false`，并硬禁用 9 个 final-test Reynolds 数。
- 资产审计通过：split contract、pressure gauge、ROM tensor 和 projection error 均已验证。

冻结检查点与 final-test 视图的哈希如下：

| 对象 | SHA-256 |
|---|---|
| `best_validation.pt` | `c89a0cbf6cc0920664875f00234afaac49bc19b1c5c5f985cbba0b09e2dbdbda` |
| final-test coefficient view | `c2a768087caf541c6421006e9c41390eb50eaccbe8138ae9099dfa72e297be7c` |

Final-test 按 `one_time_final_test_rollout` 范围解封；解封清单明确记录训练资产未改变，且禁止在看到 final-test 后更新超参数。因此，本报告不使用 final-test 进行模型选择或调参。

## 4. 模型与训练设置

### 4.1 B1 模型

- 变体：B1 Deep-FNN-H3
- 可训练参数：1,792,959
- KDA 投影参数：0
- 速度骨干：学习有限差分导数；Galerkin 项作为输入特征
- 压力骨干：Pressure–Poisson 基线加自适应代数残差/门控
- 时间历史：H3

### 4.2 训练超参数

| 配置项 | 值 |
|---|---:|
| Optimizer | AdamW（fused） |
| 最大优化步 | 8,000 |
| 初始学习率 | 1.0e-3 |
| Weight decay | 1.0e-4 |
| Micro-batch | 32 |
| Gradient accumulation | 1 |
| Gradient clip | 1.0 |
| TBPTT steps | 16 |
| AMP | BF16 |
| TF32 | 开启 |
| GPU memory fraction | 0.2 |
| Validation interval | 400 steps |
| Long validation interval | 800 steps |

### 4.3 Curriculum 适配

迁移指南中的长 horizon 训练需要结合新数据的实际轨迹长度。由于多条 train 轨迹只有 48 个状态，在 H3 历史条件下无法合法构造 K56 训练窗口，因此训练 curriculum 调整为：

| 优化步区间 | 训练 horizon |
|---|---:|
| 0–1599 | K4 |
| 1600–3199 | K8 |
| 3200–5199 | K16 |
| 5200–6799 | K24 |
| 6800–7999 | K32 |

Validation 与 final-test 轨迹长度足以支持 K56，因此 K56 被保留为长时压力测试。需要注意：K56 是超出训练最大 horizon K32 的外推式 rollout 压力测试。

### 4.4 训练资源与吞吐

| 指标 | 结果 |
|---|---:|
| 训练耗时 | 3,083.61 s（约 51.39 min） |
| 吞吐 | 155.66 optimizer steps/min |
| 峰值已分配 GPU 显存 | 0.180 GiB |

训练与集群上的其他任务并行进行，因此吞吐数据受资源竞争影响，不应直接作为独占 GPU 性能基准。

SwanLab 监控记录：[FluidicPinballV2 B1 DeepFNN run](https://swanlab.cn/@panxy1019/FluidicPinballV2_B1_DeepFNN/runs/10lty4w9)

## 5. 检查点选择与 Rollout 协议

### 5.1 检查点冻结

训练期共记录 20 次 validation。K56 长验证仅在对应 long-validation 周期具备检查点资格；step 8000 是最终且最佳的合格长验证检查点。其训练期 K56 validation score 为 0.815153，所有硬门控均通过。

冻结后，正式离线 evaluator 在 validation 与 final-test 上统一重新运行，以避免训练期窗口上限不同造成不可比。

### 5.2 正式 rollout 协议

- 每个 Reynolds 数最多评估 64 个合法窗口。
- Horizons：K1、K2、K4、K8、K16、K24、K32、K56。
- 每个 horizon 均递归 rollout，不使用未来真值回灌。
- `U error`：速度 POD 重构场相对误差。
- `p error`：压力 POD 重构场相对误差。
- `joint error = U error + p error`。
- `terminal error`：窗口末端联合误差。
- `worst-window error`：该集合所有窗口中的最大联合误差。

这些误差以 POD 子空间中的重构真值为参照，并非直接对原始 CFD 全场计算的总误差。

## 6. Validation Rollout 结果

下表均为百分比。所有 horizon 的 finite rate 均为 100%，发散窗口数均为 0。

| Horizon | U error | p error | Joint | Terminal | Worst window |
|---:|---:|---:|---:|---:|---:|
| K1 | 1.04% | 4.52% | 5.55% | 5.55% | 16.31% |
| K2 | 1.48% | 5.25% | 6.73% | 7.90% | 22.79% |
| K4 | 2.24% | 6.73% | 8.97% | 12.31% | 40.58% |
| K8 | 3.77% | 10.22% | 13.99% | 23.09% | 65.20% |
| K16 | 6.93% | 16.11% | 23.04% | 37.16% | 92.78% |
| K24 | 9.21% | 19.25% | 28.46% | 39.92% | 95.41% |
| K32 | 10.35% | 20.86% | 31.21% | 39.54% | 108.21% |
| K56 | 11.60% | 23.14% | 34.74% | 37.31% | 101.92% |

K56 的 Reynolds 数分解显示，误差跃迁发生在 Re=29.5 与 Re=33.0 之间：

| Re | U error | p error | Joint | Terminal | Worst window |
|---:|---:|---:|---:|---:|---:|
| 21.5 | 0.01% | 0.10% | 0.11% | 0.12% | 0.13% |
| 23.25 | 0.01% | 0.10% | 0.10% | 0.11% | 0.12% |
| 26.0 | 0.16% | 0.35% | 0.51% | 0.64% | 1.21% |
| 29.5 | 0.60% | 1.31% | 1.91% | 2.58% | 5.83% |
| 33.0 | 19.75% | 37.20% | 56.96% | 62.59% | 99.999% |
| 36.5 | 20.40% | 39.34% | 59.74% | 62.48% | 91.50% |
| 40.0 | 20.95% | 40.97% | 61.92% | 64.23% | 94.41% |
| 47.0 | 21.28% | 43.94% | 65.23% | 74.16% | 93.14% |
| 54.0 | 21.27% | 44.94% | 66.21% | 68.84% | 101.92% |

## 7. Final-test Rollout 结果

### 7.1 聚合结果

下表均为百分比。全部 horizon 的 finite rate 均为 100%，发散窗口数均为 0。

| Horizon | U error | p error | Joint | Terminal | Worst window |
|---:|---:|---:|---:|---:|---:|
| K1 | 1.23% | 5.28% | 6.50% | 6.50% | 16.44% |
| K2 | 1.75% | 6.20% | 7.95% | 9.40% | 22.61% |
| K4 | 2.66% | 8.01% | 10.67% | 14.73% | 39.37% |
| K8 | 4.50% | 12.21% | 16.70% | 27.53% | 67.17% |
| K16 | 8.26% | 19.12% | 27.38% | 44.19% | 93.23% |
| K24 | 10.98% | 22.88% | 33.86% | 47.37% | 97.61% |
| K32 | 12.28% | 24.82% | 37.10% | 46.93% | 99.17% |
| K56 | 13.75% | 27.45% | 41.20% | 44.42% | 101.44% |

### 7.2 K56 按 Reynolds 数分解

| Re | U error | p error | Joint | Terminal | Worst window |
|---:|---:|---:|---:|---:|---:|
| 22.25 | 0.006% | 0.097% | 0.103% | 0.108% | 0.111% |
| 24.0 | 0.009% | 0.094% | 0.103% | 0.111% | 0.123% |
| 27.5 | 0.29% | 0.62% | 0.91% | 1.18% | 2.48% |
| 31.0 | 18.74% | 35.16% | 53.90% | 59.13% | 92.54% |
| 34.5 | 19.90% | 38.24% | 58.14% | 64.95% | 99.48% |
| 38.0 | 20.71% | 40.17% | 60.88% | 63.02% | 98.64% |
| 43.0 | 21.21% | 42.53% | 63.74% | 70.99% | 94.63% |
| 50.0 | 21.43% | 44.51% | 65.94% | 68.11% | 94.92% |
| 57.0 | 21.43% | 45.66% | 67.09% | 72.19% | 101.44% |

### 7.3 Validation 与 Final-test 对比

| 指标 | Validation | Final-test | 观察 |
|---|---:|---:|---|
| K1 joint | 5.55% | 6.50% | final-test 略差 |
| K56 U | 11.60% | 13.75% | 长时速度误差增加 |
| K56 p | 23.14% | 27.45% | 压力仍是主要误差来源 |
| K56 joint | 34.74% | 41.20% | final-test 相对 validation 约恶化 18.6% |
| Divergent windows | 0 | 0 | 两个集合均数值稳定 |

## 8. 与 POD 表示误差的关系

数据集构造报告给出的 rank999 直接投影全场误差参考为：

| 子集 | 速度中位数 / P95 | 压力中位数 / P95 |
|---|---:|---:|
| Periodic validation | 0.473% / 0.668% | 0.928% / 1.27% |
| Periodic final-test | 0.473% / 0.847% | 1.00% / 1.37% |

这些数值描述 POD 表示本身相对 CFD 全场的误差；本实验 rollout 指标描述模型预测相对 POD 重构真值的误差。二者参照对象不同，不能直接相加。POD 结果表明表示基底在 validation/final-test 上仍有较低的投影误差，因此高 Re 下 50%–67% 的 K56 联合误差主要来自动力学预测，而不是 rank999 表示容量不足。

## 9. 结果分析

### 9.1 数值稳定，但科学精度不足

所有 validation 与 final-test 窗口均通过有限值与发散门控，说明 B1 学到的是稳定递推器。然而，稳定性并不等于准确性：高 Re 的预测保持有界，却落在错误的周期轨道、相位或幅值上。

### 9.2 明显的 Reynolds 数分区

低 Re 区间（约 Re≤27.5）在 K56 上仍能保持 1% 左右或更低的联合误差；validation 的 Re=29.5 也只有 1.91%。但 final-test Re=31.0 与 validation Re=33.0 分别跃升到 53.90% 与 56.96%。这提示模型在约 Re=30–33 附近没有平滑覆盖数据流形的变化。

可能原因包括：

1. H3 固定历史特征不足以稳定编码跨 Reynolds 数变化的周期相位与长期记忆。
2. B1 没有 KDA/显式参数化机制，单一 FNN 对不同动力学区域的共享映射能力不足。
3. 压力 Pressure–Poisson 残差闭合在高 Re 区间误差更大，并在递归 rollout 中持续累积。
4. 训练最大 horizon 为 K32，而 K56 是超出训练长度的压力测试；不过高 Re 在较短 horizon 已开始退化，因此 horizon 外推不是唯一原因。

### 9.3 压力误差主导

Final-test K56 中，压力误差 27.45%，约为速度误差 13.75% 的两倍。若后续方法只改善速度动力学而不改善压力闭合，联合误差仍会受到明显限制。后续比较应单独报告 U、p 和 joint，不能只观察 joint 总分。

## 10. 结论与后续建议

### 10.1 结论

B1 Deep-FNN-H3 在最新 Fluidic Pinball V2 periodic 数据集上完成了合规迁移，并通过了所有数值稳定性门控。它在低 Re 周期轨道上具有很高精度，但在约 Re=31 之后出现系统性动力学精度失效。因此：

- **可接受用途：** 稳定的迁移基线、低 Re 子区间模型、MoE/KDA 对照组。
- **不可接受用途：** 在完整 periodic Reynolds 数范围内作为高保真统一 ROM。
- **最终判定：** 工程执行成功，科学精度目标仅部分达成。

### 10.2 后续实验建议

1. 等待 Periodic MoE 训练完成，使用本报告完全相同的 64-window、K1–K56 evaluator 做冻结比较。
2. 重点比较 Re=29.5、31.0、33.0 附近的误差跃迁，判断 MoE 路由是否能分离动力学区域。
3. 若 MoE 仍在高 Re 失效，再评估带 KDA/显式参数嵌入的变体，并优先检查压力闭合。
4. 增加合法的长时间训练覆盖或采用能够处理更长历史的状态表示；任何新方案必须重新使用 validation 选模，不能基于本次已解封的 final-test 调参。
5. 后续报告应同时给出 POD 全场投影误差与 ROM rollout 误差，但继续保持二者定义分离。

## 11. 可复现性与产物

### 11.1 集群实验目录

- 实验根目录：`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/fluidic_pinball_periodic_v2_b1`
- Run：`runs/FluidicPinballV2_B1_Deep_FNN_H3_seed1248`
- Validation evaluation：`evaluation_validation`
- Final-test evaluation：`evaluation_final_test`

### 11.2 本地代码

- `C:/Users/panxy1019/Documents/Pinball/training/fluidic_pinball_periodic_v2_b1/code/prepare_b1_assets.py`
- `C:/Users/panxy1019/Documents/Pinball/training/fluidic_pinball_periodic_v2_b1/code/b1_data_contract.py`
- `C:/Users/panxy1019/Documents/Pinball/training/fluidic_pinball_periodic_v2_b1/code/train_b1_fluidic_pinball.py`
- `C:/Users/panxy1019/Documents/Pinball/training/fluidic_pinball_periodic_v2_b1/code/evaluate_b1_validation.py`
- `C:/Users/panxy1019/Documents/Pinball/training/fluidic_pinball_periodic_v2_b1/code/prepare_b1_final_test_view.py`
- `C:/Users/panxy1019/Documents/Pinball/training/fluidic_pinball_periodic_v2_b1/code/run_b1_train_and_eval.sh`

### 11.3 本地证据归档

目录：`C:/Users/panxy1019/Documents/Pinball/reports/b1_periodic_v2_artifacts`

- `validation_summary.json` / `validation_results.csv`
- `final_test_summary.json` / `final_test_results.csv`
- `final_test_unseal_manifest.json`
- `training_config.json`
- `training_throughput.json`
- `parameter_count.json`
- `validation_history.json`

以上 JSON/CSV 是本报告表格的机器可读依据，保留了正式 rollout 的完整精度，而报告中的百分比为四舍五入后的展示值。
