# CenteredSquare Periodic Specialist 长课程训练最终实验报告

日期：2026-07-29

## 1. 实验结论

方柱周期子数据集上的长课程训练已完成并停止。训练在 epoch 670 触发 early stopping；随后按用户要求取消继续延长到 epoch 720。最终交付模型使用验证集最优的 epoch 490 权重。

- 最优验证分数：`1.21152148`（epoch 490，K16）
- epoch 670 最后一次验证分数：`1.21207912`
- `final.pt` 的模型权重与 `best_validation.pt` 逐张量完全一致
- heldout 物理场 K24 重建误差：速度 `0.9395%`，压力 `1.0921%`
- K1/K4/K8/K16/K24 的所有测试窗口均为有限值，无 NaN/Inf
- 相比原 240-epoch 模型，K24 速度和压力误差分别下降约 `92.79%` 和 `92.95%`

因此，本次长课程训练显著改善了真正的物理场重建效果。训练日志中的 `val_score≈1.21` 不是物理场百分比误差；它是由 RHS、速度系数、压力系数等相对误差组合而成的模型选择分数。

## 2. 数据与划分

使用 CenteredSquare periodic 数据，训练、验证和测试划分保持不变。

| 项目 | 数量 |
|---|---:|
| 训练 Reynolds 数 | 27 |
| 验证 Reynolds 数 | 6 |
| heldout 测试 Reynolds 数 | 4 |
| 训练样本 | 3321 |
| 验证样本 | 738 |
| 测试样本 | 492 |

heldout Reynolds 数为：

- `Re=100.5`
- `Re=102.0`
- `Re=120.689651`
- `Re=144.827591`

## 3. 模型与训练设置

模型为周期专家 HPRS-MoE ROM，速度秩 `r_u=28`，压力秩 `r_p=26`，使用 RK4 闭环积分。主要训练设置如下：

| 阶段 | epoch 范围 | rollout K | rollout batch | 实际 epoch | 平均训练时间/epoch | 阶段最佳验证分数 |
|---|---:|---:|---:|---:|---:|---:|
| K4 | 1–120 | 4 | 56 | 120 | 28.33 s | 1.288724（epoch 120） |
| K8 | 121–280 | 8 | 56 | 160 | 56.83 s | 1.225053（epoch 270） |
| K12 | 281–440 | 12 | 40 | 160 | 79.21 s | 1.214165（epoch 440） |
| K16 | 441–670 | 16 | 28 | 230 | 91.45 s | 1.211521（epoch 490） |

其他关键设置：

- 主 batch size：`1024`
- 每 epoch rollout 更新次数：`2`
- 混合精度：`BF16`
- batched experts：启用
- 固定周期 regime group：`1`
- 稠密 MoE 训练：关闭
- 验证间隔：每 `10` epoch
- 训练 epoch 计算时间合计：约 `12.83 h`

epoch 281 从 K8 切换到 K12 时，原 rollout batch 56 曾触发 OOM。恢复时将 K12/K16 rollout batch 保守调整为 40/28，并从 epoch 280 的有效 `latest.pt` 恢复；模型、数据、损失和 split 均未改变。恢复后未再出现 OOM、NaN 或异常退出。

## 4. 最终系数空间验证与测试

以下结果由训练结束后的内置最终评估器使用最优 epoch 490 权重得到。

### 4.1 heldout 聚合指标

| 指标 | 四个 heldout Re 的均值 |
|---|---:|
| RHS relative L2 | 0.941805 |
| pressure head relative L2 | 0.018355 |
| one-step velocity coefficient L2 | 0.079791 |
| one-step pressure coefficient L2 | 0.077269 |
| autonomous rollout velocity coefficient L2 | 0.057603 |
| autonomous rollout pressure coefficient L2 | 0.057639 |
| rollout pressure energy relative error | 0.005360 |

### 4.2 各 heldout Reynolds 数

| Re | RHS L2 | pressure head L2 | Auto a one-step L2 | Auto b one-step L2 | Auto a rollout L2 | Auto b rollout L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 100.5 | 0.886801 | 0.008681 | 0.089739 | 0.078598 | 0.063652 | 0.055314 |
| 102.0 | 0.927452 | 0.010099 | 0.088866 | 0.079747 | 0.056370 | 0.047920 |
| 120.689651 | 0.975574 | 0.048151 | 0.067712 | 0.079742 | 0.038556 | 0.054963 |
| 144.827591 | 0.977393 | 0.006489 | 0.072846 | 0.070989 | 0.071834 | 0.072360 |

RHS 相对误差仍然偏高，但这并未直接转化为同量级的物理场误差。对当前 ROM，闭环积分、低维系数轨迹、POD 重建以及误差在时空能量中的投影共同决定最终物理场误差，因此必须以第 5 节的完整快照重建结果作为主要效果判断。

## 5. heldout 物理场重建测试

### 5.1 评估定义

从 heldout 流场快照的真实初始状态出发进行自主闭环 rollout，随后通过全局 POD 基和均值场重建速度、压力物理场，并与原始完整快照比较：

- 速度：面积加权相对 L2
- 压力：逐帧移除体积加权均值后，计算面积加权相对 L2
- 外部时间步长：`Δt=4`
- RK4 最大内部积分步长：`0.5`
- rollout 起点 stride：`24`
- 同时报告使用真实 POD 系数重建时的 POD 截断误差下限

### 5.2 多时域聚合结果

| 时域 | 物理时间 | 窗口数 | 非有限窗口 | 速度场误差 | 压力场误差 | 速度 POD 下限 | 压力 POD 下限 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1 | 4 | 20 | 0 | 0.9701% | 1.0781% | 0.4068% | 0.4566% |
| K4 | 16 | 20 | 0 | 0.8720% | 0.9799% | 0.3976% | 0.4424% |
| K8 | 32 | 16 | 0 | 0.8053% | 0.8929% | 0.3965% | 0.4283% |
| K16 | 64 | 16 | 0 | 0.8181% | 0.9328% | 0.3971% | 0.4324% |
| K24 | 96 | 16 | 0 | 0.9395% | 1.0921% | 0.4018% | 0.4354% |

K1 到 K24 没有出现误差爆炸。K24 误差约为 POD 截断下限的 2.34 倍（速度）和 2.51 倍（压力），说明剩余误差同时包含 ROM 表示误差和动力学预测误差。

### 5.3 K24 各 Reynolds 数结果

| Re | 速度场误差 | 压力场误差 | 速度 POD 下限 | 压力 POD 下限 | 窗口数 |
|---:|---:|---:|---:|---:|---:|
| 100.5 | 0.6479% | 0.5756% | 0.3721% | 0.3631% | 4 |
| 102.0 | 0.6780% | 0.6127% | 0.3921% | 0.3902% | 4 |
| 120.689651 | 0.6698% | 0.9762% | 0.3525% | 0.4812% | 4 |
| 144.827591 | 1.4672% | 1.8023% | 0.4765% | 0.4993% | 4 |

高 Reynolds 数 `Re=144.827591` 仍是最困难的 heldout 点，但 K24 速度和压力误差均低于 2%，且没有非有限窗口。

## 6. 与原 240-epoch 模型的物理场结果对比

| 时域/字段 | 原 240-epoch | 本次长课程 | 相对下降 |
|---|---:|---:|---:|
| K1 速度 | 15.06% | 0.9701% | 93.56% |
| K1 压力 | 17.79% | 1.0781% | 93.94% |
| K24 速度 | 13.03% | 0.9395% | 92.79% |
| K24 压力 | 15.48% | 1.0921% | 92.95% |

结果支持此前关于“240 个 epoch 中每个 K 阶段学习不足”的判断。延长 K4/K8/K12/K16 阶段并保证 K16 获得充分训练后，物理场误差下降了一个数量级以上。

## 7. 模型选择与停止说明

- `best_validation.pt`：epoch 490，验证分数 `1.21152148`
- `latest.pt`：epoch 670，最后验证分数 `1.21207912`
- `final_training.pt`：epoch 670 的原始训练状态
- `final.pt`：元数据记录停止于 epoch 670，但模型权重已回载为 epoch 490 最佳权重
- 已验证 `final.pt` 与 `best_validation.pt` 的 454 个模型状态张量完全一致

epoch 670 停止是 early stopping 正常触发，不是 OOM、NaN 或进程崩溃。之后曾准备继续到 epoch 720，但按用户指令在加载 epoch 670 后、执行 epoch 671 前停止，因此没有额外参数更新。

## 8. 最终产物

训练集群实验目录：

`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/centered_square_periodic_v1`

关键文件：

- 最优模型：`runs/optimized_long_seed1600/best_validation.pt`
- 最终交付模型：`runs/optimized_long_seed1600/final.pt`
- 最后训练状态：`runs/optimized_long_seed1600/final_training.pt`
- 系数空间指标：`runs/optimized_long_seed1600/centered_square_periodic_optimized_r28_p26_metrics.json`
- 系数空间摘要：`runs/optimized_long_seed1600/centered_square_periodic_optimized_r28_p26_summary.md`
- 物理场多时域结果：`runs/optimized_long_seed1600/final_physical_best/physical_field_multihorizon.json`
- 物理场评估日志：`runs/optimized_long_seed1600/final_physical_best/evaluate.log`

## 9. 最终判断

本次训练达到预期迁移目标：模型在四个未参与训练的周期 Reynolds 数上保持稳定闭环 rollout，K1–K24 完整速度场和压力场聚合误差均约为 0.8%–1.1%。后续若继续提升，优先方向不是简单增加 epoch，而是针对 `Re≈145` 的高 Re 泛化和 RHS 局部误差设计更有针对性的采样或损失；当前模型已经可以作为 CenteredSquare periodic specialist 的正式候选版本。
