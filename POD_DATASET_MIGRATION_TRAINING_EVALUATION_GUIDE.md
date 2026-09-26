# POD 数据集迁移：训练与评估方法指南

本文档总结 CenteredSquare periodic specialist 的数据处理、模型训练、GPU 优化和最终评估方法，用于迁移到新的 POD 数据集。

## 1. 方法概述

整体思路是：

1. 在新数据集上重新构建 POD 表示和物理 ROM。
2. 在 POD 系数空间训练带物理基线的 MoE 动力学模型。
3. 使用逐步增长的闭环 rollout 课程训练。
4. 使用验证集选择最佳 checkpoint。
5. 将预测系数重新投影为完整速度场和压力场，在 heldout 原始快照上做最终测试。

需要特别区分两类误差：

- **系数/算子误差**：用于训练诊断和模型选择。
- **物理场重建误差**：用于判断模型在真实流场上的最终效果。

训练日志中的综合 `val_score` 不是物理场百分比误差。

## 2. 新数据集需要提供的内容

每个 Reynolds 数或工况应包含按时间排列的：

- 速度场快照 `U(t,x)`
- 压力场快照 `p(t,x)`
- 时间坐标 `t`
- Reynolds 数或其他工况参数
- 网格面积或体积权重
- 快照所属工况和局部时间索引

在新数据集上需要重新生成：

- 速度 POD 基 `Φu`
- 压力 POD 基 `Φp`
- 速度和压力均值场
- 速度系数 `a(t)`
- 压力系数 `b(t)`
- 速度系数导数 `da/dt`
- Galerkin 或其他物理基准 RHS
- 压力基准/压力代理算子
- 归一化统计量
- 时间链 `next_idx`
- 原始快照与 POD 系数的对应索引

### 2.1 不应从旧数据集复用的内容

以下内容必须在新数据上重新计算：

- POD 基
- 均值场
- POD 系数
- ROM 算子
- pressure surrogate
- scaler
- 数据划分和索引

旧实验可以复用模型结构、损失设计、训练课程和工程优化，但不能直接套用旧 POD 坐标系中的物理量。

### 2.2 压力 gauge

压力场存在常数不唯一性。训练和评估必须使用一致的 gauge。本实验在物理场评估时，对每一帧压力移除体积加权均值：

\[
p' = p - \frac{\sum_x w_xp_x}{\sum_x w_x}.
\]

## 3. POD 秩的选择

原 CenteredSquare 实验使用：

- 速度秩 `r_u=28`
- 压力秩 `r_p=26`

新数据不应机械沿用这两个数。建议综合考虑：

1. POD 累计能量占比。
2. 使用真实 POD 系数重建原始快照时的截断误差。
3. 主要流动结构是否被保留。
4. 闭环训练时的显存和计算成本。

POD oracle 重建误差将作为模型能够达到的表示误差下限。

## 4. 数据划分

应按完整 Reynolds 数或完整工况划分，而不是随机拆分时间帧：

```text
训练工况      用于参数更新
验证工况      用于选择最佳 checkpoint 和 early stopping
heldout 工况  训练期间完全不可见，只用于最终测试
```

这样可以避免同一轨迹的相邻帧同时进入训练集和测试集，并真实评估跨参数泛化。

测试工况建议覆盖：

- 参数范围内部的插值点
- 参数范围边缘的困难点
- 条件允许时的轻度外推点

原 CenteredSquare periodic 实验使用 27 个训练 Re、6 个验证 Re 和 4 个 heldout Re。

## 5. 模型方法

模型在 POD 系数空间运行，输入通常包括：

```text
当前速度系数 a(t)
当前压力系数 b(t)
Re 或其他工况参数
周期相位
物理描述量
历史 a、b 和 RHS
```

模型不完全替代物理 ROM，而是学习物理基准模型的修正：

\[
\frac{da}{dt}
=
RHS_{\mathrm{physical}}
+
RHS_{\mathrm{MoE\ residual}}.
\]

压力分支预测 pressure closure。

### 5.1 MoE 结构

原实验采用：

- 共享编码器
- regime group router
- group 内共享专家
- group 内 Top-k 路由专家
- 速度 residual 分支
- 压力 closure 分支

主要结构参数为：

| 参数 | 数值 |
|---|---:|
| hidden dimension | 224 |
| regime groups | 3 |
| routed experts/group | 6 |
| group top-k | 1 |
| group 内 top-k | 2 |
| expert hidden | 768 |
| refinement blocks | 3 |

如果新数据仍然只包含周期子数据集，可以固定 periodic regime group。若新数据同时包含稳态、过渡态和周期态，则应重新启用 group router，不能固定为单一 group。

## 6. 损失函数

训练同时约束一步算子精度和长期闭环稳定性，主要包括：

- 一步速度系数误差
- 一步压力系数误差
- RHS/operator 误差
- pressure closure 误差
- 多步自主 rollout 误差
- 采样物理场重建误差
- 能量一致性
- 轨迹一致性
- Router load balance
- Router entropy
- Router 时间平滑
- 专家多样性
- 弱工况/regime 监督

不能只训练一步预测。较低的一步误差并不能保证模型自主积分 16～24 步后仍然稳定。

## 7. 课程训练

原实验最终采用以下课程：

| 阶段 | epoch 范围 | rollout 长度 | rollout batch | 实际 epoch |
|---|---:|---:|---:|---:|
| K4 | 1–120 | 4 | 56 | 120 |
| K8 | 121–280 | 8 | 56 | 160 |
| K12 | 281–440 | 12 | 40 | 160 |
| K16 | 441–670 | 16 | 28 | 230 |

其他关键设置：

- 主 batch size：`1024`
- 每个 epoch 的 rollout 更新次数：`2`
- 混合精度：BF16
- 验证间隔：每 10 epoch
- 优化器和 scheduler 状态随 checkpoint 保存
- 使用 `latest.pt` 支持续训
- 使用 `best_validation.pt` 保存最佳验证权重
- 最终测试使用最佳验证权重，而不是简单使用最后一个 epoch

rollout 越长，反向传播的显存需求越大，因此 rollout batch 按阶段递减。

## 8. GPU 训练优化

在 RTX 4090 上使用的主要优化包括：

- BF16 autocast
- batched expert tensor contractions
- 增大一步训练 batch
- 分阶段设置 rollout batch
- checkpoint 中的最佳模型状态存放在 CPU
- 恢复 checkpoint 后及时释放临时 GPU 张量
- 禁用不需要的 dense-MoE 路径
- 周期数据固定 regime group，减少无效 group 计算

原实验各阶段平均训练耗时：

| 阶段 | 平均耗时/epoch |
|---|---:|
| K4 | 28.33 s |
| K8 | 56.83 s |
| K12 | 79.21 s |
| K16 | 91.45 s |

新数据的 POD 秩、历史长度、样本数和专家规模改变后，应重新进行显存探测。推荐从较保守的 rollout batch 开始逐步增加。

## 9. Checkpoint 与恢复策略

建议保存：

- `latest.pt`：最近一次完整验证后的训练状态
- `best_validation.pt`：验证分数最低的模型
- `final_training.pt`：训练停止时的原始模型状态
- `final.pt`：回载最佳验证权重后的最终交付模型

checkpoint 应包含：

- 模型参数
- 优化器状态
- scheduler 状态
- scaler
- 最佳 epoch 和最佳验证分数
- 当前 epoch
- RNG 状态
- split 统计
- 完整训练参数

发生 OOM 时，应从最近的有效 `latest.pt` 恢复，只降低当前及后续阶段的 rollout batch，不改变模型、数据、损失或 split。

## 10. 第一层评估：系数空间与算子指标

训练和验证阶段记录：

- RHS relative L2
- pressure-head relative L2
- one-step `a/b` relative L2
- autonomous rollout `a/b` relative L2
- 压力能量误差
- Router 使用率
- 专家负载、熵和退化情况

这些指标用于诊断和选择 checkpoint，但综合 `val_score` 不是物理场误差。

原实验最佳验证分数为 `1.21152148`，并不代表物理场误差为 121%。

## 11. 第二层评估：完整物理场重建

最终效果必须在 heldout 原始流场快照上测量。

评估流程：

1. 从 heldout 真实初始 POD 状态出发。
2. 使用模型进行自主闭环 rollout。
3. 通过新数据集的 POD 基和均值场重建速度、压力。
4. 与对应的原始 CFD 快照直接比较。
5. 同时用真实 POD 系数重建快照，得到 POD 截断误差下限。
6. 统计 NaN/Inf 和失败窗口。

速度场面积加权相对 L2 为：

\[
E_u=
\sqrt{
\frac{\sum_{t,x}w_x\|\hat U(t,x)-U(t,x)\|^2}
{\sum_{t,x}w_x\|U(t,x)\|^2}
}.
\]

压力场使用相同形式，但在比较前应对真实压力和预测压力使用一致的 gauge。

建议至少测试：

```text
K1、K4、K8、K16、K24
```

每个时域应报告：

- 聚合速度场误差
- 聚合压力场误差
- 各 heldout 工况误差
- 速度和压力 POD 截断下限
- 测试窗口数量
- 非有限窗口数量
- 对应物理时间

## 12. 原 CenteredSquare 最终结果

| 时域 | 物理时间 | 速度场误差 | 压力场误差 | 速度 POD 下限 | 压力 POD 下限 |
|---|---:|---:|---:|---:|---:|
| K1 | 4 | 0.9701% | 1.0781% | 0.4068% | 0.4566% |
| K4 | 16 | 0.8720% | 0.9799% | 0.3976% | 0.4424% |
| K8 | 32 | 0.8053% | 0.8929% | 0.3965% | 0.4283% |
| K16 | 64 | 0.8181% | 0.9328% | 0.3971% | 0.4324% |
| K24 | 96 | 0.9395% | 1.0921% | 0.4018% | 0.4354% |

所有测试窗口均为有限值，没有出现 NaN、Inf 或闭环发散。

## 13. 新数据集迁移执行清单

### 数据检查

- [ ] 原始速度、压力、时间和工况参数完整
- [ ] 所有工况使用一致的网格或具有明确的投影关系
- [ ] 网格面积/体积权重正确
- [ ] 时间严格排序
- [ ] 原始快照和 POD 系数索引一一对应
- [ ] 压力 gauge 处理一致

### POD 与 ROM 构建

- [ ] 在新数据上计算速度 POD
- [ ] 在新数据上计算压力 POD
- [ ] 选择新的 `r_u` 和 `r_p`
- [ ] 测量 POD oracle 重建误差
- [ ] 生成 `a(t)`、`b(t)` 和 `da/dt`
- [ ] 重新生成物理 ROM/Galerkin 算子
- [ ] 重新生成 pressure surrogate
- [ ] 重新计算 scaler

### 数据划分

- [ ] 按完整工况划分 train/validation/heldout
- [ ] 相邻时间帧没有跨集合泄漏
- [ ] heldout 覆盖插值点和边缘困难点
- [ ] split 固化并写入 checkpoint

### 训练前验证

- [ ] POD oracle 能正确重建原始快照
- [ ] `next_idx` 时间链正确
- [ ] 物理基准 RHS 数值有限
- [ ] K1 前向预测数值有限
- [ ] 运行 1～2 epoch K4 smoke test
- [ ] 检查梯度、损失和显存峰值
- [ ] 确定各课程阶段的 rollout batch

### 正式训练

- [ ] 执行 K4→K8→K12→K16 课程
- [ ] 定期保存 `latest.pt`
- [ ] 按验证分数更新 `best_validation.pt`
- [ ] 监控 OOM、NaN、梯度和 GPU 利用率
- [ ] 阶段切换时特别检查显存

### 最终测试

- [ ] 使用最佳验证权重
- [ ] 运行系数空间和算子评估
- [ ] 运行 K1/K4/K8/K16/K24 物理场重建
- [ ] 报告 POD 截断下限
- [ ] 报告各 heldout 工况结果
- [ ] 确认非有限窗口数为 0
- [ ] 生成最终实验报告

## 14. 推荐验收标准

新数据集的合理验收条件可以设置为：

1. 所有 heldout rollout 窗口无 NaN/Inf。
2. K24 不出现明显误差爆炸。
3. 模型物理场误差明显低于未修正物理 ROM。
4. 模型误差与 POD 截断下限保持在可解释的倍数范围内。
5. 参数范围边缘工况没有显著失稳。
6. `best_validation.pt` 的结果可以独立重复评估。

## 15. 迁移时优先调整的参数

建议按以下顺序调整：

1. `r_u`、`r_p`
2. 时间步长和 RK4 内部最大步长
3. rollout batch
4. 各 K 阶段 epoch 数
5. history 长度
6. hidden/expert dimension
7. rollout 和物理一致性损失权重

不建议在第一次迁移时同时大幅修改模型结构、损失函数和数据构造。先保持训练方法不变，确认新数据接口和物理场评估正确，再根据结果做针对性优化。

## 16. 现有代码文件与职责

建议复制现有实现建立一个新的实验目录，保留原方柱实验作为可复现实验，不要直接覆盖。

| 文件 | 作用 | 迁移时的处理 |
|---|---|---|
| `train_square_periodic_moe.py` | 模型、数据读取、损失、课程训练、checkpoint、系数空间评估 | 尽量保持通用逻辑不变；只修改确实与数据格式或划分数量绑定的代码 |
| `run_square_periodic_training_optimized.py` | 从旧 checkpoint 继承配置，并覆盖新实验的数据路径、POD 秩、split、训练课程和 SwanLab 参数 | 新建一份新数据集专用 runner，主要修改此文件 |
| `evaluate_square_physical_fields.py` | 从预测 POD 系数重建完整速度/压力场，计算多时域物理场误差和 POD 下限 | 修改原始快照路径、必要的文件名兼容和新数据字段 |
| `resume_optimized_long.sh` | 在训练集群启动或恢复任务 | 复制后修改实验目录、checkpoint、日志名和启动参数 |

训练集群实际使用的 trainer 名称可能为 `train_square_periodic_moe_optimized.py`；它对应本地保存的优化版 trainer。复制代码时应以训练集群最终运行版本为准。

## 17. 当前代码中需要迁移的硬编码

### 17.1 Runner 中的硬编码

`run_square_periodic_training_optimized.py` 的 `values.update(...)` 当前写死了：

- `velocity_rom_periodic.npz`
- `pressure_poisson_surrogate_periodic.npz`
- `r_u=28`
- `r_p=26`
- `max_integrator_dt=0.5`
- `fixed_integrator_substeps=8`
- 6 个 validation Re
- 4 个 heldout test Re
- experiment name/tag
- SwanLab project/group/run name
- `fixed_regime_group=1`

这些值应在新 runner 中集中修改。例如：

```python
values.update(
    data_root=cli.data_root,
    tensor_path=cli.data_root / "velocity_rom_periodic.npz",
    pressure_surrogate_path=(
        cli.data_root / "pressure_poisson_surrogate_periodic.npz"
    ),
    output_dir=cli.output_dir,
    r_u=NEW_R_U,
    r_p=NEW_R_P,
    validation_re_values=NEW_VALIDATION_RE,
    test_re_selection="values",
    test_re_values=NEW_HELDOUT_RE,
    experiment_name="new_dataset_periodic_rXX_pYY",
    experiment_tag="New POD dataset periodic curriculum",
    swanlab_project="NewPODPeriodicMOE",
    swanlab_group="periodic-long-curriculum",
)
```

推荐把这些值放在专用 runner，而不是散落修改大型 trainer。这样可以继续复用训练核心，并保留每个数据集的可复现入口。

### 17.2 Trainer 中的划分断言

当前 trainer 明确要求：

```python
if (len(train_labels), len(args.validation_re_indices),
        len(args.test_re_indices)) != (27, 6, 4):
    raise AssertionError(...)
```

新数据集的工况数量改变后，这一断言必须修改。推荐改为由 runner 传入期望数量，或改成通用的不相交与非空检查：

```python
if not train_labels:
    raise ValueError("Training split is empty.")
if not args.validation_re_indices:
    raise ValueError("Validation split is empty.")
if not args.test_re_indices:
    raise ValueError("Heldout split is empty.")

if set(args.validation_re_indices) & set(args.test_re_indices):
    raise ValueError("Validation and heldout splits overlap.")
```

如果实验要求固定精确数量，也可以增加 runner 参数：

```text
--expected-train-re-count
--expected-validation-re-count
--expected-test-re-count
```

不要直接删除所有 split 检查，否则错误的 Re 映射可能造成数据泄漏。

### 17.3 POD 和索引文件名

Trainer 默认读取：

```text
pod_snapshot_index.csv
global_velocity_pod_area_weighted_l2.npz
global_pressure_pod_area_weighted_l2.npz
velocity_rom_periodic.npz
pressure_poisson_surrogate_periodic.npz
```

Trainer 对两个 POD 文件还支持以下备用名称：

```text
global_velocity_pod_weighted_l2.npz
global_pressure_pod_weighted_l2.npz
```

但当前物理场 evaluator 直接使用 `*_area_weighted_l2.npz`，没有相同的 fallback。迁移时应当：

1. 优先保持标准文件名；或者
2. 给 evaluator 增加和 trainer 相同的文件名解析逻辑。

### 17.4 数据字段约定

`pod_snapshot_index.csv` 至少需要提供：

```text
Re
Re_label
time
phase
regime
local_snapshot_index
```

速度 POD NPZ 至少需要与以下字段兼容：

```text
coeff_uv
phi_uv
cumulative_energy_uv
Re_labels
Re_values
mean_uv_regime
point_areas
```

压力 POD NPZ 至少需要与以下字段兼容：

```text
coeff_p
phi_p
cumulative_energy_p
mean_p_regime
```

原始物理场 evaluator 默认查找：

```text
snapshots_{Re_label}.npz
```

其中至少包含：

```text
U
p
```

如果新数据字段名不同，建议在数据构造阶段统一成上述接口，而不是在训练循环内部增加大量数据集特判。

## 18. 配置复用与模型权重迁移

### 18.1 当前 runner 实际做了什么

当前 runner 会执行：

```python
checkpoint = torch.load(cli.source_checkpoint, map_location="cpu")
saved_args = checkpoint["args"]
values = dict(saved_args)
```

随后覆盖新实验参数，并把这些参数传给 trainer。

因此，当前 `--source-checkpoint` 的主要用途是：

- 复用网络结构参数
- 复用损失权重
- 复用物理闭包设置
- 复用优化器相关超参数默认值

它**不等价于自动加载旧 checkpoint 的 `model_state`**。也就是说，现有迁移流程主要是“方法和配置迁移”，不能默认描述为完整模型权重微调。

### 18.2 什么时候可以迁移权重

仅在以下条件满足时才建议加载旧模型权重：

- 新旧 `r_u`、`r_p` 一致
- 输入特征维度一致
- 专家数量和隐藏维度一致
- pressure/RHS target 定义一致
- POD 坐标具有可解释的对应关系

即使 POD 秩相同，不同几何或不同数据集计算出的 POD 模态也通常不具有逐模态一一对应关系。因此，跨几何直接复制输入层、输出层和物理算子相关权重可能适得其反。

### 18.3 如需实现部分权重加载

应显式增加一个独立参数，例如：

```text
--init-model-checkpoint OLD.pt
```

并只加载名称和形状都匹配的张量：

```python
source = torch.load(args.init_model_checkpoint, map_location="cpu")
target = model.state_dict()
compatible = {
    name: value
    for name, value in source["model_state"].items()
    if name in target and target[name].shape == value.shape
}
missing, unexpected = model.load_state_dict(compatible, strict=False)
```

加载后必须记录：

- 成功加载的张量数量
- 未加载的输入/输出层
- 缺失键和额外键
- 新旧 POD 秩和输入维度

不要使用不检查形状的 `strict=False` 后直接忽略结果。对于新的几何 POD 数据，推荐首先复用配置而不复用 POD 坐标相关权重，获得可靠基线后再对部分权重迁移做消融实验。

## 19. 推荐的代码迁移步骤

### 第一步：复制实验代码

建议形成如下结构：

```text
new_pod_periodic_v1/
├── assets/
├── code/
│   ├── train_new_pod_periodic_moe.py
│   ├── run_new_pod_periodic_training.py
│   └── evaluate_new_pod_physical_fields.py
├── runs/
└── staging/raw/
```

### 第二步：只修改专用 runner

先在 runner 中修改：

- 数据目录
- ROM 和 pressure surrogate 文件名
- `r_u`、`r_p`
- validation/test Re
- experiment name
- SwanLab project/group
- curriculum 和 batch
- periodic group 是否固定

### 第三步：最小化修改 trainer

只有以下情况才修改 trainer：

- split 数量断言不适用
- 新数据字段无法转换为现有接口
- 工况参数不再是单个 Re
- regime/attractor 定义发生变化
- 物理基准 RHS 或 pressure closure 形式变化

不要为了改路径或实验名修改 trainer。

### 第四步：修改 evaluator

确认 evaluator 中的：

- POD 文件名
- 原始快照命名规则
- 原始速度/压力字段名
- 网格权重字段
- 压力 gauge
- 外部 `dt`
- RK4 内部最大步长
- snapshot index 映射

### 第五步：运行接口检查

正式训练前执行：

```text
1. 加载所有 NPZ/CSV 文件
2. 打印关键数组 shape
3. 检查 r_u/r_p 不超过可用模态数
4. 检查每个 next_idx 保持同一 Re 且时间递增
5. 检查 a、b、RHS、pressure base 全部有限
6. 用真实系数完成 POD oracle 重建
7. 运行一次模型前向和一次 RK4 step
```

## 20. 训练启动命令模板

以下命令作为新数据集长课程训练模板：

```bash
cd /path/to/new_pod_periodic_v1

export SWANLAB_API_KEY="$(awk '$1 == "password" {print $2; exit}' /root/.swanlab/.netrc)"

/root/miniconda3/envs/pt_env/bin/python \
  code/run_new_pod_periodic_training.py \
  --trainer code/train_new_pod_periodic_moe.py \
  --source-checkpoint /path/to/config_template_checkpoint.pt \
  --data-root assets \
  --output-dir runs/long_seed1600 \
  --epochs 720 \
  --batch-size 1024 \
  --rollout-batch 56 \
  --rollout-batch-by-stage 56,56,40,28 \
  --rollout-updates-per-epoch 2 \
  --curriculum-steps 4,8,12,16 \
  --curriculum-stage-epochs 120,160,160,280 \
  --amp-mode bf16 \
  --batched-experts \
  --fixed-regime-group 1 \
  --no-dense-moe-training \
  --no-compile-model \
  --eval-every 10 \
  --swanlab-mode online
```

如果新数据不是纯周期数据，应移除固定 group 或传入通用值：

```text
--fixed-regime-group -1
```

实际运行前应先用较短课程和较小 batch 做 smoke test。

## 21. 恢复训练命令模板

恢复时必须保持数据、模型、损失和 split 不变：

```bash
/root/miniconda3/envs/pt_env/bin/python \
  code/run_new_pod_periodic_training.py \
  ...与原任务完全相同的参数... \
  --resume-checkpoint runs/long_seed1600/latest.pt
```

恢复后检查日志中：

- `resume_loaded` 的 epoch 是否正确
- scheduler 的 `T_max` 是否与目标总 epoch 一致
- 当前课程 K 是否正确
- 当前 rollout batch 是否正确
- `latest.pt` 是否继续更新
- SwanLab 是否进入预期 run

若发生 OOM，只降低当前及后续课程阶段的 rollout batch，并在报告中记录修改。

## 22. 最终物理场评估命令模板

```bash
/root/miniconda3/envs/pt_env/bin/python \
  code/evaluate_new_pod_physical_fields.py \
  --trainer code/train_new_pod_periodic_moe.py \
  --checkpoint runs/long_seed1600/best_validation.pt \
  --raw-dir staging/raw \
  --output-dir runs/long_seed1600/final_physical_best \
  --horizons 1 4 8 16 24 \
  --device cuda
```

评估结束后至少核对：

```text
physical_field_multihorizon.json 存在
每个 K 的 windows > 0
每个 K 的 nonfinite_windows = 0
heldout Re 与训练设定一致
checkpoint epoch 等于最佳验证 epoch
```

## 23. 代码修改边界

迁移第一版建议保持以下内容不变：

- MoE 主体结构
- residual RHS 定义
- pressure closure 定义
- RK4 闭环逻辑
- 损失组合
- checkpoint schema
- 系数空间评估逻辑
- 物理场误差公式

优先修改：

- 数据适配层
- 路径和文件名
- POD 秩
- split
- 课程长度和 batch
- 实验命名

只有在可靠基线完成后，再分别进行模型结构、损失权重或部分权重迁移的消融实验。这样可以区分问题究竟来自数据构造、数值积分还是网络能力。

## 24. 相关实验文档

CenteredSquare periodic 长课程训练的完整结果见：

`C:\Users\panxy1019\Documents\CHANNEL\.transfer_square_periodic\FINAL_EXPERIMENT_REPORT.md`
