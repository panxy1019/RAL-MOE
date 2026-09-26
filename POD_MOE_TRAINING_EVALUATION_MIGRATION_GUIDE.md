# POD–Galerkin–MoE 训练、评估与新数据集迁移指南

## 1. 文档目的

本文总结 CenteredSquare steady specialist 实验使用的训练与评估方法，并给出迁移到新 POD 数据集时的代码复用方案。

迁移的核心不是直接加载旧 checkpoint，而是复用以下完整流程：

1. 严格的数据划分合同；
2. 仅由训练集拟合的加权 POD；
3. 面向新几何、新网格重新构造的 Galerkin ROM 与压力代理；
4. 学习物理模型残差的 `OperatorSpaceMoEROM`；
5. 从短到长的 autonomous rollout 课程训练；
6. validation 选 checkpoint、heldout 只做最终测试的评估合同。

只要几何、网格、物理变量、POD 基或 POD 阶数发生变化，就应重新构造数据资产和 ROM。通常不应直接迁移旧模型权重。

---

## 2. 当前可复用代码

本地参考代码位于：

```text
centeredsquare_steady_code/
├── train_s2b_3090.py
├── train_v16_4_v2_r32_compat.py
├── training_centeredsquare_steady_rank999.json
└── evaluate_centeredsquare_heldout.py
```

各文件职责如下：

| 文件 | 职责 | 迁移建议 |
|---|---|---|
| `train_s2b_3090.py` | 资产审计、数据缓存、模型构建、训练、validation、checkpoint、SwanLab | 作为新任务的主训练入口复用并修改 |
| `train_v16_4_v2_r32_compat.py` | POD-ROM 数组构造、Galerkin/压力算子、MoE 网络及底层 rollout 函数 | 尽量保持稳定；仅在资产 schema 或物理方程变化时修改 |
| `training_centeredsquare_steady_rank999.json` | 数据路径、阶数、网络结构、优化器、课程训练和监控配置 | 复制为新实验配置并逐项修改 |
| `evaluate_centeredsquare_heldout.py` | 冻结 checkpoint、clean heldout rollout、物理场误差及报告生成 | 复制为新数据集评估脚本，去除方柱硬编码 |

建议新建独立目录，不要直接覆盖方柱实验：

```text
new_pod_specialist_v1/
├── code/
├── configs/
├── source_artifacts/
├── runs/
├── checkpoint/
└── reports/
```

---

## 3. 数据划分合同

### 3.1 按完整工况划分

必须按完整参数工况或完整轨迹划分：

- `train`：拟合 POD、归一化量并训练网络；
- `validation`：早停、超参数比较和 checkpoint 选择；
- `heldout/test`：checkpoint 冻结后只执行一次最终评估。

禁止把同一个 Re、参数组合或轨迹中的时间帧随机分散到多个 split。三个 split 的工况标签必须零交集。

### 3.2 序列索引合同

每个快照至少要有：

- `sample_id`；
- 工况标签，例如 Re；
- split；
- 时间或时间步编号；
- `next_idx`；
- `dt_next`；
- 历史索引 `hist_idx`。

生成 K 步窗口时，每一步都必须满足：

1. `next_idx >= 0`；
2. 起点和后续快照属于同一工况；
3. 起点和后续快照属于同一 split；
4. 不允许跨工况、跨文件或跨轨迹连接。

---

## 4. POD 构造

### 4.1 训练集专属 POD

均值场、POD 基、归一化统计量只能由 train split 计算。validation 和 heldout 必须使用冻结后的训练集 POD 基投影。

速度采用体积加权内积：

$$
\langle u,v\rangle_W=u^\mathsf{T}Wv.
$$

速度系数为：

$$
a_t=\Phi_u^\mathsf{T}W(u_t-\bar u).
$$

压力先统一 gauge。本次 steady 实验采用每个快照减去体积加权空间均值：

$$
p_t'=p_t-\frac{\mathbf 1^\mathsf{T}Wp_t}
{\mathbf 1^\mathsf{T}W\mathbf 1}.
$$

随后计算压力系数：

$$
b_t=\Phi_p^\mathsf{T}W(p_t'-\bar p).
$$

### 4.2 POD 阶数

速度阶数 $r_u$ 和压力阶数 $r_p$ 应独立选择，不得假定二者相等。本次方柱基线使用：

```text
r_u = 5
r_p = 4
```

迁移时根据新训练集的累计能量重新选择，例如同时记录 99% 和 99.9% 能量阶数。不要硬编码旧数据集的阶数，也不要在代码中用 `r_u` 代替 `r_p`。

### 4.3 POD 必做审计

- POD 资产中的 `fit_split` 必须为 `train`；
- 压力资产的 `pressure_gauge` 必须与配置一致；
- 加权正交误差足够小；
- 投影—重构—再投影误差足够小；
- 快照标签顺序和 ROM 数组中的标签顺序一致；
- POD 基的可用阶数不少于配置中的 $r_u,r_p$；
- validation/heldout 未参与均值、POD 或 scaler 拟合。

---

## 5. 数据资产接口

当前主训练脚本默认读取以下资产：

```text
source_artifacts/<regime>/
├── velocity_pod_<regime>.npz
├── pressure_pod_<regime>.npz
├── normalization_<regime>.npz
├── velocity_rom_<regime>.npz
├── pressure_poisson_surrogate_<regime>.npz
└── Global_POD_AreaWeighted_L2/
    └── pod_snapshot_index.csv
```

当前 steady 代码至少依赖以下字段或等价信息：

| 资产 | 关键内容 |
|---|---|
| velocity POD | `phi_uv`、`mean_uv_regime`、`snapshot_splits`、`snapshot_Re_labels`、`fit_split` |
| pressure POD | `phi_p`、`mean_p_regime`、`fit_split`、`pressure_gauge` |
| normalization | `velocity_coeff_std`、`pressure_coeff_std`、`fit_split` |
| velocity ROM | `r_u`、`r_p` 以及 Galerkin 所需常数、线性、二次和压力耦合算子 |
| pressure surrogate | 压力 Poisson/代理所需常数、线性和二次算子 |
| snapshot index | 快照顺序、标签、split、前后继关系及时间步信息 |

如果新数据的键名不同，优先编写一个资产转换脚本，把新数据转换成上述统一 schema；这样可以减少对训练主循环的修改。

---

## 6. 半侵入式物理 ROM

本方法不是纯粹的“当前系数到下一系数”黑箱预测，而是先计算物理基线，再由神经网络学习修正量。

速度 Galerkin ROM 的一般形式为：

$$
\dot a=c(Re)+A(Re)a+H(a,a)+Pb.
$$

若黏性项依赖 $\nu=1/Re$，可将算子写成关于 $1/Re$ 的仿射形式。

压力使用压力 Poisson 或压力代理：

$$
b_{\mathrm{phy}}
=\widetilde c(Re)+\widetilde A(Re)a+\widetilde H(a,a).
$$

对于新几何、新网格、新边界条件或不同物理变量，必须重新投影并构造这些算子，不能直接复用圆柱或方柱的 ROM 张量。

ROM 构造后至少验证：

- 所有张量维度与 $r_u,r_p$ 一致；
- 参数仿射重构残差；
- Galerkin RHS 全部 finite；
- 单步物理积分不立即发散；
- ROM 使用的 POD 基、均值、压力 gauge 与训练资产完全一致。

---

## 7. 神经网络与自回归更新

### 7.1 网络输入

`OperatorSpaceMoEROM` 的输入由以下信息组合：

- 当前参数，例如 Re 或 $1/Re$；
- 参数或 phase 的 harmonic 编码；
- 当前速度 POD 系数 $a_t$；
- 当前压力 POD 系数 $b_t$；
- 当前 Galerkin RHS；
- 最近若干步历史，本次为 `history_len=3`；
- 底层兼容训练器构造的其他物理特征。

### 7.2 网络输出

网络学习两个修正量：

$$
\delta\dot a_t,\qquad \delta b_t.
$$

自回归更新为：

$$
a_{t+1}=a_t+\Delta t\left[
f_{\mathrm{Galerkin}}(a_t,b_t,Re)+\delta\dot a_t
\right],
$$

$$
b_{t+1}=b_{\mathrm{phy}}(a_{t+1},Re)+\delta b_t.
$$

然后把预测的 $a_{t+1},b_{t+1}$ 写入历史缓存，继续下一步 rollout。

**注意：模型的原始 forward 输出不是下一时刻 POD 状态，不能把它直接当作 $a_{t+1},b_{t+1}$。**

### 7.3 当前网络基线

```text
history_len                 = 3
phase_harmonics             = 4
hidden_dim                  = 224
num_blocks                  = 3
num_regime_groups           = 3
experts_per_group           = 6
shared_experts_per_group    = 1
top_k                       = 2
group_top_k                 = 1
expert_hidden               = 768
expert_blocks               = 3
quadratic_rank              = 4
quadratic_scale             = 0.05
dropout                     = 0.04
router_temperature          = 0.95
group_temperature           = 0.9
shared_scale                = 1.0
routed_scale                = 0.85
```

该结构可作为迁移基线，但对低阶 POD 可能偏大。建议保持评估合同不变，同时增加较小网络作为对照实验。

---

## 8. 损失函数

训练目标由残差监督、状态锚点、多步 rollout 和必要的路由正则组成：

$$
\mathcal L=
\mathcal L_{\mathrm{residual}}
+\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}
+\lambda_{\mathrm{roll}}\mathcal L_{\mathrm{rollout}}
+\mathcal L_{\mathrm{router/reg}}.
$$

- `residual loss`：监督速度导数修正和压力代理修正；
- `anchor loss`：约束下一步 POD 状态，避免积分漂移；
- `rollout loss`：约束连续 autonomous rollout；
- 压力和速度分别使用训练集统计量缩放。

本次配置为：

```text
rollout_pressure_weight = 2.0
rollout_gradient_ratio  = 0.1
anchor_gradient_ratio   = 0.05
```

所有 scaler 只能由 train split 中有效的训练窗口拟合，并在 validation 和 heldout 阶段冻结。

---

## 9. 训练策略

### 9.1 零输出初始化

残差专家的最后输出层初始化为零，使训练从物理 ROM 基线开始：

```text
learned residual = 0
initial prediction = physical ROM prediction
```

这能减小训练初期神经修正破坏物理基线的风险。

### 9.2 Rollout 课程训练

| 阶段 | Step | Horizon | 说明 |
|---|---:|---:|---|
| ONE | 1–400 | K1 | 单步预热，不启用 rollout loss |
| K4 | 401–1200 | K4 | 短期自回归 |
| K8 | 1201–2400 | K8 | 中期自回归 |
| K16 | 2401–4000 | K16 | 长期自回归 |

阶段切换使用约 200 step 的权重 ramp，避免损失突然变化。

### 9.3 优化器配置

```text
optimizer                    = AdamW
learning_rate                = 1.5516372391099407e-5
shared/router_lr_factor      = 0.1
weight_decay                 = 1e-4
grad_clip                    = 1.0
micro_batch                  = 16
gradient_accumulation        = 4
effective_batch              = 64
validation_every             = 200
early_stop_start             = 3200
patience_validations         = 4
relative_improvement         = 1%
maximum_steps                = 4000
```

4090 上优先测试 `(micro_batch, grad_accum)`：

```text
(16, 4), (8, 8), (4, 16)
```

保持 effective batch 为 64，并选择显存安全且吞吐最高的组合。

### 9.4 速度修正头冻结策略

方柱 steady 实验冻结了：

```text
velocity_expert_groups*
velocity_shared_experts*
```

编码器、refine blocks 和 router 使用 `0.1 × base_lr`，其余可训练部分使用基础学习率。因此方柱速度预测主要依赖 Galerkin 基线，网络重点学习压力修正。

迁移时不要机械复制这一策略：

- 如果纯 Galerkin 速度 rollout 已较准确，可继续冻结速度头；
- 如果速度误差随 K 明显增长，应解冻速度头；
- 推荐先冻结预热，再以较小学习率解冻，作为独立实验比较；
- trainable parameter audit 必须写入启动清单和报告。

---

## 10. Validation 与 checkpoint 合同

训练过程中每隔固定 step 对 validation 工况执行 clean autonomous rollout：

```text
K1, K4, K8, K16
```

不能使用 teacher forcing，也不能在中间时刻注入真实 POD 系数。

每个 horizon 至少记录：

- velocity POD relative error；
- pressure POD relative error；
- 每个工况及最坏工况误差；
- `finite_fraction`；
- `divergent_windows`；
- `pressure_drift`；
- `fixed_point_residual`；
- controlled `contraction_ratio`。

本次严格资格门为：

```text
K16 finite_fraction == 1
K16 divergent_windows == 0
K16 contraction_ratio < 1
```

通过资格门后，按如下顺序比较候选 checkpoint：

```text
(worst_pressure, worst_velocity, fixed_point_residual)
```

若没有 checkpoint 通过资格门，最终状态应写成：

```text
NO_QUALIFIED_CHECKPOINT
```

可以冻结 final checkpoint 做一次 heldout 性能刻画，但必须标记为 `FROZEN_UNQUALIFIED_CANDIDATE` 或等价表述，不能称为合格部署模型。

### 关于 contraction 指标

当 steady POD 系数非常接近零时，相对扰动归一化会使 contraction ratio 极大。迁移时建议同时报告：

- 原有相对 contraction，保持历史可比性；
- POD 绝对扰动增长；
- 物理场加权扰动增长；
- 分母下限和扰动幅值。

在验证合同确定后不要因为 heldout 结果而临时修改门槛。

---

## 11. Heldout 最终评估

### 11.1 checkpoint 冻结

先根据 validation 冻结 checkpoint，再开始 heldout。评估脚本必须记录：

- checkpoint 路径；
- step；
- SHA256；
- validation 选择证据；
- `heldout_used_for_selection = false`。

### 11.2 clean autonomous rollout

建议执行：

```text
K1, K4, K8, K16, K32, K64
```

若序列足够长，再增加完整窗口。所有后续状态都必须来自模型自身预测。

### 11.3 POD 空间误差

$$
e_a=\frac{\|\hat a-a\|_2}{\|a\|_2},
\qquad
e_b=\frac{\|\hat b-b\|_2}{\|b\|_2}.
$$

低能 POD 分量可能导致相对系数误差很大，因此不能只看该指标。

### 11.4 投影物理场误差

把预测系数和真实投影系数都重构到 POD 子空间后比较，可隔离动力学预测误差。

### 11.5 完整 CFD 物理场误差

预测速度场为：

$$
\hat u=\bar u+\Phi_u\hat a.
$$

压力场按训练时相同的 gauge 和均值定义重构。然后与原始 CFD 快照直接比较：

$$
e_u=\frac{\|\hat u-u_{\mathrm{CFD}}\|_W}
{\|u_{\mathrm{CFD}}\|_W}.
$$

完整 CFD 误差包含：

- POD 截断误差；
- POD 系数预测误差；
- 两者的交叉项。

最终报告应以完整 CFD 物理场误差为主，以 POD 系数误差和 projected physical error 为辅。

---

## 12. 迁移时需要修改的代码

### 12.1 配置文件：必须修改

复制 `training_centeredsquare_steady_rank999.json`，至少修改：

```json
{
  "experiment": "NewDataset-Steady-S2B",
  "vendor_trainer": "/absolute/path/to/train_v16_4_v2_r32_compat.py",
  "artifact_dir": "/absolute/path/to/source_artifacts/steady",
  "checkpoint_root": "/absolute/path/to/runs/baseline/checkpoints",
  "seed": 202608061,
  "r_u": 0,
  "r_p": 0,
  "pressure_gauge": "replace_with_new_contract",
  "expected_split_re_counts": {
    "train": 0,
    "validation": 0,
    "heldout": 0
  },
  "expected_train_snapshots": 0
}
```

其中 `0` 均为占位值，必须替换为新数据的真实统计。

### 12.2 `train_s2b_3090.py`：优先参数化

需要检查或修改：

1. `self.regime = "steady"` 是否仍适用；
2. `self.mechanism` 是否仍使用压力 fixed-point anchor；
3. 资产文件名与目录布局；
4. 工况标签是否仍叫 Re；
5. split 名称是否为 `train/validation/heldout`；
6. 训练窗口是否可能跨多条独立轨迹；
7. 模型输入特征是否仍包含 Re、phase 和相同历史结构；
8. 速度头是否冻结；
9. validation horizons 和资格门；
10. SwanLab project、group 和 experiment 名称。

建议把当前硬编码项移入 JSON：

- `regime`；
- `mechanism`；
- validation horizons；
- checkpoint qualification gate；
- 是否冻结 velocity experts；
- 标签字段名；
- 资产文件名；
- expected split 标签列表或哈希。

### 12.3 `train_v16_4_v2_r32_compat.py`：谨慎修改

只有以下情况才需要改底层兼容训练器：

- 新资产字段无法通过转换脚本适配；
- 物理变量数量变化；
- Galerkin 方程形式变化；
- 参数不再是 Re 或不再满足 $1/Re$ 仿射关系；
- 新数据没有 pressure surrogate；
- 输入特征或 phase 定义发生本质变化。

修改时重点检查：

- `build_arrays(...)`；
- `build_galerkin_torch(...)`；
- `build_pressure_surrogate_torch(...)`；
- `OperatorSpaceMoEROM` 的 `in_dim/out_dim/pressure_dim`；
- rollout 更新顺序；
- 所有涉及 $r_u,r_p$ 的切片和扰动向量；
- history index 构造。

### 12.4 `evaluate_centeredsquare_heldout.py`：必须去硬编码

复制后至少修改：

- 文件名和类名中的 `centeredsquare`；
- heldout 工况列表；
- 原始 heldout CFD 文件匹配规则；
- 网格权重、速度分量和压力字段读取方式；
- pressure gauge；
- 报告中的固定 Re、案例数和结论文字；
- K16 是否一定存在；
- checkpoint 冻结文件名；
- qualification gate 的读取方式。

推荐让评估脚本从资产审计 JSON 和训练配置自动读取 heldout 标签，避免在代码中写死：

```python
heldout_labels = sorted(exp.windows["heldout"])
```

报告也应根据 `heldout_labels` 动态生成，而不是写死具体 Re。

---

## 13. 推荐新增的自动审计

迁移后建议在正式训练前增加以下断言：

```text
[ ] train、validation、heldout 工况零交集
[ ] 每个快照只出现一次
[ ] next_idx 不跨标签、不跨 split、不跨轨迹
[ ] dt_next 为正且单位一致
[ ] POD、normalization 的 fit_split == train
[ ] pressure gauge 完全一致
[ ] r_u 和 r_p 分别通过维度检查
[ ] POD 快照标签顺序与 build_arrays 输出一致
[ ] ROM rank 不小于训练 rank
[ ] Galerkin 与压力代理输出全部 finite
[ ] 零残差模型可完成 K1/K4 smoke rollout
[ ] 模型输出维度分别为 r_u 和 r_p
[ ] 保存—加载 checkpoint 后模型哈希一致
[ ] resume 后 step、optimizer、scheduler 和 early-stop 状态一致
[ ] heldout 未进入 scaler、sampler、early stopping 或 selector
```

特别注意两个常见程序错误：

1. 默认假设 `r_u == r_p`，导致压力切片或扰动向量维度错误；
2. 为输出能量统计而硬编码 rank 16/32，在低阶 POD 上产生越界或错误报告。

所有 rank 相关逻辑都应写成：

```python
ru = int(config["r_u"])
rp = int(config["r_p"])
```

并分别限制到真实可用阶数。

---

## 14. 推荐运行顺序

以下命令为模板，路径需要替换为新实验的绝对路径。

### 14.1 资产审计

```bash
source activate pt_env
python code/train_s2b_3090.py \
  --config configs/training_new_dataset.json \
  --run-dir runs/baseline \
  --mode audit
```

### 14.2 GPU preflight

```bash
python code/train_s2b_3090.py \
  --config configs/training_new_dataset.json \
  --run-dir runs/baseline \
  --mode preflight
```

preflight 应验证：

- CUDA、AMP dtype 和显存；
- effective batch；
- 单步 forward/backward；
- K1/K4 rollout；
- checkpoint 保存和加载；
- 训练循环中无磁盘读取瓶颈。

### 14.3 正式训练

```bash
python code/train_s2b_3090.py \
  --config configs/training_new_dataset.json \
  --run-dir runs/baseline \
  --mode train
```

### 14.4 从 latest checkpoint 恢复

```bash
python code/train_s2b_3090.py \
  --config configs/training_new_dataset.json \
  --run-dir runs/baseline \
  --resume runs/baseline/checkpoints/latest.pt \
  --mode train
```

恢复后应确认：

- step 连续；
- optimizer 和 scheduler 已恢复；
- curriculum horizon 与 step 对应；
- 没有重新拟合 scaler；
- SwanLab 使用恢复或明确的新 run 语义；
- 没有覆盖已冻结的最终 checkpoint。

### 14.5 最终 heldout 测试

```bash
python code/evaluate_new_dataset_heldout.py \
  --root /absolute/path/to/new_pod_specialist_v1 \
  --checkpoint /absolute/path/to/frozen_checkpoint.pt \
  --raw-heldout-dir /absolute/path/to/raw_heldout \
  --output-dir /absolute/path/to/reports/final \
  --horizons 1 4 8 16 32 64 \
  --batch-size 64
```

---

## 15. SwanLab 监控建议

训练阶段至少记录：

- 当前 step、curriculum 阶段和 rollout K；
- learning rate；
- total/residual/anchor/rollout loss；
- velocity 和 pressure loss；
- gradient norm；
- GPU 显存和 step time；
- router/expert 使用率；
- validation K1/K4/K8/K16 指标；
- finite、divergence、pressure drift、fixed point、contraction；
- checkpoint qualification 状态。

稳定运行一段时间后可以停止人工高频监控，但训练脚本仍应持续记录日志和保存 `latest.pt`。建议保留低频健康检查，直至写出 `DONE.json` 或 `FAILED.json`。

---

## 16. 新数据集迁移决策

### 可以直接复用

- 工况级 split 合同；
- 训练集专属 POD/normalization 原则；
- `OperatorSpaceMoEROM` 主体结构；
- 物理残差更新方式；
- 零输出初始化；
- K1→K4→K8→K16 课程训练；
- checkpoint、SwanLab、断点恢复框架；
- clean rollout 和完整物理场误差评估框架。

### 必须重新生成

- POD 均值与基；
- POD 系数；
- 训练集归一化统计量；
- snapshot index 和序列关系；
- velocity Galerkin ROM；
- pressure Poisson/pressure surrogate；
- 原始物理场重构接口；
- 新实验 checkpoint。

### 必须重新决定

- $r_u,r_p$；
- pressure gauge；
- 是否仍使用 Re 和 $1/Re$ 仿射算子；
- history length；
- 网络容量；
- 速度修正头是否冻结；
- rollout 最大长度；
- 压力损失权重；
- qualification gate；
- validation 和 heldout 工况。

---

## 17. 最终验收清单

### 数据与资产

- [ ] split 按完整工况划分且零泄漏；
- [ ] POD/normalization 仅由训练集拟合；
- [ ] 压力 gauge、单位和网格权重一致；
- [ ] POD 重投影和 ROM 重构审计通过；
- [ ] 所有资产保存 schema、版本和 SHA256。

### 训练

- [ ] audit 通过；
- [ ] GPU preflight 通过；
- [ ] effective batch 已记录；
- [ ] 训练参数及冻结参数审计已记录；
- [ ] K1→K4→K8→K16 正常切换；
- [ ] latest checkpoint 可恢复；
- [ ] SwanLab 日志完整；
- [ ] 训练最终写出 DONE/FAILED 状态。

### checkpoint 选择

- [ ] 只使用 validation；
- [ ] qualification gate 在 heldout 前冻结；
- [ ] checkpoint 路径、step、SHA256 已记录；
- [ ] 无合格 checkpoint 时明确标记 unqualified。

### 最终评估

- [ ] heldout 从未参与训练、早停或选权重；
- [ ] 执行 K1/K4/K8/K16，条件允许时执行 K32/K64；
- [ ] 全部为 clean autonomous rollout；
- [ ] 报告 finite/divergence/drift/fixed-point/contraction；
- [ ] 同时报告 POD、projected physical 和 full CFD physical error；
- [ ] 分工况报告最坏结果；
- [ ] 报告 POD 截断误差下限；
- [ ] 最终结论与资格门一致，不夸大稳定性。

---

## 18. 一句话迁移原则

**复用模型框架、训练课程和评估合同；重新生成训练集 POD、物理 ROM、数据索引与 checkpoint；任何 heldout 信息都不能反向影响训练或 checkpoint 选择。**
