# Hopf POD 数据集迁移、训练与 Rollout 评估复用指南

## 1. 文档目的

本文档用于把当前已经完成的 CenteredSquare Hopf `r_u=r_p=11` 实验迁移到新的 POD 数据集，并指导后续修改训练代码、重新训练、验证和测试。

本文以当前实际运行代码和数据合同为准，不沿用早期方案中未经运行时验证的固定维度。迁移时应先重建新数据集的 POD/ROM 资产，再对代码做参数化改造；不要只替换 NPZ 文件后直接训练。

当前建议是：先复现普通深层 FNN 基线 B1，确认新 POD 数据和数值闭环正确，再决定是否训练 KDA 分支 B2/B3。已有方柱实验中 B1 明显优于 KDA，因此 KDA 不应被默认视为更强模型。

---

## 2. 当前实验的真实合同

### 2.1 数据划分

当前方柱 Hopf 数据按完整 Reynolds 数工况划分：

| 集合 | 工况数 | 用途 |
|---|---:|---|
| train | 23 | 拟合 POD、ROM、归一化量并训练网络 |
| validation | 6 | 选择 checkpoint、比较模型、执行准入门禁 |
| held-out test | 5 | 仅在方法通过冻结门禁后进行一次最终评估 |

任何同一 Re 工况中的时间窗都只能属于一个集合。POD 均值、POD 基、Galerkin/Pressure–Poisson 算子、特征统计量、临界平面和 radial scale 必须只使用 train 工况拟合。

### 2.2 POD 与状态变量

记：

- `r_u`：速度 POD 阶数；
- `r_p`：压力 POD 阶数；
- `a_n ∈ R^(r_u)`：速度 POD 系数；
- `b_n ∈ R^(r_p)`：压力 POD 系数；
- `g_n ∈ R^(r_u)`：Galerkin RHS；
- `Re`：Reynolds 数；
- `dt_n`：相邻快照的物理时间间隔。

当前方柱实验使用 `r_u=r_p=11`。迁移后不能假定二者相等。

### 2.3 输入特征维度

当前特征构造为：

1. 归一化 `Re` 与 `1/Re`：2 维；
2. 当前 `a`：`r_u` 维；
3. 当前 `b`：`r_p` 维；
4. 当前 Galerkin RHS `g`：`r_u` 维；
5. 11 个能量、模态分段范数和比值描述量：11 维。

因此当前块维度为：

```text
current_dim = 13 + 2*r_u + r_p
```

每个显式历史块包含 `(a_h, b_h, g_h, a-a_h, b-b_h, g-g_h)`，因此：

```text
history_block_dim = 4*r_u + 2*r_p
```

若 `history_len=3`，即当前状态加两个历史块，则：

```text
h3_dim = current_dim + 2*history_block_dim
       = 13 + 10*r_u + 5*r_p
```

当前 `r_u=r_p=11` 时：

```text
current_dim = 46
history_block_dim = 66
h3_dim = 178
```

这些公式用于审计，但代码必须以运行时构造出的张量形状为最终依据。若新特征函数发生变化，应通过一次真实 batch probe 得到维度，不能强行套用上述数值。

---

## 3. 共同的物理与数值方法

### 3.1 Galerkin 特征

速度半离散算子为：

```text
g(a,b,Re) = c(Re) + A(Re)a + H:(a⊗a) + P b
```

其中 Reynolds 相关的 `c`、`A` 在相邻 train 节点间插值，`H` 和 `P` 来自与当前 POD 基严格匹配的 train-only ROM 资产。

### 3.2 速度推进合同

当前方柱数据的快照间隔较大，直接把连续 Galerkin RHS 作为 RK4 主干会失稳。因此正式实验采用：

```text
da/dt = rhs_mean + rhs_scale ⊙ F_theta(features, memory)
```

Galerkin RHS 仍作为物理输入特征，但不直接与网络输出相加。网络学习相邻快照的有限差分导数：

```text
(a_(n+1) - a_n) / dt_n
```

预测导数再通过 RK4 推进到 `a_(n+1)`。

迁移到新 POD 数据集时必须重新做数值稳定性试验：

- 比较“直接 Galerkin + 学习残差”和“直接学习有限差分导数”；
- 在原生 `dt` 下做至少 K56 的无训练/短训练 rollout；
- 若连续 Galerkin RK4 有爆炸或系统漂移，继续采用有限差分导数合同；
- 不得仅因为新数据也来自 CFD 就默认旧合同仍正确。

### 3.3 压力闭合

压力采用 Pressure–Poisson 基线、逐模态 gate 和学习残差：

```text
b_(n+1) = sigmoid(z_gate) ⊙ P_H(a_(n+1), Re)
          + rho_p ⊙ b_state_scale
```

压力不是独立做时间积分，而是在接受新的速度宏状态后代数更新。

---

## 4. 三个对照模型

### 4.1 B1：Deep-FNN-H3

B1 使用当前状态和两个显式历史状态组成的 `h3_dim` 输入。主干是 6 个宽度为 512 的线性层，使用 LayerNorm 和 SiLU；三个输出头分别预测：

- 速度导数标准化输出：`r_u` 维；
- 压力残差：`r_p` 维；
- 压力 gate：`r_p` 维。

B1 没有 MoE、router、Top-k 或动态 memory。它是迁移时必须首先跑通的强基线。

### 4.2 B2：KDA-PR-FNN-ROM，radial-off

B2 只读取 `current_dim` 当前特征，用固定尺寸矩阵 memory 代替显式历史拼接。

当前 KDA 配置：

```text
token_dim = 128
heads = 4
d_k = 16
d_v = 16
memory states = heads*d_k*d_v = 1024
beta_max = 0.5
```

对每个 head：

```text
tau = tau_min + softplus(tau_base + delta_tau(token))
alpha = exp(clamp(-dt/tau, -20, 0))
M_bar = Diag(alpha) M
e = v - M_bar^T k
beta = beta_max * (1 - exp(-dt*softplus(rate(token))))
M_new = M_bar + beta*k*e^T
```

速度和压力分别用 `q_u`、`q_p` 查询同一写入 memory：

```text
m_u = M^T q_u
m_p = M^T q_p
```

然后把 `[current_features, m_u, m_p]` 输入与 B1 同规模的深层 FNN。

Memory 生命周期必须保持如下约束：

- 每条轨迹开始时清零，禁止跨轨迹泄漏；
- 使用三个真实初始状态按时间顺序 warm-up；
- 自主 rollout 后只用预测状态更新，不能读取未来真值；
- 每个宏时间步最多更新一次；
- 同一宏步的四个 RK4 stage 内 memory 必须冻结；
- TBPTT 每 16 步 `detach()`，但不能清空 memory 数值；
- AMP 下网络可用 BF16，memory 和物理算子保持 FP32。

### 4.3 B3：KDA radial-on

B3 与 B2 完全相同，仅加入由 train-only 速度系数拟合的临界平面 radial 正则。它是消融实验，不是默认主模型。

---

## 5. 初始化与数值稳定性要点

### 5.1 速度头必须初始化为物理零更新

输出反标准化为：

```text
velocity = velocity_std*rhs_scale + rhs_mean
```

如果最后一层只做全零初始化，则初始物理输出等于 `rhs_mean`，长 rollout 会产生系统漂移。正确初始化为：

```python
velocity_head.weight.zero_()
velocity_head.bias.copy_(-rhs_mean / rhs_scale)
```

这样初始物理导数严格为零。

### 5.2 inverse-softplus 必须避免溢出

安全实现：

```python
def inverse_softplus(x):
    x = x.clamp_min(1e-6)
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))
```

若直接对大数执行 `log(expm1(x))`，KDA 的 `tau` 或 `beta` 初始化可能出现 `inf`。

### 5.3 精度策略

- FNN/encoder：CUDA BF16 autocast；
- Galerkin、Pressure–Poisson、RK4 状态和 KDA memory：FP32；
- 允许 TF32；
- 梯度裁剪：1.0；
- 每次正式训练前检查全部参数、loss、梯度和 rollout 状态均 finite。

---

## 6. 训练方法

### 6.1 优化器与预算

当前匹配实验协议：

```text
optimizer       = AdamW
learning rate   = 1e-3
weight decay    = 1e-4
scheduler       = CosineAnnealingLR
optimizer steps = 8000
gradient clip   = 1.0
seed            = 1248
AMP             = BF16
TF32            = on
```

### 6.2 Rollout curriculum

| optimizer step | 自主 rollout 长度 |
|---:|---:|
| 0–1599 | K4 |
| 1600–3199 | K8 |
| 3200–5199 | K16 |
| 5200–6799 | K32 |
| 6800–7999 | K56 |

迁移时先确认每条 train/validation 轨迹都有足够的连续窗口。如果新数据每条轨迹不足 59 个可用状态（3 个 warm-up 加 K56），需要按物理覆盖时长重新设计 horizon，而不是让采样器重复无效窗口。

### 6.3 损失函数

当前总损失包含：

- 标准化速度 POD 系数 MSE；
- 标准化有限差分动力学 MSE；
- 压力 POD 系数 MSE；
- 速度/压力物理场相对误差；
- rollout 轨迹误差；
- 速度系数能量误差；
- KDA memory 范数弱正则；
- B3 专用 radial 幅值与增长正则。

新数据迁移时先保持现有权重不变，建立可比基线。只有在 B1 已稳定、各损失量纲和数值范围经过审计后，才新开实验调整权重。

### 6.4 GPU 利用率调优

`torch.cuda.set_per_process_memory_fraction(0.94)` 只限制本进程可用显存比例，并不自动提高 GPU 算力利用率。推荐做法：

1. 先查看同卡是否有其他任务，不能抢占或杀死不属于本实验的进程；
2. 用 `--benchmark-steps` 在最大 horizon 下测试多个 micro-batch；
3. 从较小 batch 逐步翻倍，记录吞吐和峰值显存；
4. 保留约 5%–10% 显存余量，避免验证或 K56 临时峰值 OOM；
5. 选择吞吐接近饱和且无 OOM 的最大稳定 batch；
6. `grad_accum` 只在单步 batch 放不下时使用，它不能代替真实并行 batch 的吞吐测试。

当前方柱正式训练在独占 4090 上使用 `micro_batch=1536`。这只是当前数据和模型的实测值，新 POD 阶数改变后必须重新 benchmark。

### 6.5 SwanLab 监控

正式训练使用 SwanLab online 模式，至少记录：

- `train/total` 及各损失分量；
- 学习率、总梯度范数和各参数组梯度；
- 当前 curriculum horizon；
- 峰值 GPU memory；
- validation score、finite fraction、divergent windows；
- KDA 的 `alpha/beta/tau/memory norm/write error`。

确认 loss、梯度、GPU 显存、吞吐和 KDA 诊断稳定后，可以退出 SSH/终端监控；训练进程应由独立的后台会话或作业管理器继续运行。退出 SwanLab 网页或本地查看不等于停止训练，也不要对仍在运行的训练调用 `swanlab.finish()`。

---

## 7. 验证、checkpoint 选择与最终测试

### 7.1 训练中验证

- 每 400 optimizer steps 验证；
- 常规验证使用 K4/K8/K16；
- 进入长 horizon 阶段后每 800 steps 增加 K32/K56；
- 只在 `finite_fraction=1` 且 `divergent_windows=0` 时允许选择 best checkpoint；
- best score 由 K56 的 mean joint、terminal 和 worst-window 组成。

### 7.2 统一 rollout 评估

训练结束后，对每个模型使用同一套：

```text
horizons = K1, K2, K4, K8, K16, K32, K56
validation windows per Re = 最多 64 个等间距连续窗口
checkpoint = best_validation.pt
```

报告指标：

- 速度物理场相对 L2 误差；
- 压力物理场相对 L2 误差；
- joint mean；
- K56 terminal joint；
- worst-window；
- pressure drift；
- finite fraction 和发散窗口数；
- wall-clock、窗口步吞吐；
- KDA `tau/half-life/alpha/beta/norm/max singular/effective rank/write error/query cosine`。

### 7.3 held-out test 门禁

当前冻结门禁为：B2 相对 B1 至少在 `mean joint`、`terminal` 或 `worst-window` 中一项改善 10%，同时速度或压力均值不得恶化超过 5%。未通过门禁时：

- 不加载 held-out 系数；
- 不生成 test 指标；
- 不根据 test 结果回调超参数；
- 不允许事后修改门禁阈值。

迁移到全新 POD 数据集时，可以在训练前重新预注册门禁，但一旦开始正式实验就冻结。

---

## 8. 迁移新 POD 数据集时必须修改的代码

当前主要文件：

```text
code/train_kda_pr_fnn_rom.py
code/evaluate_validation_rollout.py
code/run_single_seed_pipeline.sh
CONFIG_MANIFEST.yaml
```

### 8.1 去除全局固定 rank 和维度

当前训练器顶部写死：

```python
RANK = 11
CURRENT_DIM = 46
H3_DIM = 178
```

应改为从资产 manifest 或命令行读取：

```python
r_u = manifest["ru"]
r_p = manifest["rp"]
current_dim = runtime_current_features.shape[1]
h3_dim = runtime_h3_features.shape[1]
```

然后逐项改造：

- `ClosureHeads(r_u, r_p)`：velocity 输出 `r_u`，pressure 和 gate 输出 `r_p`；
- `DeepFNNH3(h3_dim, r_u, r_p)`；
- `KDAFNN(current_dim, r_u, r_p, ...)`；
- 所有 `reshape(-1, RANK)` 按变量语义分别使用 `r_u` 或 `r_p`；
- radial plane 的最后一维必须等于 `r_u`；
- `phi_u.shape[1] == r_u`，`phi_p.shape[1] == r_p`；
- Galerkin 输出维度必须为 `r_u`，Pressure–Poisson 输出必须为 `r_p`。

不要只把 `RANK=11` 改成另一个单值；这样无法支持 `r_u != r_p`。

### 8.2 从 split manifest 读取工况

当前 `TRAIN_RE`、`VAL_RE`、`HELDOUT_RE` 在训练器和 evaluator 中均被写死。应统一由一个冻结的 split manifest 读取，并校验：

```text
train ∩ validation = ∅
train ∩ heldout = ∅
validation ∩ heldout = ∅
loaded coefficient view 只包含 train + validation
heldout 文件路径不出现在训练进程参数中
```

所有按 Re 求 radial center、POD 统计和窗口池的循环也必须改用 manifest 中的 train 列表。

### 8.3 替换并审计 POD/ROM 资产路径

运行脚本中的以下资产必须整套替换，不能混用旧 POD 基：

- train+validation coefficient view；
- train-only Galerkin ROM；
- train-only Pressure–Poisson ROM；
- asset manifest；
- train-only fluctuation/radial contract；
- 与新数据配套的 baseline trainer/feature builder。

每个资产应记录 SHA-256。需要验证 mesh 顺序、POD 均值、POD 基、系数、Galerkin 张量和压力张量来自同一数据版本。

### 8.4 参数化 feature builder

`normalized_current()` 不能继续截取固定的 `stats["x_mean"][:46]`，应使用运行时 `current_dim`。`fit_runtime()` 应同时 probe 当前块和历史块，并保存：

```text
ru, rp, current_dim, history_block_dim, h3_dim, history_len
```

建议让数据/feature builder 返回具名 shape contract，而不是由训练器猜测。

### 8.5 参数化训练协议

当前训练器还冻结了以下内容，迁移分支中需要从 config 读取并写回 checkpoint：

- experiment name；
- `max_steps=8000` 限制；
- curriculum `STAGES`；
- validation horizons；
- validation windows per Re；
- SwanLab project/group；
- B1/B2/B3 模型超参数。

正式比较时三种模型必须共享优化器步数、训练窗口池、验证窗口和随机种子。若新数据改变预算，应为新实验建立新 config，不要偷偷覆盖旧配置。

### 8.6 同步修改 evaluator

Evaluator 必须从 checkpoint/manifest 重建模型，不得再次写死 rank、维度或 Re 列表。加载 checkpoint 后至少检查：

```text
checkpoint schema/version 一致
checkpoint asset hashes 与本次评估资产一致
model state strict=True 加载成功
normalization stats 维度正确
validation view 中不存在 heldout Re
rollout horizons 均有合法连续窗口
```

### 8.7 radial contract

若新 POD 数据不处于 Hopf 邻域、无法稳定识别二维临界平面，先禁用 B3。若保留 B3，则 `mean_a`、`plane`、`radial_scale`、`radial_floor` 必须完全由 train 工况拟合，并适配新的 `r_u`。

---

## 9. 推荐的代码重构目标

建议把迁移分支重构为单一配置驱动，而不是继续增加数据集专用常量：

```yaml
dataset:
  name: NEW_POD_DATASET
  coefficient_view: /path/to/trainval_coefficients.npz
  galerkin: /path/to/trainonly_galerkin.npz
  pressure: /path/to/trainonly_pressure.npz
  asset_manifest: /path/to/asset_manifest.json
  split_manifest: /path/to/split_manifest.json
  fluctuation_contract: /path/to/fluctuation_contract.npz

model:
  history_len: 3
  width: 512
  linear_layers: 6
  token_dim: 128
  heads: 4
  d_k: 16
  d_v: 16
  beta_max: 0.5

training:
  seed: 1248
  max_steps: 8000
  micro_batch: AUTO_BENCHMARK
  grad_accum: 1
  amp_dtype: bfloat16
  curriculum:
    - [0, 1600, 4]
    - [1600, 3200, 8]
    - [3200, 5200, 16]
    - [5200, 6800, 32]
    - [6800, 8000, 56]
```

Checkpoint 中应保存完整 config、资产 hash、运行时 shape contract、归一化统计量、best score、best step 和 held-out 状态。

---

## 10. 新数据迁移的执行顺序

### 阶段 A：数据与资产审计

- [ ] 固定 train/validation/held-out Re 列表；
- [ ] 检查完整工况隔离，无跨集合时间窗；
- [ ] 仅使用 train 拟合速度/压力 POD；
- [ ] 生成与新 POD 基匹配的 Galerkin 和 Pressure–Poisson 资产；
- [ ] 检查所有数组 finite、维度和 mesh 顺序一致；
- [ ] 生成 SHA-256 manifest；
- [ ] 建立不含 held-out 的 coefficient view。

### 阶段 B：代码参数化

- [ ] 去除 `RANK/CURRENT_DIM/H3_DIM` 固定常量；
- [ ] 支持 `r_u != r_p`；
- [ ] split、curriculum 和路径由 config/manifest 提供；
- [ ] trainer 与 evaluator 使用同一 shape contract；
- [ ] 保留旧实验目录只读，新建独立输出目录。

### 阶段 C：CPU/CUDA smoke test

- [ ] runtime shape probe 通过；
- [ ] K4、K8 前向/反向全部 finite；
- [ ] checkpoint strict round-trip 通过；
- [ ] KDA memory 每轨迹 reset；
- [ ] warm-up 更新次数正确；
- [ ] RK4 stage 内 memory 不变；
- [ ] BF16 下 memory 仍为 FP32；
- [ ] 相同 reset 和输入得到确定性结果；
- [ ] 没有未来真值进入模型输入。

### 阶段 D：吞吐 benchmark

- [ ] 确认 GPU 上其他任务和可用显存；
- [ ] 在最长 horizon 上逐级增加 micro-batch；
- [ ] 记录吞吐、显存、GPU 利用率和稳定性；
- [ ] 选取保留安全余量的最大稳定 batch。

### 阶段 E：正式训练

- [ ] 先训练 B1；
- [ ] B1 validation K56 稳定后再训练 B2；
- [ ] 仅在需要 radial 消融时训练 B3；
- [ ] SwanLab 记录完整；
- [ ] 保存 `latest.pt`、`best_validation.pt`、`final_training.pt`；
- [ ] 不读取 held-out。

### 阶段 F：最终评估

- [ ] 用统一 evaluator 比较 K1–K56；
- [ ] 输出逐 Re、逐 horizon 和逐时间步误差；
- [ ] 输出发散/finite 检查；
- [ ] KDA 输出 memory 诊断；
- [ ] 按预注册门禁决定是否解封 held-out；
- [ ] held-out 只评估一次并单独生成报告。

---

## 11. 命令模板

先在独立输出目录执行 smoke test：

```bash
/root/miniconda3/envs/pt_env/bin/python code/train_kda_pr_fnn_rom.py \
  --variant b1 \
  --baseline-trainer /path/to/new_baseline_trainer.py \
  --coefficient-view /path/to/new_trainval_coefficients.npz \
  --galerkin-path /path/to/new_trainonly_galerkin.npz \
  --pressure-path /path/to/new_trainonly_pressure.npz \
  --asset-manifest /path/to/new_asset_manifest.json \
  --fluctuation-contract /path/to/new_fluctuation_contract.npz \
  --output-root /path/to/new_experiment/smoke \
  --experiment-name NEW_B1_SMOKE \
  --smoke-only \
  --swanlab-mode disabled
```

完成参数化改造后，用最长 horizon 做短 benchmark，再确定 batch：

```bash
/root/miniconda3/envs/pt_env/bin/python code/train_kda_pr_fnn_rom.py \
  ...相同数据参数... \
  --variant b1 \
  --benchmark-steps 50 \
  --benchmark-horizon 56 \
  --micro-batch 256 \
  --swanlab-mode disabled
```

正式训练时再启用 SwanLab：

```bash
/root/miniconda3/envs/pt_env/bin/python code/train_kda_pr_fnn_rom.py \
  ...已验证的数据参数... \
  --variant b1 \
  --max-steps 8000 \
  --micro-batch <benchmark得到的稳定值> \
  --amp --amp-dtype bfloat16 --allow-tf32 --fused-adamw \
  --swanlab-mode online \
  --swanlab-project <新项目名> \
  --swanlab-group <新数据集分组>
```

注意：当前原始脚本仍包含固定 rank、Re 列表和实验名检查，以上模板只有在完成第 8 节参数化修改后才能直接用于任意新 POD 数据集。

---

## 12. 当前方柱实验结果及对迁移的启示

当前统一 K56 validation rollout 结果：

| 模型 | 速度均值 | 压力均值 | 联合均值 | 终端误差 | 最坏窗口 | 压力漂移 |
|---|---:|---:|---:|---:|---:|---:|
| B1 Deep-FNN-H3 | 0.0205% | 0.0170% | 0.0376% | 0.0501% | 0.2376% | 0.5286% |
| B2 KDA radial-off | 0.0409% | 0.0322% | 0.0731% | 0.1276% | 0.6403% | 0.7733% |
| B3 KDA radial-on | 0.0358% | 0.0306% | 0.0664% | 0.0965% | 0.6303% | 0.9096% |

B2 未通过相对 B1 的 validation 门禁，因此 held-out test 没有加载。B3 虽部分改善 B2 的长期误差，仍未超过 B1。KDA memory 没有爆炸或完全失效，但有效秩约为 2.9，形成的长记忆没有转化为更好的 validation 泛化。

因此迁移到新 POD 数据集时的默认优先级是：

1. 先验证数据、ROM 和时间推进合同；
2. 先训练 B1，建立强而简单的基线；
3. B1 不足时再验证 KDA 是否解决了明确的历史依赖问题；
4. 不建议在 KDA 未超过 B1 前直接叠加 Transformer 或 full attention。

---

## 13. 现有实现与证据文件

- 训练器：`code/train_kda_pr_fnn_rom.py`
- rollout evaluator：`code/evaluate_validation_rollout.py`
- 单 seed pipeline：`code/run_single_seed_pipeline.sh`
- 方法合同：`METHOD_KDA_PR_FNN_ROM.md`
- 代码审计：`CODE_AUDIT.md`
- smoke test 报告：`SMOKE_TEST_REPORT.md`
- 最终 rollout 报告：`rollout_evaluation_validation/FINAL_ROLLOUT_COMPARATIVE_REPORT.md`
- 聚合结果：`rollout_evaluation_validation/ROLLOUT_SUMMARY.json`
- 逐组合结果：`rollout_evaluation_validation/ROLLOUT_RESULTS.csv`
- 误差增长曲线数据：`rollout_evaluation_validation/ERROR_GROWTH_CURVES.csv`
- KDA memory 诊断：`rollout_evaluation_validation/MEMORY_DIAGNOSTICS.csv`

这份指南描述的是当前已验证实现及下一次迁移所需的重构边界。真正迁移时，应先复制到新的实验分支并保留当前方柱工程冻结不动。
