# CTDM-Galerkin-ROM：方法、验证与最终比较报告

## 1. 执行摘要

本实验在冻结的 Hopf POD--Galerkin ROM 工程上，比较固定三状态历史、仅当前状态、离散矩阵记忆和连续时间矩阵记忆。核心结论如下。

1. 离散 KDA 在给定尺度关系下确实以一阶精度收敛到
   \(\dot S=-\Gamma S+\eta k(v-S^\top k)^\top\)，而不是题面最初写出的负号写入项。负号版本的误差不随步长趋零。
2. 连续 memory 的独立 RK4 测试达到四阶；在模型级同一物理查询时刻比较中，B4 相对 B3 的 \(C_u,C_p,C_S\) 分别降低约 83.7%、77.0%、54.9%。
3. 这项时间步一致性优势没有转化为更好的 K56 长时预测。严格配对的 seed 1248 中，B4 相对 B3 的平均物理场联合误差 \(E_u+E_p\) 恶化 7.58%，worst-window 联合误差恶化 63.95%。
4. B4 的 seed 2248 和 3248 分别在第 1541、3958 步出现非有限梯度；仅 seed 1248 完成。CTDM 的完成率为 1/3，不能形成三-seed性能估计。
5. B1（普通 FNN + 固定三状态历史）是本轮最优且稳定的模型。三-seed平均 \(E_u=0.0236\%\)、\(E_p=0.0749\%\)、worst-window=0.1707%，所有窗口有限且零发散。
6. B2 说明当前状态本身已能得到稳定结果，但删除历史使 worst-window 相对 B1 增加约 91.4%。历史信息有用；本轮矩阵记忆并未提供更好的替代。
7. 预注册门控结果为 `test_access_authorized=false`。held-out test 张量没有被打开，`TEST_RESULTS.csv` 为零字节。本报告只给出 validation 结论。

因此，本轮核心假设未获支持：连续矩阵记忆改善了时间离散一致性，但预测精度和跨 seed 训练稳定性均不足，暂不应进入振荡生成元研究。

## 2. 研究问题与比较设计

本轮只验证一个问题：将离散 KDA 按

\[
D_n=\exp(-\Delta t_n\Gamma_n),\qquad
\beta_n=1-\exp(-\eta_n\Delta t_n)
\]

重新参数化为连续时间矩阵记忆，并与 POD--Galerkin 速度状态联合积分，是否比固定三状态历史和公平的离散 KDA 更适合 Hopf K56 长时预测。

比较模型为：

| 编号 | 模型 | 历史/记忆 |
|---|---|---|
| B0 | 冻结 HPRS-MoE-ROM | 原基线，三状态输入 |
| B1 | Deep-FNN-H3 | 两个历史块，无 MoE |
| B2 | Deep-FNN-current | 仅当前状态，无 memory |
| B3 | Discrete-KDA-FNN | 离散矩阵 memory |
| B4 | CTDM-Galerkin-ROM | 连续矩阵 memory，与速度联合 RK4 |

B3 与 B4 的参数张量名称、形状和总数完全一致；唯一算法差异是 macro-step 离散 memory update 与 stage-wise 连续 memory ODE。

## 3. 冻结的数据与 ROM 契约

### 3.1 Reynolds 数划分

- train：29 条完整轨迹；
- validation：\(Re=46.7,56.543246\)；
- sealed test：\(Re=47.081355,49.022357,51.786450\)。

速度和压力 POD 的均值、基、scaler 均只由 train 拟合，且 \(r_u=r_p=32\)。所有模型共享同一组 Galerkin operators、Pressure--Poisson assets、pressure gauge、adaptive pressure closure、native query times 和 physical-field evaluator。

test 访问发生在独立门控函数之后。本轮门控拒绝访问，因此没有打开 held-out POD 或投影张量。

### 3.2 运行时输入审计

冻结 B0 checkpoint 的第一层权重为 `[256,493]`，因此实际输入不是预期的 560 维。精确组成如下。

| 组成 | 维数 |
|---|---:|
| 归一化 \(Re\) 与 \(1/Re\) | 2 |
| 当前速度系数 \(a\) | 32 |
| 当前压力系数 \(b\) | 32 |
| 当前 Galerkin RHS \(g\) | 32 |
| 当前 norm/energy descriptors | 11 |
| current-only 合计 | 109 |
| 一个历史块 \((a,b,g,\dot a,\dot b,\dot g)\) | 192 |
| 两个历史块 | 384 |
| B0/B1 实际输入 | 493 |
| 已实现但只参与 loss 的 Hopf augmentation | 67 |

67 维 augmentation 是 32 维速度 fluctuation、32 维压力 fluctuation、两个临界平面坐标和一个归一化半径。正式 H4 路径没有把它接入 encoder，因此新比较遵循 checkpoint 证据：B1 使用 493 维，B2--B4 使用真实的 109 维 current-only 输入。

### 3.3 时间离散

每条轨迹使用存储的相邻 native \(\Delta t_n\)，而不是统一时间步。典型中位时间间隔约为 4.06--23.63 个物理时间单位，轨迹边界还可能出现较短残余区间。

## 4. 方法

### 4.1 Current-state encoder

B2--B4 的状态编码器为

```text
LayerNorm(109)
Linear(109,256) -> SiLU
Linear(256,256) -> SiLU
Linear(256,128) -> SiLU
```

得到 token \(z\in\mathbb R^{128}\)。B1 使用相同风格的普通深层 FNN，但输入为 493 维固定三状态历史。B1--B4 均删除 MoE、router、shared/routed experts、route loss、attention、MLA、AttnRes、oscillatory rotation、normal-form/radial/attractor loss、frequency/phase supervision 和 energy projection。

### 4.2 四头矩阵记忆

采用 4 个 heads，\(d_k=d_v=16\)。每个 head 保存

\[
S_h\in\mathbb R^{16\times16},
\]

因此每条轨迹的运行时 memory 状态为 \(4\times16\times16=1024\) 个 float32 标量。

由 token 生成

\[
k_h=\operatorname{normalize}(W_{k,h}z),\qquad
v_h=v_{\max}\tanh(W_{v,h}z),
\]

\[
\gamma_h=\gamma_{\min}+\operatorname{softplus}(W_{\gamma,h}z+b_{\gamma,h}),
\qquad
\eta_h=\eta_{\max}\sigma(w_{\eta,h}^{\top}z+b_{\eta,h}).
\]

这里使用 `gamma_min + softplus`，以保证正耗散。速度和压力使用不同 query：

\[
q_{u,h}=\operatorname{normalize}(W_{qu,h}z),\qquad
q_{p,h}=\operatorname{normalize}(W_{qp,h}z),
\]

\[
m_{u,h}=S_h^\top q_{u,h},\qquad m_{p,h}=S_h^\top q_{p,h}.
\]

四头拼接后 \(m_u,m_p\in\mathbb R^{64}\)。velocity FNN 读取 \([z,m_u]\)，pressure FNN 读取 \([z,m_p]\)。二者约为 4 层、宽度 512 的 SiLU MLP。速度/压力 residual 最终层零初始化；pressure gate 权重零初始化、bias 为 `logit(0.99)`。

### 4.3 连续极限与符号修正

公平离散模型使用

\[
S_{n+1}=(I-\beta_n k_nk_n^\top)D_nS_n+\beta_nk_nv_n^\top.
\]

展开

\[
D_n=I-\Delta t_n\Gamma_n+O(\Delta t_n^2),\qquad
\beta_n=\eta_n\Delta t_n+O(\Delta t_n^2)
\]

可得

\[
\frac{S_{n+1}-S_n}{\Delta t_n}
=-\Gamma_nS_n-\eta_nk_nk_n^\top S_n+\eta_nk_nv_n^\top+O(\Delta t_n),
\]

即

\[
\boxed{\dot S=-\Gamma S+\eta k(v-S^\top k)^\top}.
\]

所以正号写入项是给定离散式的连续极限。题面最初的负号写入项不会由该离散式导出；独立数值测试也显示其误差约 1.708--1.709，不随步长趋零。

### 4.4 B3：公平离散 KDA

每个 macro-step 只执行一次

\[
D_n=\exp(-\Delta t_n\Gamma_n),\qquad
\beta_n=1-\exp(-\eta_n\Delta t_n),
\]

\[
S_{n+1}=(I-\beta_nk_nk_n^\top)D_nS_n+\beta_nk_nv_n^\top.
\]

不存在额外 continuous stage memory update。

### 4.5 B4：速度--memory 联合 RK4

联合状态为

\[
Y=[a,\operatorname{vec}(S_1),\ldots,\operatorname{vec}(S_4)].
\]

一个 macro-step 内固定 \(b_n\)，但每个 RK4 stage 都基于 trial \(a\)、trial \(S\) 和 native \(\Delta t_n\)：

1. 重算 Galerkin RHS；
2. 重构 109 维 current-only features；
3. 重算 \(z,k,v,\gamma,\eta,q_u,q_p\)；
4. 从 trial memory 重读 \(m_u,m_p\)；
5. 同时返回 \(da/dt\) 和 \(dS/dt\)。

接受 \(a_{n+1},S_{n+1}\) 后，才调用冻结的 Pressure--Poisson 和 adaptive pressure closure 得到 \(b_{n+1}\)。实现中没有 macro-step 边界的额外离散 KDA update，没有在 RK4 内冻结 memory，也没有使用 future truth。

### 4.6 Warm-up

主评价只使用三个真实初始状态。令 \(S(t_{n-2})=0\)，在两个真实观测区间内对 \(a(t),b(t)\) 线性插值，只积分 memory；从 \(t_n\) 起完全自主 rollout K56。

warm-up=8、16 只作为容量诊断。由于 \(Re=46.7\) 轨迹长度不足以支持更长 warm-up 加 K56，这两项只对 \(Re=56.543246\) 有效，不能与两 Re 的主结果直接平均比较。

## 5. 训练协议

- 优化器：AdamW，cosine schedule；
- 每个完整 run：8000 optimizer steps；
- micro-batch：16；gradient accumulation：1；
- gradient clipping：1.0；
- AMP/TF32 开启，matrix memory 强制 float32；
- 连续轨迹片段训练，不随机打乱独立时间点；
- curriculum：K4（0--1199）→ K8（1200--2799）→ K16（2800--4799）→ K32（4800--6399）→ K56（6400--8000）；
- loss 权重：\(E_a=1\)、\(E_b=0.55\)、modal \(E_u=0.38\)、modal \(E_p=0.095\)、memory norm \(10^{-6}\)；
- seeds：1248、2248、3248；
- SwanLab：online；
- GPU：RTX 4090，单任务顺序执行。

seed 1248 用于初筛。所有 B1--B4 均 finite 且零 divergence，因此都进入 confirmatory seeds。任何非有限梯度均 fail-closed，不降低学习率、不更换 batch、不删除失败 seed。

## 6. 评价指标

- \(E_u,E_p\)：在冻结 POD mean/basis 上重构物理速度/压力场，使用面积权重的相对误差；表中以百分数显示。
- \(E_a,E_b\)：按 train-only coefficient scales 归一化后的 modal RMSE。
- terminal error：K56 最终查询时刻的 \(E_u,E_p\)。
- worst-window：每个窗口对 56 个查询时刻平均 \(E_u+E_p\)，再取最坏窗口。
- pressure drift：压力系数能量相对真值的漂移。
- divergent：非有限或 modal norm ratio 超过 10 的窗口。
- error-growth：K1、K4、K8、K16、K32、K56 的平均 \(E_u+E_p\)。
- 时间步一致性：使用 \(\Delta t,\Delta t/2,\Delta t/4\) 到达同一物理查询时刻，比较最终物理场和 memory，得到 \(C_u,C_p,C_S\)。

## 7. 数学与数值测试

### 7.1 离散到连续

终端相对误差随步长依次为

`1.1512e-2, 5.7479e-3, 2.8719e-3, 1.4354e-3`，观测阶为

`1.00205, 1.00103, 1.00051`。

结论：公平离散 KDA 对所实现的正号连续方程呈一阶收敛。

### 7.2 RK4 阶测试

观测阶为 `4.01142, 4.00486, 4.00223`，符合四阶 RK4。

### 7.3 Memory 有界性

在有界人工输入、物理时间 200、步长 0.05 下：

- 最大 \(\|S\|_F=0.993638\)；
- 终值 \(\|S\|_F=0.879372\)；
- 推导的 ultimate-bound scale 为 1.299038。

独立 memory 子系统测试通过。它不等价于耦合训练全局稳定性；后续两个 B4 seed 的非有限梯度正说明这两者必须区分。

### 7.4 Runtime smoke

B1--B4 的 CPU/CUDA forward、backward、checkpoint roundtrip 和 K4/K8/K16/K32/K56 均通过。B3 离散更新次数严格等于 horizon；B4 没有离散更新，且 RK4 stage 内 memory 非零变化。B3/B4 K56 memory 参数梯度均非零。

## 8. 参数量与计算成本

| 模型 | Trainable parameters | 每条轨迹 memory 状态 | 成功 run 平均训练时间 |
|---|---:|---:|---:|
| B0 | 32,004,147 | 0 | 1.979 h |
| B1 | 2,379,284 | 0 | 1.217 h |
| B2 | 1,984,532 | 0 | 1.026 h |
| B3 | 2,238,718 | 1,024 | 1.852 h |
| B4 | 2,238,718 | 1,024 | 2.043 h（仅成功 seed 1248） |

B3 与 B4 参数完全一致。B4 成功 run 比 B3 三-seed平均慢约 10%，但由于运行批次和系统状态不同，wall-clock 只作为同机量级参考。

## 9. Validation 结果

### 9.1 三-seed稳定模型

| 模型 | 有效 seeds | \(E_u\) | \(E_p\) | \(E_a\) | \(E_b\) | terminal joint | worst-window | pressure drift | divergence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 1 | 0.0365% | 0.2649% | 0.2337 | 0.2897 | 0.5753% | 0.8787% | 1.1412% | 0 |
| B1 | 3 | 0.0236% | 0.0749% | 0.0393 | 0.1178 | 0.1492% | 0.1707% | 0.4999% | 0 |
| B2 | 3 | 0.0621% | 0.1244% | 0.0405 | 0.1204 | 0.3340% | 0.3267% | 1.1101% | 0 |
| B3 | 3 | 0.0840% | 0.5036% | 0.0977 | 0.1612 | 0.8793% | 0.8573% | 4.7688% | 0 |
| B4 | 1 | 0.0920% | 0.4585% | 0.1096 | 0.1953 | 0.9288% | 1.1234% | 3.9676% | 0 |

B4 行只是成功的 seed 1248，不是三-seed平均。seed 2248/3248 没有合法最终 checkpoint，不能从性能统计中静默删除。

### 9.2 固定历史与 current-only

B1 的三-seed联合平均场误差为 0.0985%，B2 为 0.1865%；B2 的 worst-window 相对 B1 增加约 91.4%。因此固定三状态历史提供了明显、可重复的收益。

相对冻结 B0，B1 在更少参数下同时改善场误差、terminal、worst-window 和 pressure drift。本轮最强工程基线是普通 FNN + 固定历史，而不是 MoE 或矩阵记忆。

### 9.3 公平的 seed-1248 B3/B4 配对

只有 seed 1248 同时具有 B3 和 B4 的完整 checkpoint，因此连续/离散的公平性能判断必须使用该配对。

| 指标 | B3 discrete | B4 continuous | B4 相对变化 |
|---|---:|---:|---:|
| \(E_u\) | 0.0766% | 0.0920% | +20.17% |
| \(E_p\) | 0.4351% | 0.4585% | +5.36% |
| \(E_u+E_p\) | 0.5117% | 0.5505% | +7.58% |
| \(E_a\) | 0.0782 | 0.1096 | +40.28% |
| \(E_b\) | 0.1353 | 0.1953 | +44.30% |
| terminal joint | 0.7490% | 0.9288% | +24.00%（约） |
| worst-window | 0.6852% | 1.1234% | +63.95% |
| pressure drift | 4.2741% | 3.9676% | -7.17% |

连续模型仅在 pressure drift 上小幅改善；速度场、压力场、modal、terminal 和 worst-window 均更差。预注册的 B4-vs-B3 prediction gate 因此失败。

### 9.4 Error growth

在较难的 \(Re=56.543246\) 上：

| 模型 | K1 | K4 | K8 | K16 | K32 | K56 |
|---|---:|---:|---:|---:|---:|---:|
| B3 | 0.2266% | 0.2733% | 0.2990% | 0.3837% | 0.5762% | 0.9625% |
| B4 | 0.2225% | 0.2696% | 0.2610% | 0.3615% | 0.6567% | 1.3053% |

B4 在 K1--K16 略好，但从 K32 起反转，K56 明显更差。这与“连续积分改善局部时间一致性，但未改善长时闭环误差积累”的解释一致。

## 10. 时间步一致性

对 seed 1248、两个 validation Re 的 \(\Delta t\) 对 \(\Delta t/2\) 结果取平均：

| 模型 | \(C_u\) | \(C_p\) | \(C_S\) |
|---|---:|---:|---:|
| B3 | 1.1253e-4 | 5.0279e-4 | 2.2739e-2 |
| B4 | 1.8362e-5 | 1.1567e-4 | 1.0258e-2 |
| B4 改善 | 83.68% | 76.99% | 54.89% |

B4 的时间步一致性门通过。这个结果验证了联合 RK4 的数值动机，但不能抵消预测门和训练稳定性门的失败。

## 11. Memory 诊断

以下为 B4 成功 seed 1248、主 warm-up=3：

| Re | mean \(\gamma\) | mean \(1/\gamma\) | half-life | mean \(\eta\) | \(\|S\|_F\) | 最大奇异值 | effective rank | delta error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 46.7 | 0.02642 | 90.66 | 62.84 | 0.02705 | 2.1387 | 2.1297 | 1.6588 | 1.2210 |
| 56.543246 | 0.02670 | 170.48 | 118.17 | 0.02220 | 3.3282 | 3.3281 | 1.0722 | 0.8078 |

结论：

- \(\eta\) 和 \(\|S\|_F\) 明显非零，网络没有完全绕过 memory；
- 较高 Re 学到了更长的记忆时间尺度；
- \(Re=56.543246\) 的 effective rank 仅 1.072，memory 接近单一主奇异方向，存在明显低秩退化倾向；
- delta error 非零，memory 仍在持续接收写入，而非达到完美静态重构。

因此 `memory_nondegenerate` 按预注册阈值通过，但“通过阈值”不等价于 memory 表示丰富；特别是高 Re 的有效秩已非常接近 1。

## 12. Warm-up 容量诊断

在 \(Re=56.543246\) 上，B4 的 worst-window 从 warm-up=3 的 1.1234% 变为：

- warm-up=8：1.1182%；
- warm-up=16：1.1132%。

延长 warm-up 仅带来不足 1% 的相对改善，同时 \(E_u\)、\(E_p\) 和 terminal 并未系统改善。这说明主要瓶颈不是两区间 warm-up 容量不足。

## 13. 跨 seed 训练稳定性

| 模型 | seed 1248 | seed 2248 | seed 3248 | 完成率 |
|---|---|---|---|---:|
| B1 | 完成 | 完成 | 完成 | 3/3 |
| B2 | 完成 | 完成 | 完成 | 3/3 |
| B3 | 完成 | 完成 | 完成 | 3/3 |
| B4 | 完成 | step 1541 非有限梯度 | step 3958 非有限梯度 | 1/3 |

两次 B4 失败发生在不同 curriculum 阶段：seed 2248 位于 K8，seed 3248 位于 K16。训练器已启用 float32 memory 和 gradient clipping=1.0，仍然出现非有限梯度。

这不能由独立 memory boundedness test 排除，因为该测试假设有界人工输入并只研究 memory 子系统；正式训练涉及 encoder、Galerkin residual、压力闭合和长 rollout 的耦合反向传播。现有证据只能确定“耦合优化出现不稳定”，不能仅凭日志唯一定位为 ODE stiffness、特征爆炸或压力路径。

## 14. Validation freeze 与 test 访问

冻结清单记录：

```text
confirmatory_attempts_complete = true
three_seed_complete            = false
B4_vs_B3_prediction_gate       = false
B4_timestep_consistency_gate   = true
B4_memory_nondegenerate        = true
test_access_authorized         = false
```

所有计划 seeds 都已尝试，但 B4 两次 fail-closed，因此三个有效 seed 未完成；此外 seed-1248配对预测门本身也失败。test 访问没有授权，`authorized_checkpoint_sha256s` 为空，`TEST_RESULTS.csv` 为零字节。

## 15. 对十个预注册问题的回答

1. **离散 KDA 是否收敛到实现的连续方程？** 是。对正号写入连续方程观测到约一阶收敛；负号方程不收敛。
2. **联合 RK4 是否具备预期时间步一致性？** 独立制造解达到四阶；模型级 B4 的 \(C_u,C_p,C_S\) 均显著优于 B3。
3. **Memory 是否有界？** 独立有界输入测试通过，成功 seed 的 rollout memory norm 有限；但两个 confirmatory seeds 的训练梯度非有限，所以不能宣称完整耦合训练全局稳定。
4. **Current-only FNN 是否说明 memory 必要？** 不支持“memory 必要”。B2 三-seed稳定且显著优于 B3/B4；不过 B1 又优于 B2，说明历史信息有用，只是本轮矩阵 memory 没有有效利用它。
5. **KDA 是否优于固定三状态历史？** 否。B3 worst-window 相对 B1 恶化约 402%，且 pressure drift 更大。
6. **连续 KDA 是否优于公平离散 KDA？** 否。seed-1248配对中联合场误差恶化 7.58%，worst-window 恶化 63.95%；另外 B4 仅 1/3 seeds 完成。
7. **改善来自哪里？** 仅观察到 pressure drift 小幅改善和显著时间步一致性改善；速度/压力场误差、modal error、terminal 与 worst-window 没有改善。
8. **学到的物理记忆时间尺度是多少？** 成功 seed 中 mean \(1/\gamma\) 约为 90.7（Re 46.7）和 170.5（Re 56.54），half-life 约为 62.8 和 118.2 个物理时间单位。
9. **Memory 是否被绕过或退化？** 没有完全绕过：\(\eta\)、memory norm 和梯度均非零。但高 Re effective rank 约 1.07，表现出接近 rank-1 的退化。
10. **是否值得进入振荡生成元研究？** 目前不值得。应先解决跨 seed 非有限梯度，并证明 memory 相对 B1/B2/B3 的配对预测收益。

## 16. 局限性

- 因预注册门控拒绝 test，本报告不包含 held-out 泛化结论。
- B4 只有一个完整 seed；它的数值只用于 seed-1248配对和机制诊断，不能当作稳定均值。
- validation 只有两个 Re，其中长 warm-up 只在一个 Re 可用。
- 非有限梯度的根因尚未通过逐 stage Jacobian、梯度来源分解或 stiffness 指标唯一定位。
- wall-clock 受共享集群状态影响，只用于量级比较。

## 17. 建议的下一阶段

不建议直接加入 oscillatory generator。更合理的下一步是一个独立的稳定性实验，而不是事后修改本轮结果：

1. 在不打开 test 的前提下，对 B4 失败 seeds 重放至失败前，记录每个 RK4 stage 的 \(\gamma,\eta,\|S\|,\|\partial L/\partial S\|\) 和 Galerkin residual；
2. 区分 forward nonfinite、backward overflow 和长 horizon Jacobian 放大；
3. 预注册稳定化手段，例如有界 \(\gamma\)、更严格的 memory/state norm 控制或 stiffness-aware solver；
4. 重新从三个全新 seeds 比较 B1/B2/B3/B4，仍以配对 prediction gate 为主；
5. 只有连续模型同时满足三-seed完成、预测收益和时间步一致性收益，才进入振荡生成元研究。

## 18. 可复现性与产物

- Python 3.11.15；PyTorch 2.13.0+cu126；CUDA 12.6；cuDNN 91002；
- SwanLab 0.9.0；NVIDIA RTX 4090 24,564 MiB；driver 595.71.05；
- 成功 validation checkpoint 与 B0 共 11 个 frozen hashes 见 `CHECKPOINT_HASHES.json`；
- 两个失败 run 的日志 SHA-256 分别记录在对应 `*_FAILURE.json`；
- 完整命令见 `EXECUTION_COMMANDS.md`；
- 原始逐 seed validation JSON 位于 `validation_raw/`。

本报告结论只依赖冻结 validation、数学测试和失败清单，不使用 held-out test。
