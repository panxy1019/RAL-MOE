# 方柱周期数据集 KDA-inspired CDM-GROM 第一阶段实验报告

日期：2026-08-27  
状态：第一阶段验证完成；结论为 **FAIL（未达到进入 M4 的门槛）**

## 1. 结论摘要

本轮在方柱绕流 `periodic` 数据集上完成了冻结 FVM–Galerkin 基线、瞬时缺陷构造、训练集相位对识别、非神经连续 delta-memory 拟合，以及 6 个验证 Reynolds 数上从 `t=0` 到 `t=500` 的自由滚动。

主要结论如下。

1. 连续 KDA 状态方程的推导在给定缩放下成立，代码中的矩阵维数、delta erase、切向相位注入和联合 RK4 也与理论一致。
2. 训练集识别出的主振荡对为零基索引 `(1, 2)`，即 POD 模态 2 和 3，而不是预先假定的模态对。
3. 新的 solver-consistent instantaneous defect 与旧 flow-map defect 几乎不相关：平均相关系数仅 `0.02045`，说明二者不是可互换的训练标签。
4. M3 最优候选为 `gamma=0.25, eta=1.0, ridge=1e-6`，6/6 工况均有限且有界，平均 `J500=0.228064`。
5. M3 相比 M1 仅改善约 `0.43%`，但比 M0 差 `3.73%`，比同为零初始化的 M2 差约 `29.20%`，因此没有满足预先冻结的成功条件。
6. 均值流基线的平均 `J500=0.163412`，优于所有动态 ROM。这不是均值流恢复了周期动力学，而是当前绝对全场归一化指标会奖励振荡衰减。必须同时查看频率、相位和振幅误差。
7. 按测试计划，M3 未成功时不得继续 M4；因此本轮没有加载 held-out，也没有运行 amplitude+phase 或 full-residual 扩展。

## 2. 数据隔离与冻结配置

### 2.1 数据划分

- Train：27 个 periodic Reynolds 数，共 3402 个快照；用于 POD、缺陷、相位对、归一化、候选时间尺度和线性读出拟合。
- Validation：`Re = 99, 101.5, 110.344827586, 125.862068966, 141.379310345, 150`；只在候选全部生成并冻结后加载。
- Held-out：未下载、未加载、未用于任何选择或报告数值。

### 2.2 冻结物理基线

- 速度 POD 阶数：28。
- 压力 POD/线性压力闭合阶数：24。
- Galerkin 空间离散：OpenFOAM/FVM 一致的离散张量。
- 稳定化：无人工 damping。
- 自由滚动积分：联合四阶 RK4，最大子步长 `0.05`。
- FVM 张量 SHA-256：`d5e849b393a5b2649c19cb1431af738260a97bc593dd1f378647dd3659265b6b`。

所有新增文件和运行结果均位于独立目录，没有修改其他训练任务或共享环境中的项目代码。

## 3. 方法实现

### 3.1 Instantaneous defect

使用 OpenFOAM `romSpatialRhs` 在原始 CFD 快照上计算并投影离散空间 RHS：

```text
g(t) = adot_CFD_projected(t) - F_FVM(a_CFD(t); Re)
```

27/27 个训练工况均成功且数值有限。汇总诊断为：

- instantaneous defect 相对 L2 均值：`0.474555`；
- old flow-map defect 与 instantaneous defect 的平均相关系数：`0.0204533`。

低相关性验证了理论材料中的判断：旧 flow-map 标签是有限时间平均缺陷，不能作为连续时间闭合的精确瞬时标签。

### 3.2 相位对与目标

只使用训练集中 `Re >= 110` 的充分周期数据，在前 12 个 POD 模态内搜索相位对。冻结结果：

- phase pair：零基索引 `(1, 2)`；
- winner score：`0.402655`；
- 跨 Re 频率变异系数：`0.022701`；
- runner-up `(7, 8)` score：`0.101458`。

目标采用：

```text
delta_omega = omega_CFD - omega_FVM
delta_rho_dot = rho_dot_CFD - rho_dot_FVM  # 本轮仅作诊断
```

训练集 RMS：`delta_omega = 0.173998`，`delta_rho_dot = 0.0297482`。相位缺陷自相关的 `1/e` 时间约为 8，因此冻结：

- `gamma in {0.0625, 0.125, 0.25}`；
- `eta in {0.1, 0.25, 0.5, 1.0}`；
- `ridge in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1}`。

### 3.3 KDA-inspired 连续记忆

第一版地址和值为：

```text
phi = [1, cos(theta), sin(theta), cos(2 theta), sin(2 theta), rho_scaled, Re_scaled]
k = q = phi / ||phi||_2
v = [omega_FVM_scaled, rho_dot_FVM_scaled, rho_scaled, 1]
```

矩阵记忆 `S in R^(7 x 4)` 满足：

```text
Sdot = -gamma S + eta k (v - S^T k)^T
```

训练时沿 CFD 轨迹 teacher forcing，逐条轨迹采用零初始化；读出用闭式 ridge 回归，不含神经网络训练。自由滚动时在每个 RK4 stage 同时更新 `a` 和 `S`。

相位修正为：

```text
r_p = -delta_omega_hat * a_q
r_q =  delta_omega_hat * a_p
```

它对选定平面的瞬时径向导数贡献严格为零，对瞬时角速度贡献严格为 `delta_omega_hat`。

### 3.4 对照模型

- M0：冻结 FVM–GROM，无闭合。
- M1：无记忆 Markov phase correction。
- M2：旧 fixed-pole CDM，固定率 `[0.01, 0.05]`，ridge `10`，保持旧 flow-map 训练目标。
- M3：本轮 KDA-inspired phase-only continuous delta-memory。
- Mean flow：退化基线。
- M4：仅当 M3 明确胜过 M0/M1/M2 后运行；本轮未触发。

## 4. 验证指标

统一主指标为质量加权的绝对全速度场误差：

```text
J500 = sqrt(
  sum_t ||U_ROM(t) - U_CFD(t)||_M^2
  / sum_t ||U_CFD(t)||_M^2
)
```

计算时包含 28 阶 POD 无法解析的投影残差。因此本报告的 `J500` 不能与旧报告中“中心化系数相对误差”数值直接比较。

所有验证均从真实 `a(0)` 开始，覆盖 `t=0...500`。M3 主实验采用零 memory。M2 同时报告旧初始化和零初始化敏感性；零初始化版本用于最严格的同协议比较。

## 5. 主结果

### 5.1 聚合结果

| 模型 | 初始化 | 稳定性 | mean J500 | median J500 | max J500 |
|---|---|---:|---:|---:|---:|
| Mean flow | 不适用 | 6/6 | **0.163412** | — | — |
| M2 fixed-pole | zero | 6/6 | **0.176525** | 0.177374 | 0.278178 |
| M2 fixed-pole | legacy `a0/scale` | 6/6 | 0.178640 | 0.189567 | 0.279686 |
| M0 frozen | 不适用 | 6/6 | 0.219857 | 0.226068 | 0.301669 |
| M3 KDA memory | zero | 6/6 | 0.228064 | 0.223695 | 0.269953 |
| M1 Markov phase | 不适用 | 6/6 | 0.229055 | 0.224733 | 0.271414 |

M3 的相对比较：

- 对 M1：`-0.43%`，只有极小改善；
- 对 M0：`+3.73%`，更差；
- 对零初始化 M2：约 `+29.20%`，更差；
- 对 mean flow：约 `+39.56%`，更差。

### 5.2 每个 Reynolds 数的 J500

| Re | Mean flow | M0 | M1 | M2 legacy | M3 |
|---:|---:|---:|---:|---:|---:|
| 99.000000 | 0.128726 | 0.146708 | 0.196041 | **0.071502** | 0.194517 |
| 101.500000 | 0.130596 | 0.158669 | 0.200022 | **0.094425** | 0.199110 |
| 110.344828 | 0.143553 | 0.195213 | 0.214158 | **0.150856** | 0.211187 |
| 125.862069 | **0.171713** | 0.259959 | 0.235309 | 0.228278 | 0.236202 |
| 141.379310 | **0.196917** | 0.256923 | 0.257389 | 0.247095 | 0.257415 |
| 150.000000 | **0.208964** | 0.301669 | 0.271414 | 0.279686 | 0.269953 |

表中 M2 每工况采用 legacy 初始化，以保持旧模型原定义；零初始化聚合结果已经单独给出，结论不变。

### 5.3 动力学次级指标

| 模型 | 平均频率相对误差 | 平均振幅相对误差 | 平均末时刻绝对相位误差 |
|---|---:|---:|---:|
| M0 | 0.27 | 0.64 | 65.18 rad |
| M1 | 0.16 | 0.72 | 11.86 rad |
| M2 legacy | 0.42 | 0.56 | 83.70 rad |
| M3 | **0.15** | 0.71 | **11.69 rad** |

M3 确实显著降低了 M0 的长期相位漂移，并略优于 M1；但它没有恢复正确振幅。尤其在低 Re，M1/M3 的相位闭合会通过非线性 FVM 向量场间接改变径向和其他模态，即使注入项本身严格切向，也可能造成很大的振幅误差。

M2 的低 `J500` 也不能解释成周期动力学优秀。例如：

- Re=99：`J500=0.0715`，但频率误差 `97.1%`、振幅误差 `98.5%`；
- Re=101.5：`J500=0.0944`，但频率误差 `78.1%`、振幅误差 `99.0%`。

这说明 M2 在近 Hopf 工况主要通过压低振荡获得较低绝对全场误差，正是理论文稿警告的 degeneration。

## 6. 预注册成功条件判定

测试计划要求 M3 同时满足：

1. 6/6 验证轨迹有限、有界；
2. mean `J500` 低于 M0；
3. mean `J500` 低于 M1；
4. mean `J500` 低于 M2。

本轮判定：

| 条件 | 结果 |
|---|---|
| 稳定 6/6 | PASS |
| M3 < M0 | FAIL |
| M3 < M1 | PASS，但仅 0.43% |
| M3 < M2 | FAIL |

因此第一阶段总判定为 **FAIL**。按冻结协议，不运行 M4，不进行第二阶段结构消融，不访问 held-out。

## 7. 理论与实现严格审计

### 7.1 推导中没有发现的错误

离散更新在以下缩放下：

```text
D_n = exp(-h Gamma_n)
beta_n = h eta_n
```

一阶展开得到：

```text
(S_(n+1)-S_n)/h
= -Gamma S + eta k(v-S^T k)^T + O(h)
```

交叉项为 `O(h^2)`，因此连续极限正确。代码测试还验证了离散 KDA 随步长缩小时收敛到连续 ODE。

记忆能量估计也正确：当 `Gamma >= gamma0 I`、`eta >= 0` 且 `v` 有界时，memory subsystem 具有耗散上界。当前所有候选 `gamma>0`、`eta>0`，验证中 memory 有限。

### 7.2 必须保留的理论限制

1. memory 有界不等于耦合 ROM 有界；切向读出仍可能经非线性基线诱发幅值和其他模态变化。
2. `omega` 和 `rho_dot` 在 `rho -> 0` 附近存在坐标奇异性。实现使用小正数截断以避免除零，但这不是严格物理正则化。
3. teacher forcing 只保证沿真实轨迹的读出拟合，不保证自由滚动时访问到的 memory/address 分布一致，存在明显 exposure bias。
4. `S(0)=0` 会引入 memory transient。当前最优 M3 的 `gamma=0.25` 对应约 4 个时间单位的被动衰减尺度，相对 500 的总窗口很短；M2 零初始化复核也表明初始化并不能解释模型排序。
5. 当前 address 主要由同一极限环上的相位和幅值决定，memory readout 很可能接近一个扩展 Markov 映射。M3 只比 M1 改善 0.43%，目前没有足够证据支持“缺陷具有 M1 无法表达的关键历史依赖”。
6. 瞬时标签在连续理论上更正确，但主选择目标是长时自由滚动 `J500`。二者并不等价；低 teacher-forced RMSE 不能自动转化成长时优势。
7. `J500` 的绝对全场分母由均值流能量主导，会奖励振荡塌缩。后续任何结论必须把 `J500` 与相位、频率、振幅和升阻力谱等指标联合报告。

### 7.3 假设判定

- H1“长期误差含系统性 phase-speed defect”：得到部分支持。M1/M3 明显降低相位误差。
- H2“phase-speed defect 不完全由当前状态决定”：未得到支持。M3 与 M1 几乎相同。
- H3“state-addressed delta memory 优于 fixed exponential memory”：被当前验证结果否定。M3 明显不如 M2 的主指标。

## 8. 训练与监控状态

- 运行集群：`10.10.164.243:20381`。
- 环境：项目自带 `pt_env`，NVIDIA RTX 4090，float64 CUDA。
- SwanLab：M0/M1/M3 正式网格与 M2 补充验证均生成了本地 run 记录；因集群没有在线认证且隔离环境缺少 `swanboard` dashboard 插件，没有可用的在线网页面板，但标量与运行文件已落盘。
- 所有验证脚本均正常调用 `swanlab.finish()` 或自然结束；当前无验证/训练进程，GPU 已恢复空闲。
- 没有更改共享虚拟环境；SwanLab 依赖安装在项目私有 `vendor/` 下。

## 9. 推荐下一步

当前不建议直接扩大 M3 参数网格或进入 M4。优先级应为：

1. 将主目标改成多指标门槛：保留绝对 `J500`，同时强制频率、振幅、相位和升阻力统计不过度退化。
2. 对 M0 的误差做切向/径向/其余 26 模态的能量分解，确认低 Re 与高 Re 是否属于不同失效机制。
3. 先建立更强但仍纯物理的 periodic 校准基线，例如能量保持/耗散约束的线性或二次校准，再判断记忆是否仍有独立增益。
4. 若继续 KDA 路线，先做 M1 与 M3 的条件残差检验：在给定当前 `phi` 后，检查 `delta_omega` 残差是否仍与历史量显著相关。只有检验为真，继续 memory 才有理论依据。
5. 若重点是准确周期动力学，应使用 periodic steady-memory 初始化，并以极限环频率、振幅、相位和力系数共同选择，而不是只最小化绝对全场 `J500`。

## 10. 可复现实验文件

- `validation_grid_result.json`：M0/M1/M3 的 92 模型验证网格与逐工况指标。
- `m2_latest_metric_result.json`：旧初始化 M2 的统一指标结果。
- `m2_zero_init_latest_metric_result.json`：零初始化公平性复核。
- `artifacts_train_defect_summary.json`：27 个训练工况瞬时缺陷汇总。
- `artifacts_phase_pair.json`：train-only phase pair 选择结果。
- `transfer_runtime/PHASE_TARGET_SUMMARY.json`：相位目标、归一化与候选时间尺度。
- `rollout/evaluate_validation_grid.py`：M0/M1/M3 GPU 联合验证。
- `rollout/evaluate_m2_latest_metric.py`：M2 统一口径与初始化敏感性验证。
- `tests/test_core.py`：连续/离散 memory、切向注入和联合 RK4 单元测试。

集群副本位于：

```text
/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/
centeredsquare_kda_cdm_rom_20260827
```
