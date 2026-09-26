# 方柱绕流 ROM 实验方法与效果总览

日期：2026-08-03  
项目：CenteredSquare 方柱扰流；OpenFOAM 13；二维不可压缩圆柱/方柱类外流的 POD 降阶建模

## 1. 当前结论

当前最可信、可作为后续 CDM-GROM 研究基础的模型是：

> **periodic 子数据集上的 r28/r24 OpenFOAM 离散一致 FVM–POD–Galerkin ROM，配合训练集拟合的线性压力系数闭合。**

它不是神经网络模型，也不是严格的纯 Pressure–Poisson ROM；更准确地说，它是“侵入式有限体积 Galerkin 速度算子 + 低阶线性压力闭合”的校准型物理 ROM。

该基线在 validation 和冻结后 held-out 的全部工况中均稳定积分到 `t=500`。在充分发展的周期工况上，能够较好恢复极限环振幅和主频；但在 Hopf 临界附近仍低估振幅，并存在长期相位漂移。

在此基线上首次尝试的固定地址、实指数核 Continuous Delta-Memory GROM（CDM-GROM）未通过 validation 的物理质量门槛：虽然少数候选略微降低了长期逐点 L2，却显著恶化振幅和频率。因此当前 CDM 不冻结、不进行新的 held-out 测试，正式基础模型仍是上述 periodic FVM–GROM。

---

## 2. 数据、算例与划分

### 2.1 正式 OpenFOAM 源算例

经核验，正式 Re=100 原始算例为：

```text
/home/ray/Desktop/centeredSquare/runs/cn09_graded_20260708_000142/Re100
```

而下列路径是旧的不同算例，不能用于重建当前数据集的离散算子：

```text
/home/ray/Desktop/centeredSquare/Re100
```

NPZ 快照与正式算例在 Re=100、t=100 的速度最大绝对差为 `5.96e-8`，因此数据文件、网格和正式 CFD 案例是一致的。

### 2.2 离散格式与边界条件

正式算例的关键离散设置为：

- 对流：`div(phi,U) Gauss linearUpwind grad(U)`；
- 扩散：`laplacian ... linear corrected`；
- 法向梯度：`snGrad corrected`；
- 入口：抛物线速度；
- 上下壁和方柱：无滑移；
- 速度出口：零梯度；
- 压力出口：固定零压；其他压力边界：零梯度。

### 2.3 periodic 子集

periodic 专家域为 Re=98.5–150，共 37 个案例：

| 角色 | 数量 / Re |
|---|---|
| train | 27 个 Re，3402 个快照 |
| validation | 99.0、101.5、110.3448、125.8621、141.3793、150.0 |
| held-out | 100.5、102.0、120.6897、144.8276 |

POD 的均值与模态仅由 train 构造。模型阶数、压力闭合、阻尼和积分设置在 held-out 场被读取前完成冻结；held-out 只用于一次最终测试。

---

## 3. 为什么需要重建 POD–Galerkin 算子

旧的纯物理 ROM 预测很差，根本原因不是 Galerkin 降阶理论天然无效，而是它在 VTK 单元中心上用连续微分近似重建空间算子，没有复现 OpenFOAM 的：

1. 有限体积面通量；
2. `linearUpwind` 对流离散及其限制行为；
3. 非正交网格修正；
4. 固定入口/出口和障碍物边界条件。

因此旧算子与 CFD 的投影动力学之间存在 O(1) 级别结构误差。后续 CDM 不能负责“修复错误物理主算子”；首先必须让物理基线和原求解器的离散形式一致。

---

## 4. 重建的离散一致 FVM–POD–Galerkin 方法

### 4.1 OpenFOAM 原位右端计算

为避免连续微分近似，编译了 OpenFOAM 13 工具 `romSpatialRhs`。它在原网格和原边界条件上直接计算：

\[
R(U,p;\nu)= -\operatorname{div}(\phi,U)
+\nu\operatorname{laplacian}(U)-\nabla p,
\qquad \phi=\operatorname{flux}(U).
\]

然后将该离散右端投影到速度 POD 模态。通过均值状态、单模态正负扰动、双模态混合扰动和压力模态扰动，使用有限差分装配 ROM 系数。

### 4.2 ROM 方程

速度系数 `a(t)` 的模型写为：

\[
\dot a = c_c+A_c a+H_c(a,a)
+\nu(c_d+A_d a)+c_p+P b(a,\nu).
\]

其中：

- `c_c, A_c, H_c`：有限体积对流项的常数、线性、二次部分；
- `c_d, A_d`：粘性扩散项；
- `c_p, P`：压力梯度投影；
- `b(a,nu)`：压力 POD 系数闭合；
- `nu=1/Re`。

对于 periodic 基线：速度阶数为 `r=28`，压力阶数为 `r_p=24`，探针扰动为 `eps=0.05`，总计使用 1617 个离散探针状态。

### 4.3 自洽性检查

periodic r28/r24 张量装配满足：

| 检查项 | 结果 |
|---|---:|
| 速度质量 Gram 与单位阵最大偏差 | `2.12e-9` |
| 扩散轴向非线性残差最大范数 | `1.99e-8` |
| Re=100, t=100：张量 RHS 与直接 OpenFOAM 投影 RHS 相对差 | `7.63%` |

最后一项无法降至机器精度的主要原因是 `linearUpwind` 的分段/限制行为并不是全局二次多项式，固定二次张量只能近似表达它。

### 4.4 压力闭合

压力 POD 使用 OpenFOAM 固定出口零压规范，避免逐快照再减空间均值。压力系数采用线性闭合：

\[
b(a,\nu)=W[1,\nu,a,\nu a].
\]

训练结果：

| 项目 | 数值 |
|---|---:|
| 压力阶数 | 24 |
| ridge | 0.01 |
| 特征数 | 58 |
| validation 平均压力系数相对误差 | 1.82% |
| validation 最大压力系数相对误差 | 3.43% |

这一步不是神经网络，也不是将压力方程完全替换成经验模型；它是对压力 POD 系数的低阶、参数依赖线性恢复。

---

## 5. Hopf 子集重建结果

Hopf 子集使用 r11 速度 / r11 压力。修正后的 FVM–Galerkin 模型已能避免旧模型的灾难性行为，但临界区难以同时兼顾稳定和振荡幅值：

- 无附加阻尼的线性压力模型可跑到 `t=500`，但 validation 长期速度相对 L2 约为 `1.97`；
- 加入训练/validation 选择的线性涡黏稳定化

\[
-\gamma(a-a_c),
\]

并取 `gamma=0.3` 后，平均长期误差降至 `0.934`，所有验证案例均稳定完成；
- 但该阻尼会压低高 Re 的 Hopf 振荡，因此不把 Hopf r11 模型作为正式基线，也不使用 held-out 继续调参。

Hopf 稳定化后的 validation 速度长期相对 L2：

| Re | 误差 |
|---:|---:|
| 94.5 | 0.662 |
| 95.25 | 0.667 |
| 95.5 | 0.674 |
| 97.5 | 0.895 |
| 99.0 | 0.968 |
| 101.5 | 0.948 |

结论：Hopf 重建是对旧方法的重要修正，但其绝对精度尚不足以作为 CDM 的优先基础。

---

## 6. 正式 periodic FVM–GROM 基线

### 6.1 冻结设置

| 项目 | 冻结值 |
|---|---|
| 速度 / 压力阶数 | r28 / r24 |
| 压力闭合 | 线性，ridge=0.01 |
| 额外涡黏阻尼 | 0 |
| 积分器 | RK4 |
| 最大内部步长 | 0.05 |
| warm-up | 2 个快照间隔，随后自由滚动 |
| 张量 SHA-256 | `d5e849b393a5b2649c19cb1431af738260a97bc593dd1f378647dd3659265b6b` |

### 6.2 validation 效果

所有 6 个 validation 工况稳定到 `t=500`。

| Re | 单步速度误差 | 长期逐点速度 L2 | 振幅误差 | 主频误差 |
|---:|---:|---:|---:|---:|
| 99.0000 | 4.99% | 0.554 | 53.03% | 13.33% |
| 101.5000 | 5.96% | 0.826 | 46.54% | 6.90% |
| 110.3448 | 5.32% | 1.319 | 21.12% | 3.57% |
| 125.8621 | 6.24% | 1.348 | 0.69% | 3.70% |
| 141.3793 | 6.28% | 1.401 | 3.02% | 3.85% |
| 150.0000 | 7.12% | 1.394 | 3.65% | 3.85% |
| **平均** | **5.99%** | **1.140** | **21.34%** | **5.87%** |

### 6.3 冻结后 held-out 效果

4 个 held-out 工况均稳定到 `t=500`：

| Re | 单步速度误差 | 长期逐点速度 L2 | 振幅误差 | 主频误差 | 终态系数范数：预测 / 真值 |
|---:|---:|---:|---:|---:|---:|
| 100.5000 | 5.76% | 0.784 | 50.77% | 6.90% | 0.245 / 0.237 |
| 102.0000 | 6.01% | 0.830 | 44.13% | 6.90% | 0.254 / 0.245 |
| 120.6897 | 6.12% | 1.473 | 4.04% | 3.70% | 0.373 / 0.367 |
| 144.8276 | 6.57% | 1.416 | 3.40% | 3.85% | 0.517 / 0.496 |
| **平均** | **6.11%** | **1.126** | **25.58%** | **5.34%** | — |

同时，held-out 平均长期压力相对误差为 `0.898`。

### 6.4 如何解读“长期 L2 大于 1”

这不等同于模型完全失效。对于周期吸引子，主频只要相差数个百分点，在 `t=0–500` 的长时间窗口中就会不断积累相位差；即使振幅、频率、最终能量包络仍较接近，逐时刻 L2 也可能超过 1。

因此本项目必须同时看：

1. 单步误差：局部动力学是否正确；
2. 长时稳定性：是否出现数值爆炸或错误吸引子；
3. 极限环振幅和主频：周期轨道是否正确；
4. 长期逐点 L2：相位保持能力。

在充分周期的 Re≥120.69 held-out 区间，振幅误差不超过 4.1%、主频误差不超过 3.9%，说明该基线已经能够再现高 Re 周期运动的主导物理；其主要缺点是临界附近振幅低估和长期相位漂移。

---

## 7. 基于 periodic 基线的 CDM-GROM 首次研究

### 7.1 目标

CDM 不替代 FVM–Galerkin 主算子，而只拟合其在训练轨迹上的一步流映射残差：

\[
q_n=
\frac{a_{n+1}^{\mathrm{CFD}}-
\Phi_{\Delta t}^{\mathrm{FVM}}(a_n)}{\Delta t}.
\]

这里 `Phi` 是冻结的 FVM–GROM 在一个快照间隔上的 RK4 映射。训练只使用 periodic train，从中每隔若干时间转移抽样，最终使用 837 个训练转移。

### 7.2 连续 Delta-memory 状态

对每一个衰减率 `lambda_j`，引入 28 维最小固定地址连续记忆状态：

\[
\dot z_j=\lambda_j(\hat a-z_j).
\]

其中 `ahat` 为标准化 resolved POD 系数。闭合项由当前状态、Re 特征以及历史差 `z_j-ahat` 的线性读出组成：

\[
\dot a=F_{\mathrm{FVM}}(a;\nu)+r(a,z;\nu).
\]

消去 `z_j` 后，模型对应连续物理时间中的有限指数卷积核；它不是固定历史窗口，也不是离散 token 递推。

候选包括：

| 衰减率集合 | 记忆维数 |
|---|---:|
| 空集 | 0（代数残差校准对照） |
| [0.02] | 28 |
| [0.01, 0.05] | 56 |
| [0.005, 0.02, 0.08] | 84 |

对每个结构搜索 ridge=`0.1, 1, 10, 100`。所有候选仅在固定的 periodic validation 工况上滚动至 `t=500`，held-out 在此阶段没有被读取。

### 7.3 CDM 结果

按“validation 长期逐点速度 L2 最小”选出的最佳项是：

```text
rates = []
ridge = 1
memory dimension = 0
```

即最佳项根本没有连续记忆，只是代数残差校准。其指标为：

| 指标 | FVM–GROM 基线 | 零记忆校准 |
|---|---:|---:|
| 长期速度 L2 | 1.1403 | 1.0704 |
| 振幅误差 | 0.2134 | 0.4448 |
| 主频误差 | 0.0587 | 0.2439 |
| 单步速度误差 | 0.0599 | 0.0948 |

它仅把长期逐点 L2 降低约 6.1%，却使振幅误差增至约 2.08 倍、主频误差增至约 4.16 倍、单步误差增至约 1.58 倍。因此不能将该下降解释为动力学质量提升。

最佳真正 CDM 候选为：

```text
rates = [0.01, 0.05]
ridge = 10
memory dimension = 56
```

| 指标 | FVM–GROM 基线 | 最佳真实 CDM |
|---|---:|---:|
| 长期速度 L2 | 1.1403 | 1.0861 |
| 长期压力 L2 | 0.9425 | 0.9191 |
| 振幅误差 | 0.2134 | 0.4074 |
| 主频误差 | 0.0587 | 0.2764 |
| validation 数值稳定 | 是 | 是 |

它在长期逐点 L2 上改善约 4.75%，但振幅误差恶化约 90.9%、主频误差恶化约 371%。

### 7.4 物理门槛与判定

为避免只用一个长时 L2 指标作出错误结论，使用以下 validation 门槛：

1. 所有工况数值稳定；
2. 长期速度 L2 优于基线；
3. 振幅误差相对基线恶化不超过 10%；
4. 主频误差相对基线恶化不超过 10%。

没有任何真正 CDM 候选通过。因此：

- 当前 CDM 结构：**validation 拒绝**；
- CDM 冻结：**否**；
- CDM held-out 测试：**不运行**；
- periodic FVM–GROM 基线：**保持冻结且继续使用**。

---

## 8. 当前方法的边界与误差来源

### 已解决的问题

- 找到并使用了正确的正式 OpenFOAM 构建案例；
- 让速度主算子与 OpenFOAM 有限体积离散和边界条件保持一致；
- 消除了旧连续微分 VTK 算子的主要结构失配；
- 获得了在高 Re 周期轨道上稳定、振幅和频率均较准确的物理基线；
- 建立了防止 held-out 参与调参的冻结与哈希校验流程。

### 仍未解决的问题

1. `linearUpwind` 的分段非多项式性质不能被固定二次张量完全表达；
2. Re≈100–102 的 Hopf 临界附近增长率/振幅仍被明显低估；
3. 高 Re 主频仍有约 3.7%–3.9% 偏差，导致长期相位漂移；
4. 当前线性压力闭合虽有较小静态系数误差，但不是严格离散 Pressure–Poisson 消元；
5. 当前实指数固定地址 CDM 把“截断记忆、`linearUpwind` 张量残差、压力闭合误差和粗快照流映射误差”混合进同一残差，无法有效分离它们。

---

## 9. 推荐的后续路线

建议继续保持“先物理基线、再记忆闭合”的顺序：

1. 保留现有 OpenFOAM 离散 FVM 算子；
2. 研究可表达 `linearUpwind` 分段行为的分区/非多项式对流算子；
3. 采用 shift mode 或按 Re 构造平衡基流的参数化 POD，以改善 Hopf 临界增长率；
4. 构造严格的离散 Pressure–Poisson ROM，并显式处理速度边界 lifting；
5. 若继续研究 CDM，优先使用带共轭复极点的二维旋转记忆块：

\[
\dot z=(\sigma I+\omega J)z+C\phi(a),
\]

其中 `omega` 应先由 train 残差的频谱确定；然后只用 validation 对单步误差、振幅、频率和稳定性设置联合门槛。

不建议继续增加单纯实指数衰减记忆通道，因为第一次系统性验证已显示它无法改善周期相位动力学。

---

## 10. 代码、数据与报告位置

### 本地

```text
C:\Users\panxy1019\Documents\CHANNEL\centeredsquare_pod_galerkin_rebuild_20260803
C:\Users\panxy1019\Documents\CHANNEL\centeredsquare_periodic_cdm_grom_20260803
```

关键文件：

- `centeredsquare_pod_galerkin_rebuild_20260803/reports/POD_GALERKIN_REBUILD_REPORT.md`：FVM–POD–Galerkin 重建报告；
- `centeredsquare_pod_galerkin_rebuild_20260803/artifacts/periodic/frozen/FROZEN_METHOD.json`：正式 periodic 基线的冻结设置与哈希；
- `centeredsquare_pod_galerkin_rebuild_20260803/artifacts/periodic/evaluation/heldout/HELDOUT_METRICS.json`：最终 held-out 指标；
- `centeredsquare_periodic_cdm_grom_20260803/reports/PERIODIC_CDM_VALIDATION_REPORT.md`：CDM 验证报告；
- `centeredsquare_periodic_cdm_grom_20260803/artifacts/validation_v1/VALIDATION_GATE.json`：CDM 物理门槛审计。

### 虚拟机

```text
/home/ray/Desktop/centeredSquare/pod_galerkin_rebuild_v1
/home/ray/Desktop/centeredSquare/pod_galerkin_periodic_v1
/home/ray/Desktop/centeredSquare/pod_galerkin_rebuild_code
/home/ray/Desktop/centeredSquare/centeredsquare_periodic_cdm_grom_20260803
```

## 11. 最终状态

| 项目 | 状态 |
|---|---|
| 原始数据与算例核验 | 通过 |
| 离散一致 FVM–POD–Galerkin 重建 | 通过 |
| Hopf 物理基线 | 已修正，但不作为正式基线 |
| periodic 物理基线 | 已冻结，推荐使用 |
| periodic held-out 测试 | 已完成 |
| 固定地址实指数 CDM | validation 拒绝 |
| CDM held-out 测试 | 未运行，避免数据泄漏 |
