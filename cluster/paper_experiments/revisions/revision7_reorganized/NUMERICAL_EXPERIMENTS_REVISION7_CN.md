# 4. 数值实验

> **版本说明。** 本文是基于冻结 revision 6、原始结果文件与审计资产重新组织的独立 revision 7。未启动训练，未重跑 evaluator，未修改 checkpoint，也未补算缺失指标。除相位（rad 或 cycle）、权重和计数外，表中相对误差均按原始 relative error × 100% 表示。正文仅保留六张主表；逐 Re 细表、失败记录和哈希移至附录或配套审计文件。

## 4.1 流动设置、数据集与评价协议

### 4.1.1 CFD 配置

研究对象为二维不可压缩圆柱绕流。圆柱直径 \(D=1\)，运动黏度 \(\nu=10^{-3}\)，入口速度按 \(U_\infty=Re\,\nu/D\) 设置。冻结 OpenFOAM 13 case 的计算域为
\[
x\in[-10,20],\qquad y\in[-10,10],
\]
并在 \(z\in[-0.05,0.05]\) 上以单层网格实现二维计算。圆柱中心位于原点、半径为 0.5。速度边界条件为：入口定值速度、出口零法向梯度、圆柱无滑移、上下边界 slip、前后面 empty；压力边界条件为：出口定值零压，入口、上下边界和圆柱采用零法向梯度，前后面 empty。

三套局部资产使用相同的 97,368 点网格、点顺序和 lumped point-area 权重。冻结 case 的求解器时间步从 \(\Delta t=0.002\) 起步，采用 `adjustTimeStep=yes`、`maxCo=0.8` 和 `maxDeltaT=0.2`。数据库写出间隔随 Re 和轨迹合同变化，而不是统一常数；因此本文把 \(K\) 解释为原生数据库步数。已有冻结时间表显示，例如 Steady K56 对应的物理时长随 Re 从 431.274 变化到 234.604，Hopf K56 从 1309.781 变化到 1179.953，Periodic K48 从 598.701 变化到 177.959。不同流态的 KLong 不能直接解释为相同物理时间。

### 4.1.2 局部坐标图与数据隔离

Steady、Hopf 和 Periodic specialist 分别保留独立的 train-only 流态均值、速度/压力 POD 基、scaler、Galerkin 算子、pressure closure、history builder 和原生积分器。所有正式对比均取 \(r_u=r_p=32\)，history length 为 3。压力在投影和评价前逐快照减去面积加权均值，即 `subtract_area_mean_per_snapshot`。三个 specialist 的 train、validation 和 test 按完整 Reynolds 数和完整轨迹隔离，不随机拆分同一轨迹窗口。

Steady 数据覆盖 \(Re=20.0\) 至 46.440072；原始隔离为 14 个 train Re、2 个 validation Re 和 4 个 held-out Re。Hopf 扩展资产覆盖 \(Re=45.5\) 至 59.201432；隔离为 29/2/3 个 train/validation/test Re，其中正式 test 为 47.081356、49.022357 和 51.786450。Periodic 资产包含 63 条 Re 轨迹，隔离为 53/6/4，正式 held-out 为 70.314635、100.352251、149.059229 和 189.862278。融合模块仍按 complete-Re/complete-trajectory 隔离；S–H sealed test cache 为 42.359071、43.50 和 43.90。

`Global MoE†` 使用全局 POD、mean 和 scaler 的 all-Re representation，其中包含 held-out Re。它因此是 **transductive all-Re representation baseline**，不是严格 train-only POD 的 inductive held-out baseline。该符号在全文保持不变。

### 4.1.3 评价口径

本文仅在原生长时域上报告主结果：Steady、Hopf 与 Global MoE† 使用 K56，Periodic specialist 使用 K48。物理场误差是面积加权全场相对 \(L_2\)；POD 系数误差衡量去均值后的模态动力学。Global MoE† 的全场误差包含 Re-specific mean-field 能量，可能显著小于系数误差；两者回答不同问题，不能互相替代。

Strict attractor 判据按流态定义：

- Steady 联合门要求 finite fraction=1、发散窗为 0、pressure fixed-point error \(\le5\%\) 且 pressure drift \(\le5\%\)。扰动收缩还需考察 paired perturbation gain；gain 小于 1 才表示对应方向收缩。
- Hopf 联合门要求 K56 速度/压力场误差不超过 5%、振幅和能量漂移不超过 10%、频率误差不超过 5%、相位漂移绝对值不超过 0.25 cycle、归一化轨道距离不超过 10%，并同时满足 finite=1、零发散和无 false growth。
- Periodic 联合门要求 K48 速度/压力场误差不超过 5%、振幅和能量漂移不超过 10%、Strouhal 误差不超过 5%、末周期漂移绝对值不超过 0.25 cycle、归一化轨道距离不超过 10%，并满足 finite=1 和零发散。

上述定义意味着“finite”只是 strict PASS 的必要条件，低全场误差也不能替代正确的相位、频率、轨道或固定点动力学。

## 4.2 流态专属 specialist 的能力

本节只回答三个完整 Proposed specialist 是否能在各自局部坐标图内稳定工作，不混入 Global、E2、T2-C、Vanilla 或 DataOnly。

### 4.2.1 原生坐标图中的长时物理场预测

**表 1　Proposed specialist 的原生长时场结果。** Steady 行使用冻结的四 held-out 原生汇总；Hopf 和 Periodic 分别使用正式 held-out K56/K48 汇总。不同流态之间不构造 overall mean。

| Regime | Proposed specialist | Test Re | 原生时域 | 速度场误差 | 压力场误差 | 联合误差 | Worst | Finite | 发散窗 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Steady | Proposed Steady Specialist MoE | 24.630436, 32.740068, 39.685479, 45.142703 | K56 | 0.1245% | 7.0362% | 3.5803% | 9.4823% | 1.000 | 0 |
| Hopf | Proposed Hopf Specialist MoE | 47.081356, 49.022357, 51.786450 | K56 | 0.0127% | 0.1697% | 0.0912% | 0.2305% | 1.000 | 0 |
| Periodic | Proposed Periodic Specialist MoE | 70.314635, 100.352251, 149.059229, 189.862278 | K48 | 0.8132% | 3.4445% | 2.1289% | 3.6537% | 1.000 | 0 |

Steady rollout 在 K56 上保持有限，但该事实不等同于压力固定点 strict 保持。Hopf 在三个正式 held-out Re 上以零发散完成 K56，说明局部 Hopf 模型具有稳定的未见参数场预测能力。Periodic 在四个 held-out Re 上完成 K48，且三个 Re 通过完整周期吸引子门。三者构成可运行的局部动力学部件，但其能力边界并不相同。

### 4.2.2 原生坐标图中的吸引子保持

**表 2　Proposed specialist 的吸引子证据。** “范围”严格沿用冻结 evaluator；未在同一合同下冻结的字段标为 N/A。

| Regime / population | Fixed point / drift | 振幅、频率、相位与轨道 | Tail / terminal | Perturbation / false oscillation | Strict 结果 |
|---|---|---|---|---|---|
| Steady 全 train/validation/heldout（20 Re） | pressure FP 0.7889%–60.7548%；drift 8.8935%–145.6516% | 不适用 | 同一全 split strict scan 未冻结独立 tail/terminal 汇总，N/A | K56 worst paired gain=585.262；仍 \(>1\)。false-oscillation 未冻结独立二值计数，N/A | 联合门 **0/20 PASS** |
| Hopf 正式 held-out（47.081356, 49.022357, 51.786450） | 不适用 | K56 finite=1、零发散；未通过完整 strict conjunction | 由 Hopf 周期诊断取代 | 无 false growth 是 strict 条件之一 | **0/3 PASS** |
| Hopf 训练点机制案例（49.3, 49.6, 50.0） | 不适用 | RMS 振幅 1.3430%–2.9146%；频率 1.5220%–1.6477%；相位 -0.1166 至 -0.1136 cycle；轨道 6.4660%–7.2697% | 周期闭合由相位与轨道联合评价 | finite=1、发散窗=0、false growth=否 | **3/3 PASS（train）** |
| Periodic held-out strict cases（100.352249, 149.059235, 189.862274） | 不适用 | RMS 振幅 0.4335%–0.8792%；频率 0.1320%–0.3816%；phase RMS 0.0223–0.0702 rad；轨道 2.4061%–4.4317% | 末周期漂移 -0.0241 至 0.0029 cycle | finite=1、发散窗=0 | **3/3 PASS；完整 held-out 3/4** |

#### Steady

完整扫描没有任何 Re 同时满足压力 fixed-point error 和 pressure drift 的 5% 联合门。因而不能写成“Proposed Steady 已严格保持压力固定点”。冻结对照仍显示，Proposed S3-B 的 held-out K56 pressure fixed-point worst 为 27.1317%，低于 Vanilla 的 49.4685%；K56 worst paired gain 从 107702.891 降至 585.262。该证据支持“固定点误差和扰动放大显著降低”，但因 worst gain 仍远大于 1，不支持“所有扰动方向严格收缩”。

#### Hopf：正式 held-out 与 in-sample strict mechanism cases

正式 held-out Re=47.081356、49.022357、51.786450 的 K56 场预测均有限且零发散，但 strict attractor 计数为 0/3。Re=49.3、49.6、50.0 则同时通过场误差、振幅、频率、相位、轨道、能量、finite、divergence 与 false-growth 联合门。这三个 Re 属于训练点，只证明模型能在代表性参数点保持 Hopf 机制，不证明未见 Re 的 strict 泛化。

#### Periodic

Re=100.352249、149.059235、189.862274 均在 held-out K48 通过联合门；Re=70.314635 因 K48 压力误差 5.9601% 超过 5% 阈值而未通过，尽管其数值有限且周期并未发散。因此 Periodic 证据是 3/4 held-out strict preservation，而非无条件 4/4。

总体而言，三个 Proposed specialist 是可运行但能力域不同的局部动力学组件：Hopf 的严格案例目前属于 in-sample mechanism evidence，Steady 尚未通过压力固定点与漂移联合门，Periodic 则具有 held-out 周期吸引子证据。

## 4.3 完整系统比较

完整系统仅包括 `Global MoE†`、E2 Top-1 multi-chart routing 和 Proposed admissibility-constrained multi-chart MoE-ROM。Proposed 系统冻结为：

```text
Steady core        → S-only
S–H overlap        → T2-C window-conditioned convex physical-field fusion
Hopf core          → H-only
H–P / Periodic     → P-only / native-time hard routing
```

T2-C 只是 Proposed 系统在 S–H overlap 内的局部机制，不是完整方法的总称。

### 4.3.1 分区域长时场精度

**表 3　三个完整系统的分区域结果。** Core 区域中 E2/Proposed 的数值来自其确定性 Top-1 输出与冻结 native specialist 的输出同一性，不是新增 evaluator；H–P 边界没有合法 test 双路 cache，故只报告 validation admissibility。不同 horizon 的行不作直接排名。

| 区域 | 完整系统 | Re / split | 时域 / 误差空间 | 速度 | 压力 | 联合 | Worst | Finite / 发散 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---|---|
| Steady core | Global MoE† | 24.630436, 32.740067, 39.685478, 45.142704 test | K56 / 物理场 | 0.1809% | 0.6639% | 0.4224% | 0.8327% | 1.000 / 0 | coeff u/p=30.4250%/60.5898% |
| Steady core | E2 Top-1 | 同上 | K56 / native S physical field | 0.1245% | 7.0362% | 3.5803% | 9.4823% | 1.000 / 0 | 确定性 S-only 输出 |
| Steady core | Proposed system | 同上 | K56 / native S physical field | 0.1245% | 7.0362% | 3.5803% | 9.4823% | 1.000 / 0 | S-only |
| S–H boundary | Global MoE† | sealed Re 无完整精确匹配 | N/A | N/A | N/A | N/A | N/A | N/A | 禁止用近邻 Re 替代 |
| S–H boundary | E2 Top-1 | 42.359071, 43.50, 43.90 test | K56 / sealed physical field | 0.0627% | 4.6083% | 4.6710% | 15.6584% | 1.000 / 0 | Top-1 |
| S–H boundary | Proposed system | 同上 | K56 / sealed physical field | 0.0429% | 3.0021% | 3.0450% | 14.5023% | 1.000 / 0 | T2-C |
| Hopf core | Global MoE† | 47.081356, 49.022358, 51.786449 test | K56 / 物理场 | 0.0187% | 0.1880% | 0.1033% | 0.1532% | 1.000 / 0 | coeff u/p=55.1998%/46.9329% |
| Hopf core | E2 Top-1 | 47.081356, 49.022357, 51.786450 test | K56 / native H physical field | 0.0127% | 0.1697% | 0.0912% | 0.2305% | 1.000 / 0 | 确定性 H-only 输出 |
| Hopf core | Proposed system | 同上 | K56 / native H physical field | 0.0127% | 0.1697% | 0.0912% | 0.2305% | 1.000 / 0 | H-only |
| H–P boundary | Global MoE† | P-native validation | N/A | N/A | N/A | N/A | N/A | N/A | 未运行同一双路门 |
| H–P boundary | E2 Top-1 | P-native validation 6 Re | K56 admissibility | N/A | N/A | N/A | N/A | P 6/6 finite；H 0/6 | P-only |
| H–P boundary | Proposed system | 同上 | K56 admissibility | N/A | N/A | N/A | N/A | \(m_H=0,m_P=1\) | fail closed，未训练凸门 |
| Periodic core | Global MoE† | 70.314636, 100.352249, 149.059235, 189.862274 test | K56 / 物理场 | 1.2147% | 4.9544% | 3.0846% | 4.8076% | 1.000 / 0 | coeff u/p=13.5459%/16.1058% |
| Periodic core | E2 Top-1 | 70.314635, 100.352249, 149.059235, 189.862274 test | K48 / native P physical field | 0.8132% | 3.4445% | 2.1289% | 3.6537% | 1.000 / 0 | P-only |
| Periodic core | Proposed system | 同上 | K48 / native P physical field | 0.8132% | 3.4445% | 2.1289% | 3.6537% | 1.000 / 0 | P-only；与 E2 相同 |

Global MoE† 的物理场误差在部分区域很小，但其系数误差明显更大，且 representation 含 held-out Re。表 3 因而不是严格 inductive 的全域胜负表。K48 与 K56 也未混合成总体均值。

### 4.3.2 S–H 边界：瞬态精度与局部吸引子

**表 4　sealed S–H test cache 上的逐 Re 结果。** 场、tail 和 orbit 列为百分数；\(\alpha\) 为无量纲权重。Routing regret 与 per-window oracle gap 使用冻结报告中的 all-horizon overall-mean 定义，而非 K56-only 定义。

| Re | 方法 | K56 joint | Tail center | Amp. abs. | Orbit | Tail inc. | Terminal inc. | \(\alpha_S/\alpha_H\) | Routing regret | Per-window oracle gap |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 42.359071 | E2 Top-1 | 4.6503% | 0.0513% | 0.0018% | 1.9375% | 0.0046% | 0.0054% | N/A | N/A | N/A |
| 42.359071 | T2-C | 4.6433% | 0.0511% | 0.0018% | 1.9392% | 0.0046% | 0.0054% | 0.9987 / 0.0013 | -0.0084% | 0.1019% |
| 43.500000 | E2 Top-1 | 2.4877% | 0.0551% | 0.0009% | 1.6198% | 0.0064% | 0.0085% | N/A | N/A | N/A |
| 43.500000 | T2-C | 3.2343% | 0.0394% | 0.0002% | 0.2458% | 0.0020% | 0.0017% | 0.0861 / 0.9139 | 0.3431% | 1.2198% |
| 43.900000 | E2 Top-1 | 6.8749% | 0.0574% | 0.0009% | 1.6135% | 0.0063% | 0.0072% | N/A | N/A | N/A |
| 43.900000 | T2-C | 1.2574% | 0.0215% | 0.0005% | 0.5202% | 0.0015% | 0.0016% | 0.1891 / 0.8109 | -1.1903% | 0.000017% |

Re=42.359071 时，T2-C 的平均 \(\alpha_S=0.9987\)，几乎退化为 S-only，从而基本保持低 Re 核心侧。Re=43.50 是必须保留的反例：T2-C 的 K56 全场误差从 2.4877% 增至 3.2343%，但 tail center、振幅、轨道、tail increment 和 terminal increment 均改善。Re=43.90 则同时改善 K56 全场误差和局部吸引子，并接近 per-window convex oracle。由此可见，transient physical-field fidelity 与 local-attractor fidelity 必须分别评价；结果不支持“T2-C 对所有 Re、窗口和 worst 指标全面占优”。

### 4.3.3 H–P admissibility 退化

在 P-native development 域中，Periodic specialist 对 53/53 train 和 6/6 validation Re 均完成有限 K56；Hopf 仅对 5/53 train Re 有限，并在 validation 上为 0/6。联合有效的五个 Re 全部属于 train，因而不存在可用于模型选择的 validation-supported H–P band。正式门据此冻结为
\[
m_H=0,\qquad m_P=1.
\]

Proposed 系统在 H–P 边界机械退化为 P-only/native-time hard routing；E2 和 Proposed 在该区域输出相同。没有创建 H–P native output cache，没有启动 T2-C、RiskPrediction 或 LookAhead 训练，也不存在 H–P best/last checkpoint。这不是未完成的融合实验，而是 admissibility principle 的 fail-closed 输出；不能写成“H–P 获得了软融合增益”。

## 4.4 Specialist 设计消融

本节只比较 Vanilla-FNN-MoE、DataOnly-MoE 和 Proposed Specialist MoE。Global、E2 与 T2-C 不进入消融主表。

### 4.4.1 Expert 架构与 physics–data 消融

Vanilla 仅将 proposed expert 的特殊残差、线性与低秩二次支路替换为标准 FNN；POD、router、expert 数量、输入、history、RK4、物理算子、loss、split、预算和 evaluator 保持不变。DataOnly 则移除连续时间局部向量场、RK4、Galerkin RHS、物理投影、pressure-correction physics 和 physical residual loss，直接学习离散状态映射。

**表 5　Specialist 长时消融。** Steady 使用三方法共同可评估 Re；Hopf 和 Periodic 使用各自冻结测试人口。

| Regime | 方法 | Test Re | 时域 | 速度 | 压力 | 联合 | Worst | Finite / 发散 | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Steady | Vanilla-FNN-MoE | 24.630436, 32.740068, 39.685479 | K56 | 0.0890% | 4.1482% | 2.1186% | 3.8043% | 1.000 / 0 | user-frozen step 6200 |
| Steady | DataOnly-MoE | 同上 | K56 | 0.2569% | 14368.8903% | 7184.5736% | 13637.9888% | 1.000 / 3 | DONE with divergence |
| Steady | Proposed Specialist MoE | 同上 | K56 | 0.1370% | 8.3817% | 4.2593% | 9.4299% | 1.000 / 0 | DONE |
| Hopf | Vanilla-FNN-MoE | validation fail closed | K16/K56 | N/A | N/A | N/A | N/A | non-finite | 未开放 test |
| Hopf | DataOnly-MoE | 47.081356, 49.022357, 51.786450 | K56 | 0.2204% | 1408.8483% | 704.5344% | 791.9889% | 1.000 / 39 | DONE with divergence |
| Hopf | Proposed Specialist MoE | 同上 | K56 | 0.0127% | 0.1697% | 0.0912% | 0.2305% | 1.000 / 0 | DONE |
| Periodic | Vanilla-FNN-MoE | 70.314635, 100.352251, 149.059229, 189.862278 | K48 | 2.2109% | 9.5010% | 5.8559% | 8.2047% | 1.000 / 0 | DONE |
| Periodic | DataOnly-MoE | 同上 | K48 | 1.0107% | 4.2341% | 2.6224% | 6.2089% | 1.000 / 0 | DONE |
| Periodic | Proposed Specialist MoE | 同上 | K48 | 0.8132% | 3.4445% | 2.1289% | 3.6537% | 1.000 / 0 | DONE |

Steady 的 clean-field 均值并不支持 Proposed 全面优于 Vanilla；Proposed 的主要证据来自固定点与扰动鲁棒性。Hopf 中，Vanilla 在 validation 长时非有限，DataOnly 出现 39 个发散窗，而 Proposed 在 test K56 上零发散，说明连续时间 physics–data 结构对 Hopf 长期稳定性具有关键作用。Periodic 中 Proposed 的场误差和 worst 均优于 Vanilla，并低于 DataOnly，但周期 strict 结果还需单独考察。

### 4.4.2 吸引子消融

**表 6　Specialist 吸引子消融。** 不同流态使用各自 evaluator；“N/A”与“FAIL”严格区分。

| Regime / population | Vanilla-FNN-MoE | DataOnly-MoE | Proposed Specialist MoE | 客观解释 |
|---|---|---|---|---|
| Steady held-out / full scan | pressure FP worst 49.4685%；K56 worst paired gain 107702.891；无统一 strict pass count | 三个共同 test Re 各 1 个 norm-divergent window；同 perturbation-bank strict N/A | pressure FP worst 27.1317%；K56 worst gain 585.262；完整 20 Re 联合门 0 PASS | Proposed 显著降低误差与放大，但未证明全方向收缩 |
| Hopf train mechanism Re=49.3,49.6,50.0 | validation 非有限，test/case study N/A | 0/3 PASS；频率误差 93.7201%–97.0839%；相位漂移 -6.6525 至 -6.2293 cycle；每 Re 5 发散窗；false growth=是 | 3/3 PASS；频率 1.5220%–1.6477%；相位 -0.1166 至 -0.1136 cycle；零发散；false growth=否 | 强证据支持 physics–data 结构，但三点属于 train |
| Periodic selected held-out Re=100.352249,149.059235,189.862274 | 1/3 PASS（完整 held-out 1/4） | 3/3 PASS | 3/3 PASS（完整 held-out 3/4） | Proposed 相对 Vanilla 更鲁棒；DataOnly 也能保持这三个极限环 |

Periodic 的 DataOnly 在三个指定 held-out Re 上均获得 strict PASS，必须保留这一负向消融结果。因此可以写 Proposed 在多数场误差、相位或轨道指标上更准确，且相对 Vanilla 的 held-out 鲁棒性更高；不能写成“只有 Proposed 能保持周期极限环”。

### 4.4.3 综合解释

特殊 expert 架构的贡献具有流态依赖性：它在 Periodic 相对 Vanilla 的场误差、worst 和 strict 通过率上表现清晰，在 Steady clean-field 均值上却没有全面优势。Physics–data 结构的最强证据来自 Hopf：移除连续时间物理推进后出现大压力误差、39 个长时发散窗以及指定 Re 的 false growth。Periodic DataOnly 的 3/3 指定 strict PASS 则表明，周期吸引子在该数据覆盖下也可由离散映射学习；这限制了“物理结构是周期保持唯一原因”的主张。

## 4.5 限制与结论边界

### 可以支持

- 三个流态专属 specialist 构成具有不同动力学能力域的局部 ROM。
- Physics–data 结构对 Hopf 的长期稳定性和代表性训练点的严格动力学保持具有关键作用。
- Proposed Periodic 相对 Vanilla 在 held-out 极限环保持上更鲁棒。
- S–H T2-C 可在共同可行域内改善平均局部吸引子表现，并在 Re=43.90 同时改善场误差。
- Admissibility gate 能在候选 expert 缺乏 validation-supported rollout 时自动退化为 Top-1。

### 不可以支持

- Proposed 在所有流态、误差、Re、窗口和 worst 指标全面占优。
- Steady 已实现全方向严格固定点收缩。
- Hopf 的 Re=49.3、49.6、50.0 证明了 held-out strict 泛化。
- Global MoE† 与 Proposed 已完成严格 inductive、同 representation、同 horizon 的全域排名。
- H–P 已实现有效软融合。
- Router 已学习真实 S→H→P 连续时间迁移。

本轮所有消融只使用单一固定 seed，不支持跨 seed 显著性。Global MoE† 使用 all-Re representation。S–H 历史 test 曾被查看，因而不能称为全程 blind test。Steady pressure fixed-point+drift 联合 strict pass 为 0。H–P 没有共同 validation-supported 可行域，也没有训练合法软融合门。Router 数据不包含经认证的真实 S→H→P 连续演化轨迹。最后，不同流态的 KLong 对应不同原生时间，不能据此计算未经校准的统一全域平均。

## 4.6 可追溯性

每张主表和关键结论的源文件、键、Re、horizon、误差空间、split 和 SHA256 均记录于 `REVISION7_SOURCE_MAP.json`。Revision 6 主稿 SHA256 为 `f4a5fd04b981a90f665f9e8abb462ff508d9011bd13075f986aeb7c1ededb35c`，本 revision 7 未覆盖 revision 6。

逻辑检查结果为：

```text
局部 specialist 可用
→ 完整系统比较
→ S–H/H–P 边界机制
→ specialist 内部消融
→ 限制与结论边界
```

该顺序已满足本轮重组合同。

---

# 附录 A　从正文移出的详细资产

## A.1 Global MoE† 全部逐 Re 结果

| Regime | Re | K56 速度场 | K56 压力场 | 速度系数 | 压力系数 | Finite / 发散 |
|---|---:|---:|---:|---:|---:|---|
| Steady | 24.630436 | 0.3404% | 1.3249% | 44.2564% | 101.0296% | 1.000 / 0 |
| Steady | 32.740067 | 0.1800% | 0.6831% | 31.2366% | 74.5202% | 1.000 / 0 |
| Steady | 39.685478 | 0.1186% | 0.3976% | 25.4158% | 41.5445% | 1.000 / 0 |
| Steady | 45.142704 | 0.0845% | 0.2500% | 20.7911% | 25.2651% | 1.000 / 0 |
| Hopf | 47.081356 | 0.0268% | 0.2199% | 46.6749% | 29.9005% | 1.000 / 0 |
| Hopf | 49.022358 | 0.0183% | 0.2882% | 24.1395% | 32.6218% | 1.000 / 0 |
| Hopf | 51.786449 | 0.0110% | 0.0560% | 94.7850% | 78.2764% | 1.000 / 0 |
| Periodic | 70.314636 | 1.5601% | 8.0550% | 23.7912% | 28.0363% | 1.000 / 0 |
| Periodic | 100.352249 | 1.1484% | 4.3387% | 11.7726% | 13.4181% | 1.000 / 0 |
| Periodic | 149.059235 | 0.7461% | 2.7039% | 6.7727% | 8.2213% | 1.000 / 0 |
| Periodic | 189.862274 | 1.4043% | 4.7200% | 11.8473% | 14.7476% | 1.000 / 0 |

这些行只说明冻结 Global wrapper 在相应数据库节点上的 K56 行为。Global 没有运行各局部 specialist 的同合同 strict evaluator。

## A.2 Proposed Steady 全 split pressure fixed-point 扫描

联合门为 pressure fixed-point error \(\le5\%\)、pressure drift \(\le5\%\)、finite=1、发散窗=0。

| Split | Re | 速度场 | 压力场 | Pressure FP | Pressure drift | Strict |
|---|---:|---:|---:|---:|---:|---|
| train | 20.000000 | 0.3643% | 33.0479% | 60.7548% | 31.2046% | FAIL |
| train | 22.535675 | 0.2468% | 26.1227% | 38.8054% | 34.9110% | FAIL |
| train | 26.667332 | 0.1953% | 16.5687% | 18.5122% | 33.1805% | FAIL |
| train | 30.720428 | 0.1197% | 8.0281% | 8.7574% | 36.0012% | FAIL |
| train | 34.737568 | 0.0469% | 2.9230% | 2.2360% | 28.5002% | FAIL |
| train | 36.657768 | 0.0275% | 1.3850% | 0.7889% | 84.3329% | FAIL |
| train | 38.357250 | 0.0304% | 1.4185% | 2.1137% | 145.6516% | FAIL |
| train | 40.711525 | 0.0456% | 1.6044% | 3.8781% | 27.2780% | FAIL |
| train | 41.576576 | 0.0474% | 1.5962% | 4.5306% | 23.6479% | FAIL |
| train | 42.359070 | 0.0595% | 1.6744% | 5.2393% | 17.0534% | FAIL |
| train | 43.093925 | 0.0480% | 1.9856% | 5.8982% | 23.8484% | FAIL |
| train | 44.478352 | 0.0640% | 1.9377% | 5.5003% | 12.5380% | FAIL |
| train | 45.795193 | 0.0721% | 2.0826% | 7.9359% | 8.8935% | FAIL |
| train | 46.440071 | 0.0652% | 2.3200% | 8.2121% | 18.7922% | FAIL |
| validation | 28.695137 | 0.1508% | 11.0850% | 13.7375% | 32.4259% | FAIL |
| validation | 43.797401 | 0.0538% | 1.9672% | 6.5408% | 22.0419% | FAIL |
| heldout | 24.630436 | 0.2984% | 18.6741% | 27.1048% | 35.5309% | FAIL |
| heldout | 32.740067 | 0.0748% | 5.0818% | 5.0171% | 21.9365% | FAIL |
| heldout | 39.685478 | 0.0384% | 1.5265% | 3.1507% | 63.4742% | FAIL |
| heldout | 45.142704 | 0.0618% | 2.2503% | 5.6249% | 75.2493% | FAIL |

联合 strict 通过数为 0。仅满足 pressure fixed-point error <5% 的次级六 Re 为 train 34.737568、36.657768、38.357250、40.711525、41.576576，以及 heldout 39.685479；它们均因 drift 超阈值而不是 strict pass。

## A.3 DataOnly Hopf 指定训练 Re

| Re | K56 速度 | K56 压力 | RMS 振幅 | 频率 | 相位漂移/cycle | 轨道 | 发散窗 | False growth | Strict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 49.3 | 0.1361% | 1118.1745% | 711.6247% | 96.9669% | -6.2293 | 586.6544% | 5 | 是 | FAIL |
| 49.6 | 0.0783% | 1183.8779% | 530.2295% | 97.0839% | -6.4035 | 429.4772% | 5 | 是 | FAIL |
| 50.0 | 0.0743% | 1627.9665% | 139.3650% | 93.7201% | -6.6525 | 122.2574% | 5 | 是 | FAIL |

## A.4 DataOnly Periodic 指定 held-out Re

| Re | K48 速度 | K48 压力 | RMS 振幅 | 频率 | Phase RMS/rad | 末周期漂移 | 轨道 | Strict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100.352249 | 0.8891% | 3.1271% | 0.7753% | 0.0852% | 0.0376 | -0.0097 | 4.7728% | PASS |
| 149.059235 | 0.7854% | 2.3207% | 0.3444% | 0.0905% | 0.0326 | -0.0097 | 4.3304% | PASS |
| 189.862274 | 0.8509% | 2.2511% | 0.5757% | 0.0507% | 0.0314 | 0.0009 | 4.6841% | PASS |

## A.5 主要 N/A 与失败原因

| 项目 | 状态 | 原因 |
|---|---|---|
| Global MoE† on sealed S–H cache | N/A | 没有三个精确 sealed Re 的同 evaluator 结果，禁止近邻替代 |
| Global on Hopf 49.3/49.6/50.0 | N/A | Global 数据库没有这些精确 Re |
| Hopf Vanilla test | FAIL CLOSED | validation K16/K56 长时非有限，未开放 test |
| Global Steady strict fixed-point | N/A | 冻结接口不支持数据库终点后的同合同 continuation |
| H–P soft fusion | FAIL CLOSED | Hopf 在 P-native validation 为 0/6 finite K56；无 validation-supported overlap |
| Router true regime migration | 未认证 | 训练数据没有真实 S→H→P 连续迁移轨迹 |

完整 SHA256、源键与表格映射见 `REVISION7_SOURCE_MAP.json`；未解决缺口见 `REVISION7_UNRESOLVED_LIMITATIONS.md`。
