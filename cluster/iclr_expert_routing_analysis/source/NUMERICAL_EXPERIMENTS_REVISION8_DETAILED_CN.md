# 4. 数值实验（revision 8：逐 Test Re 详细版）

> **版本与口径。** 本稿是独立新增的 revision 8，不覆盖 revision 6 或 revision 7。未训练模型、未运行 evaluator、未修改 checkpoint，也未补算缺失指标。除相位（rad/cycle）、权重、计数和 finite fraction 外，全部误差均为冻结 relative error ×100%。正文不再优先给出跨 Re 平均值；凡涉及 Test Re，均逐 Re 列出物理场误差与可用的 POD 系数误差。

## 4.1 评价合同与缺失值规则

Steady 与 Hopf specialist 使用 K56，Periodic specialist 使用 K48；Global MoE† 的冻结补充 evaluator 使用 K56。K 表示原生数据库步数，不代表跨流态统一的物理时间。三个 specialist 保留各自 POD、mean、scaler、history、Galerkin、pressure closure 与原生推进合同。压力采用冻结的逐快照面积加权去均值 gauge。

表中缺失标记含义如下：

1. `N/A¹`：DataOnly 冻结 evaluator 未保存 POD coefficient error；
2. `N/A²`：Hopf Vanilla 在 validation K16/K56 非有限，按合同未开放 test；
3. `N/A³`：对应冻结 evaluator 未运行或未保存该吸引子字段；
4. `N/A⁴`：Global wrapper 只保存长时场/POD 系数指标，没有运行 local-specialist strict attractor evaluator；
5. `N/A⁵`：Global 数据库无 Hopf 指定 Re=49.3/49.6/50.0 的精确资产；
6. `N/A⁶`：S–H sealed cache 只保存 joint field 与吸引子聚合，没有速度/压力分量或 POD 系数误差。

`N/A` 不是零误差，也不是 PASS。本文不从联合误差反推速度/压力，不从物理场误差反推 POD 系数，也不把不同 horizon 混成统一均值。

## 4.2 长时物理场与 POD 系数预测

### 4.2.1 流态专用 specialist 消融与 Global 对照

**表 1　全部冻结方法的逐 Test Re 长时误差。**

| Regime | 方法 | Test Re | 时域 | 速度场误差 | 压力场误差 | 速度 POD 系数误差 | 压力 POD 系数误差 | Finite | 发散窗 |
|---|---|---|---|---|---|---|---|---|---|
| Steady | Proposed Specialist MoE | 24.630436 | K56 | 0.2995% | 18.5603% | 0.4201% | 15.1009% | 1.000 | 0 |
| Steady | Proposed Specialist MoE | 32.740068 | K56 | 0.0730% | 5.0620% | 0.5373% | 16.4108% | 1.000 | 0 |
| Steady | Proposed Specialist MoE | 39.685479 | K56 | 0.0384% | 1.5226% | 0.3999% | 15.5819% | 1.000 | 0 |
| Steady | Proposed Specialist MoE | 45.142703 | K56 | 0.0618% | 1.9665% | 0.2562% | 6.5857% | 1.000 | 0 |
| Steady | Vanilla-FNN-MoE | 24.630436 | K56 | 0.1723% | 7.4363% | 0.2613% | 6.0504% | 1.000 | 0 |
| Steady | Vanilla-FNN-MoE | 32.740068 | K56 | 0.0619% | 2.7920% | 0.3858% | 9.0424% | 1.000 | 0 |
| Steady | Vanilla-FNN-MoE | 39.685479 | K56 | 0.0330% | 2.2163% | 0.3482% | 22.6792% | 1.000 | 0 |
| Steady | Vanilla-FNN-MoE | 45.142703 | K56 | 0.0699% | 3.9201% | 0.2480% | 13.1228% | 1.000 | 0 |
| Steady | DataOnly-MoE | 24.630436 | K56 | 0.5122% | 27275.4654% | N/A¹ | N/A¹ | 1.000 | 1 |
| Steady | DataOnly-MoE | 32.740068 | K56 | 0.1790% | 10365.5271% | N/A¹ | N/A¹ | 1.000 | 1 |
| Steady | DataOnly-MoE | 39.685479 | K56 | 0.0795% | 5465.6783% | N/A¹ | N/A¹ | 1.000 | 1 |
| Steady | DataOnly-MoE | 45.142703 | K56 | N/A | N/A | N/A | N/A | N/A | N/A |
| Steady | Global MoE† | 24.630436 | K56 | 0.3404% | 1.3249% | 44.2564% | 101.0296% | 1.000 | 0 |
| Steady | Global MoE† | 32.740068 | K56 | 0.1800% | 0.6831% | 31.2366% | 74.5202% | 1.000 | 0 |
| Steady | Global MoE† | 39.685479 | K56 | 0.1186% | 0.3976% | 25.4158% | 41.5445% | 1.000 | 0 |
| Steady | Global MoE† | 45.142703 | K56 | 0.0845% | 0.2500% | 20.7911% | 25.2651% | 1.000 | 0 |
| Hopf | Proposed Specialist MoE | 47.081356 | K56 | 0.0032% | 0.0355% | 0.0418% | 0.2164% | 1.000 | 0 |
| Hopf | Proposed Specialist MoE | 49.022357 | K56 | 0.0288% | 0.4321% | 0.8479% | 5.7742% | 1.000 | 0 |
| Hopf | Proposed Specialist MoE | 51.786450 | K56 | 0.0061% | 0.0414% | 0.2830% | 1.1130% | 1.000 | 0 |
| Hopf | Vanilla-FNN-MoE | 47.081356 | K56 | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² |
| Hopf | Vanilla-FNN-MoE | 49.022357 | K56 | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² |
| Hopf | Vanilla-FNN-MoE | 51.786450 | K56 | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² |
| Hopf | DataOnly-MoE | 47.081356 | K56 | 0.1097% | 1478.7163% | N/A¹ | N/A¹ | 1.000 | 13 |
| Hopf | DataOnly-MoE | 49.022357 | K56 | 0.1620% | 1164.2404% | N/A¹ | N/A¹ | 1.000 | 13 |
| Hopf | DataOnly-MoE | 51.786450 | K56 | 0.3895% | 1583.5883% | N/A¹ | N/A¹ | 1.000 | 13 |
| Hopf | Global MoE† | 47.081356 | K56 | 0.0268% | 0.2199% | 46.6749% | 29.9005% | 1.000 | 0 |
| Hopf | Global MoE† | 49.022357 | K56 | 0.0183% | 0.2882% | 24.1395% | 32.6218% | 1.000 | 0 |
| Hopf | Global MoE† | 51.786450 | K56 | 0.0110% | 0.0560% | 94.7850% | 78.2764% | 1.000 | 0 |
| Periodic | Proposed Specialist MoE | 70.314635 | K48 | 1.3473% | 5.9601% | 1.9281% | 2.3977% | 1.000 | 0 |
| Periodic | Proposed Specialist MoE | 100.352251 | K48 | 0.3972% | 1.7083% | 1.8687% | 2.3345% | 1.000 | 0 |
| Periodic | Proposed Specialist MoE | 149.059229 | K48 | 0.3809% | 1.6825% | 1.6843% | 3.8576% | 1.000 | 0 |
| Periodic | Proposed Specialist MoE | 189.862278 | K48 | 1.1274% | 4.4272% | 2.9151% | 6.7689% | 1.000 | 0 |
| Periodic | Vanilla-FNN-MoE | 70.314635 | K48 | 3.0836% | 13.3257% | 4.4136% | 5.3593% | 1.000 | 0 |
| Periodic | Vanilla-FNN-MoE | 100.352251 | K48 | 1.0393% | 3.9659% | 4.8897% | 5.4197% | 1.000 | 0 |
| Periodic | Vanilla-FNN-MoE | 149.059229 | K48 | 2.1460% | 8.2360% | 9.4908% | 18.8801% | 1.000 | 0 |
| Periodic | Vanilla-FNN-MoE | 189.862278 | K48 | 2.5746% | 12.4765% | 6.6568% | 19.0759% | 1.000 | 0 |
| Periodic | DataOnly-MoE | 70.314635 | K48 | 1.8104% | 10.6074% | N/A¹ | N/A¹ | 1.000 | 0 |
| Periodic | DataOnly-MoE | 100.352251 | K48 | 0.7633% | 2.7966% | N/A¹ | N/A¹ | 1.000 | 0 |
| Periodic | DataOnly-MoE | 149.059229 | K48 | 0.8276% | 2.2308% | N/A¹ | N/A¹ | 1.000 | 0 |
| Periodic | DataOnly-MoE | 189.862278 | K48 | 0.6416% | 1.3018% | N/A¹ | N/A¹ | 1.000 | 0 |
| Periodic | Global MoE† | 70.314635 | K56 | 1.5601% | 8.0550% | 23.7912% | 28.0363% | 1.000 | 0 |
| Periodic | Global MoE† | 100.352251 | K56 | 1.1484% | 4.3387% | 11.7726% | 13.4181% | 1.000 | 0 |
| Periodic | Global MoE† | 149.059229 | K56 | 0.7461% | 2.7039% | 6.7727% | 8.2213% | 1.000 | 0 |
| Periodic | Global MoE† | 189.862278 | K56 | 1.4043% | 4.7200% | 11.8473% | 14.7476% | 1.000 | 0 |

Steady 的逐 Re 结果表明，Vanilla 在部分 clean-field 指标上优于 Proposed，因此不能声称 Proposed 在 Steady 场误差上全面支配 Vanilla；Proposed 的主要优势应结合下节固定点和扰动鲁棒性评价。DataOnly 在三个已归档测试 Re 上压力误差分别达到 27275.4654%、10365.5271% 和 5465.6783%，且各含一个发散窗；Re=45.142703 没有冻结 DataOnly test 结果，故明确保留 N/A。

Hopf Proposed 在三个正式 held-out Re 上均 finite=1、零发散，且速度/压力物理场误差分别为 0.0032%/0.0355%、0.0288%/0.4321% 和 0.0061%/0.0414%。DataOnly 虽为有限数值，但每个 Re 均有 13 个发散窗，压力误差超过 1100%。这组逐 Re 证据直接说明连续时间 physics–data 结构和 pressure closure 对 Hopf 长时稳定性的重要性。

Periodic Proposed 在四个 held-out Re 上均完成 K48。相对 Vanilla，Proposed 在四个 Re 的速度与压力场误差均更低；相对 DataOnly，Proposed 在 Re=70.314635、100.352251 和 149.059229 的两类场误差更低，而 Re=189.862278 的 DataOnly 速度/压力场误差更低。后者必须保留，说明物理结构优势不能只靠单点场误差判断。

Global MoE† 使用包含 held-out Re 的 all-Re representation，属于 transductive baseline。其物理场误差有时很低，但 POD 系数误差明显较大，例如 Hopf Re=51.786450 的速度/压力 POD 误差为 94.7850%/78.2764%。因此物理场误差与去均值模态动力学误差回答不同问题，不能相互替代，也不能把 Global† 当作严格 train-only POD baseline。

### 4.2.2 完整系统的逐 Re 解释

在 Steady、Hopf 与 Periodic 核心域，E2 和最终 Proposed system 均确定性退化为对应 native specialist；其逐 Re 数值已经在表 1 的 Proposed 行完整列出，不再以平均值重复。S–H 边界是唯一启用输出级软融合的区域，逐 Re 结果如下。

**表 2　S–H sealed test cache 的逐 Re 场误差与吸引子误差。**

| Test Re | 方法 | 时域 | Joint field | 速度场 | 压力场 | 速度 POD | 压力 POD | 尾段中心 | 振幅绝对 | 轨道几何 | Tail increment | Terminal increment | αS | αH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42.359071 | E2 Top-1 | K56 | 4.6503% | N/A⁶ | N/A⁶ | N/A⁶ | N/A⁶ | 0.0513% | 0.0018% | 1.9375% | 0.0046% | 0.0054% | N/A | N/A |
| 42.359071 | T2-C | K56 | 4.6433% | N/A⁶ | N/A⁶ | N/A⁶ | N/A⁶ | 0.0511% | 0.0018% | 1.9392% | 0.0046% | 0.0054% | 0.9987 | 0.0013 |
| 43.500000 | E2 Top-1 | K56 | 2.4877% | N/A⁶ | N/A⁶ | N/A⁶ | N/A⁶ | 0.0551% | 0.0009% | 1.6198% | 0.0064% | 0.0085% | N/A | N/A |
| 43.500000 | T2-C | K56 | 3.2343% | N/A⁶ | N/A⁶ | N/A⁶ | N/A⁶ | 0.0394% | 0.0002% | 0.2458% | 0.0020% | 0.0017% | 0.0861 | 0.9139 |
| 43.900000 | E2 Top-1 | K56 | 6.8749% | N/A⁶ | N/A⁶ | N/A⁶ | N/A⁶ | 0.0574% | 0.0009% | 1.6135% | 0.0063% | 0.0072% | N/A | N/A |
| 43.900000 | T2-C | K56 | 1.2574% | N/A⁶ | N/A⁶ | N/A⁶ | N/A⁶ | 0.0215% | 0.0005% | 0.5202% | 0.0015% | 0.0016% | 0.1891 | 0.8109 |

Re=42.359071 时，T2-C 的 αS=0.9987，基本退化为 S-only，K56 joint 仅由 4.6503% 降至 4.6433%。Re=43.50 时，T2-C 的 joint 从 2.4877% 增至 3.2343%，但尾段中心、振幅和轨道几何均改善；这是不能删除的反例。Re=43.90 时，joint 从 6.8749% 降至 1.2574%，同时局部吸引子指标改善。三点共同说明门控权重随 Re/初始状态迁移，而场误差与吸引子误差必须双轨报告。

H–P 边界没有合法 test 双路 cache：P specialist 在 P-native validation 上 6/6 finite，而 H 为 0/6，因此冻结 admissibility 为 mH=0、mP=1。最终系统使用 P-only/native-time hard routing，没有训练 H–P T2-C，也没有可报告的 H–P Top-2 test error。该 N/A 是 fail-closed 方法输出，不是遗漏一个成功实验。

## 4.3 吸引子与渐近动力学保持

### 4.3.1 Steady：固定点和扰动吸引性逐 Re 消融

**表 3　Steady 全部 held-out Test Re 的逐方法固定点诊断。**

| 方法 | Test Re | 时域 | 速度场误差 | 压力场误差 | 速度 POD 误差 | 压力 POD 误差 | 压力固定点误差 | 压力漂移 | Finite | 发散窗 | Strict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed Specialist MoE | 24.630436 | K56 | 0.2994822135475797% | 18.560328512913493% | 0.4201% | 15.1009% | 27.13174775732543% | 16.995640099048615% | 1.0 | 0 | FAIL |
| Proposed Specialist MoE | 32.740068 | K56 | 0.07302204843975468% | 5.062044458455372% | 0.5373% | 16.4108% | 5.0118740503653125% | 20.4062357544899% | 1.0 | 0 | FAIL |
| Proposed Specialist MoE | 39.685479 | K56 | 0.038420137200809105% | 1.5225849078872071% | 0.3999% | 15.5819% | 3.136636949094276% | 47.06015586853027% | 1.0 | 0 | FAIL |
| Proposed Specialist MoE | 45.142703 | K56 | 0.06182482042412395% | 1.9664616391550986% | 0.2562% | 6.5857% | 5.630533131903716% | 11.44980937242508% | 1.0 | 0 | FAIL |
| Vanilla-FNN-MoE | 24.630436 | K56 | 0.17226024037521165% | 7.436343222917023% | 0.2613% | 6.0504% | 49.46854183242194% | 4.068085923790932% | 1.0 | 0 | FAIL |
| Vanilla-FNN-MoE | 32.740068 | K56 | 0.06192744917043081% | 2.7919828498609878% | 0.3858% | 9.0424% | 9.574123856583194% | 17.118127644062042% | 1.0 | 0 | FAIL |
| Vanilla-FNN-MoE | 39.685479 | K56 | 0.03295748240101586% | 2.2163099686041177% | 0.3482% | 22.6792% | 2.5341750513468395% | 75.17428994178772% | 1.0 | 0 | FAIL |
| Vanilla-FNN-MoE | 45.142703 | K56 | 0.06992014404865077% | 3.9201255119243923% | 0.2480% | 13.1228% | 5.27953256292543% | 42.338404059410095% | 1.0 | 0 | FAIL |
| DataOnly-MoE | 24.630436 | K56 | 0.5122% | 27275.4654% | N/A¹ | N/A¹ | N/A³ | N/A³ | 1.000 | 1 | N/A³ |
| DataOnly-MoE | 32.740068 | K56 | 0.1790% | 10365.5271% | N/A¹ | N/A¹ | N/A³ | N/A³ | 1.000 | 1 | N/A³ |
| DataOnly-MoE | 39.685479 | K56 | 0.2769% | 5726.5426% | N/A¹ | N/A¹ | 35545.9642% | 342902.0508% | 1.000 | 1 | FAIL |
| DataOnly-MoE | 45.142703 | K56 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Global MoE† | 24.630436 | K56 | 0.3404% | 1.3249% | 44.2564% | 101.0296% | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ |
| Global MoE† | 32.740068 | K56 | 0.1800% | 0.6831% | 31.2366% | 74.5202% | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ |
| Global MoE† | 39.685479 | K56 | 0.1186% | 0.3976% | 25.4158% | 41.5445% | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ |
| Global MoE† | 45.142703 | K56 | 0.0845% | 0.2500% | 20.7911% | 25.2651% | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ |

Proposed 与 Vanilla 的四个 held-out Re 均未同时满足 pressure fixed-point error≤5% 和 pressure drift≤5%，因此 strict 均为 FAIL。Proposed 在 Re=24.630436 显著降低压力 fixed-point error（27.1317% 对 49.4685%）；在 Re=39.685479 则固定点误差略高于 Vanilla（3.1366% 对 2.5342%），且两者 drift 都很大。完整 20-Re Proposed 扫描为 0/20 strict PASS。扰动库的 K56 worst paired gain 从 Vanilla 的 107702.891 降至 Proposed 的 585.262，但仍远大于 1。故可支持“显著降低最坏扰动放大”，不能支持“所有方向严格收缩”。

DataOnly 只有 Re=39.685479 运行了同一类 terminal fixed-point supplement，其 pressure fixed-point error=35545.9642%、drift=342902.0508%，并含发散窗；另外三个 held-out Re 没有该 strict evaluator。Global† 没有 local Steady continuation/fixed-point 合同，故 strict 字段逐 Re 标为 N/A，而不是根据低场误差判 PASS。

### 4.3.2 Hopf：正式 held-out 与三个指定严格案例

**表 4a　Hopf 正式 held-out Test Re 的逐方法吸引子诊断。**

| 方法 | Re | split | 速度场 | 压力场 | 速度 POD | 压力 POD | RMS振幅 | 峰峰值 | 频率 | Phase RMS(rad) | 末端相位(cycle) | 轨道距离 | 速度能量漂移 | 压力能量漂移 | Finite | 发散窗 | False growth | Strict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed Specialist MoE | 47.081356 | heldout | 0.0032% | 0.0355% | 0.0418% | 0.2164% | 0.5370% | 24.8119% | 17.7378% | 0.1429 | 0.0039 | 5.1224% | 0.0033% | 0.0152% | 1.000 | 0 | 否 | FAIL |
| Proposed Specialist MoE | 49.022357 | heldout | 0.0288% | 0.4321% | 0.8479% | 5.7742% | 73.9674% | 378.1346% | 1.1090% | 0.8568 | -0.0665 | 46.8896% | 0.0233% | 0.2188% | 1.000 | 0 | 否 | FAIL |
| Proposed Specialist MoE | 51.786450 | heldout | 0.0061% | 0.0414% | 0.2830% | 1.1130% | 23.9803% | 34.0758% | 1.1814% | 0.3790 | -0.0900 | 25.7239% | 0.0057% | 0.0679% | 1.000 | 0 | 否 | FAIL |
| Vanilla-FNN-MoE | 47.081356 | test | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² |
| Vanilla-FNN-MoE | 49.022357 | test | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² |
| Vanilla-FNN-MoE | 51.786450 | test | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² |
| DataOnly-MoE | 47.081356 | test | 0.1097% | 1478.7163% | N/A¹ | N/A¹ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | 1.000 | 13 | N/A³ | N/A³ |
| DataOnly-MoE | 49.022357 | test | 0.1620% | 1164.2404% | N/A¹ | N/A¹ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | 1.000 | 13 | N/A³ | N/A³ |
| DataOnly-MoE | 51.786450 | test | 0.3895% | 1583.5883% | N/A¹ | N/A¹ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | 1.000 | 13 | N/A³ | N/A³ |
| Global MoE† | 47.081356 | test | 0.0268% | 0.2199% | 46.6749% | 29.9005% | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ | N/A⁴ |
| Global MoE† | 49.022357 | test | 0.0183% | 0.2882% | 24.1395% | 32.6218% | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ | N/A⁴ |
| Global MoE† | 51.786450 | test | 0.0110% | 0.0560% | 94.7850% | 78.2764% | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ | N/A⁴ |

Proposed 在三个正式 held-out Re 上均能稳定完成 K56，但 strict 均为 FAIL：Re=47.081356 的频率误差 17.7378% 和峰峰值误差 24.8119% 超阈值；Re=49.022357 的 RMS/峰峰值/轨道误差分别为 73.9674%、378.1346% 和 46.8896%；Re=51.786450 的 RMS/峰峰值/轨道误差为 23.9803%、34.0758% 和 25.7239%。因此低物理场误差不能替代吸引子联合门。

**表 4b　Hopf 指定 Re=49.3、49.6、50.0 的逐方法严格 K56 案例。**

| 方法 | Re | split | 速度场 | 压力场 | 速度 POD | 压力 POD | RMS振幅 | 峰峰值 | 频率 | Phase RMS(rad) | 末端相位(cycle) | 轨道距离 | 速度能量漂移 | 压力能量漂移 | Finite | 发散窗 | False growth | Strict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed Specialist MoE | 49.300000 | train | 0.0013% | 0.0095% | 0.0451% | 0.1507% | 2.9146% | 2.5017% | 1.6477% | 0.5015 | -0.1166 | 7.2697% | 0.0007% | 0.0100% | 1.000 | 0 | 否 | PASS |
| Proposed Specialist MoE | 49.600000 | train | 0.0016% | 0.0091% | 0.0734% | 0.1810% | 1.4693% | 6.6160% | 1.6169% | 0.4948 | -0.1136 | 6.4963% | 0.0011% | 0.0010% | 1.000 | 0 | 否 | PASS |
| Proposed Specialist MoE | 50.000000 | train | 0.0020% | 0.0171% | 0.1446% | 0.5056% | 1.3430% | 1.3787% | 1.5220% | 0.4713 | -0.1138 | 6.4660% | 0.0016% | 0.0078% | 1.000 | 0 | 否 | PASS |
| Vanilla-FNN-MoE | 49.300000 | train | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² |
| Vanilla-FNN-MoE | 49.600000 | train | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² |
| Vanilla-FNN-MoE | 50.000000 | train | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² | N/A² |
| DataOnly-MoE | 49.300000 | train | 0.1361% | 1118.1745% | N/A¹ | N/A¹ | 711.6247% | 401.2360% | 96.9669% | 21.9143 | -6.2293 | 586.6544% | 0.2344% | 12511.4524% | 1.000 | 5 | 是 | FAIL |
| DataOnly-MoE | 49.600000 | train | 0.0783% | 1183.8779% | N/A¹ | N/A¹ | 530.2295% | 212.0275% | 97.0839% | 22.8912 | -6.4035 | 429.4772% | 0.0521% | 14023.4425% | 1.000 | 5 | 是 | FAIL |
| DataOnly-MoE | 50.000000 | train | 0.0743% | 1627.9665% | N/A¹ | N/A¹ | 139.3650% | 117.4328% | 93.7201% | 24.2909 | -6.6525 | 122.2574% | 0.0175% | 26509.9687% | 1.000 | 5 | 是 | FAIL |
| Global MoE† | 49.300000 | train | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ |
| Global MoE† | 49.600000 | train | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ |
| Global MoE† | 50.000000 | train | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ | N/A⁵ |

Proposed 在三个指定 Re 上均 PASS：物理场、POD 系数、振幅、频率、相位、轨道、能量、finite、divergence 和 false-growth 全部同时满足冻结门。DataOnly 在三个 Re 上均 FAIL，压力误差为 1118.1745%–1627.9665%，频率误差为 93.7201%–97.0839%，每个 Re 有 5 个发散窗且 false growth=是。Vanilla 因 validation fail-closed 未评估；Global† 无精确 49.3/49.6/50.0 资产。需要明确：这三个 Proposed strict 案例属于 train Re，是机制保持证据，不是 held-out 泛化证据。

### 4.3.3 Periodic：四个 held-out Re 的逐方法周期吸引子消融

**表 5　Periodic 全部 Test Re 的逐方法周期吸引子指标。**

| 方法 | Test Re | 时域/合同 | 速度场 | 压力场 | 速度 POD | 压力 POD | RMS振幅 | 峰峰值 | 频率 | Phase RMS(rad) | 末周期漂移 | 轨道距离 | 速度能量漂移 | 压力能量漂移 | Finite | 发散窗 | Strict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed Specialist MoE | 70.314635 | K48 | 1.3473% | 5.9601% | 1.9281% | 2.3977% | 1.3963% | 2.2966% | 0.5528% | 0.1079 | 0.0237 | 7.8199% | 0.1652% | 1.1547% | 1.000 | 0 | FAIL |
| Proposed Specialist MoE | 100.352251 | K48 | 0.3972% | 1.7083% | 1.8687% | 2.3345% | 0.4335% | 0.1580% | 0.1366% | 0.0223 | -0.0088 | 2.4061% | 0.0712% | 0.7429% | 1.000 | 0 | PASS |
| Proposed Specialist MoE | 149.059229 | K48 | 0.3809% | 1.6825% | 1.6843% | 3.8576% | 0.5809% | 0.5302% | 0.1320% | 0.0229 | 0.0029 | 2.7589% | 0.0030% | 0.8959% | 1.000 | 0 | PASS |
| Proposed Specialist MoE | 189.862278 | K48 | 1.1274% | 4.4272% | 2.9151% | 6.7689% | 0.8792% | 1.5040% | 0.3816% | 0.0702 | -0.0241 | 4.4317% | 0.0311% | 0.8300% | 1.000 | 0 | PASS |
| Vanilla-FNN-MoE | 70.314635 | K48 | 3.0836% | 13.3257% | 4.4136% | 5.3593% | 7.5012% | 6.5734% | 1.7981% | 0.3057 | 0.0476 | 10.7218% | 0.0921% | 1.6463% | 1.000 | 0 | FAIL |
| Vanilla-FNN-MoE | 100.352251 | K48 | 1.0393% | 3.9659% | 4.8897% | 5.4197% | 1.3306% | 1.1235% | 0.3158% | 0.0794 | 0.0137 | 3.3292% | 0.0963% | 0.9602% | 1.000 | 0 | PASS |
| Vanilla-FNN-MoE | 149.059229 | K48 | 2.1460% | 8.2360% | 9.4908% | 18.8801% | 1.5512% | 0.7637% | 0.9201% | 0.1865 | -0.0513 | 6.6541% | 0.0933% | 6.5878% | 1.000 | 0 | FAIL |
| Vanilla-FNN-MoE | 189.862278 | K48 | 2.5746% | 12.4765% | 6.6568% | 19.0759% | 4.6511% | 3.0524% | 0.7024% | 0.1443 | 0.0450 | 9.1529% | 0.1173% | 13.2444% | 1.000 | 0 | FAIL |
| DataOnly-MoE | 70.314635 | K48 long-field only | 1.8104% | 10.6074% | N/A¹ | N/A¹ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ | N/A³ |
| DataOnly-MoE | 100.352251 | K48 strict supplement | 0.8891% | 3.1271% | N/A¹ | N/A¹ | 0.7753% | 2.0613% | 0.0852% | 0.0376 | -0.0097 | 4.7728% | 0.4974% | 0.8651% | 1.000 | 0 | PASS |
| DataOnly-MoE | 149.059229 | K48 strict supplement | 0.7854% | 2.3207% | N/A¹ | N/A¹ | 0.3444% | 0.5408% | 0.0905% | 0.0326 | -0.0097 | 4.3304% | 0.8042% | 2.3279% | 1.000 | 0 | PASS |
| DataOnly-MoE | 189.862278 | K48 strict supplement | 0.8509% | 2.2511% | N/A¹ | N/A¹ | 0.5757% | 0.8865% | 0.0507% | 0.0314 | 0.0009 | 4.6841% | 0.4620% | 1.8254% | 1.000 | 0 | PASS |
| Global MoE† | 70.314635 | K56 | 1.5601% | 8.0550% | 23.7912% | 28.0363% | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ |
| Global MoE† | 100.352251 | K56 | 1.1484% | 4.3387% | 11.7726% | 13.4181% | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ |
| Global MoE† | 149.059229 | K56 | 0.7461% | 2.7039% | 6.7727% | 8.2213% | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ |
| Global MoE† | 189.862278 | K56 | 1.4043% | 4.7200% | 11.8473% | 14.7476% | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | N/A⁴ | 1.000 | 0 | N/A⁴ |

Proposed 在 Re=100.352251、149.059229、189.862278 上 strict PASS，在 Re=70.314635 因压力场误差 5.9601% 超过 5% 门限而 FAIL，完整 held-out 通过率为 3/4。Vanilla 仅 Re=100.352251 PASS，其余三个 FAIL。DataOnly 的严格补充 evaluator 覆盖 100.352249、149.059235、189.862274，三者均 PASS；Re=70.314635 只有 long-field K48 结果而无 strict cycle diagnostics，故为 N/A 而不是 FAIL。

DataOnly 在表 1 与表 5 的三个共同 Re 上出现两套略有差异的物理场数值：表 1 来自统一 long-field K48 evaluator，表 5 来自同时生成 cycle strict diagnostics 的冻结补充 evaluator。两者使用不同的冻结窗口聚合合同，故分别按原资产报告，不互相覆盖，也不挑选其中较有利的一套替代另一套。

这组逐 Re 结果支持 Proposed 相对 Vanilla 的周期鲁棒性优势，但也保留了关键负结果：DataOnly 在三个已认证 Periodic Re 上同样保持极限环，且 Re=189.862274 的场误差低于 Proposed。因此不能声称物理结构是 Periodic attractor preservation 的唯一来源；更准确的结论是 Proposed 在完整四点 held-out 覆盖和相对 Vanilla 鲁棒性上更强。

## 4.4 讨论、可支持主张与限制

逐 Re 展开后，结论比平均表更清晰。第一，Proposed 的主要优势具有流态依赖性：Hopf 上是长时压力稳定性和指定案例的严格机制保持，Periodic 上是相对 Vanilla 的四点一致改进与 3/4 strict coverage，Steady 上则是扰动放大显著降低而非 clean-field 全面最优。第二，T2-C 的收益集中在 S–H 高侧 Re=43.90，同时 Re=43.50 暴露 transient-field 与 attractor 指标不一致。第三，Global† 的低物理场误差不能掩盖其较大的 POD 系数误差和 all-Re representation 属性。

本研究可以支持：

- 三个局部 specialist 在各自原生坐标图内具有不同而可验证的能力域；
- Proposed 相对 DataOnly 显著改善 Hopf K56 压力稳定性；
- Proposed 相对 Vanilla 提高 Periodic held-out 周期吸引子通过率；
- Proposed Steady 显著降低最坏扰动增益，但尚未达到全方向严格收缩；
- T2-C 仅在 validation-supported S–H 共同可行域启用，H–P 正确退化为 P-only。

本研究不能支持：

- Proposed 在所有 Re、所有误差和所有 worst case 上全面占优；
- Hopf 的 49.3/49.6/50.0 是 held-out strict 泛化；
- DataOnly 在 Periodic 完全不能保持吸引子；
- Global† 与多坐标系统已在同 representation、同 horizon、同 evaluator 下完成全域公平排名；
- H–P 已获得软融合收益，或 Router 已学习真实 S→H→P 连续时间迁移。

## 4.5 可追溯性

本 revision 的每一行均直接映射到冻结 JSON/CSV。`REVISION8_SOURCE_MAP.json` 记录源文件、SHA256、horizon、split 和缺失字段原因；`REVISION8_DETAILED_ROWS.csv` 保存表 1 的机器可读逐 Re 行。revision 6 与 revision 7 均未被覆盖。
