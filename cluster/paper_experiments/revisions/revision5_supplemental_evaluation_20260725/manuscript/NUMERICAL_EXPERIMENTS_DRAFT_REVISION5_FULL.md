# 数值实验部分草稿（中文 revision 4：Global 与边界融合重组版）

> 本文件是新版本，不覆盖 revision 3。所有相对误差均为原始 relative error × 100%。本文只展示 KLong。带 † 的 Global MoE 数值来自其冻结归档的 K24 POD 系数域 evaluator；其余 specialist 与融合结果来自 K48/K56 面积加权物理场 evaluator。两类数值用于并列呈现模型行为，但不能直接作等口径大小排序。

## 4.1 实验协议、数据隔离与评价口径

Steady、Hopf、Periodic specialist 分别保留自己的 POD 基、均值场、scaler、Galerkin 算子、pressure closure、history builder 和原生积分合同。Vanilla-FNN-MoE 仅替换 expert 网络；DataOnly-MoE 移除连续时间物理推进并学习离散映射；Global MoE 使用一套全局 POD 空间和统一全域网络。所有可用于 checkpoint 选择的决策只使用 train/validation，测试集不用于调参。

KLong 在 Steady/Hopf specialist 中为 K56，在 Periodic specialist 中为 K48，在冻结 Global MoE 归档中为 K24。它们是各自原生数据库步数，不代表统一物理时间。由于原始采样合同和 evaluator 不同，本节保留“时域/误差空间”列，避免把 K24 系数误差伪装为 K56 物理场误差。

## 4.2 长时物理场预测

### 4.2.1 流态专用 specialist、消融与 Global MoE

| 流态 | 方法 | 测试 Re | 长时域 | 误差空间 | 速度误差 | 压力误差 | 联合误差 | worst | finite | 发散窗 |
|---|---|---|---|---|---|---|---|---|---|---|
| Steady | DataOnly-MoE | 24.630436, 32.740068, 39.685479（共同可评估集） | K56 | 面积加权物理场 | 0.2569% | 14368.8903% | 7184.5736% | 13637.9888% | 1.000 | 3 |
| Steady | Proposed Specialist MoE | 24.630436, 32.740068, 39.685479（共同可评估集） | K56 | 面积加权物理场 | 0.1370% | 8.3817% | 4.2593% | 9.4299% | 1.000 | 0 |
| Steady | Vanilla-FNN-MoE | 24.630436, 32.740068, 39.685479（共同可评估集） | K56 | 面积加权物理场 | 0.0890% | 4.1482% | 2.1186% | 3.8043% | 1.000 | 0 |
| Steady | Global MoE† | 24.630436, 32.740067, 39.685478 | K24 | POD 系数域† | 22.4582% | 46.6276% | 34.5429% | — | — | — |
| Hopf | DataOnly-MoE | 47.081356, 49.022357, 51.786450 | K56 | 面积加权物理场 | 0.2204% | 1408.8483% | 704.5344% | 791.9889% | 1.000 | 39 |
| Hopf | Proposed Specialist MoE | 47.081356, 49.022357, 51.786450 | K56 | 面积加权物理场 | 0.0127% | 0.1697% | 0.0912% | 0.2305% | 1.000 | 0 |
| Hopf | Vanilla-FNN-MoE | 47.081356, 49.022357, 51.786450 | K56 | 面积加权物理场 | 验证失败，未开放 test | — | — | — | — | — |
| Hopf | Global MoE† | 47.081356, 49.022358, 51.786449 | K24 | POD 系数域† | 49.5338% | 46.2694% | 47.9016% | — | — | — |
| Periodic | DataOnly-MoE | 70.314635, 100.352251, 149.059229, 189.862278 | K48 | 面积加权物理场 | 1.0107% | 4.2341% | 2.6224% | 6.2089% | 1.000 | 0 |
| Periodic | Proposed Specialist MoE | 70.314635, 100.352251, 149.059229, 189.862278 | K48 | 面积加权物理场 | 0.8132% | 3.4445% | 2.1289% | 3.6537% | 1.000 | 0 |
| Periodic | Vanilla-FNN-MoE | 70.314635, 100.352251, 149.059229, 189.862278 | K48 | 面积加权物理场 | 2.2109% | 9.5010% | 5.8559% | 8.2047% | 1.000 | 0 |
| Periodic | Global MoE† | 70.314636, 100.352249, 149.059235, 189.862274 | K24 | POD 系数域† | 4.9334% | 6.1781% | 5.5557% | — | — | — |

Global MoE 已按要求纳入本表，并明确给出速度与压力误差。其 K24 结果在三个流态上分别反映全局模型的速度/压力 POD 系数 rollout 误差；由于没有经过同一 K48/K56 物理场重构 evaluator，本稿不据此宣称 Global 与 specialist 的严格数值排名。这个限制是测量口径限制，而不是删除 Global 结果。

Steady 的 Proposed 权威模型是 S3-B contraction checkpoint。在共同可评价的三个 Re 上，Vanilla 的 clean-field 平均误差较低，因此不能声称 Proposed 在 Steady 所有场误差上全面占优；Proposed 的主要优势应结合固定点和扰动吸引性判断。Hopf 中，Proposed 在 held-out K56 保持 finite=1 且零发散，而 DataOnly 出现 39 个 norm-divergent windows，Vanilla 在 validation 已非有限。Periodic 中，Proposed 在相同四个 held-out Re 的 K48 联合误差和 worst-case 均优于 Vanilla，并在吸引子联合门上取得更高通过率。

### 4.2.2 长时场误差的解释边界

物理场误差衡量整段轨迹的逐时刻预测质量，但不能单独判断渐近吸引子的中心、振幅、频率、相位和轨道几何是否保持。尤其在 S–H 边界 Re=43.50，T2-C 的全窗口误差略有退化，但尾段中心和轨道几何明显改善。因此下一节将 attractor preservation 独立列出，而不把它折叠进单一 joint error。

## 4.3 吸引子与渐近动力学保持

### 4.3.1 Specialist attractor preservation

#### (a) Steady 固定点与局部扰动

| 方法 | 评价 Re | 时域/误差空间 | 速度误差 | 压力误差 | 压力固定点误差 | 漂移/诊断 | finite | 发散窗 | Strict 结论 |
|---|---|---|---|---|---|---|---|---|---|
| Proposed Specialist MoE (S3-B) | 24.630440 | K56 / 物理场 | 0.2995% | 18.5603% | 27.1317% | 16.9956% | 1.000 | 0 | 未建立三方法同口径二值门 |
| Proposed Specialist MoE (S3-B) | 32.740070 | K56 / 物理场 | 0.0730% | 5.0620% | 5.0119% | 20.4062% | 1.000 | 0 | 未建立三方法同口径二值门 |
| Proposed Specialist MoE (S3-B) | 39.685480 | K56 / 物理场 | 0.0384% | 1.5226% | 3.1366% | 47.0602% | 1.000 | 0 | 未建立三方法同口径二值门 |
| Proposed Specialist MoE (S3-B) | 45.142700 | K56 / 物理场 | 0.0618% | 1.9665% | 5.6305% | 11.4498% | 1.000 | 0 | 未建立三方法同口径二值门 |
| Vanilla-FNN-MoE | 24.630440 | K56 / 物理场 | 0.1723% | 7.4363% | 49.4685% | 4.0681% | 1.000 | 0 | 未建立三方法同口径二值门 |
| Vanilla-FNN-MoE | 32.740070 | K56 / 物理场 | 0.0619% | 2.7920% | 9.5741% | 17.1181% | 1.000 | 0 | 未建立三方法同口径二值门 |
| Vanilla-FNN-MoE | 39.685480 | K56 / 物理场 | 0.0330% | 2.2163% | 2.5342% | 75.1743% | 1.000 | 0 | 未建立三方法同口径二值门 |
| Vanilla-FNN-MoE | 45.142700 | K56 / 物理场 | 0.0699% | 3.9201% | 5.2795% | 42.3384% | 1.000 | 0 | 未建立三方法同口径二值门 |
| DataOnly-MoE | 24.630436 | K56 / 物理场 | 0.5122% | 27275.4654% | — | — | 1.000 | 1 | 未建立三方法同口径二值门 |
| DataOnly-MoE | 32.740067 | K56 / 物理场 | 0.1790% | 10365.5271% | — | — | 1.000 | 1 | 未建立三方法同口径二值门 |
| DataOnly-MoE | 39.685478 | K56 / 物理场 | 0.0795% | 5465.6783% | — | — | 1.000 | 1 | 未建立三方法同口径二值门 |
| Global MoE† | 24.630436 | K24 / POD 系数域† | 30.6107% | 62.7433% | — | 压力能量 127.0489%; 伪振荡峰值比 207.75× | — | — | 未运行 Steady 同口径 strict evaluator |
| Global MoE† | 32.740067 | K24 / POD 系数域† | 20.6771% | 46.7996% | — | 压力能量 66.6479%; 伪振荡峰值比 98.11× | — | — | 未运行 Steady 同口径 strict evaluator |
| Global MoE† | 39.685478 | K24 / POD 系数域† | 16.0867% | 30.3400% | — | 压力能量 30.6771%; 伪振荡峰值比 40.88× | — | — | 未运行 Steady 同口径 strict evaluator |

Proposed S3-B 的 K56 pressure fixed-point worst 为 27.1317%，Vanilla 为 49.4685%；更严格的 perturbation-bank K56 最坏 paired gain 从 107702.891 降至 585.262。该结果支持“固定点与扰动鲁棒性显著改善”，但由于最坏 gain 仍大于 1，不支持“所有扰动方向严格收缩”。Global 的低 Re 归档出现较大的伪振荡 overshoot，因此其存在于表中并不等于通过 Steady strict-attractor 判据。

#### (b) Hopf 指定严格 K56 保持案例

| 指定 Re | 方法 | 时域/误差空间 | 速度误差 | 压力误差 | RMS 振幅 | 峰峰值 | 频率误差 | 末端相位漂移(周期) | 轨道距离 | 能量漂移/不可用原因 | finite/发散窗 | false growth | Strict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 49.3 | Proposed Specialist MoE | K56 / 物理场 | 0.0013% | 0.0095% | 2.9146% | 2.5017% | 1.6477% | -0.1166 | 7.2697% | u 0.0007%, p 0.0100% | 1.000 / 0 | 否 | PASS |
| 49.3 | Vanilla-FNN-MoE | — | — | — | — | — | — | — | — | validation K16/K56 非有限，按 fail-closed 合同未开放指定案例评估 | — | — | N/A |
| 49.3 | DataOnly-MoE | — | — | — | — | — | — | — | — | 冻结归档未在 49.3/49.6/50.0 运行同一 strict evaluator | — | — | N/A |
| 49.3 | Global MoE† | — | — | — | — | — | — | — | — | Global held-out 资产不含这三个精确 Re，且仅有 K24 系数域 evaluator | — | — | N/A |
| 49.6 | Proposed Specialist MoE | K56 / 物理场 | 0.0016% | 0.0091% | 1.4693% | 6.6160% | 1.6169% | -0.1136 | 6.4963% | u 0.0011%, p 0.0010% | 1.000 / 0 | 否 | PASS |
| 49.6 | Vanilla-FNN-MoE | — | — | — | — | — | — | — | — | validation K16/K56 非有限，按 fail-closed 合同未开放指定案例评估 | — | — | N/A |
| 49.6 | DataOnly-MoE | — | — | — | — | — | — | — | — | 冻结归档未在 49.3/49.6/50.0 运行同一 strict evaluator | — | — | N/A |
| 49.6 | Global MoE† | — | — | — | — | — | — | — | — | Global held-out 资产不含这三个精确 Re，且仅有 K24 系数域 evaluator | — | — | N/A |
| 50.0 | Proposed Specialist MoE | K56 / 物理场 | 0.0020% | 0.0171% | 1.3430% | 1.3787% | 1.5220% | -0.1138 | 6.4660% | u 0.0016%, p 0.0078% | 1.000 / 0 | 否 | PASS |
| 50.0 | Vanilla-FNN-MoE | — | — | — | — | — | — | — | — | validation K16/K56 非有限，按 fail-closed 合同未开放指定案例评估 | — | — | N/A |
| 50.0 | DataOnly-MoE | — | — | — | — | — | — | — | — | 冻结归档未在 49.3/49.6/50.0 运行同一 strict evaluator | — | — | N/A |
| 50.0 | Global MoE† | — | — | — | — | — | — | — | — | Global held-out 资产不含这三个精确 Re，且仅有 K24 系数域 evaluator | — | — | N/A |

Proposed 在 Re=49.3、49.6、50.0 上均通过完整 strict conjunction，包括物理场、振幅、频率、相位、轨道、能量、finite、divergence 和 false-growth 条件。这三个 Re 参与原始训练，故其科学含义是证明代表性参数点的严格保持能力，而不是 held-out 泛化。Vanilla、DataOnly 和 Global 已在同一子表中出现；然而现有冻结产物没有三者在这三个精确 Re 上的同口径 strict 结果，必须标为 N/A，不能用邻近 Re 或不同 horizon 冒充。因而目前可宣称 Proposed 的三个严格案例，但不能据该表宣称它在这三个案例上定量优于全部消融。

#### (c) Periodic 三个严格保持测试案例

| 测试 Re | 方法 | 时域/误差空间 | 速度误差 | 压力误差 | RMS 振幅 | 峰峰值 | 频率误差 | 相位误差 | 末周期漂移 | 轨道距离 | finite/发散窗 | Strict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100.352249 | Proposed Specialist MoE | K48 / 物理场 | 0.3972% | 1.7083% | 0.4335% | 0.1580% | 0.1366% | 0.0223 | -0.0088 | 2.4061% | 1.000 / 0 | PASS |
| 100.352249 | Vanilla-FNN-MoE | K48 / 物理场 | 1.0393% | 3.9659% | 1.3306% | 1.1235% | 0.3158% | 0.0794 | 0.0137 | 3.3292% | 1.000 / 0 | PASS |
| 100.352249 | DataOnly-MoE | K48 / 物理场 | 0.7633% | 2.7966% | — | — | — | — | — | — | 1.000 / 0 | N/A：未运行周期 strict evaluator |
| 100.352249 | Global MoE† | K24 / POD 系数域† | 2.8343% | 3.6636% | 0.9267% | — | 0.7457% | 0.0159 | — | — | — | N/A：时域与 strict 定义不同 |
| 149.059235 | Proposed Specialist MoE | K48 / 物理场 | 0.3809% | 1.6825% | 0.5809% | 0.5302% | 0.1320% | 0.0229 | 0.0029 | 2.7589% | 1.000 / 0 | PASS |
| 149.059235 | Vanilla-FNN-MoE | K48 / 物理场 | 2.1460% | 8.2360% | 1.5512% | 0.7637% | 0.9201% | 0.1865 | -0.0513 | 6.6541% | 1.000 / 0 | FAIL |
| 149.059235 | DataOnly-MoE | K48 / 物理场 | 0.8276% | 2.2308% | — | — | — | — | — | — | 1.000 / 0 | N/A：未运行周期 strict evaluator |
| 149.059235 | Global MoE† | K24 / POD 系数域† | 2.5471% | 3.2347% | 0.7537% | — | 0.5134% | 0.0093 | — | — | — | N/A：时域与 strict 定义不同 |
| 189.862274 | Proposed Specialist MoE | K48 / 物理场 | 1.1274% | 4.4272% | 0.8792% | 1.5040% | 0.3816% | 0.0702 | -0.0241 | 4.4317% | 1.000 / 0 | PASS |
| 189.862274 | Vanilla-FNN-MoE | K48 / 物理场 | 2.5746% | 12.4765% | 4.6511% | 3.0524% | 0.7024% | 0.1443 | 0.0450 | 9.1529% | 1.000 / 0 | FAIL |
| 189.862274 | DataOnly-MoE | K48 / 物理场 | 0.6416% | 1.3018% | — | — | — | — | — | — | 1.000 / 0 | N/A：未运行周期 strict evaluator |
| 189.862274 | Global MoE† | K24 / POD 系数域† | 4.1608% | 5.0987% | 1.2367% | — | 0.7290% | 0.0132 | — | — | — | N/A：时域与 strict 定义不同 |

Periodic 的三条 Re 均为 held-out。Proposed 在三者上均 PASS；Vanilla 只在 Re=100.352249 上 PASS，在更高 Re 的相位/轨道或压力相关条件下失败。DataOnly 在相同 Re 有 K48 物理场误差，但没有运行周期 strict evaluator；Global 在相同 Re 有 K24 系数域振幅、频率和相位诊断，但 horizon 与定义不同。因此表中同时展示四种方法，并把“缺少同口径认证”和“实际 FAIL”严格区分。

### 4.3.2 吸引子优势的综合解释

Hopf 的核心证据是 Proposed 在指定案例中同时保持振幅、频率、相位和轨道，而不是只获得很小的瞬态场误差；Periodic 的核心证据是 Proposed 在三个 held-out Re 通过联合门，并将严格通过率从 Vanilla 的 1/4 提升到 3/4。Steady 的证据性质不同：它表现为固定点误差与扰动放大显著下降，但尚未达到全方向严格收缩。三类流态因此不能共用一个未经定义的“Strict”标签。

## 4.4 Admissibility-constrained 边界融合算法

### 4.4.1 统一边界机制

边界层先执行动力学 admissibility gate，再决定 Top-1 或 Top-2。S–H 的 S-native development 域同时支持冻结 S/H K56 rollout，因此允许 T2-C 在公共物理场做窗口条件凸融合。H–P 的 P-native validation 中，Hopf 为 0/6 finite K56，而 Periodic 合法，因此 mask 固定为 mH=0、mP=1；所谓 T2-C 系统在此边界机械退化为 P-only，而不是训练一个不合法的 H–P 融合门。

### 4.4.2 长时物理场与吸引子联合对比

| 边界 | 方法 | 评价 Re | 时域/误差空间 | 速度误差 | 压力误差 | 联合误差 | 吸引子/渐近指标 | 实际机制与结论 |
|---|---|---|---|---|---|---|---|---|
| S–H | E2 Top-1 | 42.359071, 43.500000, 43.900000 | K56 / 物理场 | 0.0627% | 4.6083% | 4.6710% | 尾段中心 0.0546%; 轨道 1.7236% | Top-1 |
| S–H | T2-C | 42.359071, 43.500000, 43.900000 | K56 / 物理场 | 0.0429% | 3.0021% | 3.0450% | 尾段中心 0.0373%; 轨道 0.9017% | 共同可行域内窗口级凸融合 |
| S–H | Global MoE† | 无同一 sealed-cache 结果 | K24 / POD 系数域† | — | — | — | — | 不能用其他 Re 替代边界 test |
| H–P | E2 Top-1 | 70.314635, 100.352249, 149.059235, 189.862274 | K48 / 物理场 | 0.8132% | 3.4445% | 2.1289% | Periodic strict 3/4 PASS | P-only/native-time hard routing |
| H–P | T2-C（admissibility 退化） | 70.314635, 100.352249, 149.059235, 189.862274 | K48 / 物理场 | 0.8132% | 3.4445% | 2.1289% | Periodic strict 3/4 PASS | mH=0, mP=1；未训练 H–P 凸门 |
| H–P | Global MoE† | 70.314636, 100.352249, 149.059235, 189.862274 | K24 / POD 系数域† | 4.9334% | 6.1781% | 5.5557% | 仅 Global 原生振幅/相位诊断；无同口径 strict 判定 | 全局单模型；不可与 K48 物理场直接排名 |

这张表同时覆盖 S–H 和 H–P。S–H 中 T2-C 将平均 K56 联合物理场误差从 4.6710% 降至 3.0450%，并改善平均尾段中心与轨道几何；Global 尚未在同一 S–H sealed cache 上评估，所以不能填入伪造的边界数值。H–P 中 E2 与 admissibility-constrained T2-C 产生相同的 P-only 输出：这不是缺失实验，而是 mH=0 后的预注册稀疏退化。Global 在 Periodic held-out Re 有结果，但仍是 K24 系数域口径。

### 4.4.3 S–H 逐 Re 行为与权重

| 测试 Re | 方法 | K56 联合误差 | 尾段中心误差 | 振幅绝对误差 | 轨道几何误差 | αS | αH |
|---|---|---|---|---|---|---|---|
| 42.359071 | E2 Top-1 | 4.6503% | 0.0513% | 0.0018% | 1.9375% | — | — |
| 42.359071 | T2-C | 4.6433% | 0.0511% | 0.0018% | 1.9392% | 0.9987 | 0.0013 |
| 43.500000 | E2 Top-1 | 2.4877% | 0.0551% | 0.0009% | 1.6198% | — | — |
| 43.500000 | T2-C | 3.2343% | 0.0394% | 0.0002% | 0.2458% | 0.0861 | 0.9139 |
| 43.900000 | E2 Top-1 | 6.8749% | 0.0574% | 0.0009% | 1.6135% | — | — |
| 43.900000 | T2-C | 1.2574% | 0.0215% | 0.0005% | 0.5202% | 0.1891 | 0.8109 |

T2-C 并非在每个 Re 上都改善全窗口误差。Re=42.359071 基本退化为 S-only；Re=43.50 增大 H 权重后吸引子指标改善但 K56 总误差上升；Re=43.90 则同时改善总误差与渐近指标。这一逐 Re 结果支持“能力感知、状态条件融合”，同时否定“融合必然逐点支配”的过强表述。

### 4.4.4 H–P fail-closed 结果

H–P 当前仅允许 P-only/native-time hard routing。Hopf 在 P-native validation 没有共同稳定支持域，因此 RiskPrediction、LookAhead 和 T2-C 均不应启动。论文应把该结果表述为统一 admissibility 原则的有效输出：当候选 specialist 不具备有限、稳定、validation-supported rollout 时，其融合权重归零。

## 4.5 限制与可支持的论文结论

本轮使用单一固定 seed，不能声称跨 seed 显著性；历史 S–H test 曾被查看，不能声称全程 blind test；Hopf 三个指定严格案例属于训练 Re；Global 尚未通过与 specialist 完全相同的 K48/K56 物理场 evaluator；H–P 没有共同可行域；现有 Router 数据不包含经认证的真实跨流态连续迁移轨迹。

实验支持的结论是：physics–data specialist 在 Hopf/Periodic 的长期稳定性和吸引子保持上提供关键收益；Steady contraction 机制明显改善固定点与扰动鲁棒性；T2-C 只在 S–H 共同可行域启用；H–P 按 admissibility mask 退化为 P-only。实验不支持 Proposed 在所有指标全面支配、Steady 全方向严格收缩、Global 与多坐标系统已完成同口径全域排名，或 Router 已学习真实 S→H→P 时间迁移。

## 4.6 可追溯性

Revision 3 已在独立目录冻结并生成 SHA256；本 revision 4 只写入 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/revisions/revision4_global_fusion_20260724`。Global 原始指标来自 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/V16_1_SteadyPressureAnchor32/reproduction/heldout_eval_frozen_checkpoint_20260723/V16_1_SteadyPressureAnchor32_ru32_rp32_reproduced_metrics.json`；specialist、T2-C、attractor 与 H–P fail-closed 的来源路径均记录在同目录 `REVISION4_AUDIT.json`。


---

# 数值实验部分补充稿（revision 5）

> 本文件在 revision 4 基础上追加新的同 horizon 评价，不覆盖旧版本。全文相对误差统一为 `relative error × 100%`。

## 4.2.3 Global MoE† 的 K56 同时域复核

此前 Global MoE† 仅有 K24 系数域结果，因此只能作为不同口径的旁证。本轮保持 checkpoint、全局 POD、scaler、Galerkin 与 pressure closure 不变，将 rollout horizon 延长到 K56，并增加面积加权物理场误差。结果显示，Global 在四个 Steady held-out Re、三个 Hopf held-out Re、四个 Periodic held-out Re和五个额外 Steady 候选 Re 上均形成完整有限窗口。故原表中的 Global K56 “—”可以由新的逐 Re 数值替换。

需要强调的是，Global 全场误差以含均值场的物理范数归一化，而系数误差聚焦脉动模态；在 Hopf 区域，二者可相差数个数量级。因此应同时报告，不能只选较小的一列。Global 的 K56 成功也不等价于通过局部 specialist 的 strict attractor evaluator，因为后者还要求终点续推、漂移、相位和轨道等合同。

## 4.3.3 Steady 压力固定点联合门

我们对 Proposed Steady 的全部 train、validation 与 heldout Re 执行同一 K56 扫描。预注册联合门要求输出有限、零发散、压力固定点物理相对误差不超过 5%，且压力漂移系数相对误差不超过 5%。扫描没有产生任何联合 PASS。六个 Re 的终态固定点误差低于 5%，但漂移仍超阈值，因此只能称为“fixed-point-qualified”，不能称为严格压力固定点保持。

在这六个次级诊断点上，DataOnly 的压力预测和漂移显著恶化，并在每个 K56 窗口触发 norm-divergence；Global 完成有限 K56 且全场误差较小，但由于终点之后缺少合法数据库索引，不能执行同合同 fixed-point continuation。该对照说明 Proposed 的固定点锚定相对 DataOnly 确有实质作用，但现有证据支持的是相对鲁棒性改善，而不是严格渐近收敛证明，也不足以对 Global 授予或否决同一 strict 标签。

## 4.3.4 DataOnly 的 Hopf 与 Periodic 补充吸引子结果

在 Hopf 指定 Re=49.3、49.6、50.0 上，DataOnly 的 K56 strict 结果均为 FAIL：频率相对误差约为 94%–97%，末端相位漂移约为 -6.2 至 -6.7 个周期，且三个 Re 均出现 false growth。与 Proposed 在相同三点的 strict PASS 相比，该消融清楚表明，连续时间局部向量场、RK4、Galerkin 与压力物理结构对于 Hopf 吸引子保持不可由离散数据映射自动替代。

另一方面，在 Periodic held-out Re=100.352249、149.059235、189.862274 上，DataOnly 均通过 K48 strict cycle 门。这一负向/竞争性结果必须保留。它表明成熟周期区间的数据驱动离散映射在当前 horizon 上足以维持周期几何；Proposed 的论证重点应放在更低误差、跨流态一致性和边界稳定性，而不是声称 DataOnly 无法保持周期吸引子。

## 4.5.1 修订后的结论边界

综合补充实验，可支持的表述为：Proposed 的物理结构对 Hopf 严格保持和 Steady 固定点鲁棒性具有关键贡献；Global 单模型在 Steady K56 失稳；DataOnly 在 Hopf 指定案例失败，但在三个 Periodic held-out 案例成功。因而组件有效性具有流态依赖性，不能用单一“全面优于”概括。完整逐 Re 数值见 `REVISION5_SUPPLEMENTAL_EVALUATION_ZH.md`。
