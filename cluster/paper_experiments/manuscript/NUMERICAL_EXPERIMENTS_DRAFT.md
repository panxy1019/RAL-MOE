# 数值实验部分草稿（中文修订版）

## 4.1 流动构型、数据划分与多坐标图 POD 空间

本文研究跨越 Steady、Hopf 与 Periodic 三类尾流动力学的参数化降阶预测问题。三个 specialist 分别保留其原生速度 POD、压力 POD、均值场、scaler、Galerkin 算子、pressure closure、历史特征构造器以及原生时间推进合同。不同局部坐标图中的模态系数不直接混合；只有各 specialist 独立推进并重构到同一物理网格后，才允许在经过认证的相邻共同可行域中实施输出级凸融合。

训练、验证和测试均以完整 Reynolds 数和完整轨迹隔离。新训练的 Vanilla-FNN-MoE 与 DataOnly-MoE 每个流态只使用一个固定随机种子，因此下文是确定性单次实验比较，不报告不存在的跨 seed 均值与标准差。Steady 的权威 Proposed 模型采用跨硬件审计后保留的 S3-B contraction checkpoint；后续 S4 fixed-point-anchor 候选因 4090 上的 P95 contraction 保护门失败，不作为最终 Proposed specialist。

本文所有物理场相对误差均以百分数表示。主结果只展示长时自主预测 KLong，而不再同时罗列 K1 和 K16。Steady、Hopf 使用 K56；Periodic 为保证三种消融具备严格一致的 horizon，统一使用 K48。

| 流态 | 评价 Re | KLong | 数据库 time 坐标跨度（沿用原数据单位） |
|---|---|---|---|
| Steady | 24.630436 | K56 | 431.273646 |
| Steady | 32.740068 | K56 | 324.271778 |
| Steady | 39.685479 | K56 | 268.691205 |
| Steady | 45.142703 | K56 | 234.604068 |
| Hopf | 47.081355 | K56 | 1309.780761 |
| Hopf | 49.022357 | K56 | 1230.472006 |
| Hopf | 51.786450 | K56 | 1179.953180 |
| Hopf | 49.300000 | K56 | 1225.216155 |
| Hopf | 49.600000 | K56 | 1221.156848 |
| Hopf | 50.000000 | K56 | 1209.083683 |
| Periodic | 70.314635 | K48 | 598.700951 |
| Periodic | 100.352251 | K48 | 375.724543 |
| Periodic | 149.059229 | K48 | 232.525249 |
| Periodic | 189.862278 | K48 | 177.959440 |

需要强调的是，KLong 是“原生数据库步数”，而不是统一物理时间。由于三个数据库的采样合同不同，甚至同一流态不同 Re 的原生时间间隔也不同，表中逐 Re 给出从首个合法窗口计算的原始 `time` 坐标跨度；其单位沿用各数据库原合同，不预先解释为可跨数据库比较的统一物理单位。后续若使用无量纲时间比较，应单独定义统一的无量纲化，而不能把不同 specialist 的 K56 直接解释为相同时间。

## 4.2 长时物理场预测

### 4.2.1 流态专用 specialist 消融

| 流态 | 方法 | 测试 Re | KLong | 速度误差 | 压力误差 | 联合误差 | worst | finite | 发散窗 |
|---|---|---|---|---|---|---|---|---|---|
| Steady | Vanilla-FNN-MoE | 24.630436, 32.740068, 39.685479（共同可评估集） | K56 | 0.0890% | 4.1482% | 2.1186% | 3.8043% | 1.000 | 0 |
| Steady | DataOnly-MoE | 24.630436, 32.740068, 39.685479（共同可评估集） | K56 | 0.2569% | 14368.8903% | 7184.5736% | 13637.9888% | 1.000 | 3 |
| Steady | Proposed Specialist MoE | 24.630436, 32.740068, 39.685479（共同可评估集） | K56 | 0.1370% | 8.3817% | 4.2593% | 9.4299% | 1.000 | 0 |
| Hopf | Vanilla-FNN-MoE | 47.081356, 49.022357, 51.786450 | K56 | 验证失败，未开放 test | — | — | — | — | — |
| Hopf | DataOnly-MoE | 47.081356, 49.022357, 51.786450 | K56 | 0.2204% | 1408.8483% | 704.5344% | 791.9889% | 1.000 | 39 |
| Hopf | Proposed Specialist MoE | 47.081356, 49.022357, 51.786450 | K56 | 0.0127% | 0.1697% | 0.0912% | 0.2305% | 1.000 | 0 |
| Periodic | Vanilla-FNN-MoE | 70.314635, 100.352251, 149.059229, 189.862278 | K48 | 2.2109% | 9.5010% | 5.8559% | 8.2047% | 1.000 | 0 |
| Periodic | DataOnly-MoE | 70.314635, 100.352251, 149.059229, 189.862278 | K48 | 1.0107% | 4.2341% | 2.6224% | 6.2089% | 1.000 | 0 |
| Periodic | Proposed Specialist MoE | 70.314635, 100.352251, 149.059229, 189.862278 | K48 | 0.8132% | 3.4445% | 2.1289% | 3.6537% | 1.000 | 0 |

在 Steady 流态中，为保证三种方法使用同一评价人口，平均值只基于共同具备 K56 窗口的 Re=24.630436、32.740068、39.685479。Vanilla 的提前冻结 checkpoint 在部分 clean-field 指标上具有较小误差，因此结果不支持“特殊 expert 结构在 Steady 所有场误差上全面优于普通 FNN”。然而 DataOnly 的压力误差达到远高于可接受范围的水平，并出现 norm-divergent windows，说明仅学习离散数据映射不能稳定复现 pressure closure。Physics–Data specialist 对 DataOnly 的主要优势来自显式连续时间动力学、Galerkin 物理项、pressure coupling 与 rollout 约束共同提供的稳定性，而不能简单归因于网络宽度。

在 Hopf 流态中，Proposed 在 held-out Re=47.081356、49.022357、51.786450 上完成 K56 自主 rollout，finite fraction 为 1 且无发散。DataOnly 虽保持数值有限，但具有 39 个 K56 divergent windows，压力误差也显著高于 Proposed。Vanilla 在 validation K16/K56 已出现非有限轨迹，按照 validation-first 合同禁止进入 test。该结果说明 Hopf specialist 中的 normal-form/physics–data expert、连续时间推进与压力 closure 并非可被普通 FNN 或纯离散映射无损替换。

在 Periodic 流态中，三种方法均在 Re=70.314635、100.352251、149.059229、189.862278 上采用 K48。Proposed 相对 Vanilla 明显降低速度、压力、联合误差及 worst-case；相对 DataOnly 的优势不仅体现在物理场误差，还体现在下节经过严格认证的振幅、频率、相位和轨道闭合保持。这表明周期流态中的 physics–data coupling 对长期相位推进和极限环几何尤其重要。

### 4.2.2 完整参数域系统比较

Global MoE 已有冻结 checkpoint 和 held-out 结果，但其归档 evaluator 主要输出模态一步及 rollout 误差；E2/T2-C 的 S–H cache 使用面积加权物理场误差。两组数值合同并不相同，因此本文不把它们强行合并为一个全域排名表。当前能够严格比较的完整系统结果集中在 S–H 边界：E2 Top-1 的平均 K56 联合误差为 4.6710%，T2-C 为 3.0450%。H–P 边界则因 Hopf 在 P-native validation 上 0/6 条 K56 轨迹保持有限而 fail closed，统一系统退化为 P-only/native-time hard routing。

### 4.2.3 嵌入式路由诊断

E2 是固定 Re、整轨迹级的 temporal-consistent regime router。它在 11 条测试轨迹上的 balanced accuracy 和 macro-F1 均为 1，但 S–H 临界附近的最小 margin 仅 0.004595。该结果证明 E2 可以作为部署合同清晰的 Top-1 基线，但不能证明模型学习了真实的 S→H→P 动态流态迁移。

| 测试 Re | 方法 | K56联合误差 | 尾段中心误差 | 振幅绝对误差 | 轨道几何误差 | αS | αH |
|---|---|---|---|---|---|---|---|
| 42.359071 | E2 Top-1 | 4.6503% | 0.0513% | 0.0018% | 1.9375% | — | — |
| 42.359071 | T2-C | 4.6433% | 0.0511% | 0.0018% | 1.9392% | 0.9987 | 0.0013 |
| 43.500000 | E2 Top-1 | 2.4877% | 0.0551% | 0.0009% | 1.6198% | — | — |
| 43.500000 | T2-C | 3.2343% | 0.0394% | 0.0002% | 0.2458% | 0.0861 | 0.9139 |
| 43.900000 | E2 Top-1 | 6.8749% | 0.0574% | 0.0009% | 1.6135% | — | — |
| 43.900000 | T2-C | 1.2574% | 0.0215% | 0.0005% | 0.5202% | 0.1891 | 0.8109 |

T2-C 的权重由每个 K56 查询窗口的初始三步物理历史决定，并在窗口内部保持固定。低 Re=42.359071 的平均 αS 接近 1；Re=43.50 和 43.90 则提高 αH。该随状态变化的权重模式表明门控不是简单学习一个全局固定比例。RiskPrediction 与 LookAhead 在当前 sealed cache 中最终输出与共同凸门一致，删除 critic 或短 rollout 后误差不变，因此复杂评分模块没有展示额外价值，最终保留更简单的 T2-C。

## 4.3 吸引子与渐近动力学保持

### 4.3.1 Specialist attractor preservation

为避免一个宽表掩盖不同流态的动力学含义，本节将 Steady 固定点、Hopf 极限环和 Periodic 周期轨道拆分为三个子表。

#### (a) Steady 固定点与局部扰动

| 方法 | 评价 Re | K56速度 | K56压力 | 压力固定点误差 | 压力漂移 | finite | clean发散窗 |
|---|---|---|---|---|---|---|---|
| Proposed Specialist MoE (S3-B) | 24.630440 | 0.2995% | 18.5603% | 27.1317% | 16.9956% | 1.000 | 0 |
| Proposed Specialist MoE (S3-B) | 32.740070 | 0.0730% | 5.0620% | 5.0119% | 20.4062% | 1.000 | 0 |
| Proposed Specialist MoE (S3-B) | 39.685480 | 0.0384% | 1.5226% | 3.1366% | 47.0602% | 1.000 | 0 |
| Proposed Specialist MoE (S3-B) | 45.142700 | 0.0618% | 1.9665% | 5.6305% | 11.4498% | 1.000 | 0 |
| Vanilla-FNN-MoE | 24.630440 | 0.1723% | 7.4363% | 49.4685% | 4.0681% | 1.000 | 0 |
| Vanilla-FNN-MoE | 32.740070 | 0.0619% | 2.7920% | 9.5741% | 17.1181% | 1.000 | 0 |
| Vanilla-FNN-MoE | 39.685480 | 0.0330% | 2.2163% | 2.5342% | 75.1743% | 1.000 | 0 |
| Vanilla-FNN-MoE | 45.142700 | 0.0699% | 3.9201% | 5.2795% | 42.3384% | 1.000 | 0 |
| DataOnly-MoE | 24.630436 | 0.5122% | 27275.4654% | — | — | 1.000 | 1 |
| DataOnly-MoE | 32.740067 | 0.1790% | 10365.5271% | — | — | 1.000 | 1 |
| DataOnly-MoE | 39.685478 | 0.0795% | 5465.6783% | — | — | 1.000 | 1 |

Proposed S3-B 的 K56 pressure fixed-point worst 为 27.1317%，Vanilla 为 49.4685%，相对降低 45.1535%。在更严格的 perturbation-bank 评价中，K56 最坏 paired gain 从 107702.891 降至 585.262，相对降低 99.4566%。这说明 proposed contraction training 和物理耦合显著减弱了最坏方向的扰动放大，是 Steady specialist 的核心吸引子优势。

但该结果不能被写成“Strict attractor 全部通过”：两种模型的最坏 paired gain 仍大于 1，说明仍存在放大方向；DataOnly 又没有执行相同 perturbation-bank evaluator。因此本研究对 Steady 的准确表述是“固定点误差和扰动鲁棒性显著改善”，而不是“所有方向严格收缩”。这一保守表述反而能清楚区分 clean-field accuracy、数值有限性和局部吸引性三个不同概念。

#### (b) Hopf 指定严格保持案例

| 指定 Re | K56速度 | K56压力 | RMS振幅误差 | 峰峰值误差 | 频率误差 | 末端相位漂移(周期) | 轨道距离 | 速度能量漂移 | 压力能量漂移 | finite | 发散窗 | false growth | Strict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 49.299999 | 0.0013% | 0.0095% | 2.9146% | 2.5017% | 1.6477% | -0.1166 | 7.2697% | 0.0007% | 0.0100% | 1.000 | 0 | 否 | PASS |
| 49.599998 | 0.0016% | 0.0091% | 1.4693% | 6.6160% | 1.6169% | -0.1136 | 6.4963% | 0.0011% | 0.0010% | 1.000 | 0 | 否 | PASS |
| 50.000000 | 0.0020% | 0.0171% | 1.3430% | 1.3787% | 1.5220% | -0.1138 | 6.4660% | 0.0016% | 0.0078% | 1.000 | 0 | 否 | PASS |

Re=49.3、49.6、50.0 的 Proposed Hopf specialist 均通过完整 K56 strict conjunction。每个案例同时满足速度/压力场误差不超过 5%、振幅误差不超过 10%、频率误差不超过 5%、末端相位漂移小于 0.25 周期、轨道距离不超过 10%、能量漂移合格、finite fraction=1、零 divergence 且无 false growth。结果说明在这些指定参数点，模型能够同时保持物理场、极限环振幅、频率、相位和局部轨道几何。

这些 Re 在原始 split 中参与训练，因此它们用于证明“模型在代表性 Hopf 参数点具备严格 K56 吸引子保持能力”，而不是证明跨 Re 的 held-out 泛化。独立 validation/held-out 上的完整严格通过率仍应在限制中如实给出。

#### (c) Periodic 的三个严格保持测试案例

| 测试 Re | K48速度 | K48压力 | RMS振幅误差 | 峰峰值误差 | 频率误差 | 相位RMS(rad) | 末周期漂移 | 轨道距离 | 速度能量漂移 | 压力能量漂移 | finite | 发散窗 | Strict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100.352249 | 0.3972% | 1.7083% | 0.4335% | 0.1580% | 0.1366% | 0.0223 | -0.0088 | 2.4061% | 0.0712% | 0.7429% | 1.000 | 0 | PASS |
| 149.059235 | 0.3809% | 1.6825% | 0.5809% | 0.5302% | 0.1320% | 0.0229 | 0.0029 | 2.7589% | 0.0030% | 0.8959% | 1.000 | 0 | PASS |
| 189.862274 | 1.1274% | 4.4272% | 0.8792% | 1.5040% | 0.3816% | 0.0702 | -0.0241 | 4.4317% | 0.0311% | 0.8300% | 1.000 | 0 | PASS |

三个 Re 均来自 Periodic held-out 集，并通过原生 K48 周期吸引子联合门。Proposed 在保持低物理场误差的同时，RMS/峰峰值振幅、频率、相位漂移及轨道距离均较小，表明模型没有通过把振荡衰减到均值场来“获得”低总场误差，而是真正保留了极限环的主要几何和时序结构。相较于 Vanilla 仅 1/4 held-out 通过严格周期门，Proposed 达到 3/4，支持特殊 expert 和 physics–data coupling 对周期吸引子保持的有效性。

### 4.3.2 S–H 边界吸引子比较

| 测试 Re | 方法 | K56联合误差 | 尾段中心误差 | 振幅绝对误差 | 轨道几何误差 | αS | αH |
|---|---|---|---|---|---|---|---|
| 42.359071 | E2 Top-1 | 4.6503% | 0.0513% | 0.0018% | 1.9375% | — | — |
| 42.359071 | T2-C | 4.6433% | 0.0511% | 0.0018% | 1.9392% | 0.9987 | 0.0013 |
| 43.500000 | E2 Top-1 | 2.4877% | 0.0551% | 0.0009% | 1.6198% | — | — |
| 43.500000 | T2-C | 3.2343% | 0.0394% | 0.0002% | 0.2458% | 0.0861 | 0.9139 |
| 43.900000 | E2 Top-1 | 6.8749% | 0.0574% | 0.0009% | 1.6135% | — | — |
| 43.900000 | T2-C | 1.2574% | 0.0215% | 0.0005% | 0.5202% | 0.1891 | 0.8109 |

### 4.3.3 瞬态误差与吸引子误差的不一致

Re=43.50 是必须保留的反例：T2-C 的全 K56 场误差高于 E2，但尾段中心、振幅和局部轨道几何均优于 E2。这说明瞬态逐时刻拟合和渐近动力结构保持并不等价。Re=43.90 则在两类指标上同时改善。论文因此不使用单一 joint field error 替代吸引子评价，也不以个别吸引子改善掩盖全窗口误差退化。

## 4.4 泛化能力、计算成本与限制

所有测试均在 checkpoint 和 validation 选择冻结后执行；但历史上 S–H test 曾被查看，因此不能宣称完整 blind test。新消融仅使用单一固定 seed，不能给出跨 seed 统计显著性。T2-C 在 S–H 窗口需要两个 frozen specialists 各自执行一次原生 rollout；离开重叠区后 admissibility mask 自动退化为单专家成本。

当前最重要的限制包括：Steady 尚未实现所有扰动方向严格收缩；Hopf 三个严格案例属于指定训练 Re；Periodic 的第四个 held-out Re=70.314635 未通过完整 strict conjunction；H–P 没有 validation-supported 共同可行域；Global MoE 尚未与多坐标系统通过同一全域物理场 evaluator 比较；现有 Router 数据没有真实 startup 或缓变 Re(t) 转移轨迹。

综合而言，实验支持“受动力学可行性约束的吸引子专门化多坐标稀疏 MoE-ROM”这一方法结论：Physics–Data specialist 在 Hopf/Periodic 的长期稳定性和吸引子保持上提供关键收益；Steady contraction 组件显著改善固定点及扰动鲁棒性；T2-C 只在 S–H 共同可行域启用；当 H–P 缺少共同 validation 支撑时，系统正确地退化为 P-only，而不是强制融合。

## 4.5 结果文件与可追溯性

Steady Proposed 的权威结果来自 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/evaluation_inputs/s3_heldout_metrics.json`；Steady Vanilla 与 DataOnly 分别来自 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/steady/vanilla-fnn/frozen_user_stop_step6200_20260724/evaluation/test/metrics.json` 和 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/steady/data-only/evaluation/test/metrics.json`。Hopf 严格吸引子结果来自 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/native_attractor_audit_20260723_v1/results/HOPF_NATIVE_ATTRACTOR_AUDIT.json`，Hopf DataOnly 来自 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/hopf/data-only/evaluation/test/metrics.json`，Vanilla 的 validation fail-closed 证据来自 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/hopf_v6/hopf/vanilla-fnn/evaluation/validation/metrics.json`。Periodic Proposed、Vanilla 和补充 K48 DataOnly 结果分别来自 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json`、`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/periodic_v4/periodic/vanilla-fnn/evaluation/test/native/periodic_r32_multihorizon_evaluation.json` 和 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/periodic/data-only/evaluation/test_k48_supplement_20260724/metrics.json`。S–H 场误差、matched oracle 与吸引子结果分别来自 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/final_test_evaluation/EVALUATION_REPORT.json`、`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/supplements/per_re_matched_oracle_v4/PER_RE_MATCHED_ORACLE_REPORT.json` 和 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/supplements/t2c_attractor_v2/T2C_ATTRACTOR_ANALYSIS.json`。H–P fail-closed 证据来自 `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_hp_20260723_v1/H_P_RECOVERY_REPORT_20260723.md`。
