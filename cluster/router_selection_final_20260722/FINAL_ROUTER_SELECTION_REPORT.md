# E2/E3 Router 最终选择报告

日期：2026-07-22  
最终决策：`KEEP_E2_AS_FORMAL_BASELINE`

## 1. 门控结果

| 门控 | 结果 |
|---|---|
| 冻结 E2 best/config/SHA256 | PASS；checkpoint SHA256 `f0286a9974f16b513bb42cd94232c397123dcd3e0a01eab475ff1593f7881dcd` |
| E2 validation 统一 wrapper Top-1 smoke | PASS；10/10 路由与 oracle 一致 |
| 实际调用三套 native validation rollout | PASS；S/H/P 均 finite=1、divergent=0 |
| E2 正式 baseline 标记 | PASS；只读批准记录已落盘 |
| E3 chart-independent 输入合同 | PASS；固定 64 点公共物理观测，不接收 chart/label |
| E3 common mesh/area/pressure gauge | PASS；三套资产 exact |
| E3 原 split、原 seed/budget 重训 | PASS；8000 steps，best step 5500 |
| E3 checkpoint strict reload | PASS；概率最大复现差 `5.96e-8` |

## 2. E2 unified smoke

冻结 E2 对全部 10 条 validation Re 轨迹进行一次 trajectory-level Top-1 选择，然后直接调用所选 specialist 的 native validation 路径。wrapper 没有改写 scaler、feature/history、积分器或 pressure closure。

| Specialist | Validation Re | Native 路径 | Finite | Divergent |
|---|---:|---|---:|---:|
| Steady | 2 | `evaluate_horizon(validation,K4)` | 1.0 | 0 |
| Hopf | 2 | `train_h4_expanded.validate` | 1.0 | 0 |
| Periodic | 6 | `integrate_autonomous_step_np`, K4 | 1.0 | 0 |

因此 E2 已正式标记为 `E2_ReOnly_TrajectoryTop1_UnifiedSystem`。该标记证明路由与集成合同，不覆盖 specialist 各自的科学硬门。

## 3. 修复后的 E3

E3 的 8 个描述符现在只由 Re、最近三步时间戳，以及公共网格上固定 64 个速度/压力传感器观测计算。传感器由几何坐标确定；descriptor 函数不接收 chart 或类别标签。训练数据库中的 POD 只用于把历史数据存储格式解码成物理传感器值。

跨 chart 表示审计使用同一个源物理状态投影到目标 chart，没有拼接不同 Re：

| 方向 | descriptor relative-L2 mean | 解释 |
|---|---:|---|
| S→H | `2.77e-6` | 表示一致性良好 |
| H→S | `2.11e-7` | 表示一致性良好 |
| H→P | `1.30e-3` | 无真实 Re overlap，仅外推诊断 |
| P→H | `1.48e-4` | 无真实 Re overlap，仅外推诊断 |

## 4. Validation 决策比较

| 指标 | E2 | 修复后 E3 |
|---|---:|---:|
| Balanced accuracy | 1.000 | 1.000 |
| Macro-F1 | 1.000 | 1.000 |
| NLL | 0.159584 | 0.020914 |
| Brier | 0.086836 | 0.005225 |
| ECE-10 | 0.129758 | 0.019458 |
| 最小 Top-1 margin | 0.205955 | 0.685445 |
| 非相邻 S-P 次高排序 | 0 | 6 |

E3 的校准和 margin 明显改善，但没有任何 validation/test 轨迹从错误选择变为正确选择；两者的 accuracy、balanced accuracy、macro-F1 和混淆矩阵完全相同。当前也没有新增经认证的边界或跨流态轨迹，且 E3 更复杂、raw secondary ranking 仍产生非相邻 S-P。

依据预先约定的“相当则保留更简单、合同更清晰的 E2”，本轮不把概率置信度提升单独解释为 E3 在主任务上明显优于 E2。E3 checkpoint 和审计作为有效消融结果保留，但不替换 baseline。

## 5. Top-2 决策

- 后续相邻 Top-2 的 Router 候选保持 **E2**。
- 本轮没有执行 Top-2。
- 实现 Top-2 前必须显式施加 S–H–P adjacency mask；禁止直接 S–P 组合。
- 只允许 S–H、H–P 相邻边界的输出级物理场融合，遵循修正后的论文设计，不恢复连续 RHS 融合路线。
- 仍只能称为 `temporal-consistent regime router`，不得声称真实 S→H→P 时间迁移。

## 6. 产物

- E2 unified smoke：`/root/panxy/particalMOE/e2_unified_validation_smoke_20260722`
- E2 frozen baseline：`/root/panxy/particalMOE/trajectory_router_e1_e2_e3_20260722/frozen_E2_baseline_candidate`
- E3 repaired run：`/root/panxy/particalMOE/trajectory_router_e3_chart_independent_20260722`
- E3 SwanLab：[run 1wi4t1c3](https://swanlab.cn/@panxy1019/V17_TrajectoryLevel_Hierarchical_MoE/runs/1wi4t1c3)
