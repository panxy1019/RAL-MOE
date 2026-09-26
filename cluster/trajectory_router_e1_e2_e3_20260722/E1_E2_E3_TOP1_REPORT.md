# E1/E2/E3 trajectory-level Top-1 实验终态报告

日期：2026-07-22  
状态：三组实验均已完成，未追加训练或修改 checkpoint。

## 1. 结论

- **E1 Oracle**：通过，给出当前固定 Re、整轨迹 Top-1 任务的上界。
- **E2 Re-only**：在当前 held-out 集上通过，11/11 条轨迹均选择正确 specialist，可作为当前最简、部署合同明确的 Top-1 基线；但 S/H 边界样本 `Re=45.142702578` 的 Top-1 margin 仅 `0.00459456`，边界鲁棒性尚未证明。
- **E3 Re + physical descriptors**：作为消融实验通过，11/11 正确，置信度和校准数值优于 E2；但当前描述符由数据所属 specialist 的局部 POD 系数、基和均值构造，尚未认证为“路由前、未知 chart 时”可从统一物理观测独立计算。因此暂不能把 E3 宣布为可部署 Router，也不能据此认定它优于 E2。
- 本轮没有真实连续跨流态轨迹，只能称为 **`temporal-consistent regime router`**；不能声称已学习真实 S→H→P 时间迁移。

综合判定：**E2 是当前可用 Top-1 基线；E3 是待修正输入合同后复核的候选。**

## 2. 实验合同与数据隔离

三组实验均执行整条固定-Re轨迹级路由，一旦选择 specialist，整条 rollout 保持 Top-1，不在时间步内切换。三个 native specialists 未进入 Router optimizer，训练时也未加载；其 held-out 能力由 E0 结果文件的 SHA-256 固定绑定。

| 项目 | 数值 |
|---|---:|
| 每个 Re 的历史窗口数 | 32 |
| 训练窗口 / 完整 Re 轨迹 | 3072 / 96 |
| 验证窗口 / 完整 Re 轨迹 | 320 / 10 |
| 测试窗口 / 完整 Re 轨迹 | 352 / 11 |
| complete-Re isolation | 通过 |
| 同轨迹跨 split | 未发现 |

测试轨迹组成：Steady 4、Hopf 3、Periodic 4。三个实验使用相同 split。

## 3. 统一结果

下列校准指标按 11 条测试轨迹的平均概率补算；ECE 使用 10 个等宽 confidence bins。样本仅 11 条，因此 ECE 只作诊断，不作统计显著性结论。

| 实验 | Balanced Acc. | Macro-F1 | NLL | Brier | ECE-10 | 平均置信度 | 最小 Top-1 margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 Oracle | 1.000 | 1.000 | 0 | 0 | 0 | 1.000 | 1.000 |
| E2 Re-only | 1.000 | 1.000 | 0.183816 | 0.093280 | 0.150093 | 0.849907 | 0.004595 |
| E3 Re+descriptors | 1.000 | 1.000 | 0.027928 | 0.012463 | 0.024133 | 0.975867 | 0.476421 |

三组测试混淆矩阵完全相同，类别顺序为 `[Steady, Hopf, Periodic]`：

```text
[[4, 0, 0],
 [0, 3, 0],
 [0, 0, 4]]
```

## 4. 边界行为

E2 的核心/远离边界样本置信度高，但在 S/H 边界附近表现出预期的不确定性：

| Re | 真值 | E2 Top-1 概率 | E2 margin | E3 Top-1 概率 | E3 margin |
|---:|---|---:|---:|---:|---:|
| 45.142702578 | Steady | 0.498299 | 0.004595 | 0.738210 | 0.476421 |
| 47.0813545644 | Hopf | 0.633087 | 0.281103 | 0.996851 | 0.993703 |
| 70.3146353337 | Periodic | 0.825245 | 0.650602 | 1.000000 | 1.000000 |

E3 的较大 margin 不能独立证明泛化改善，因为其描述符计算目前知道样本来自哪个 local chart。需要先消除这一前置假设，再做同 split、同 seed budget 的复核。

## 5. 非相邻 Top-2 诊断

- E2：验证/测试的非相邻 S–P Top-2 计数均为 0。
- E3：验证为 6，测试为 4；均来自 Periodic 轨迹中 Steady 概率略高于 Hopf，虽然两者都极小，且不影响正确的 Periodic Top-1。
- E1 的测试计数为 4 是 one-hot 概率中两个零值并列导致的排序伪影，不应解释为 Oracle 违反拓扑。

因此，`illegal_S_P_top2_count` 不是本轮 Top-1 的失败条件。但未来若开展边界 Top-2，必须显式施加 S–H–P 邻接 mask 后再归一化，不能直接使用未约束的第二大 logit。

## 6. Checkpoint 与可复现性

| 文件 | step | SHA-256 |
|---|---:|---|
| E2 best.pt | 6300 | `f0286a9974f16b513bb42cd94232c397123dcd3e0a01eab475ff1593f7881dcd` |
| E2 last.pt | 8000 | `b74d2711f07e97d070d16bfa542357c79e4472e8cfabf99901612dd90fc4f001` |
| E3 best.pt | 5500 | `087b977b439262f5ffe7b33620c8bae799c0ff01a538a5071a7dbaca7309aaed` |
| E3 last.pt | 8000 | `20a77e441d066f086186f902e3490512c5b3cafa35eaf5434ae2b3043777a0ce` |

四个 checkpoint 均已用 CPU `torch.load` 成功读取，参数形状与输入维度一致。E1 无可训练参数，因此没有 Router checkpoint。

E0 绑定哈希：

- Steady：`7df7abd889d62b62acd864b35b476ac3759ade77f246d5886aa790b7517d50d1`
- Hopf：`1272698ae9abd8354cce087e3d5764f8d639e515d4ed5318be3ac0a5dd02dae7`
- Periodic：`b17be4ec3459dfab11b760f1d3f326fb1b153161b9e65ca7c87e887990f2fc1e`

三组结果引用完全相同的 E0 证据。

## 7. 解释边界与未完成证明

本轮 Router 评估通过 E0 哈希绑定选中 specialist 的既有 native rollout 结果，没有重新执行一个统一 wrapper 的端到端 rollout。因此“E2/E3 与 Oracle gap 为 0”严格指 **11 条轨迹的选择决策相同**；它不是新的一次跨工程接口端到端场重构证明。

E3 在部署前需要完成两项最小认证：

1. 将 8 个描述符改为仅从统一的物理场/固定传感器观测与最近两步历史计算，计算时不得知道来源 chart 标签。
2. 对同一物理状态分别经 S/H/P 表示后计算描述符，报告跨 chart 差异；若差异不可忽略，应直接使用物理场或固定传感器定义，而不是 local-POD 重构定义。

在此之前，不建议用 E3 直接推进 Top-2。若下一轮继续，优先顺序应是：先完成上述 E3 输入合同复核与统一 wrapper 的端到端 Top-1 smoke test，再在真实相邻边界数据上设计 S–H、H–P 的输出级 Top-2 实验；不要构造跨 Re 伪时间序列。

## 8. SwanLab

- [E1 run](https://swanlab.cn/@panxy1019/V17_TrajectoryLevel_Hierarchical_MoE/runs/0qq7v0d8)
- [E2 run](https://swanlab.cn/@panxy1019/V17_TrajectoryLevel_Hierarchical_MoE/runs/iojalakv)
- [E3 run](https://swanlab.cn/@panxy1019/V17_TrajectoryLevel_Hierarchical_MoE/runs/btxch0lj)

原始终态 JSON 位于本报告同目录的 `raw/`。
