# Periodic local r32/rp32 HPRS-MoE-ROM heldout 评估报告

## 1. 结论摘要

本报告评估 `periodic_v16_public_r32` 的 best-validation checkpoint（epoch 85，best validation score 0.285295）。该 checkpoint 只完成了 K4 和 K8 训练，尚未经过 K16 curriculum。

四个 heldout Reynolds 数为：

- Re = 70.314635（developing periodic）
- Re = 100.352251（mature periodic）
- Re = 149.059229（mature periodic）
- Re = 189.862278（high-Re periodic）

主要结论：

- 四个 Re 在全部 13 个共同窗口、全部 K1/K4/K8/K16/K24/K48 horizon 上均保持 100% finite；divergent windows 为 0，未出现首次发散 step。
- K48 velocity area-weighted L2 全部小于 1.35%。
- K48 pressure area-weighted L2 在三个成熟/高 Re 周期点为 1.68%–4.43%；developing periodic 的 Re=70.314635 为 5.96%，略高于 5% preserved 阈值。
- K48 极限环振幅、频率、相位、轨道和能量在四个 Re 上总体稳定。Re=70.314635 的失败不是数值发散，而是 pressure physical-field L2 单项超阈值。
- 严格联合判定为 preserved 3/4；未保留点为 Re=70.314635。

因此，当前模型已经很好地保持成熟周期区域，但 developing periodic 的长程压力精度仍差约 0.96 个百分点。建议保留 epoch-85 checkpoint 作为不可覆盖的基线，并继续一轮受控 K16 训练；K16 checkpoint 必须按固定 K48 validation 选择，不能继续使用当前 one-step 主导的 validation score 单独早停。

## 2. 模型和评估对象

| 项目 | 配置 |
|---|---|
| 模型 | Periodic-only HPRS-MoE-ROM |
| POD | Periodic local ru=32, rp=32 |
| 速度推进 | Galerkin + learned residual + RK4 |
| 压力 | Pressure Poisson surrogate + AdaptiveGate closure |
| checkpoint | `best_validation.pt` |
| checkpoint epoch | 85 |
| 已训练 curriculum | K4、K8；未训练 K16 |
| 训练/验证/heldout | 53/6/4 Re，零交集 |
| pressure gauge | subtract area mean per snapshot |
| 评估窗口 | 每个 Re 13 个共同起点，stride=8，均可完整推进 K48 |

评估中的速度和压力状态、历史状态全部由预测值递推；Re 保持为给定参数，phase/time 使用数据库时钟。因而这是 state-autonomous rollout，但不是“未知时钟/未知 phase”的自由积分。

物理场 area-weighted L2 使用 POD 基、regime mean 和 point-area quadratic form 精确计算。它衡量 r32 POD 子空间内的重构真值与预测值，不包含相对于原始 CFD 场的 POD 截断误差。

## 3. 物理场精度

以下全部相对误差均乘以 100，以百分比表示。

### 3.1 Re = 70.314635

| Horizon | Velocity area-L2 | Pressure area-L2 |
|---:|---:|---:|
| K1 | 0.241% | 1.762% |
| K4 | 0.341% | 2.246% |
| K8 | 0.401% | 2.430% |
| K16 | 0.533% | 3.049% |
| K24 | 0.671% | 3.685% |
| K48 | **1.347%** | **5.960%** |

### 3.2 Re = 100.352251

| Horizon | Velocity area-L2 | Pressure area-L2 |
|---:|---:|---:|
| K1 | 0.154% | 0.830% |
| K4 | 0.172% | 0.746% |
| K8 | 0.187% | 0.814% |
| K16 | 0.223% | 0.959% |
| K24 | 0.263% | 1.116% |
| K48 | **0.397%** | **1.708%** |

### 3.3 Re = 149.059229

| Horizon | Velocity area-L2 | Pressure area-L2 |
|---:|---:|---:|
| K1 | 0.121% | 0.582% |
| K4 | 0.143% | 0.704% |
| K8 | 0.161% | 0.811% |
| K16 | 0.186% | 0.952% |
| K24 | 0.215% | 1.098% |
| K48 | **0.381%** | **1.683%** |

### 3.4 Re = 189.862278

| Horizon | Velocity area-L2 | Pressure area-L2 |
|---:|---:|---:|
| K1 | 0.228% | 0.976% |
| K4 | 0.299% | 1.487% |
| K8 | 0.317% | 1.425% |
| K16 | 0.379% | 1.603% |
| K24 | 0.475% | 2.000% |
| K48 | **1.127%** | **4.427%** |

物理场结果表明，velocity 长程误差增长缓慢；主要剩余风险是 pressure，并集中在 developing periodic 与最高 Re 的 K48。

## 4. 模态和闭合诊断

### 4.1 K48 coefficient relative L2

| Re | Velocity coefficient L2 | Pressure coefficient L2 |
|---:|---:|---:|
| 70.314635 | 1.928% | 2.398% |
| 100.352251 | 1.869% | 2.335% |
| 149.059229 | 1.684% | 3.858% |
| 189.862278 | 2.915% | 6.769% |

### 4.2 One-step operator诊断

| Re | RHS relative error | Pressure closure error |
|---:|---:|---:|
| 70.314635 | 18.552% | 0.470% |
| 100.352251 | 22.144% | 0.514% |
| 149.059229 | 21.011% | 0.505% |
| 189.862278 | 18.543% | 0.640% |

RHS error明显高于最终状态误差，说明 RK4 积分、Galerkin base 与学习残差的组合对状态轨迹具有一定误差抵消。继续训练时不能只追求 RHS 指标下降，还必须同时检查 autonomous K48/K96 稳定性。

## 5. 数值稳定性

| Re | Horizon | Finite fraction | Divergent windows | 首次发散 step |
|---:|---|---:|---:|---|
| 70.314635 | K1–K48 | 100% | 0/13 | 无 |
| 100.352251 | K1–K48 | 100% | 0/13 | 无 |
| 149.059229 | K1–K48 | 100% | 0/13 | 无 |
| 189.862278 | K1–K48 | 100% | 0/13 | 无 |

divergent window 定义为出现 non-finite 状态，或预测 velocity/pressure modal norm 超过该真值窗口最大 norm 的 10 倍。四个 Re 均未触发。

## 6. 极限环、频率、相位与轨道

诊断坐标由每个 heldout Re 的完整真值 modal trajectory 确定二维主轨道平面；该坐标只用于评估，不参与训练。频率由 K48 窗口中展开相位的时间斜率计算。

| Re | RMS振幅误差 | Peak-to-peak误差 | 真值主频/St | 预测主频/St | St误差 | Phase RMS | 末端漂移 | Orbit distance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70.314635 | 1.396% | 2.297% | 0.009294 | 0.009339 | 0.553% | 0.108 rad | +0.0237 cycle | 0.0782 |
| 100.352251 | 0.433% | 0.158% | 0.015791 | 0.015769 | 0.137% | 0.022 rad | -0.0088 cycle | 0.0241 |
| 149.059229 | 0.581% | 0.530% | 0.026394 | 0.026380 | 0.132% | 0.023 rad | +0.0029 cycle | 0.0276 |
| 189.862278 | 0.879% | 1.504% | 0.035446 | 0.035581 | 0.382% | 0.070 rad | -0.0241 cycle | 0.0443 |

成熟周期区的振幅与频率保持非常好。Re=70.314635 的 orbit distance 和 phase error 较高，但仍处于受控范围，没有出现相位持续失锁或轨道离开吸引子。

## 7. 能量漂移

下表为 K48 预测场平均能量相对真值平均能量的偏差。

| Re | Velocity energy drift | Pressure energy drift |
|---:|---:|---:|
| 70.314635 | 0.165% | 1.155% |
| 100.352251 | 0.071% | 0.743% |
| 149.059229 | 0.003% | 0.896% |
| 189.862278 | 0.031% | 0.830% |

能量漂移均明显低于 10% 阈值，未发现能量爆炸或系统性衰减。

## 8. 最终 preserved 判定

联合判据固定为：K48 velocity/pressure area-L2 均不超过 5%；finite fraction=100%；divergent windows=0；RMS与peak-to-peak振幅误差不超过10%；St误差不超过5%；末端相位漂移不超过0.25 cycle；normalized orbit distance不超过0.10；velocity/pressure energy drift均不超过10%。

| Re | 判定 | 主要依据 |
|---:|---|---|
| 70.314635 | **Not preserved** | K48 pressure area-L2=5.960%，其余稳定性/极限环项通过 |
| 100.352251 | **Preserved** | 全部指标通过 |
| 149.059229 | **Preserved** | 全部指标通过 |
| 189.862278 | **Preserved** | 全部指标通过；K48 pressure=4.427%，接近5%阈值 |

最终 heldout preserved count：**3/4**。

## 9. 是否继续 K16 训练

建议继续，但采用受控 continuation，而不是覆盖当前模型重新开始：

1. 固定保存 epoch-85 `best_validation.pt`，作为当前 3/4 preserved 基线。
2. 从 epoch-180 `final_training.pt` 恢复 optimizer、scheduler 和 RNG，进入真正的 K16 阶段；当前训练在 epoch 180 早停，实际上从未执行 K16。
3. K16 至少训练一个预先固定的最小预算后才允许早停；早停与 best checkpoint 必须使用固定 K1/K4/K8/K16/K24/K48 validation 曲线，而不是当前 one-step 主导的混合 score。
4. 主要目标是把 developing periodic 的 K48 pressure 从5.960%压到5%以内，同时要求 Re=100/149/189 的 K48 velocity、pressure、频率和轨道指标不回退。
5. 若 K16 continuation 使成熟周期区任一 Re 从 preserved 退化为 not preserved，应保留当前 epoch-85 模型，不接受新 checkpoint。

当前证据支持继续 K16：问题是局部、幅度较小且与缺失 K16 训练直接相关；但继续训练的收益必须由固定多 horizon validation 证明。

## 10. 产物

- 正式评估 JSON：`multihorizon_evaluation/periodic_r32_multihorizon_evaluation.json`
- 评估日志：`multihorizon_evaluation/evaluation_v2.log`
- 被替换的首版短窗 FFT 诊断：`multihorizon_evaluation/periodic_r32_multihorizon_evaluation_v1_signal_a0.json`
- best checkpoint：`best_validation.pt`
- 可恢复训练末态：`final_training.pt`

