# CenteredSquare Steady Specialist 最终实验报告

- 生成时间：`2026-07-28T14:56:54+0800`
- 模型：`CenteredSquare-Steady-S2B-rank999`
- POD：rank999，`ru=5`、`rp=4`
- 压力规范：`subtract_volume_mean_per_snapshot`
- SwanLab：https://swanlab.cn/@panxy1019/V17indepentMOEV2/runs/mk26kj3q

## 1. 数据与冻结合同

| Split | Re | 案例数 | 快照数 |
|---|---|---:|---:|
| Train | 见 `ASSET_AUDIT.json` | 60 | 7560 |
| Validation | 55, 75, 90, 94.5, 95.25 | 5 | 630 |
| Heldout/Test | 60, 85, 95.1, 95.3 | 4 | 504 |

训练、validation、heldout 的 Re 集合零交集。POD、均值与 normalization 仅由 train split 拟合；heldout 未用于训练、早停或 checkpoint 选择。

## 2. Checkpoint 结论

- 训练于 step `3200` 正常早停。
- 冻结 checkpoint：`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/centeredsquare_steady_specialist_v1/checkpoint/frozen_centeredsquare_steady_step3200.pt`
- SHA256：`23c7b18fb44316fca28bef0aeac4458235c100b99b2b0c4503d7ae5770cc6fe9`
- 预注册状态：`FROZEN_UNQUALIFIED_CANDIDATE`。
- validation 中通过 contraction `<1` 门的 checkpoint 数：`0`。

因此该权重用于一次性 heldout 性能刻画，但不能标记为通过旧圆柱 contraction 硬门的 qualified deployment checkpoint。

## 3. Heldout clean autonomous rollout

下表是四个 heldout Re 合并后的误差；relative L2 已乘 100。`Full CFD physical` 直接相对原始 CFD 快照，包含 POD 截断误差。

| K | 窗口数 | Raw POD U | Raw POD p | Projected physical U | Projected physical p | Full CFD physical U | Full CFD physical p | Finite | Divergent |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 492 | 16.3801% | 3.2915% | 0.5050% | 0.4771% | 0.6111% | 0.6103% | 100.0% | 0 |
| 4 | 480 | 30.2213% | 8.3274% | 0.9171% | 1.2067% | 0.9730% | 1.2521% | 100.0% | 0 |
| 8 | 464 | 38.2555% | 9.7535% | 1.1564% | 1.4123% | 1.1971% | 1.4440% | 100.0% | 0 |
| 16 | 432 | 49.7494% | 9.8928% | 1.5018% | 1.4319% | 1.5311% | 1.4583% | 100.0% | 0 |
| 32 | 368 | 65.2516% | 11.4117% | 1.9687% | 1.6515% | 1.9903% | 1.6723% | 100.0% | 0 |
| 64 | 240 | 76.3684% | 13.0815% | 2.3037% | 1.8930% | 2.3219% | 1.9106% | 100.0% | 0 |

### K16 分 Re 的 Full CFD physical relative L2

| Heldout Re | Velocity | Pressure | POD floor U | POD floor p |
|---:|---:|---:|---:|---:|
| 60 | 1.4510% | 1.4876% | 0.0602% | 0.0914% |
| 85 | 1.0176% | 0.5641% | 0.0921% | 0.1291% |
| 95.1 | 1.7555% | 1.7285% | 0.3929% | 0.3892% |
| 95.3 | 1.7743% | 1.7637% | 0.4337% | 0.4179% |

## 4. 稳定性指标

- K16 controlled contraction worst：`582.232`。
- K16 pressure drift worst：`1.19121`。
- K16 fixed-point residual worst：`1.59628`。
- K16 finite fraction：`100.0%`；divergent windows：`0`。

旧圆柱 contraction 门使用相对极小 steady POD 系数的归一化扰动比；在本方柱数据上该比值远大于 1。clean rollout 保持有限且无发散，但不能据此宣称局部扰动收缩。

## 5. 资产与数值审计

- Train POD 系数复投影相对差：velocity `1.470e-07`，pressure `3.874e-07`。
- ROM 的 1/Re 仿射重建残差最大约 `1.016e-15`。
- Heldout residual 与 POD modes 的最大加权内积：velocity `1.408e-08`，pressure `2.142e-08`。
- `latest.pt` 与 `final.pt` 模型权重 SHA256 相同：`True`。

## 6. 最终判断

该模型在全部 heldout clean rollout 窗口上保持 finite 且无发散，并给出了可复现的方柱 steady 预测基线；但 validation 与 heldout 的 controlled contraction 均未通过 <1 硬门，因此结果应标记为 stable clean-rollout baseline / unqualified contraction，而不是合格的局部吸引子部署模型。

完整逐窗口/逐 Re 数值见 `heldout_metrics.json`；checkpoint 选择证据见 `checkpoint_selection.json`。
