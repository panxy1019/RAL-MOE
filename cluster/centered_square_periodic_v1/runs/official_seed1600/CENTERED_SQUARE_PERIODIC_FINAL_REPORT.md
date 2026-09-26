# CenteredSquare Periodic Specialist：最终训练、验证与测试报告

## 结论

方柱绕流 Periodic specialist 已在 RTX 4090 的 `pt_env` 环境中完成从零训练、验证和冻结 heldout 测试。最终采用验证最优的 epoch-205 模型（而非末轮 epoch-240 参数）；`final.pt` 与 `best_validation.pt` 的模型状态哈希完全一致。

模型在四个 heldout Re 上的所有 K24 autonomous rollout 均保持有限值。压力 closure 表现强：POD 系数相对 L2 平均为 **1.682%**，相对原始 Pressure-Poisson surrogate 的平均 **368.37%** 有显著改善。速度动态和长时 rollout 仍有较大误差，且随着 Re 增大而变差，因此该模型可作为稳定、可用的方柱 Periodic 初始 specialist，但尚未达到高精度长时预测标准。

## 数据与划分合同

数据根目录：`/root/centeredSquare_three_regime_overlap_v1`。

| 项目 | 数值 |
|---|---:|
| Periodic 域 Re 范围 | 98.5--150 |
| Re 个数 | 37 |
| 总快照数 | 4,662 |
| POD 有效 history-3 样本 | 4,551 |
| Train / validation / heldout Re | 27 / 6 / 4 |
| Train / validation / heldout 样本 | 3,321 / 738 / 492 |
| Velocity / pressure POD rank | 28 / 26 |
| 压力 gauge | 每快照减 volume-weighted mean |
| Heldout Re | 100.5, 102.0, 120.689655, 144.827586 |

POD 均值、基和归一化统计量仅由 27 个 train Re 拟合。验证与 heldout Re 仅被投影到冻结 POD 基中，未参与拟合。

## 训练配置

基线架构沿用已交接的 HPRS-MoE Periodic specialist：共享物理编码器、3 个 regime group、每 group 6 个 routed expert + 1 个 shared expert，velocity/pressure 独立稀疏路由，pressure closure 为 `adaptive_gate`。

| 项目 | 数值 |
|---|---:|
| 随机种子 | 1600 |
| Epoch | 240 |
| 批大小 | 256 |
| 学习率 / weight decay | 5.5e-4 / 1.5e-4 |
| 积分器 | RK4 |
| 训练 rollout curriculum | K4 -> K8 -> K12 -> K16 |
| 最终测试 rollout horizon | K24 |
| Rollout 采样频率 | 每 8 个 mini-batch 一次 |
| SwanLab | online，运行见下方链接 |

方柱快照间隔为 `dt=4`，而测得脱落周期约为 5.39--5.66。直接用一个 RK4 步推进 `dt=4` 会在初始物理 ROM rollout 中溢出。为保持连续 Galerkin ROM 的数值稳定性，正式运行将每个外部步拆为最多 8 个 `dt<=0.5` 的内部 RK4 子步；外部数据时钟、标签和 split 均未改变。

## 验证集结果与模型选择

验证分数定义为 `RHS relative L2 + coefficient relative L2 + 0.35 * pressure relative L2`。分数越低越好。

| Epoch | Train loss | Validation score | RHS L2 | Coefficient L2 | Pressure L2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 213.2814 | 10.6484 | 5.2321 | 4.5036 | 2.6074 |
| 5 | 19.6653 | 3.9630 | 2.1266 | 1.7691 | 0.1922 |
| 60 | 0.3394 | 1.4342 | 1.0674 | 0.3589 | 0.0223 |
| 115 | 0.2793 | 1.2558 | 0.9676 | 0.2824 | 0.0167 |
| **205 (best)** | **0.1872** | **1.2221** | **0.9788** | **0.2390** | **0.0121** |
| 240 | 0.2182 | 1.2247 | 0.9808 | 0.2399 | 0.0117 |

训练稳定收敛，没有记录到 NaN 或 OOM。模型最终回滚到 epoch-205 验证最优状态后进行 heldout 测试。

## Heldout 最终测试

下表均为冻结 heldout Re 的 local-POD coefficient 空间相对 L2。K24 指 24 个外部快照步的 autonomous rollout；每个 Re 评估 4 个有效窗口，均为有限值。

| Heldout Re | RHS L2 | Pressure closure L2 | K1 velocity | K1 pressure | K24 velocity | K24 pressure | K24 pressure energy error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100.5 | 89.39% | 0.89% | 61.83% | 58.39% | 42.78% | 34.35% | 13.17% |
| 102.0 | 92.80% | 0.94% | 78.25% | 75.39% | 65.78% | 58.07% | 7.63% |
| 120.690 | 98.64% | 4.50% | 121.97% | 130.04% | 93.32% | 97.31% | 8.44% |
| 144.828 | 98.90% | 0.39% | 117.16% | 119.37% | 114.90% | 115.56% | 39.07% |
| **平均** | **94.93%** | **1.68%** | **94.80%** | **95.79%** | **79.20%** | **76.32%** | **17.08%** |

压力 closure 对比尤为明显：原 Pressure-Poisson surrogate 的 heldout 相对 L2 分别为 496.62%、481.33%、305.86% 和 189.67%，而 MoE closure 分别降为 0.89%、0.94%、4.50% 和 0.39%。

速度部分也优于 Galerkin-only RHS：四个 Re 的 Galerkin RHS 相对 L2 为 573.40%--611.38%，MoE 降为 89.39%--98.90%，但高 Re 的 autonomous trajectory 仍明显困难。这说明当前结果适合作为后续 Periodic chart 的稳定起点，尚不宜将 K24 高 Re 预测视为高保真重建。

> 说明：本表是训练器在局部 POD 系数/ROM 评价口径下输出的误差。它不是直接从重建 CFD 物理场计算的 volume-weighted field L2；因此不能与圆柱交接文档中的物理场百分比逐项直接比较。

## 最终一致性检查

- `best_validation.pt`：epoch 205，best validation score `1.2220910639`。
- `final.pt`：epoch 240 的训练记录，但加载的模型状态与 `best_validation.pt` 一致；模型状态 SHA-256 为 `de738a711a21ea5cd5fc39966faf2a3d233b78ab73492bc65d0ed47530b44ada`。
- 四个 heldout Re 均输出 4 个 K24 autonomous rollout 窗口，说明评估器没有因非有限预测丢弃窗口。
- 输入数据包、最终 checkpoint 和最终 metrics 的 SHA-256 清单见 `FINAL_ARTIFACT_SHA256SUMS.txt`。

## 产物与复现

实验根目录：

`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/centered_square_periodic_v1`

| 用途 | 路径 |
|---|---|
| 最佳模型 | `runs/official_seed1600/best_validation.pt` |
| 最终交付模型（与最佳模型状态一致） | `runs/official_seed1600/final.pt` |
| 全量测试指标 | `runs/official_seed1600/centered_square_periodic_v16_public_r28_p26_metrics.json` |
| 测试误差表 | `runs/official_seed1600/centered_square_periodic_v16_public_r28_p26_error_vs_re.csv` |
| 训练器自动摘要 | `runs/official_seed1600/centered_square_periodic_v16_public_r28_p26_summary.md` |
| 本报告 | `runs/official_seed1600/CENTERED_SQUARE_PERIODIC_FINAL_REPORT.md` |
| 哈希清单 | `runs/official_seed1600/FINAL_ARTIFACT_SHA256SUMS.txt` |
| SwanLab | https://swanlab.cn/@panxy1019/V17SquarePeriodicMOE/runs/ejlmkjqz |

复现主命令：

```bash
cd /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/centered_square_periodic_v1
/root/miniconda3/envs/pt_env/bin/python code/run_square_periodic_training.py \
  --trainer code/train_square_periodic_moe.py \
  --source-checkpoint /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt \
  --data-root assets \
  --output-dir runs/official_seed1600 \
  --epochs 240 --swanlab-mode online
```

