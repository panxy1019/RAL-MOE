# CenteredSquare Hopf H4 最终实验报告

## 1. 结论摘要

本次实验完成了方柱绕流 Hopf 子数据集上的重新训练、validation checkpoint 冻结和
5 个 held-out Reynolds 数的 K1/K2/K4/K8/K16/K24/K48 自主滚动测试。

- 冻结 checkpoint：step 7200，validation score `2.423865345`；
  `numeric_gate=true`、`hopf_gate=true`、`hard_gate=true`。
- K48 held-out：5/5 工况 `finite_fraction=100%`，总发散窗口为 0。
- K48 平均物理场误差：速度 `0.4387%`，
  压力 `0.4844%`。
- K48 平均径向 RMS 误差：`8.8808%`。
- 5/5 工况的 RMS 振幅误差低于 10%，5/5 频率误差低于 2%。
- 主要不足位于临界点附近的 P2P 相对误差和增长方向判断，而不是数值发散或总场重构。

## 2. 数据与隔离协议

- 数据集：CenteredSquare，OpenFOAM 13 `icoFoam`，CrankNicolson 0.9，
  graded mesh，9400 cells。
- Hopf 专家域：Re 94–102，共 34 个完整工况。
- 划分：23 train / 6 validation / 5 held-out；按完整 Re 工况隔离。
- 样本数：train+validation `3654`，
  held-out `630`。
- POD：速度/压力均采用 train-only、volume-weighted r11 基。
- held-out Re：95.1、95.3、96.5、100.5、102.0。
- checkpoint 冻结前训练程序只加载 train+validation coefficient view；
  held-out view 在冻结文件复制和 SHA 审计后才由独立 evaluator 首次加载。
- POD 系数重投影最大绝对差：速度
  `8.494e-07`，
  压力 `2.273e-06`。

## 3. 方法迁移与稳定化

圆柱实验使用连续时间 Galerkin ROM 作为 RK4 加性主干。方柱数据的相邻存储快照物理时间
间隔较大，直接迁移时正式预检在 Re=97.75 已出现约 191.8 倍模态范数膨胀；全域扫描也表明
这不是可通过删除少量窗口解决的问题。

最终采用的最小稳定化方案为：

1. 保留 Galerkin 和 pressure-ROM 算子、历史 RHS、Re 与模态状态作为网络物理特征；
2. 不把不稳定的连续时间 ROM 直接加到离散状态更新；
3. 从稳定的零更新初始化学习 snapshot-to-snapshot 有限差分动力学；
4. 保留 H3 fluctuation/radial loss、H4 normal-form loss、等 Re 采样和 K1→K8 curriculum。

因此本实验属于“operator-feature HPRS-MoE”，不是原圆柱连续时间加性 ROM 主干的逐字复制。

## 4. 训练与 checkpoint 选择

- 随机种子：`1248`。
- 优化步数：8000；micro-batch：`46`。
- curriculum：K1 0–1199，K2 1200–2799，K4 2800–4799，K8 4800–7999。
- AMP/TF32：`True` / `True`。
- 训练耗时：`3.351` 小时，平均
  `39.785` steps/min。
- SwanLab：[在线运行记录](https://swanlab.cn/@panxy1019/CenteredSquare_Hopf_H4/runs/xmb9155b)，上传完成 25,811 条记录。
- 40 次 validation 均保持 finite 且零 divergence。
- 仅 step 7200 同时通过 numeric 与 Hopf hard gate；step 7800/8000 虽 score 更低，
  但 Hopf gate 未通过，因此没有用于最终测试。
- 冻结 checkpoint SHA-256：`07af06ec5506e5b6a12d9d00f557a4e48fa722be3c2a183df752f0883ad40beb`。

### Step 7200 validation K48

| Re | 速度总场误差 | 压力总场误差 | 径向 RMS 误差 | Growth-sign | 最大范数比 |
|---:|---:|---:|---:|---:|---:|
| 94.50 | 0.1594% | 0.4507% | 20.8255% | 51.8617% | 4.625 |
| 95.25 | 0.2191% | 0.4410% | 27.8843% | 56.1028% | 5.795 |
| 95.50 | 0.2152% | 0.4649% | 23.8208% | 66.2312% | 6.623 |
| 97.50 | 0.5181% | 0.6869% | 12.1224% | 58.5610% | 6.350 |
| 99.00 | 1.2044% | 1.2450% | 14.2262% | 64.1187% | 3.754 |
| 101.50 | 1.0772% | 1.0619% | 14.4971% | 68.4893% | 1.517 |

## 5. Held-out K48 结果

每个 Re 使用 64 个合法自主滚动窗口。

| Re | 速度物理场误差 | 压力物理场误差 | 径向 RMS | RMS 振幅 | P2P 振幅 | 频率误差 | Growth-sign | 最大范数比 | 发散 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 95.10 | 0.1446% | 0.3452% | 14.0717% | 9.7485% | 281.1316% | 1.8299% | 32.4126% | 6.351 | 0 |
| 95.30 | 0.0912% | 0.2282% | 8.7200% | 4.5164% | 642.2729% | 1.6434% | 45.9093% | 6.738 | 0 |
| 96.50 | 0.2586% | 0.2962% | 6.3984% | 1.9778% | 11.0518% | 1.7684% | 50.0000% | 8.428 | 0 |
| 100.50 | 0.7324% | 0.6749% | 7.4254% | 1.7144% | 5.4500% | 1.5477% | 71.2142% | 2.291 | 0 |
| 102.00 | 0.9669% | 0.8777% | 7.7882% | 3.0757% | 3.6533% | 1.4495% | 70.2239% | 1.387 | 0 |

## 6. 结果解释

### 数值稳定性与总场

K48 的所有 320 个最终窗口均为有限值，发散数为 0，最大范数比均低于 evaluator 的
10 倍阈值。五个 held-out Re 的速度和压力物理场误差均低于 1%，说明该稳定化迁移在
总场重构与长滚动数值稳定性方面有效。

### 频率与振幅

五个 Re 的频率误差为约 1.45%–1.83%，RMS 振幅误差为约 1.71%–9.75%，整体较好。
Re=100.5 和 102.0 的 P2P 误差分别约 5.45% 和 3.65%。Re=96.5 为 11.05%，略高于
10% 参考线。

Re=95.1 和 95.3 的 P2P 相对误差分别约 281% 和 642%。这两个点位于 Hopf onset
`Re_H≈95.312` 附近，真实窗口内 P2P 极小，导致相对指标分母敏感；对应 RMS 振幅误差
仍只有 9.75% 和 4.52%，径向 RMS 误差为 14.07% 和 8.72%。因此应将其解释为
近临界微小峰峰值未被精确保持，而不是整体吸引子幅值爆炸。

### 增长方向与 normal-form 可解释性

Growth-sign accuracy 仅为
`32.41%`–
`71.21%`，
临界附近尤其偏弱。训练后的 normal-form `mu` 在全部训练 Re 上仍为负，
`beta` 最大值仅 `4.316e-18`，平衡半径均为 0。
这说明最终预测能力主要来自 MoE 数据动力学，H4 normal-form head 未形成可物理解读的
Hopf 分岔参数化。该 checkpoint 不应宣称为“normal-form 参数识别成功”。

## 7. 最终判断

本次迁移在以下目标上通过：

- 数据划分和 train-only POD/ROM 隔离；
- 8000 步稳定训练与 SwanLab 完整记录；
- validation-only checkpoint 选择；
- 5/5 held-out K48 有限、零发散；
- 全部 held-out 速度/压力物理场误差低于 1%；
- 全部 held-out RMS 振幅误差低于 10%、频率误差低于 2%。

仍未完全解决：

- Hopf onset 附近的微小 P2P 振幅保持；
- 临界增长方向的逐步判别；
- H4 normal-form 参数的物理可解释性。

因此最准确的结论是：**该模型是稳定且总场精度良好的方柱 Hopf operator-feature
MoE-ROM，并能较好保持频率与 RMS 振幅；但不能宣称已经完整识别或严格保持临界
normal-form 动力学。**

## 8. 产物

- `selected_validation_step7200.pt`：冻结 checkpoint。
- `checkpoint_selection_manifest.json`：validation-only 选择证据。
- `heldout_metrics.json`：逐 Re、逐 horizon 测试指标。
- `FINAL_CENTEREDSQUARE_HOPF_H4_SUMMARY.json`：机器可读摘要。
- `FINAL_CENTEREDSQUARE_HOPF_H4_REPORT.md`：本报告。
- `artifact_inventory.json`：关键产物 SHA-256 清单。
