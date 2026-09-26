# HopfExpanded34 H4 最终实验报告

## 1. 训练与冻结

- 实验：`HopfExpanded34_H4_NormalFormRadial_r32`
- 正式训练：return code 0，完成 8000 optimizer steps。
- checkpoint 选择只使用 validation Re `46.7, 56.543246`；三个 test Re 在冻结前未读取。
- 冻结 checkpoint：step 7200，validation score `0.878767498`，hard gate 通过。
- checkpoint SHA-256：`02148741ed8bc9fbec88709f69b10492e263514b2edcee76652485b399962235`。
- 训练耗时：1.979 小时，平均 67.36 steps/min。

## 2. Test 终评协议

- Test Re：`47.081355, 49.022357, 51.786450`。
- 这些 Re 曾用于前序方法诊断，因此本报告称为最终评价集，不称为严格一次性 blind test。
- 使用冻结 checkpoint、state-autonomous rollout 和新 29-Re train-only POD/ROM/统计量。
- 评价 horizon：K1/K2/K4/K8/K16/K24/K48；表内相对误差均乘以 100，以百分数表示。

## 3. K48 主要结果

| Re | 速度物理场误差 | 压力物理场误差 | RMS 振幅误差 | P2P 振幅误差 | Growth-sign | 频率误差 | Orbit distance | False growth | 严格保持 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 47.081355 | 0.0024% | 0.0284% | 0.5769% | 13.0542% | 76.8730% | 37.2568% | 0.0587 | 否 | 未通过 |
| 49.022357 | 0.0165% | 0.2785% | 38.9436% | 378.1110% | 67.6667% | 1.2036% | 0.2421 | 否 | 未通过 |
| 51.786450 | 0.0052% | 0.0394% | 21.0241% | 26.8319% | 94.1272% | 1.2623% | 0.2922 | 否 | 未通过 |

三组 test Re 均为 finite_fraction=100%、divergent windows=0。严格 attractor-preserved 为 `0/3`。

严格判据失败项：

- Re=47.081355：P2P amplitude、frequency
- Re=49.022357：RMS amplitude、P2P amplitude、orbit distance
- Re=51.786450：RMS amplitude、P2P amplitude、orbit distance

## 4. 结果分析

模型的数值稳定性和总物理场重构很好：三个 Re 的 K48 速度误差均低于 0.017%，压力误差均低于 0.279%，且没有发散。主要问题仍是临界二维波动动力学，而不是压力闭合或总场。

- Re=47.081355 已消除 false growth：真实平均对数增长 `3.00864e-05`，预测 `-0.00206555`；RMS 振幅误差仅 0.5769%。其 P2P 误差 13.0542% 略超 10% 门限。真实频率接近零，因此 37.2568% 的相对频率误差不宜作为有效旋转频率结论。
- Re=49.022357 的频率误差仅 1.2036%，但 RMS/P2P 振幅误差分别为 38.9436%/378.1110%，并伴随 orbit distance `0.2421`，说明轨道尺度和形状没有保持。
- Re=51.786450 的增长方向准确率为 94.1272%、频率误差 1.2623%，但 RMS/P2P 振幅误差和 orbit distance 仍超门限，表现为稳定但幅值偏弱的吸引子。

## 5. 结论

本次新数据 H4 达成了 `3/3` K48 finite、`3/3` 零 divergence，并保持了很低的总场速度/压力误差；同时消除了 Re=47.081355 的 false growth。严格 Hopf attractor preserved 未通过，核心短板是 Re=49.022357 和 51.786450 的振幅/轨道尺度，以及 Re=47.081355 的轻度 P2P 偏差。当前结果不支持把该 checkpoint 宣称为三组 test Re 的完整 Hopf 吸引子保持模型。
