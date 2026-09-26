# CenteredSquare E2 + T2-C 融合训练报告

- 状态：development 训练完成；heldout 物理场与指标未访问。
- 合同：trajectory/query-level E2；相邻边界独立；K24 固定窗口权重；只做物理输出凸融合；无反馈。
- E2：完成，validation 选择 step `3300`。
- S–H preflight：`72` 窗口，finite=100%，divergent=0。
- S–H gate：best step `5500`，validation joint mean `0.01063933`。
- H–P preflight：`48` 窗口，finite=100%，divergent=0。
- H–P gate：best step `3500`，validation joint mean `0.00790629`。

## Validation 对比

| 边界 | candidate 1 | candidate 2 | fixed 0.5 | E2 blend | T2-C | per-window oracle |
|---|---:|---:|---:|---:|---:|---:|
| SH | 0.03802627 | 0.01535906 | 0.02329357 | 0.01820808 | 0.01063933 | 0.01063355 |
| HP | 0.01052313 | 0.02286325 | 0.01332418 | 0.01508062 | 0.00790629 | 0.00776155 |

## SwanLab

- https://swanlab.cn/@panxy1019/CenteredSquare_Fusion_E2_T2C/runs/8rntelli
- https://swanlab.cn/@panxy1019/CenteredSquare_Fusion_E2_T2C/runs/b9gl1e6n
- https://swanlab.cn/@panxy1019/CenteredSquare_Fusion_E2_T2C/runs/ej4w150k

## 解释边界

- 本报告只冻结 development 训练与 validation 选择，不包含 final-test 结论。
- Steady 专家原报告中的 contraction 未合格状态保持不变，融合 gate 不掩盖该限制。
- H–P 是本数据集上的新认证执行，不继承旧数据集的 H–P 结论。
- 用户提供的既有报告已披露 heldout Re 与专家指标；本 strict run 自身未加载 heldout 物理场或计算 heldout 指标。
