# C：独立局部坐标系到共同物理空间

此版强调跨坐标系组装这个区别：局部基底不相容，因此局部动力学独立运行，不能直接混合系数；重构到同一物理网格之后才做 T2-C。

图中以 H-P 相邻对为例，S-H 和 Top-1 仍由左上角的规则涵盖。两条独立轨迹都进入各自的解码器。上方 pass-through 是 Top-1 的替代分支，不与 Top-2 同时输出两份结果。

为减少文字，图中未展开共享专家与组内 Top-2 的内层拓扑，也未展开速度 RK 和压力代数映射。ROM + sparse MoE 表示整个局部物理-数据模型，不表示压力接受常微分方程积分。小平面、曲线和网格仅为示意。

推荐图注：RAL-MoE-ROM selects one specialist or an admissible adjacent pair from a parameter-only regime preference. The illustrated H-P specialists encode the same physical history and evolve independently with ROM backbones and sparse MoE closures. Each prediction is reconstructed before Top-1 pass-through or Top-2 physical-space fusion. The assembly weight depends on the parameter, pair-normalized preference and a chart-independent history descriptor. Fusion does not feed back to local states; pressure remains algebraically reconstructed.
