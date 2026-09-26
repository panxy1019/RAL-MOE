# ql-ROM-style Circular Periodic 交付包

先读 `qlrom_comparison_report.md`，再读 `source_and_adaptation.md`。本包是固定32维父POD空间内的独立受限实现，不是官方或完整物理空间ql-ROM复现。

## 使用结果

`table_rows_pod.tex` 与 `table_rows_raw_cfd.tex` 为两种独立参考口径的LaTeX组件/联合误差行及caption，不要混用。未修改任何论文或既有实验结果。

运行 `python build_report.py` 从保存的逐步CSV重建报告/表格并验证聚合；仅使用Python标准库，无需GPU、网络或旧checkpoint。会重建本包中的报告/表格与交付校验，不改变实验CSV或模型。

## 实验复现

`results/code/` 保存实际使用脚本，`qlrom_experiment_frozen.py` 对应协议最初SHA；`qlrom_experiment.py` 后续仅增加可选的物理初始状态与预构算子参数供计时，差分单列保存。脚本中的集群ROOT、父POD/投影算子目录和旧实验R是实际绝对路径；换机器需显式配置这些路径与新输出目录。

不要对已有冻结目录重新执行prepare/validate/test并覆盖结果。新实验需使用独立OUT、重建协议，再依次执行prepare、unit、validate、check_convergence、test、reaggregate、raw_reevaluate、timing_neural/timing_non_neural、final_diagnostics。计时五个方法串行执行，避免并发争用；调用前确认GPU和CPU负载。

依赖：原集群PyTorch 2.11.0+cu126，NumPy/SciPy以及新目录隔离安装的scikit-learn1.7.2、joblib、threadpoolctl；精确BLAS/CPU/GPU环境见results/timing.json。旧神经计时会读取原评价代码及原checkpoint，结果中保存其SHA256。

## 文件范围

包内有新局部模型、预测、逐步误差、冻结窗口与协议、审计证据、代码、报告和LaTeX。没有打包既有神经checkpoint、父POD基/物理算子、依赖环境或四个约80MB的原始场；它们路径/hash在manifest里。原始场已另存本地 `C:\Users\panxy1019\Documents\CHANNEL\qlrom_20260923\raw_reference\` 与集群新实验raw_reference目录，下载URL和hash见results/raw_reference_audit.json。

结果包不含论文HTML；来源链接见source_and_adaptation.md。本次没有为独立代码额外赋予开源许可证，也不把论文许可证当成代码许可证。
