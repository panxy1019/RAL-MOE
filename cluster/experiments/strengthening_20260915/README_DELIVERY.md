# 本次交付入口

1. 先读 `FINAL_REPORT_CN.md`：最终统计、负结果、协议边界与八项清单。
2. 论文插入内容：`paper_ready_inserts.tex`，保留其中的 reference、pressure gauge、auxiliary feature 和 runtime caveat。
3. 投稿图片使用 PDF；PNG 用于快速预览。
   - `constant_alpha_v1/figures/router_fusion_weights.pdf`
   - `constant_alpha_v1/figures/fusion_error_growth.pdf`
   - `aligned_rollout_figures/four_method_error_growth.pdf`
   - `aligned_rollout_figures/per_Re_physical_time_curves.pdf`
4. 数值与复核入口：`FINAL_PERIODIC_K48.json`、`FINAL_NATIVE_RUNTIME.json`、`FINAL_CHECKPOINT_AUDIT.json`、`post_training_status.json`。

## 避免混用口径

- `constant_alpha_v1` 是 centered-square overlap K24。
- `final_periodic_*` 是 Circular Periodic K48；其中 Global 文件的原生精度指标不是跨图表最终精度，最终应使用 `global_common_truth_v1`。
- `aligned_rollout_figures/aggregate_curves.csv` 是最终四方法可比较曲线。
- `EXPERIMENT_REPORT.md` 是历史阶段记录；最终状态以 `FINAL_REPORT_CN.md` 为准。
- 本包不是可直接替换论文的完整 Overleaf 工程，也未包含原始数据/大型 checkpoint。

## 复现脚本

实验脚本依赖原集群项目树，其固定根目录记录在 `dense_periodic.py` 中。不要在原输出目录盲目重跑；先指定新的 `--output` 路径以保留结果。

- `constant_alpha.py`：固定权重解析拟合和敏感性比较。
- `dense_periodic.py`：原生 Periodic trainer 的 Dense 替换。
- `evaluate_periodic_dense.py`：统一窗口评估和原生计时。
- `global_common_truth.py`：跨 POD 基、压力去均值后的共同参考评估。
- `verify_global_replay.py`：独立重放核对。
- `final_audit.py`：checkpoint 身份和参数统计。
- `test_scheduler.py`：调度退出状态测试；需在 Linux 原项目环境运行。
- `plot_fusion.py`、`plot_aligned.py`：本地绘图，需要 NumPy 和 Matplotlib。

`launch_*` 脚本是运行管理历史，不是建议再次执行的交付入口。所有本批次计算任务已结束。

`PACKAGE_MANIFEST.json` 记录包内各文件的 SHA256。包内不含私人登录凭据。
