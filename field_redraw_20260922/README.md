# 三案例物理场重绘包（2026-09-22）

## 精简六图版

按后续要求新增 `output_minimal/`，仅保留案例名称及 colorbar 数值，去掉所有说明文字、caption、坐标数字和刻度。`six_figures_minimal.zip` 仅含六张 400 dpi PNG。原始数组、指标和色标范围与完整标注版相同，旧版输出保持不动。

重绘命令：`python scripts/plot_field_comparison.py --minimal`。

## 先看哪里

- `output/preview_iclr.pdf`：使用当前论文原版 ICLR 2027 样式、139.7 mm 正文宽度编译的六页选图预览。
- `output/field_overview_three_cases_pressure.pdf` / `velocity.pdf`：方案 A，三案例 × 参考/预测/误差。
- `output/fusion_comparison_hp_square_pinball_pressure.pdf` / `velocity.pdf`：方案 B，H–P，Square 与 Pinball 分块排列。
- `output/fusion_comparison_sh_square_pinball_pressure.pdf` / `velocity.pdf`：另一重叠区 S–H 的补充版。
- 相同文件名的 PNG 为 400 dpi 预览；PDF 内文字可选取，科学场栅格化。

建议优先看三案例压力总览和 H–P 压力融合对比；速度版与 S–H 版作为备选/附录，不建议把六张全部放入正文。未改动原论文、旧 PNG、训练检查点，也没有上传 Overleaf。

## 真实数据和模型身份

| 面板 | Re | 窗口 start | k/K | t0 → t | 预测身份 |
|---|---:|---:|---:|---|---|
| Circular P | 70.3146 | 976 | 48/48 | 1384.4960 → 1983.1969 | 原初始化、seed 1248 的 Periodic sparse-MoE 局部模型 |
| Square H–P | 100.5000 | 506 | 24/24 | 8 → 104 | 冻结 T2-C，Periodic/Hopf sparse-MoE 候选 |
| Pinball H–P | 22.2500 | 2 | 56/56 | 36.75 → 50.75 | 冻结 T2-C，Periodic/Hopf Deep-FNN-H3 候选 |
| Square S–H | 95.1000 | 7814 | 24/24 | 8 → 104 | 冻结 T2-C，Steady/Hopf sparse-MoE 候选 |
| Pinball S–H | 17.0000 | 514 | 56/56 | 0.75 → 14.75 | 冻结 T2-C，Steady/Hopf Deep-FNN-H3 候选 |

选样规则先固定为：最低 held-out Re、稳定 start 顺序中的首个有效窗口、该窗口完整 rollout 终点。不按融合收益选帧。Circular 与 Square 直接解码旧预测缓存；Pinball 恢复两个固定窗口的 K56 推理，所有候选/E2/gate 哈希与历史 final-test seal 一致。没有训练。

全部参考均为 **POD reconstruction**，不是 raw CFD/DNS。总览混合局部预测和融合，不能作为统一架构消融或单网络跨几何泛化证据。

## 数值与尺度

- 速度场为 `hypot(u,v)`，速度误差为 `hypot(u_pred-u_ref,v_pred-v_ref)`，不是速度模长之差。
- 压力为运动学压力；每个场各自按原积分权重去全域均值，不拟合参考偏置。
- **保留原始数据尺度**，不另做 `u/U`、`p/U²` 变换。Circular 资产报告明确 `nu=0.001` 和 raw physical POD；不能把三例统称为同一无量纲幅值。每例独立色标，只能行内比较。
- 同行各预测场共用全域联合范围；压力范围以零对称；误差从零到同行最大值。无 percentile clipping、平滑、相位对齐或填充固体。
- ROI 按几何固定，所有方法完全相同。指标使用完整计算域，不是裁剪 ROI。
- `data/snapshot_metrics.csv` 是逐快照指标，单位为百分比；`panel_manifest.json` 中 Eu/Ep 为比例。联合误差明确为 `Eu+Ep`。不可当作论文全窗口平均。
- 图内不放误差百分比，避免过密；精确指标保留在 CSV。示例：Pinball H–P 该终点 T2-C 的 Eu=0.0208%、Ep=0.0924%，单独 Periodic 为 0.0117%、0.0958%；不能声称该帧融合在所有量上优于每个候选。

## 后续改图：不需要服务器

`data/` 已包含全部最终物理数组、原 VTK 拓扑导出的三角形、原始点/单元映射和权重。改配色、尺寸、ROI、字体时只运行渲染程序，不需要 PyTorch、VTK 或 GPU。

```powershell
python -m pip install numpy matplotlib
python scripts/plot_field_comparison.py --config config/figure_config.json
```

本机也可使用：

```powershell
& 'C:/Users/panxy1019/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/plot_field_comparison.py
```

脚本会优先使用本目录 `.runtime/` 内已安装的绘图库。本机实测 Python 3.12、NumPy 2.5.3、Matplotlib 3.11.2。迁移时安装依赖即可，不必复制 `.runtime/`。

只改 `config/figure_config.json` 中版式、配色和 ROI；`scripts/plot_field_comparison.py` 中 `overview()`、`fusion()` 分别控制两种版式，`cbar()` 控制科学计数法刻度。不要为增强某个方法的视觉优势单独修改其色标。

每次渲染会更新本重绘包的 PDF/PNG、`panel_manifest.json`、`snapshot_metrics.csv` 和 `render_qa.json`，不会修改包外文件。

## 需要更换窗口/模型时

这不再是单纯改版式：必须重新生成物理数组并更新时间、来源、哈希和误差。

服务器上的本次新数据目录：

`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/experiments/field_redraw_20260922/data`

运行环境：

`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/.runtime/pt_env/bin/python`

代码职责：

1. `scripts/extract_snapshots.py --case circular|square|pinball --output <new-data-dir>`：读取原缓存，或恢复固定 Pinball 窗口；导出 `[u,v,p]` 和来源 JSON。
2. `scripts/build_geometry.py`：读取原始 VTK，导出真实网格连接并验证 POD/网格排序；脚本中 `R/O` 为该服务器路径，迁移时修改。
3. `scripts/enrich_provenance.py`：补充原始单位文档、seed 和几何哈希证据；迁移时修改 `R/D`。
4. `scripts/download_data.py`：下载本次新结果；凭据仅从临时环境变量 `FIELD_SSH_PASSWORD` 读取，不写入文件。新机器/端口需要修改辅助脚本的连接参数。
5. `scripts/plot_field_comparison.py`：完全本地重绘并再次核对数值指标。

`extract_snapshots.py` 依赖服务器上各案例原评价代码；不是独立训练包。所有实际检查点、POD 基、缓存与网格路径/哈希见 `data/panel_manifest.json`，没有把大检查点/POD 基复制到本交付包。

## LaTeX

`captions.tex` 和 `figure_snippet.tex` 提供可插入片段；当前论文未被替换。总览应置于跨案例总览或独立场图附录，不能放在仅名为 Square 的小节。

`preview_iclr.tex` 使用从原论文复制的 `iclr2027_conference.sty`。本机编译：

```powershell
& '../RAL_MOE_ROM_ICLR2027/build/tools/tectonic-0.17.0/tectonic.exe' --keep-logs --outdir output preview_iclr.tex
```

更改图片后需重新编译并检查预览，不要把论文中的图缩小到低于设计字号。

尚未纳入的材料和限制见 `data_gaps.md`，验证记录见 `qa_report.md`。
