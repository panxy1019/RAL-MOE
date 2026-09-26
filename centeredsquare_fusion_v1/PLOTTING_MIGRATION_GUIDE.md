# CenteredSquare SH/HP 融合绘图程序与迁移指南

更新日期：2026-09-20

## 1. 目标与当前状态

当前绘图链路只生成 SH 和 HP 两个边界的论文图。每个边界生成两张图：

1. 物理场图：`Truth (CFD)`、两个子专家、`Proposed RA-LMoE-ROM`，两列分别为速度模长和压力。
2. 误差图：两个子专家与融合结果，两列分别为逐点速度绝对误差和逐点压力绝对误差。

当前冻结图已经生成并完成视觉检查。后续迁移时应以本目录中的当前适配程序为主，以 `reference_plotting/` 中的圆柱绕流程序为版式参考，不要反过来修改旧参考程序。

## 2. 权威程序与职责

| 文件 | 身份 | 作用 |
|---|---|---|
| `plot_boundary_fusion_fields.py` | 当前主程序 | 读取 cache、POD 与 VTK，选帧、重构物理场、融合、计算误差并绘制两张图。 |
| `export_boundary_figure_weights.py` | 当前辅助程序 | 从冻结 E2 router 和 T2-C gate 导出逐窗口融合权重 CSV。 |
| `config_sh_boundary_figure.json` | 当前 SH 配置 | Steady/Hopf 基底、cache、权重表、标签、顺序和秩。 |
| `config_hp_boundary_figure.json` | 当前 HP 配置 | Periodic/Hopf 基底、cache、权重表、标签、顺序和秩。 |
| `common.py` | 共享运行依赖 | `ConvexGate`、router 概率、SHA256 等公共逻辑。 |
| `reference_plotting/plot_field_figures.py` | 只读参考 | 原圆柱绕流 `v10b` 版式程序。不可作为当前方柱数据的直接入口。 |
| `reference_plotting/run_all_figures.py` | 只读参考 | 原四类图片批处理入口。当前 SH/HP 绘图不使用它。 |

精确文件哈希见 [`PLOTTING_FILE_MANIFEST.json`](PLOTTING_FILE_MANIFEST.json)。

## 3. 本地目录结构

```text
centeredsquare_fusion_v1/
├── plot_boundary_fusion_fields.py          # 当前绘图主程序
├── export_boundary_figure_weights.py       # 冻结权重导出
├── config_sh_boundary_figure.json          # SH 配置
├── config_hp_boundary_figure.json          # HP 配置
├── common.py                               # gate/router 公共代码
├── PLOTTING_MIGRATION_GUIDE.md             # 本文档
├── PLOTTING_FILE_MANIFEST.json             # 当前关键文件 SHA256
├── reference_plotting/
│   ├── plot_field_figures.py               # 原圆柱参考程序
│   ├── run_all_figures.py                  # 原批处理程序
│   ├── config_sh_fusion.json               # 原 SH 参考配置
│   ├── SH_reference_physical.png           # 原版式物理场参考图
│   ├── SH_reference_errors.png             # 原版式误差参考图
│   ├── SH_reference_manifest.json          # 原参考图审计清单
│   └── centeredSquare_CN09_graded_Re100_internal_final_reference.vtk
└── results/E2_T2C_K24_20260730_STRICT_V3/
    ├── heldout_evaluation_20260730_V1/
    │   ├── cache_sh/sh_heldout_cache.npz
    │   └── cache_hp/hp_heldout_cache.npz
    └── paper_boundary_figures_20260730_V1/
        ├── inputs/
        │   ├── SH_T2C_weights.csv
        │   ├── SH_T2C_weights.manifest.json
        │   ├── HP_T2C_weights.csv
        │   ├── HP_T2C_weights.manifest.json
        │   └── centeredSquare_CN09_graded_Re100_internal_final_reference.vtk
        ├── SH/
        │   ├── SH_2.png                    # 当前本地物理场图
        │   ├── SH_2_1.png                  # 当前本地误差图
        │   ├── candidate_scan.json
        │   └── FIELD_FIGURE_MANIFEST.json
        └── HP/
            ├── HP_2.png                    # 当前本地物理场图
            ├── HP_2_1.png                  # 当前本地误差图
            ├── candidate_scan.json
            └── FIELD_FIGURE_MANIFEST.json
```

### 输出重命名说明

当前四张本地 PNG 被重命名过，但文件内容与原始 manifest 完全一致：

| 当前本地文件 | 原始生成名 |
|---|---|
| `SH/SH_2.png` | `CenteredSquare_SH_boundary_T2C_physical_fields_ultra_compact_wide.png` |
| `SH/SH_2_1.png` | `CenteredSquare_SH_boundary_T2C_pointwise_errors_ultra_compact_wide.png` |
| `HP/HP_2.png` | `CenteredSquare_HP_boundary_T2C_physical_fields_ultra_compact_wide.png` |
| `HP/HP_2_1.png` | `CenteredSquare_HP_boundary_T2C_pointwise_errors_ultra_compact_wide.png` |

因此边界目录中的旧 `FIELD_FIGURE_MANIFEST.json` 仍记录原始生成名；当前本地别名与实际哈希由顶层 `PLOTTING_FILE_MANIFEST.json` 记录。迁移新数据集时建议保留程序自动生成的长文件名，避免再次出现名称与 manifest 不一致。

## 4. 历史远端位置与当前可用性

训练服务器历史项目根目录：

```text
/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
```

历史代码位置：

```text
<ROOT>/centeredsquare_fusion_v1/
```

历史冻结运行位置：

```text
<ROOT>/centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3/
```

历史绘图输出位置：

```text
<RUN>/paper_boundary_figures_20260730_V1/
```

这些训练服务器路径在 2026-07-30 完成过验证。2026-09-20 本轮通过 SSH 密钥重新连接 `root@10.10.164.243:20381`，主机名为 `7b9b20463be9`，确认训练根目录及冻结绘图输入仍存在。此前登录失败不能推断服务器或数据失效；本文不保存密码。

reference VTK 的历史虚拟机来源为：

```text
/home/ray/Desktop/centeredSquare/three_regime_overlap_v1/subsets/steady/reference_vtk/
centeredSquare_CN09_graded_Re100_internal_final_reference.vtk
```

2026-09-20 该历史路径已不存在。本地 `reference_plotting/` 与冻结结果 `inputs/` 中均保存了 VTK 副本；后续迁移请使用本地副本。文档不保存任何 SSH 密码。

## 5. 科学数据合同

### 5.1 当前两个边界

| 边界 | candidate 1 | candidate 2 | truth 基底 | 系数秩 |
|---|---|---|---|---|
| SH | Steady | Hopf | Steady | Steady `(r_u,r_p)=(5,4)`；Hopf `(11,11)` |
| HP | Periodic | Hopf | Periodic | Periodic `(28,26)`；Hopf `(11,11)` |

融合权重 `alpha` 始终表示 candidate 1 的权重：

```text
x_T2C = alpha * x_candidate_1 + (1 - alpha) * x_candidate_2
```

融合发生在重构后的物理场，不在不同 POD 坐标中直接混合系数，也不反馈给子专家。

### 5.2 cache 至少需要的数组

```text
re, split, starts, timestamps,
truth_a, truth_b,
candidate_1_a, candidate_1_b,
candidate_2_a, candidate_2_b,
quad_u, quad_p,
pair_indices
```

当前 heldout cache 都是 K24。`quad_u` 和 `quad_p` 的最后一维顺序为：

```text
[candidate_1_error_squared,
 candidate_2_error_squared,
 cross_term,
 truth_energy]
```

### 5.3 网格和自由度

- 当前 VTK 为 OpenFOAM cell-centered 网格，共 9400 个 cells。
- POD 和 VTK 必须保持相同 cell 顺序；不能通过翻转图片或调整坐标轴掩盖顺序错误。
- 当前速度自由度严格为交错布局：`[u1,v1,u2,v2,...]`。
- 两份配置必须保留 `"velocity_layout": "interleaved"`。
- 压力在重构后使用相同的面积加权零均值 gauge。
- 绘图使用 VTK 中面切片的真实单元拓扑；不要改回 cell-center Delaunay，否则会产生跨分块棋盘/交叉伪影。

## 6. 选帧合同

默认策略优先检查 K24 末帧；若末帧不满足预注册资格，则向前回溯。排序依次考虑：

1. 最低 `eligibility_tier`；
2. 更大的融合 joint advantage；
3. 更强的真值结构分数；
4. 更小的窗口编号作为确定性 tie-break。

当前选择结果：

| 边界 | Re | window | time index | physical time | tier | 是否末帧 |
|---|---:|---:|---:|---:|---:|---|
| SH | 95.1 | 0 | 1 | 16.0 | 0 | 否，按规则回溯 |
| HP | 100.5 | 5 | 23 | 384.0 | 0 | 是 |

完整候选排序保存在相应 `candidate_scan.json` 中。若论文需要固定帧，必须显式传入 `--re` 和 `--time-index`，并在 manifest 中保留记录。

## 7. 复现命令

以下命令中的路径用占位符表示，迁移后必须替换为新位置。

### 7.1 导出冻结融合权重

该步骤需要 `numpy` 和 `torch`，并需要冻结的 router/gate checkpoints：

```bash
python export_boundary_figure_weights.py \
  --run-root <RUN_ROOT> \
  --cache <SH_HELDOUT_CACHE> \
  --boundary SH \
  --output-csv <OUTPUT>/inputs/SH_T2C_weights.csv

python export_boundary_figure_weights.py \
  --run-root <RUN_ROOT> \
  --cache <HP_HELDOUT_CACHE> \
  --boundary HP \
  --output-csv <OUTPUT>/inputs/HP_T2C_weights.csv
```

### 7.2 生成 SH 图

绘图环境需要 `numpy`、`matplotlib`、`pyvista` 和 `cmocean`：

```bash
python plot_boundary_fusion_fields.py \
  --mode boundary_fusion \
  --config config_sh_boundary_figure.json \
  --template-vtk <REFERENCE_VTK> \
  --output-dir <OUTPUT>/SH \
  --candidate-scan \
  --split heldout \
  --error-cmap haline_r \
  --png-only \
  --dpi 400 \
  --xlim 2 17 \
  --ylim 0 4
```

### 7.3 生成 HP 图

```bash
python plot_boundary_fusion_fields.py \
  --mode boundary_fusion \
  --config config_hp_boundary_figure.json \
  --template-vtk <REFERENCE_VTK> \
  --output-dir <OUTPUT>/HP \
  --candidate-scan \
  --split heldout \
  --error-cmap haline_r \
  --png-only \
  --dpi 400 \
  --xlim 2 17 \
  --ylim 0 4
```

旧训练服务器中 `pt_env` 可运行权重导出，但当时没有 Matplotlib；绘图曾使用系统 `/usr/bin/python3`。迁移到新机器时建议建立独立绘图环境，不要向冻结训练环境临时安装包。

## 8. 修改图片时改哪里

| 需求 | 修改位置 |
|---|---|
| 改方法名、换行方式 | 两个 JSON 的 `method_labels`。 |
| 改方法顺序 | 两个 JSON 的 `plot_order`，同时检查 `comparator_methods`。 |
| 改输出文件前缀 | JSON 的 `figure_stem`。 |
| 改裁剪范围 | CLI `--xlim`、`--ylim`；当前默认在主程序第 98–99 行附近。 |
| 改 DPI | CLI `--dpi`。 |
| 固定 Re/帧 | CLI `--re`、`--time-index`、`--split heldout`。 |
| 改物理场配色 | `plot_figures()` 中 `cividis` 与 `RdBu_r`。 |
| 改误差配色 | CLI `--error-cmap`；论文当前使用 `haline_r`。 |
| 改画布尺寸 | `plot_figures()` 中物理场 `figsize=(14.6, 7.0)`，误差图高度 `1.65*N_method+0.75`。 |
| 改 panel 纵横比 | `draw_panel(..., box_aspect=0.32)`。 |
| 改边距/行列间距 | `fig.subplots_adjust(...)`。 |
| 改色标范围 | `robust_limits()`；物理场由 truth 的 0.5%/99.5% 分位控制，压力关于 0 对称。 |
| 改 POD 文件、键名、秩 | 对应 JSON 的 `bases`。 |
| 改速度布局 | JSON 的 `velocity_layout`；只有确认新数据合同后才能改。 |

不要为了图片更“好看”修改误差定义、pressure gauge、POD 重构、candidate 顺序或选帧排序。

## 9. 新数据集迁移步骤

1. 复制两个当前 JSON 为新配置，不覆盖现有配置。
2. 由冻结 evaluator 生成满足第 5.2 节合同的新 cache。
3. 核对 candidate 1/2 与 gate `pair_indices` 的顺序。
4. 在新配置中更新 cache、权重 CSV、POD、键名、秩和 truth 基底。
5. 明确新速度自由度是 `blocked` 还是 `interleaved`。
6. 准备与 POD 自由度严格同序的 VTK；核对 cell 数、cell-center 和障碍位置。
7. 导出新权重，并保存权重 manifest。
8. 先运行 candidate scan，再生成图片。
9. 逐张检查入口、障碍、尾迹方向、速度分量、压力高低压区和网格伪影。
10. 核对 PNG 与 `FIELD_FIGURE_MANIFEST.json` 的 SHA256，并更新顶层文件清单。

遇到维度不一致、权重缺失/越界、NaN/Inf、VTK/POD 顺序不明、pressure gauge 不同或 `haline_r` fallback 时必须停止，不应继续输出论文图。

## 10. 最小验证

```powershell
python -m py_compile `
  .\plot_boundary_fusion_fields.py `
  .\export_boundary_figure_weights.py

Get-Content -Raw .\config_sh_boundary_figure.json | ConvertFrom-Json | Out-Null
Get-Content -Raw .\config_hp_boundary_figure.json | ConvertFrom-Json | Out-Null
```

生成图片后还必须进行视觉检查；仅脚本成功退出不能证明网格和自由度映射正确。

## 11. 下一步

后续若只修改版式，先复制配置和输出目录，再按第 8 节修改。若迁移数据集，先完成第 5 和第 9 节的数据合同检查，再接触绘图参数。
