# 流场绘图程序、科学绘图原则与新数据集迁移指南

## 1. 文档目的

本文档说明当前圆柱绕流论文流场图的生成程序、输入数据合同、选帧规则、物理场重构与误差定义、配色和版式原则，以及将绘图流程迁移到新数据集时需要修改和核验的内容。

当前绘图体系由以下文件组成：

```text
paper_experiments/visualization/
├── plot_field_figures.py
├── run_all_figures.py
├── config_steady_specialist.json
├── config_hopf_specialist.json
├── config_periodic_specialist.json
├── config_sh_fusion.json
└── PER_WINDOW_T2C_WEIGHTS.csv
```

核心原则是：

> VTK 文件只提供公共网格拓扑；真实场和预测场始终由冻结 rollout 系数、原生 POD 基和均值场重构。绘图程序不训练模型、不修改 checkpoint，也不重新定义科学评价合同。

---

## 2. 当前输出范围

`run_all_figures.py` 依次处理四种情况：

1. Steady specialist；
2. Hopf specialist；
3. Periodic specialist；
4. S–H boundary fusion。

每种情况输出两张 PNG：

1. 物理场对比图；
2. 逐点绝对误差图。

因此一次完整运行共输出 8 张 PNG。当前 ultra-compact 版本文件名为：

```text
<figure_stem>_physical_fields_ultra_compact_wide.png
<figure_stem>_pointwise_errors_ultra_compact_wide.png
```

每个情况的输出目录还包含：

```text
candidate_scan.json
FIELD_FIGURE_MANIFEST.json
```

可选使用 `--save-vtk` 额外保存选中帧的 `.vtp` 文件。

---

## 3. 程序职责划分

### 3.1 `plot_field_figures.py`

单个配置对应一次绘图任务。它负责：

1. 读取 JSON 配置；
2. 加载 rollout NPZ；
3. 加载速度和压力 POD 基、均值场及面积权重；
4. 读取作为拓扑模板的 VTK；
5. 对候选 Reynolds 数、窗口和时间帧进行扫描；
6. 根据预注册规则选择展示帧；
7. 从 POD 系数重构速度和压力物理场；
8. 对 S–H 情况在物理场层面执行冻结凸融合；
9. 计算或读取与物理场误差等价的评价量；
10. 绘制物理场图和逐点误差图；
11. 保存候选扫描、资产 SHA256 和输出清单。

### 3.2 `run_all_figures.py`

这是四种情况的批处理入口。它维护：

```python
CONFIGS = {
    "steady": ...,
    "hopf": ...,
    "periodic": ...,
    "sh_fusion": ...,
}
```

并为每个情况建立独立输出子目录。当前 `VTK_TEMPLATE` 在该文件中是硬编码路径；迁移数据集时必须更新它，或改为新网格的绝对路径。

---

## 4. 输入资产合同

### 4.1 Rollout bundle

specialist 配置默认要求 NPZ 中至少存在：

```text
re
split
times
truth_a
truth_b
<method>_a
<method>_b
```

推荐形状为：

```text
re:          [N_window]
split:       [N_window]
times:       [N_window, N_time]
truth_a:     [N_window, K, r_u]
truth_b:     [N_window, K, r_p]
method_a:    [N_window, K, r_u]
method_b:    [N_window, K, r_p]
```

其中：

- `a` 是速度 POD 系数；
- `b` 是压力 POD 系数；
- `K` 是实际保存的 rollout 查询帧数；
- `split` 应明确记录 `train`、`validation` 或 `heldout`；
- CLI 中的 `--split test` 会映射到 bundle 内的 `heldout`。

绘图程序不会从 checkpoint 执行 rollout。新数据必须先由冻结 evaluator 生成符合上述合同的 bundle。

### 4.2 POD 资产

速度 POD NPZ 默认键名：

```text
phi_uv
mean_uv_regime
point_areas
```

压力 POD NPZ 默认键名：

```text
phi_p
mean_p_regime
```

如新资产使用不同键名，可在配置中指定：

```json
{
  "phi_u_key": "new_phi_u_key",
  "mean_u_key": "new_mean_u_key",
  "area_key": "new_area_key",
  "phi_p_key": "new_phi_p_key",
  "mean_p_key": "new_mean_p_key"
}
```

当前重构约定为：

```text
phi_uv.shape = [r_u, 2N]
phi_p.shape  = [r_p, N]
mean_uv.shape = [2N]
mean_p.shape  = [N]
point_areas.shape = [N]
```

速度自由度顺序必须严格为：

```text
[u_1, ..., u_N, v_1, ..., v_N]
```

物理场重构为：

```math
\boldsymbol{u}
=
\overline{\boldsymbol{u}} + \boldsymbol{a}^{\mathsf T}\Phi_u,
```

```math
p
=
\overline{p} + \boldsymbol{b}^{\mathsf T}\Phi_p.
```

如果新数据采用节点交错、分量交错或其他自由度排列，必须先转换为当前合同，不能只修改图形显示顺序。

### 4.3 VTK 网格模板

VTK 只提供：

- 点坐标；
- 三角形拓扑；
- POD 自由度与可视化网格之间的点编号映射。

程序优先读取 `vtkOriginalPointIds`。若该字段不存在，则仅在 surface 提取没有改变点数、点坐标和点顺序时允许直接使用当前编号。

必须满足：

```text
max(vtkOriginalPointIds) < N_POD
```

新 VTK 必须与 POD 基使用相同公共网格和相同节点编号。几何外观相同并不足以证明自由度顺序一致。

### 4.4 S–H fusion 额外资产

S–H 配置还需要：

```text
s_a, s_b
h_a, h_b
quad_u, quad_p
start 或其他窗口键
PER_WINDOW_T2C_WEIGHTS.csv
```

权重表通过：

```text
(round(Re, 6), window_key)
```

定位轨迹级 `alpha_S`。最终权重为：

```math
\alpha_H = 1-\alpha_S.
```

S–H 预测必须先分别使用 Steady 和 Hopf 原生 POD 资产重构到公共物理网格，再进行：

```math
\widehat{x}_{T2C}
=
\alpha_S\widehat{x}_S
+
(1-\alpha_S)\widehat{x}_H.
```

禁止直接融合 Steady 和 Hopf 的 POD 系数。

`quad_u` 和 `quad_p` 用于在候选扫描阶段快速、精确计算凸组合的相对场误差。其最后一维当前应表达：

```text
[S误差平方, H误差平方, S/H误差交叉项, 真值场能量]
```

迁移时不得改变元素顺序而不修改 evaluator 和绘图代码。

---

## 5. 压力 gauge 合同

当前程序使用面积加权零均值压力 gauge：

```math
p
\leftarrow
p-\frac{\sum_i A_i p_i}{\sum_i A_i}.
```

该处理发生在：

1. 加载压力均值和压力 POD 基时；
2. 每次重构完整压力场之后。

因此新数据必须满足：

- 使用同一组 `point_areas`；
- truth 和所有 prediction 使用相同 gauge；
- 不能一部分数据使用固定参考点 gauge，另一部分使用面积零均值 gauge；
- 不得仅为改善图片外观而单独平移某个方法的压力。

如果新 evaluator 已经应用相同面积零均值 gauge，重复执行不会改变结果；如果使用了其他 gauge，应在进入绘图 bundle 前统一。

---

## 6. 误差定义

### 6.1 图中标注的整体相对误差

速度相对物理场误差：

```math
\epsilon_u
=
\sqrt{
\frac{
\sum_i A_i\left[
(\widehat{u}_i-u_i)^2+
(\widehat{v}_i-v_i)^2
\right]
}{
\sum_i A_i(u_i^2+v_i^2)
}
}.
```

压力相对物理场误差：

```math
\epsilon_p
=
\sqrt{
\frac{
\sum_i A_i(\widehat{p}_i-p_i)^2
}{
\sum_i A_i p_i^2
}
}.
```

联合指标仅用于候选帧排序：

```math
\epsilon_{\mathrm{joint}}
=
\frac{1}{2}(\epsilon_u+\epsilon_p).
```

当前图片中的 `epsilon_u` 和 `epsilon_p` 是无量纲比例，使用统一 LaTeX 科学计数法显示，**没有乘以 100**。论文表格如使用百分比，需要在表格生成阶段明确乘以 100，不能混淆两种表示。

### 6.2 逐点误差图

速度逐点误差：

```math
|\Delta\boldsymbol{u}_i|
=
\sqrt{
(\widehat{u}_i-u_i)^2+
(\widehat{v}_i-v_i)^2
}.
```

压力逐点误差：

```math
|\Delta p_i|
=
|\widehat{p}_i-p_i|.
```

逐点误差图显示绝对误差，不对每个节点除以局部真值。这样可以避免真值接近零时产生虚假的局部相对误差奇点。

### 6.3 系数域快速评价

当 truth 和所有方法使用同一 POD 基时，候选扫描使用由 POD Gram 矩阵和均值交叉项构成的二次型计算场误差。该计算与完整物理场面积加权相对误差等价，但避免为每个候选帧反复重构全部网格。

选定最终展示帧后，程序仍会显式重构物理场并绘图。

---

## 7. 候选帧选择原则

默认策略是优先最后一个 rollout 帧，同时允许在没有合格末帧时回溯候选帧。

specialist 模式中：

```text
目标方法：Proposed Specialist MoE
比较方法：FNN-MoE、DataOnly-MoE
```

S–H fusion 模式中：

```text
目标方法：Proposed RA-LMoE-ROM
比较方法：Steady Specialist、Hopf Specialist
```

资格等级为：

| Tier | 条件 |
|---|---|
| 0 | 目标方法的速度误差和压力误差分别严格优于两个比较方法 |
| 1 | 目标方法的联合误差和速度误差严格优于两个比较方法 |
| 2 | 目标方法的联合误差严格优于两个比较方法 |
| 不合格 | 不满足上述任何条件 |

最终排序依次考虑：

1. 优先合格的最后一帧；
2. 更低的 `eligibility_tier`；
3. 更大的最小联合误差优势；
4. 更强的真值结构分数；
5. 更小的窗口编号作为确定性 tie-break。

需要明确：

> 该流程选择的是用于论文定性展示的代表性/优势帧，不是整体统计结论。总体性能结论必须来自冻结 evaluator 的完整逐 Re、逐窗口结果。

如需固定选帧，可显式指定：

```bash
--re 43.5 --time-index 55 --split heldout
```

---

## 8. 配色与共享色标原则

当前最终配色为：

| 内容 | Colormap | 归一化 |
|---|---|---|
| 速度模长 | `cividis` | CFD truth 的 0.5%–99.5% 分位数 |
| 压力 | `RdBu_r` | 以零为中心，CFD truth 绝对分位范围对称 |
| 逐点误差 | `haline_r` | 所有比较方法共享 0 到稳健上界 |

`haline_r` 依赖 `cmocean`。如果环境中没有安装 `cmocean`，程序会警告并回退为 `magma`。若论文要求固定最终配色，迁移后必须确认日志中没有 fallback。

共享色标原则：

1. 同一张图中所有方法使用相同速度色标；
2. 同一张图中所有方法使用相同压力色标；
3. 同一张误差图中所有方法共享同一个速度误差色标；
4. 同一张误差图中所有方法共享同一个压力误差色标；
5. 不允许为每个方法单独自动拉伸色标；
6. 发散方法允许饱和显示，不能用其极值压平其他正常方法的结构。

色条统一使用 LaTeX 科学计数法，并保留物理变量标签：

```text
Velocity magnitude |u|
Pressure p
Pointwise error |Delta u|
Pointwise error |Delta p|
```

---

## 9. 当前几何保真紧凑版式合同

`batlowW_style_haline_ultra_compact_v10b` 曾使用强制
`ax.set_box_aspect(0.32)` 压扁 panel。该设置会改变物理坐标的显示比例，
使圆柱截面由圆变成椭圆，因此该版本不应作为论文最终图。

当前代码采用几何保真紧凑版式：

- 每个 panel 使用 `ax.set_aspect("equal", adjustable="box")`；
- x、y 方向一个物理长度单位具有相同屏幕长度；
- 只压缩标题、行距、色条和外边距，不压缩物理结构；
- 物理场整图 `figsize=(11.8, 7.15)`；
- 误差图高度为 `1.65 * N_method + 0.75`；
- 两列分别为速度和压力；
- 变量列标题只在顶部出现一次；
- 方法名称放在第一列左侧并竖排；
- 长方法名拆行以防止相邻行重叠；
- panel 角标 `(a)–(h)` 或 `(a)–(f)` 保留；
- 误差值放在对应 panel 右上角；
- 两个共享水平色条放在整图底部；
- PNG 输出使用 400 dpi；
- `bbox_inches="tight"` 和较小 `pad_inches` 减少外部空白。

当前经过真实场视觉核验的输出版本为：

```text
results/batlowW_style_haline_geometry_preserved_compact_v11
```

由于当前视窗为 `x in [-1.5, 10.0]`、`y in [-3.0, 3.0]`，几何保真
panel 的自然高宽比为 `6/11.5`。不能在保持该完整视窗的同时把 panel
任意压到 `0.32`；若论文版面仍嫌高，应优先重排面板或在有科学依据时
统一裁剪视窗，而不能非等比例缩放坐标轴。

版式参数可以因论文栏宽微调，但以下内容不能因排版改变：

- 面板顺序；
- 数值和误差定义；
- pressure gauge；
- POD 重构；
- 共享色标；
- 方法标签含义；
- 选帧记录。

---

## 10. 新数据集配置模板

### 10.1 Specialist 模板

建议复制旧配置到新文件，不直接覆盖旧配置：

```json
{
  "schema": "field_figure_config/v1",
  "mode": "specialist",
  "figure_stem": "NewDataset_Steady",
  "bundle": "/ABSOLUTE/PATH/new_k56_rollouts.npz",
  "time_offset": 0,
  "arrays": {
    "re": "re",
    "split": "split",
    "times": "times",
    "truth_a": "truth_a",
    "truth_b": "truth_b"
  },
  "truth_basis": "steady",
  "bases": {
    "steady": {
      "velocity": "/ABSOLUTE/PATH/velocity_pod.npz",
      "pressure": "/ABSOLUTE/PATH/pressure_pod.npz",
      "r_u": 32,
      "r_p": 32
    }
  },
  "methods": {
    "vanilla": {
      "a": "vanilla_a",
      "b": "vanilla_b",
      "basis": "steady"
    },
    "dataonly": {
      "a": "dataonly_a",
      "b": "dataonly_b",
      "basis": "steady"
    },
    "proposed": {
      "a": "proposed_a",
      "b": "proposed_b",
      "basis": "steady"
    }
  },
  "plot_order": ["vanilla", "dataonly", "proposed"]
}
```

Hopf 和 Periodic 只需更换 `figure_stem`、bundle、basis 名称和 POD 路径。预测时域可以不同，例如 Periodic 当前使用 K48，其文件名和 `times` 必须反映真实时域。

### 10.2 S–H fusion 模板

```json
{
  "schema": "field_figure_config/v1",
  "mode": "sh_fusion",
  "figure_stem": "NewDataset_SH_boundary",
  "bundle": "/ABSOLUTE/PATH/new_sh_cache.npz",
  "weights_csv": "/ABSOLUTE/PATH/new_t2c_weights.csv",
  "weight_key_column": "start",
  "weight_alpha_column": "T2_C_alpha_S",
  "time_offset": 1,
  "arrays": {
    "re": "re",
    "split": "split",
    "times": "times",
    "truth_a": "true_a",
    "truth_b": "true_b",
    "weight_key": "start",
    "quad_u": "quad_u",
    "quad_p": "quad_p"
  },
  "truth_basis": "steady",
  "bases": {
    "steady": {
      "velocity": "/ABSOLUTE/PATH/steady_velocity_pod.npz",
      "pressure": "/ABSOLUTE/PATH/steady_pressure_pod.npz",
      "r_u": 32,
      "r_p": 32
    },
    "hopf": {
      "velocity": "/ABSOLUTE/PATH/hopf_velocity_pod.npz",
      "pressure": "/ABSOLUTE/PATH/hopf_pressure_pod.npz",
      "r_u": 32,
      "r_p": 32
    }
  },
  "methods": {
    "s_only": {
      "a": "s_a",
      "b": "s_b",
      "basis": "steady"
    },
    "h_only": {
      "a": "h_a",
      "b": "h_b",
      "basis": "hopf"
    },
    "t2c": {
      "s_a": "s_a",
      "s_b": "s_b",
      "s_basis": "steady",
      "h_a": "h_a",
      "h_b": "h_b",
      "h_basis": "hopf"
    }
  },
  "plot_order": ["s_only", "h_only", "t2c"]
}
```

---

## 11. 迁移前必须执行的检查

### 11.1 文件与键检查

确认：

- 所有 JSON 中的绝对路径存在；
- rollout bundle 包含配置声明的全部数组；
- POD NPZ 包含配置声明的全部键；
- `weights_csv` 中每个 S–H 窗口都有唯一权重；
- 输出使用全新的目录，不覆盖旧图和旧 manifest。

### 11.2 Shape 检查

确认：

```text
N_window 一致
K 一致
r_u 与 phi_uv 的截取维度一致
r_p 与 phi_p 的截取维度一致
phi_uv.shape[1] == 2N
phi_p.shape[1] == N
point_areas.shape == [N]
```

### 11.3 数值检查

确认：

- 所有 truth 和 prediction 系数为 finite；
- `point_areas` 全部有限且为正；
- 面积总和大于零；
- 压力 gauge 修正后面积加权均值接近零；
- 重构场不含 NaN/Inf；
- 每种方法的 relative L2 分母大于数值阈值；
- S–H 权重满足 `0 <= alpha_S <= 1`。

### 11.4 网格和顺序检查

至少随机抽查一个 snapshot：

1. 从新 POD 系数重构物理场；
2. 与新数据集原始 CFD snapshot 对比；
3. 检查圆柱位置、入口、尾迹方向和压力高低压区；
4. 检查节点编号映射；
5. 检查速度分量是否被交换；
6. 检查压力是否仅存在 gauge 差异。

如果网格顺序不一致，禁止通过转置、翻转图片或调整 `xlim/ylim` 掩盖问题。

### 11.5 时间检查

`physical_time` 的读取位置为：

```text
times[window, time_index + time_offset]
```

因此必须验证：

- `time_offset` 与 rollout bundle 的历史/预测定义一致；
- K56、K48 等标签与实际查询帧数一致；
- 不把数组索引当作物理时间；
- 不因迁移重新解释原生 specialist 的时间步长。

---

## 12. 推荐运行流程

### 12.1 单个配置预检

先只扫描候选：

```bash
python3 plot_field_figures.py \
  --mode specialist \
  --config /ABSOLUTE/PATH/config_new_steady.json \
  --template-vtk /ABSOLUTE/PATH/template.vtk \
  --output-dir /ABSOLUTE/PATH/preflight_output \
  --candidate-scan \
  --error-cmap haline_r \
  --png-only
```

如需固定 Re 和时间帧：

```bash
python3 plot_field_figures.py \
  --mode specialist \
  --config /ABSOLUTE/PATH/config_new_steady.json \
  --template-vtk /ABSOLUTE/PATH/template.vtk \
  --output-dir /ABSOLUTE/PATH/fixed_frame_output \
  --re 43.5 \
  --time-index 55 \
  --split heldout \
  --error-cmap haline_r \
  --png-only
```

S–H 使用：

```bash
python3 plot_field_figures.py \
  --mode sh_fusion \
  --config /ABSOLUTE/PATH/config_new_sh.json \
  --template-vtk /ABSOLUTE/PATH/template.vtk \
  --output-dir /ABSOLUTE/PATH/sh_output \
  --error-cmap haline_r \
  --png-only
```

### 12.2 四种情况批量生成

更新 `run_all_figures.py` 中：

1. `CONFIGS`；
2. `VTK_TEMPLATE`。

然后运行：

```bash
python3 run_all_figures.py \
  --output-dir /ABSOLUTE/PATH/new_dataset_figures \
  --error-cmap haline_r \
  --png-only
```

在迁移后的 `10.10.164.243:20060` 集群上，系统 `/usr/bin/python3`
不包含绘图依赖。已建立独立绘图环境：

```text
/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/
particalMOE/.runtime/plot_env_v1/bin/python
```

该集群应使用：

```bash
PLOT_PY=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/.runtime/plot_env_v1/bin/python

MPLBACKEND=Agg "$PLOT_PY" run_all_figures.py \
  --output-dir /ABSOLUTE/PATH/new_dataset_figures \
  --error-cmap haline_r \
  --png-only
```

已认证依赖版本为：

```text
Python 3.11.15
NumPy 2.4.4
Matplotlib 3.11.1
PyVista 0.48.4
cmocean package 4.0.3
VTK 9.6.2
```

预期终态：

```text
PNG_COUNT = 8
PDF_COUNT = 0
四个 FIELD_FIGURE_MANIFEST.json
四个 candidate_scan.json
```

---

## 13. 输出审计

`FIELD_FIGURE_MANIFEST.json` 保存：

- 配置路径及 SHA256；
- rollout bundle 路径及 SHA256；
- VTK 模板路径及 SHA256；
- T2-C 权重表路径及 SHA256；
- 每套速度/压力 POD 资产路径及 SHA256；
- 选中窗口、Re、split、时间索引和物理时间；
- 每个方法的速度、压力和联合误差；
- 色图、DPI 和输出文件；
- 每个输出文件的 SHA256。

迁移完成后，应把 manifest 与图片一起冻结。仅保存图片而不保存 manifest，会失去对选帧和输入资产的可追溯性。

---

## 14. Fail-closed 条件

出现以下任一情况时应停止绘图并修复资产，而不是继续生成论文图片：

1. POD 维度与系数维度不一致；
2. VTK 点顺序无法与 POD 自由度顺序对应；
3. truth 与 prediction 使用不同 pressure gauge；
4. 面积权重缺失、非正或与原 POD 不一致；
5. rollout bundle 含 NaN/Inf；
6. Re、split、窗口或时间戳无法对齐；
7. S–H 权重缺失、重复或超出 `[0,1]`；
8. S–H 两个模型不是在严格相同物理查询时刻输出；
9. 新数据使用不同网格却仍沿用旧 POD 基；
10. `haline_r` 不可用却未注意到 colormap fallback；
11. candidate scan 找不到任何满足预注册资格的展示帧；
12. test/heldout 数据被用于训练、调参或重新选择科学结论。
13. 圆柱或其他已知等距几何在图片中发生非等比例拉伸。

---

## 15. 论文使用边界

这些图片适合用于：

- 展示速度尾迹、涡脱落和压力结构；
- 展示不同方法的空间误差分布；
- 展示 Proposed specialist 或 S–H fusion 在选中代表帧上的局部优势；
- 辅助解释消融模型的典型失真或发散形态。

这些图片不能单独支持：

- 所有 Reynolds 数上全面优于基线；
- 所有窗口或 worst case 全面占优；
- 长期吸引子保持结论；
- H–P Top-2 合法性；
- 真实跨流态动态迁移能力。

上述结论必须由冻结的完整 rollout evaluator、逐 Re 指标、逐窗口指标和吸引子诊断共同支持。

---

## 16. 最简迁移清单

```text
[ ] 复制而非覆盖旧 JSON 配置
[ ] 更新 rollout bundle 路径
[ ] 更新速度/压力 POD 路径
[ ] 更新 VTK_TEMPLATE
[ ] 核验 NPZ 键名和 shape
[ ] 核验 [u_all, v_all] 自由度顺序
[ ] 核验 point_areas
[ ] 核验 pressure gauge
[ ] 核验 Re/split/times/time_offset
[ ] 核验所有数组 finite
[ ] S–H 时核验公共物理时间和冻结权重
[ ] 单配置 preflight
[ ] 视觉检查网格、圆柱位置和尾迹方向
[ ] 批量生成 8 张 PNG
[ ] 检查 4 个 manifest 和 4 个 candidate scan
[ ] 冻结输入与输出 SHA256
```
