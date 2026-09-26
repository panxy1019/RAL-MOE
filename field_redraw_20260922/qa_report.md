# 验证记录

## 已执行

- 按冻结规则选定 5 个快照；选择代码不读取误差排序。Circular/Square 从原预测缓存解码，Pinball 只恢复 S–H、H–P 各一个 K56 窗口的冻结推理。
- Square 两候选哈希与原 PREFLIGHT 对照；gate 绑定 E2 哈希；重新计算的 alpha 与历史权重 CSV 相差小于 `1e-6`。
- Pinball 两组候选、E2、gate 的哈希均存在于原 final-test seal。恢复耗时写入各面板 JSON，没有训练或模型选择。
- E2 为三个类别概率的真实全局 argmax，四个展示窗口的选择均位于预设候选对内；不使用 oracle 或测试误差替代。
- Square/Pinball 用直接物理数组积分重算 Eu/Ep，与原 quadratic statistics 公式核对，逐项差异小于 `1e-7`。Circular 与旧同窗口终点 CSV 核对，差异小于 `1e-5`。本地渲染再以独立 einsum 实现核对导出指标，差异小于 `1e-12`。
- 全部导出场无 NaN/Inf，积分权重为正，压力全域加权均值绝对值小于 `1e-6`；无固体填零或跨障碍物插值。
- Circular VTK 与 POD 点坐标完全一致；取原网格一个挤出平面的原始面三角形。Square 9400 个单元、Pinball 70194 个单元，保留原单元映射。Square 最大坐标偏差 `1.90735e-6`，Pinball `4.82428e-5`；Pinball 同时核对原 mesh hash 和积分权重。
- 压力用零中心发散色图、速度和误差用顺序色图。同行物理场/误差分别共享尺度，颜色范围来自完整计算域，无百分位截断。
- 所有图宽 5.5 in = 139.7 mm，图文基准字号 8–8.5 pt、刻度 7 pt；所有流场轴的 x/y 数据单位像素长度相等。
- `render_qa.json` 检查文字位于画布内，并保存 PDF/PNG 哈希；`validate_package.py` 核对全部导出数组、网格、图文件哈希与有限性/gauge/模板溢出。
- 当前原版 ICLR 样式实际 `textwidth=397.48499 TeX pt`。六页预览已用 Poppler 按 96 dpi 渲染并逐页检查，最终日志无 `Float too large`、`Overfull` 或 `Underfull`。导出的原图均为 400 dpi，PDF 文本为矢量字体。

## 必须随图保留的限制

1. 三例总览不是统一 T2-C 消融。Circular 为 original-init Periodic 局部模型，Pinball 候选明确是 Deep-FNN-H3。
2. 所有参考为 POD 重构，而非 raw CFD。速度显示模长，但误差计算两分量向量范数。
3. Circular 使用 raw physical scale 与固定 `nu=0.001`，没有将其强行改称与其他几何相同的无量纲场。保留原始尺度、逐例独立色标。
4. Pinball S–H 的共同全范围色标由前柱附近少量单元的大误差控制：E2 最大点速度误差约 `1.4761`、压力误差 `0.8229`；位置约 `(-1.82,-0.01)`。大部分误差面板因而很暗。这不是缺数据或将误差设为零；原极值均保留。详见 `data/independent_qa.json`。
5. 固定帧并不总是融合优于每个候选。例如 Square S–H、Pinball H–P 的该帧联合误差均高于 Candidate 1；与全窗口平均结论不能混为一谈。主图只与真实 E2 Top-1 对照。
6. 未对色盲视觉做专门仿真；采用 cividis、RdBu_r、magma，方法身份由固定列名而非颜色区分。
7. 没有覆盖旧成果、调整原论文结构或上传外部服务。作者需选择主文版本；`figure_snippet.tex` 仅供插入。

## 重跑验证

```powershell
python scripts/plot_field_comparison.py
# 重新编译 preview_iclr.tex 后：
python scripts/validate_package.py
```

`scripts/build_geometry.py` 需要原 VTK/POD 文件；仅修改图形外观时无需重跑它。
