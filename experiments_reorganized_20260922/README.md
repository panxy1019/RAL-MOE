# 实验与附录重组（2026-09-22）

基于 round3_20260921/Overleaf_after_batch3_20260921。原目录未修改。

## 上传

将 changed_files 下的七个 LaTeX 文件按相对路径放入原项目。
本轮删除重复的正文概览表，保留原附录 Table 5（重新编号为 Table 4），并在 implementation.tex 表注区分原有局部评价与新增对照/融合实验的预测长度。
main.tex 仅调整实验附录输入顺序及 reproducibility 中的附录说明；摘要、Introduction、Related work 和方法文件未修改。
如果线上 main.tex 已有后续改动，请只合并其附录输入列表和 reproducibility 引用，不整文件覆盖。

正文入口为 sections/experiments.tex。原目录只有 experiments1.tex，而 main.tex 引用了 experiments；本交付补齐正确入口。
旧实验附录文件仍可留在项目中，但不要与 experimental_*.tex 同时 input，否则会出现重复内容和标签。

## 结构

1. Local prediction accuracy and finite-horizon reliability。
2. Cross-regime routing and physical-space fusion。
3. Internal routing and physical-field diagnostics。
4. Computational cost and timing boundaries。

后半部分附录对应为 experimental_local、experimental_fusion、experimental_diagnostics、experimental_cost。
保留原有关键标签、实验数据与图片依赖，新增总分组标签。

## 证据边界

- Pinball POD–Galerkin 比较不能解释为仅增加 sparse-MoE 的消融：已绑定 Periodic 为 Deep-FNN-H3；独立 S/H 旧行尚未完成身份闭环。正文保留原值并明确标为 archived comparison。
- 当前预测窗口不足以建立任意长期稳定性；未虚构同条件 bare-Galerkin/additive-MoE 新实验。
- Circular Dense 优于 proposed、二次修复未稳定改善、Hopf 未完整三种子均在正文保留。
- Square 主比较统一全窗口，终端结果及其原有结论放在附录并注明口径。
- 残差为离线重构场诊断，不是原 CFD 算子残差；计时不是完整层级管线加速比。

## 检查

在 qa_project 独立副本使用 Tectonic 完整编译，退出码 0；PDF 仅用于审阅。
本轮未重训、未新增实验、未修改图中数据。未完成全部页面的逐页视觉检查。
