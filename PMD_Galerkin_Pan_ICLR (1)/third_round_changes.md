# 第三轮修改结果

依据 `RAL_MoE_ROM_third_round_revision_checklist.md`，继续修改最新目录 `PMD_Galerkin_Pan_ICLR (1)`。本轮不重写方法、不修改实验数据、不修改 checkpoint/model identity。

## P0

- 为图表两侧显式添加段落边界，避免行内浮动体插入句子。表格使用标准 float 包的 `[H]` 保持在完整段落之间；Figure 2/5/6 同样固定在其段落之外。Figure 5 移到附录 I 的完整诊断段落之后。未修改 conference style。
- Table 3/8/12 使用 Equal blend、E2-probability blend、T2-C mu-only、T2-C、Convex oracle；Table 3 的较长表头换行，不缩写方法名。
- Reynolds-number trajectories 改为 parameter trajectories。
- Table 13 改为 Active experts、Distinct Top-2 pairs、Dominant-pair share (%)，caption 引用现有 entropy 方程；H 定义不变。
- Table 7 增加 dash 对应未满足 validation-stage rollout criterion、未报告 held-out result 的说明。

## P1

- bilinear responses 改 quadratic responses，保留原二次项数学定义。
- 保留 projected Galerkin backbone / projected ROM backbone 的层级。
- regime coverage、data splits、validation-selected comparison、Periodic-core 全文统一。
- T2-C 物理场组合称 convex fusion；Delta_psi 明确称 logit correction。
- worst-Re 选择准则改为 largest Reynolds-number-wise full-window mean joint error，保留 full-window 含义。
- 摘要首次使用 hybrid physics--data specialist，后续主要称 specialist；3.2 标题为 Regime-local hybrid specialists。

## P2

- 标题源码已经正确使用 RAL-MoE-ROM，未修改模板的标题大小写转换。
- specialist 标签语境中的 Steady/Periodic 大小写统一；普通物理描述保留小写。
- Table 12 明确写 E2-probability blend；沿用 reduced time step。
- 使用标准 `\raggedbottom` 避免强行拉伸段落间空白；不改字体、边距、行距或 conference style。

## 核验与交付

- 全文 27 页，结论在第 9 页。附录因图表保持完整而重新分页。
- 14 个表格的数据行数字序列与本轮修改前相同；所有 equation/align 数学环境逐字一致，六维 descriptor 定义不变。
- 两个 bibliography 文件哈希不变。
- undefined references/citations、重复标签、overfull boxes 均为 0。17 条 underfull 提示均来自 bibliography，不修改其内容或样式。
- Appendix A--J、Table 1--14、Figure 1--6 编号保留。
- 可上传包：`output/overleaf_third_round_20260913.zip`，入口 `main.tex`。包含论文源码、模板、参考文献及全部图像，不含内部审计/备份/旧版本/PDF 日志。
- PDF 预览：`output/third_round_pdf/main.pdf`。
- 原稿快照：`review_third_20260913/before/`；diff 与机器核验记录保存在同一 review 目录。

未解决的作者确认项见 `unresolved_items.md`；不能把语言与排版检查通过理解为科学身份问题已解决。
