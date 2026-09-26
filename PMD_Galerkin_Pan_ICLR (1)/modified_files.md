# 本轮修改文件

日期：2026-09-13。工作目录：`C:/Users/panxy1019/Documents/CHANNEL/PMD_Galerkin_Pan_ICLR (1)`。
依据：`RAL_MoE_ROM_second_round_consistency_review.md`。范围为语言、当前稿件内部数学/术语一致性、交叉引用和必要的编译修复；没有重写实验或替换数据。

| 文件 | 修改内容 |
|---|---|
| `main.tex` | 摘要首句润色；区分 parameter region 与 reduced chart；精简 sparse correction 表述；统一 applicable。 |
| `sections/introduction.tex` | mean fields、independent local ROMs、linear expert branch、autonomous rollout；明确每个 parameter region 对应 chart 与 specialist。 |
| `sections/related_work.tex` | 修复引用前多余句点及缺空格；区分 chart 与自主 specialist；调整 cross-chart assembly 标题。 |
| `sections/method.tex` | 保留 sparse mixture 公式，将 expert 参数化指向附录 C；压力公式按当前附录 C 改成 gated PP contribution + correction；统一 m-star、applicability、gauge-corrected 和 chart-specific dimensions；分开引用 advancement 与 optimization 附录。 |
| `sections/experiments.tex` | 统一 autonomous rollout 和 one-step prediction；弱化 validation-stage stability 表述；修复连接句、引用空格及 Table 3 后缺空格；不改结果数字。 |
| `appendix/rom_derivation.tex` | 修复未定义的 app:specialist；affine reconstruction 同时指向附录 A 的完整定义；规范 Galerkin backbone 与 gauge 术语。 |
| `appendix/specialist_details.tex` | pressure correction、chart-specific training set、reduced time step；修复 Eq. 引用格式。 |
| `appendix/routing_details.tex` | 修复 rollout.The；完整保留六维 descriptor 及其他公式。 |
| `appendix/implementation.tex` | verified 改为 reported training configurations；区分配置课程/预算与选中 checkpoint；Selected step 改 Selected 以兼容 epochs；supplementary implementation；接收从附录 G 移来的 gate 训练协议。 |
| `appendix/circular_cylinder.tex` | 统一 predefined validation-stage rollout criterion；整理 caption 空格。 |
| `appendix/square_cylinder.tex` | 合并重复 seed variability 段落，保留三种子和 sample SD 定义；训练协议移至 E；Best single 与正文 Best fixed 对齐；移除指向不含 oracle protocol 的附录 D 引用；修复六个非数学模式中的粗体数字。 |
| `appendix/fluidic_pinball.tex` | one-step prediction / autonomous rollout；修复四个非数学模式中的粗体数字。 |
| `appendix/internal_routing.tex` | 修复 K=56 后缺空格；统一 reduced time step/autonomous rollout；明确 routing mass 不直接给出 physical interpretation。 |

未修改：`sections/conclusion.tex`、`appendix/parameter_counts.tex`、两个 `.bib`、模板、图像、实验数值、checkpoint 名称和六维描述量定义。流程图位图内的旧术语单独列为待处理项。

## 差异与回退材料

该目录不是 Git 仓库，没有初始化仓库或创建提交。
通过 `git diff --no-index` 与修改前快照生成：

- `review_20260913/revision.diff`：完整源文件差异。
- `review_20260913/diff_summary.txt`：13 个 tex 文件，新增 105 行、删除 122 行（按行统计，不是科学内容增删量）。
- `review_20260913/before/`：修改前 main、sections、appendix 与 bibliography 快照。没有删除原始 zip。

## 编译与核验

成品：`output/pdf/main.pdf`，26 页，结论仍在第 9 页。
原稿 baseline 编译在附录 G 因 `\mathbf allowed only in math mode` 中止。本轮修复该错误，也修复附录 H 同类用法。最终编译成功，无 undefined references/citations、无 duplicated labels、无 overfull boxes。

完整核验：`review_20260913/verification.json`。仍有 21 条 underfull 排版提示（包括 bibliography 行间疏松），不属于 overfull/溢出版面；未为了消除这些提示更改参考文献或模板。已渲染全稿 26 页检查总体布局，并放大复核方法公式、训练配置表与三种子表。
