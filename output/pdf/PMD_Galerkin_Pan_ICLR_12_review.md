# RAL-MoE-ROM 稿件校对清单

检查对象：`PMD_Galerkin_Pan_ICLR (12).pdf`，26 页。检查日期：2026-09-26。

已通读正文、参考文献及附录；对疑似空格、标点和图中文字问题进行了页面图像核对。页码为 PDF 印刷页码；有行号处使用左侧审稿行号。未修改原 PDF。以下区分明确错误、表达建议和需要作者核实的内容；不等同于数学证明或实验复现审查。

## 1. 明确的语法、拼写、标点问题

| 位置 | 原文/定位词 | 建议修改 | 说明 |
|---|---|---|---|
| p.1，摘要 | `history-conditioned specialists fusion` | `history-conditioned specialist fusion` 或 `history-conditioned fusion of specialists` | 名词作前置修饰语应使用单数。 |
| p.2，L079 | `sparsely structured MOEs` | `structured sparse MoEs` | 与全文术语一致；原词序也容易被理解成“结构稀疏”。 |
| p.2，L081 | `derived from the projection-based.` | 至少补成 `derived from the projection-based ROM.`；但是否真是 derived 还需核实，见第 4 节 | 形容词后缺少中心名词。 |
| p.2，L083–084 | `Within a prescribed adjacent specialists pair, these preferences support either Top-1 specialist or neighboring specialists .` | `Within a prescribed pair of adjacent specialists, these preferences support either Top-1 selection or the selection of both specialists.` | 复合名词、并列结构、选择动作缺失，且句号前多余空格。随后一句再说明 fusion 即可。 |
| p.3，Figure 1(b) | `Pressure-Possion` | `Pressure-Poisson` | 图内明确拼写错误，普通正文搜索可能找不到。 |
| p.3，Figure 1 图注，L128 | `For a neighbouring specialists selection` | `When a pair of neighboring specialists is selected` | 语法不自然；也统一英美拼写。 |
| p.3，L137 | `state evolution(Lu et al., 2021; …)` | `state evolution (Lu et al., 2021; …)` | 文内引文前缺空格；页面上确实存在。 |
| p.4，L162 | `applies router at two levels` | `applies routing at two levels` | router 是可数名词，此处表达 routing 更自然。 |
| p.4，L181–182 | `In our benchmarks, parameters μ = Re and ν(μ) = Re⁻¹.` | `In our benchmarks, we set μ = Re and ν(μ) = Re⁻¹.` | 原句缺少谓语。 |
| p.4，Eq. (2) 后，L185 | `while u … and p … denote …` | 将 Eq. (2) 末尾句号改逗号，下一行用 `where u … and p … denote …` | 当前为句号后接从属连词，且这里是在定义符号。 |
| p.4，L186 | `The subscript h denote` | `The subscript h denotes` | 主谓一致。 |
| p.4，L190 | `separate velocity and pressure basis` | `separate velocity and pressure bases` | 两套基，使用 basis 的复数 bases。 |
| p.4，Eq. (4)，L203 | 公式结尾逗号，下一句独立以 `Appendix B provides` 开始 | 将公式末尾逗号改为句号 | 当前上下句没有语法承接关系。 |
| p.4，L205 | `pressure–Poisson equation(PPE)` | `pressure–Poisson equation (PPE)` | 括号前缺空格。 |
| p.5，L264–265 | `… is PPE equation defined in Appendix B; Both terms …` | `… is the algebraic pressure map defined in Appendix B. Both terms …` | 缺冠词、分号后不应无故大写 Both；该符号实际表示映射，见 Eq. (46)。 |
| p.6，Eq. (16) 后 | `â… denote recursively predicted coefficients, whereas a… symbols denote reference coefficients.` | `The quantities â… and b̂… denote recursively predicted coefficients, whereas a… and b… denote the corresponding reference coefficients.` | 原句符号作主语时数的一致性不清楚，且漏解释压力系数；统一解释全部四个量。 |
| p.6，§3.4 | `Both specialist encode` | `Both specialists encode` | both 后用复数。 |
| p.7，L326 | `four part` | `four parts` | 数量与名词不一致。 |
| p.7，L326–327、§4.1 标题 | `local specialists prediction` / `LOCAL SPECIALISTS PREDICTION ACCURACY` | `local-specialist predictions` / `LOCAL-SPECIALIST PREDICTION ACCURACY` | 或用所有格 `local specialists’ predictions`。 |
| p.7，L328 | `invariants ,` | `invariants,` | 删除逗号前空格。 |
| p.7，L328–329 | `Three test-case configurations: fluidic pinball, flow past a cylinder and flow past a square, are summarized …` | `The three test-case configurations—fluidic pinball, flow past a cylinder, and flow past a square—are summarized in Appendix D.` | 原文冒号和逗号混用，破坏主谓结构。也可使用括号。 |
| p.7，L334–335 | `Table 1 shows local specialists achieves` | `Table 1 shows that the local specialists achieve` | 主谓一致。 |
| p.7，L358–359 | `Dense model … while Structured model …` | `The Dense model … while the Structured model …` | 单数可数名词需要限定词。全文同类位置一起改。 |
| p.7，L361 | `under joint error metric` | `using the joint error metric` | 缺冠词且搭配可改进。 |
| p.7，L372 | `the effectiveness of the local specialist accuracy :` | `the contribution of input-dependent sparse routing to local prediction accuracy:` | 原搭配不成立；同时删除冒号前空格。 |
| p.8，L394 | `we fix the local specialist and the E2 router` | `we fix the local specialists and the E2 router` | 本段涉及多个冻结的 specialist。 |
| p.8，L428 | `Contribution of History-conditioned.` | `Contribution of history conditioning.` | conditioned 为形容词，不能单独作 of 的宾语。 |
| p.9，L437 | `but hopf specialist has` | `but the Hopf specialist has` | 缺冠词；Hopf 为专名。 |
| p.9，L446、L450–451 | `Periodic specialist …`、`in Periodic specialist`、`in Hopf specialist` | `the Periodic specialist …`、`in the Periodic specialist`、`in the Hopf specialist` | 同类冠词问题。p.24 Table 10 后、p.25 Appendix H 开头也有。 |
| p.9，L461 | `0.8652 s for Structured. model Despite …` | `0.8652 s for the Structured model. Despite …` | 句号插到了名词短语中间。 |
| p.9，L465 | `comparison .` | `comparison.` | 删除句号前空格。 |
| p.12，L643 | `systemas in Eq. (2)` | `system as in Eq. (2)`；更自然为 `system given in Eq. (2)` | 页面中确实缺空格。 |
| p.14，L735–736 | `Substituting equation 32 into equation 40, and applying …, yields …` | `Substituting the reconstructions in Eqs. (31)–(32) into Eq. (40) and applying … yields …:` | 删除主语与谓语之间的多余逗号；统一公式引用格式；速度重构引用也应补充，见第 2 节。 |
| p.17，Specialist architectures | `D = … is feature width` | `D = … is the feature width` | 缺冠词。 |
| p.17，Table 6 图注 | `Periodic regime velocity and pressure errors on flow past a cylinder case(%).` | `Velocity and pressure errors (%) in the periodic regime for flow past a circular cylinder.` | case 的搭配和括号前空格均需调整。 |
| p.17，Table 6 后 | `Lower errors than structured model …` | `Lower errors than those of the Structured model …` | 比较对象应是 errors 与 errors，而非 errors 与 model。 |
| p.18，Table 7 前 | `T2-C has the lowest terminal velocity and the more accurate individual specialist differs …` | `T2-C achieves lower terminal-step velocity and pressure errors than either individual specialist. The more accurate individual specialist differs between overlaps: …` | `velocity` 后漏掉 error；原句容易误解成预测的速度最小。重写也避免把 oracle 纳入“最低”的范围。 |
| p.20，Figure 3 图注 | `The Square case is set Re = 100 … while Pinball at Re = 22 …` | `For the Square case, Re = 100, K = 24, and t = 104 (rollout duration: 96); for the Pinball case, Re = 22, K = 56, and t = 50 (rollout duration: 14).` | 第一部分缺 at，后一部分表达不完整。 |
| p.21，Figure 4 图注 | `Circular case is set at …` | `The circular-cylinder case uses …` | 缺冠词；几何名称写完整更清楚。 |
| p.26，Table 12 图注，L1351 | `… on flow past a Circular .` | `Local prediction cost for the circular-cylinder periodic benchmark at K = 48.` | Circular 为形容词，后缺 cylinder；句号前多余空格。 |
| p.26，L1363–1365 | `with Dense and Structured model, batch size one and K = 48 rollout` | `with the Dense and Structured models at batch size one over a K = 48 rollout` | 两个模型应用复数，并补齐介词。 |
| p.26，L1365 | `ten repetition` | `ten repetitions` | 复数错误。 |
| p.26，L1367、L1370–1371 | `Structured model`、`Dense model` | `the Structured model`、`the Dense model` | 同类冠词错误。 |
| p.26，L1381 | `RAL-MOE-ROM` | `RAL-MoE-ROM` | 正文模型名大小写统一；全大写标题样式不需要机械替换。 |

## 2. 文内引用、图表引用与指向问题

1. **p.12，L647：Section 3.2 应改为 Section 3.3。** 原文把 time advancement 指向 §3.2；实际时间积分说明和 Eqs. (14)–(15) 位于 §3.3。
2. **p.6，§3.3：“Benchmark-specific time-stepping schemes are provided in Appendix D.” 当前附录 D 并未兑现这一指向。** D 提供 CFD 时间步、采样间隔、优化器和网络设置，但没有逐 specialist 的 ROM 积分方案表。应补上真实方案/子步设置，或指向实际包含这些内容的位置。不要把 CFD time stepping 当作 ROM time stepping。
3. **p.14，L735：只说把 Eq. (32) 代入 Eq. (40) 不完整。** Eq. (40) 同时含速度和压力；导出含 a 的 Eq. (41) 还使用了 Eq. (31) 的速度重构。建议明确为 `the velocity and pressure reconstructions in Eqs. (31)–(32)`。
4. **p.3，Figure 1(a)：标签与正文不一致。** 图内是 `Specialist s`、`Specialist h`、`Specialist t`；正文采用 S/H/P。若三框表示三种流态，应统一为 `Specialist S`、`Specialist H`、`Specialist P`；尤其检查 t 是否为旧版本标签。
5. **p.3，Figure 1(b)：`nonlinear + affine + low-rank bilinear` 与 Eq. (8) 术语不完全一致。** 正文说无偏置 linear branch 和 quadratic branch。建议图内改成 `nonlinear + linear + low-rank quadratic`，除非确实有未在正文说明的 affine bias。
6. **p.7 Table 1：比较结论应限定有基线数据的设置。** Steady 的 POD–Galerkin 全是破折号。`consistently reduce both velocity and pressure errors` 不能读成已对全部四个设置完成基线比较。建议加 `in all settings where POD–Galerkin results are available`。
7. **Table 1、Table 2、Table 3 缺少独立可读的指标/单位说明。** 建议 Table 1 列头加 (%)；Table 2 和 Table 3 的 caption 明确 `joint error E_u + E_p (%)`。Table 2 还应解释 ± 是跨随机种子的标准差、标准误还是其他统计量，以及种子数。正文/附录的 seed 信息不足以代替表注中的统计量定义。
8. **p.19 Table 8：建议 caption 明确 terminal-step。** 它接续 Table 7，且参数分组平均值对应 Table 7 的终点误差，不是 Table 3 的 full-window 指标。当前 `at K = 24` 容易混淆。
9. **p.19 Figure 2 的 ± 同样应定义。** p.8 提到 three paired gate-training seeds，但误差条具体是什么仍应明确。`±0.0000` 建议增加有效位或说明为舍入结果，不能让读者理解成方差严格为零。

人工对照未发现明显的“正文作者—年份引用在参考文献中找不到”或 `??` 型未解析引用。Li et al. (2021a) 对应 cluster-based network model，Li et al. (2021b) 对应 FNO，当前对应关系一致。参考文献条目也均能在正文或附录找到用途。此结论基于编译后的 PDF，不包含 BibTeX key 或 LaTeX 编译日志检查。

## 3. 参考文献的发表版本、元数据和引用适配性

### 3.1 arXiv 是否需要替换

**没有发现仍明确标为“arXiv preprint”而漏写已发表 venue 的条目。唯一直接使用 arXiv URL 的是 FNO，但它已经写了 ICLR 2021。** 因此这里是链接规范化问题，不是遗漏会议发表信息。

| 条目 | 当前状态 | 核查结论/建议 |
|---|---|---|
| Li et al. (2021b), FNO，p.12 L600–603 | venue 为 ICLR 2021；URL 为 arXiv:2010.08895 | 已发表情况可见[作者发表列表](https://kovachki.github.io/publications/)。建议 URL 改为[正式 OpenReview 记录](https://openreview.net/forum?id=c8P9NQVtmnO)，保留 2021。此次 OpenReview 正文访问出现验证页，发表信息另以作者主页交叉确认。 |
| Colanera & Magri (2025), ql-ROM，p.11 | CMAME 447:118393，DOI 完整 | 已使用正式发表版；[机构存档的出版 PDF](https://iris.sissa.it/retrieve/869ce5bc-d11a-4aec-a5ca-776c644cf249/1-s2.0-S0045782525006656-main_compressed.pdf)确认。无需改成 arXiv。 |
| Ben-Shabat et al. (2024), Neural Experts，p.10 | NeurIPS 37，页码、DOI、官方 PDF | 已是正式版；见[NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b83fae17d73b079b1b98ab200276db9f-Abstract-Conference.html)。 |
| Chalapathi et al. (2024)，p.10 | ICLR 2024，官方 proceedings PDF | 已是正式版；见[ICLR 页面](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9aeda582add763c41c7b39691ce19ab0-Abstract-Conference.html)。 |
| Wang et al. (2025), MoE-POT，p.12 | NeurIPS 38，DOI、官方 PDF | 已是正式版；见[NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2d23a9991a6f64482bf395628e279f5f-Abstract-Conference.html)。 |
| Siyang Li et al. (2026), iMOOE，p.12 | ICLR 2026，官方 proceedings PDF | 已是正式版；见[ICLR 页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html)。 |
| Fedus et al. (2022)，p.11 | JMLR 23(120):1–39 | 与[JMLR 正式记录](https://jmlr.org/papers/v23/21-0998.html)一致。 |
| Kovachki et al. (2023)，p.11 | JMLR 24(89):1–97 | 与[JMLR 正式记录](https://jmlr.org/papers/v24/21-1524.html)一致。 |

其余参考文献在当前 PDF 中均已给出期刊或会议名称。这里的逐项在线确认重点放在 arXiv 链接、近年会议论文、疑似缺项和论述匹配问题；不能据此声称每一条旧文献的每位作者、卷期、页码和 DOI 都完成了独立外部核验。

### 3.2 参考文献格式/元数据修正

- **p.12，L608–609，Pan & Xiao (2024)：建议补 DOI `10.1016/j.jcp.2024.113452`。** 作者、年份、卷 519 和文章号 113452 与[出版社记录](https://www.sciencedirect.com/science/article/abs/pii/S0021999124007009)一致。没有 DOI 不构成错误引用，但全文多数期刊条目都有 DOI，补齐更一致。
- **p.12，L630，`huanshuo dong` 建议规范为 `Huanshuo Dong`。** 这不是作者识别错误：[NeurIPS 官方元数据](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2d23a9991a6f64482bf395628e279f5f-Abstract-Conference.html)本身就是小写。可规范大小写。该论文某个 OpenReview PDF 的其他姓名拼写与 proceedings 元数据也不同，不应只凭该 PDF 擅改你条目中的 Fei Zha / Yan Jiang。
- **p.12，Li (2026) 和 Wang (2025) 的标题中 `pde` 应保护为 `{PDE}`。** 当前 sentence-case 样式把缩写变成小写。
- **p.11，Colanera & Magri 标题中的 `ql-rom` 应保护为 `{ql-ROM}`。** 正文 p.7 又写 `QL-ROM`，也建议统一成原方法名 ql-ROM。
- **p.12，Sirovich (1987) 的 `part i` 建议规范成 `Part I`。** 尤其罗马数字 I 不应变成小写 i。可在 BibTeX 中用 `{Part I}` 保护。
- 当前 ICLR proceedings 条目包含 `volume 2024/2026`、编辑和长页码，不应只凭形式陌生就认定“引用错误”。可统一成简洁的 `In International Conference on Learning Representations, YEAR`，但这是风格选择；不要凭空补期号或 DOI。
- DOI 中出现比正式年份更早的年份，不自动构成错误，例如 Fresca 2022 / DOI 含 2021，Geelen 2023 / DOI 含 2022。不能仅按 DOI 字符串改出版年份。

### 3.3 文献是否支撑对应论述

以下是论述适配性建议，不把它们混同于作者或年份写错。

1. **p.1 的 truncation/closure 论述与 Amsallem & Farhat (2012) 的对应偏弱。** [原论文摘要](https://doi.org/10.1002/nme.4274)主要讨论线性投影 ROM 的稳定化方法，而非直接研究被截断非线性模态的 closure error。p.2 用它支撑 stabilization 更合适。建议把“闭合误差”和“稳定性问题”拆开，各配直接文献，或缩窄 p.1 的断言。
2. **p.3，L159–160 列举三类 SciML 应用，却同时引用 Neural Experts，匹配不精确。** [Neural Experts 官方摘要](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b83fae17d73b079b1b98ab200276db9f-Abstract-Conference.html)主要是隐式神经表示及表面、图像、音频重构，并非该句列出的物理硬约束、神经算子预训练或 PDE 预测。建议加入 `implicit neural representations`，并逐项就近放引文。

可将后一处改成：

> MoE-based conditional specialization has also been explored in implicit neural representations (Ben-Shabat et al., 2024), physics-informed hard constraints (Chalapathi et al., 2024), neural-operator pretraining (Wang et al., 2025), and PDE forecasting (Li et al., 2026).

## 4. 技术措辞与一致性：需要作者确认，不建议机械替换

1. **p.2 `derived from` 与 Eq. (8) 的关系。** 若 linear/quadratic branches 是独立训练的参数，而仅借鉴 Galerkin 结构，建议写 `inspired by the structure of the projection-based ROM`。只有系数确实由投影算子推导或初始化时才保留 derived，并说明映射关系。
2. **p.5 L216 和 p.17 `Features include … projected Galerkin equation`。** 输入是方程右端的数值 g，而不是 equation 本身。建议使用 `evaluations of the projected Galerkin vector field` 或 `projected Galerkin right-hand-side values`；历史信息也应在描述中体现。
3. **p.6 `three-layer MLP—a 1–24–3 MLP` 易引起计数歧义。** 123 个参数与一个隐藏层的 1→24→3 网络吻合。建议 `an MLP with one hidden layer of width 24 (1–24–3; 123 parameters)`。无需改参数总数。
4. **p.14 L753 把 reduced pressure coefficients 称为 pressure field。** Q 的输出是压力系数，物理场还需要 Eq. (32) 解码。建议写 `The reduced pressure coefficients are recovered as b^{PP,(r)} = Q_r^{PP}(a^{(r)}; μ), and the pressure field is reconstructed using Eq. (32).` 这是对象混用，不只是英语润色。
5. **p.20–21 代表性图像与 Table 8 的测试设置关系应说明。** Figure 3 的 Square 是 Re=100，Table 8 的 H–P 是 Re=100.5 和 102.0；可能是独立可视化样例，不能仅据此断定数据错误，但最好说明该窗口来自哪个 split，是否不属于那 16 个汇总测试窗口。
6. **p.21 Figure 4 rollout duration=598，与 p.26 的 598.7009 如指同一窗口，建议统一精度。** 可写 approximately 599，或统一为 598.7；如果不同窗口应明确。
7. **p.7 `plays a decisive role`、`confirming … robustness`、`nearly fails` 不是语法错误，但语气偏强。** 当前证据是限定基准和有限时域的对比，建议分别改成 `contributes substantially`、`supporting its effectiveness over the evaluated horizons`、`yields a joint error of approximately 114%`。
8. **p.3 的 `Recent work … More recently …` 重复。** 合并成 `Recent work has further emphasized explicitly local reduced dynamics. For example, ql-ROM …` 即可。
9. **p.8 的 §4.2 标题单独带句号，其他节标题不带。** 删除末尾句号；p.7 `Autonomous Rollout Error Comparison` 和 p.8 `Comparison for the cross-regime routing` 则补齐行内小标题句号，并统一 sentence case。
10. **术语统一。** 使用 RAL-MoE-ROM、MoE、ql-ROM；正文多数采用美式 spelling，图注的 neighbouring 可改 neighboring。Steady/Hopf/Periodic 若作为模型标签可保留大写，一般形容流态时用 steady/periodic 小写。模型名 Local-Specialist / Local Specialist / Proposed 建议说明等价或统一。

## 5. 推荐优先处理顺序

1. 图 1 的 Poisson 拼写和 S/H/P 标签；p.2 残句；p.9 `Structured. model Despite`。
2. p.12 的 §3.2→§3.3 指向、p.6 对 Appendix D 的未兑现指向，以及 p.14 系数/场的混用。
3. 全文主谓一致、复数、冠词和多余空格。
4. FNO 链接、BibTeX 缩写保护、Pan & Xiao DOI；两处引用适配性调整。
5. 表格指标和 ± 定义、图注与汇总实验关系、术语和小标题风格。

本文仅提供校对与修改建议。需要依赖训练实现、原始日志或作者意图的项目已单独标记，未当作已证实的数学或实验错误。
