# ICLR 2027 投稿检查报告

检查日期：2026-09-24。

最终版面依据：`C:/Users/panxy1019/Downloads/PMD_Galerkin_Pan_ICLR (7).pdf`，共 26 页。源码定位依据：`C:/Users/panxy1019/Downloads/PMD_Galerkin_Pan_ICLR (5)`。这两者不是完全相同的版本，例如 Conclusion 和若干实验段落不同；以下 PDF 页码针对 (7)，源码位置只作定位参考。未修改原稿，未重新编译。

## 1. 投稿格式与分页

- 源码使用 `\documentclass{article}`、`\usepackage{iclr2027_conference,times}` 和 `iclr2027_conference.bst`，匿名模式开启，`\iclrfinalcopy` 被注释。
- 将目录中的 `.sty`、`.bst` 与官方 2027 ZIP 中对应文件逐字节比较（统一换行后），两者均一致。没有发现将旧年份模板改名使用的情况。
- (7).pdf 的全部页面都是 US Letter，612 × 792 pt；有匿名作者、审稿页眉、行号和页码。PDF Author 元数据为空；已嵌入的外链是文献链接，未发现个人仓库链接。
- 第 9 页包含第 4.4 节、完整 Conclusion、AI use statement 和第一条参考文献。正文在第 9 页结束，符合初投稿正文最多 9 页的规定。参考文献延续到第 12 页；附录随后开始。整份 PDF 26 页本身不是超页。
- **AI use statement 不必另起一页。** 官方模板在该标题前没有强制分页命令；它是必需项、不计正文页数、最多 1 页。你目前接在 Conclusion 后的做法可以保留。
- 若希望正文与声明更易区分，可在 Conclusion 后使用 `\clearpage`，这是排版选择，不是官方硬性要求；正文内容仍必须全部留在第 9 页以内。
- Ethics statement 与 Reproducibility statement 为推荐项，不计正文页数。复现声明很适合本稿，可指向附录 A–H 和实际提供的匿名代码；不能声称已经公开并不存在的材料。
- 未发现 `??` 型未解析引用。未见大面积文字重叠或裁切；图 1 和图 6 字号较小，建议提高可读性。

官方依据：[Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)、[2027 官方模板源码](https://raw.githubusercontent.com/ICLR/Master-Template/master/iclr2027/iclr2027_conference.tex)、[官方模板 ZIP](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)。

可选结构：

```latex
\input{sections/conclusion} % 正文至此必须不超过 9 页

% \clearpage % 可选；不是硬性规定
\subsection*{AI use statement}
...

\subsection*{Reproducibility statement}
...

\bibliography{references,legacy_references}
\bibliographystyle{iclr2027_conference}

\clearpage % 可选，让附录从新页开始
\appendix
\input{appendix/rom_derivation}
```

## 2. 必须优先修正的文字、标点和引用格式

以下是明确发现的问题，不把 PDF 文本提取时丢失的所有空格误判为源稿错误。表内引用前缺空格等问题已与源码交叉核对。

|位置（PDF）|现有文字/问题|建议修改|
|---|---|---|
|第 1 页摘要|`fusion..`|`fusion.`|
|第 1 页摘要|`routing further improve prediction`|`routing further improves prediction`；如果效果主要来自融合，可写 `routing and fusion further improve prediction`。|
|第 1 页 Introduction|`However,projection-based`|`However, projection-based`|
|第 1 页 Introduction|`prediction.\citep{amsallem2012}.`|`prediction \citep{amsallem2012}.`；引文应置于该句末尾句号之前。|
|第 1 页 Introduction|`interpretability\citep{...}`|`interpretability \citep{...}`|
|第 2 页方法概述|`each specialist, use`|`each specialist uses`，去掉主谓之间逗号。|
|第 2 页概述及贡献点|`projected-based ROM`|`projection-based ROM`，两处统一。|
|第 2 页方法概述|`sparsely structured MOEs`|统一成 `a structured sparse MoE correction`。|
|第 2 页方法概述|`branches derived from the projection-based.`|句子残缺；若这些分支是可训练结构，可写 `branches motivated by the polynomial structure of the projected dynamics.`，不要未经核实声称它们直接由投影推导。|
|第 3 页 Related Work|`evolution\citep{lu2021,kovachki2023}`|`evolution \citep{lu2021,kovachki2023}`|
|第 3 页 Related Work|`Recent work ... More recently ...`|相邻两句重复引入时间关系，可合并。|
|第 3 页 Related Work|`applies router at two levels`|`applies routing at two levels`|
|第 4 页 §3.1|`parameters mu=Re`|`the parameter is mu=Re`，或 `we set mu=Re`。|
|第 4 页式 (2) 后|句号后接小写 `while`|改成独立句 `Here, ...`；不要留下句子片段。|
|第 4 页式 (2) 后|`The subscript h denote`|`The subscript h denotes`|
|第 4 页式 (2) 后|`represent th viscous`|`represent the viscous`|
|第 4 页 POD 段落|`separate velocity and pressure basis`|`separate velocity and pressure bases`|
|第 4 页式 (4) 后|`pressure poisson equation(PPE)`|`pressure–Poisson equation (PPE)`；Poisson 首字母大写，括号前加空格。|
|第 5 页式 (13) 后|`is PPE equation defined in Appendix B; Both terms`|`is the reduced PPE map defined in Appendix B. Both terms ...`，或分号后用小写 `both`。|
|第 6 页式 (16) 后|`a-hat ... denote ... whereas a ... symbols denote`|改为 `The hatted coefficients denote predictions, whereas the unhatted coefficients denote reference values.`，同时解释速度和压力。|
|第 6 页 §3.4|`Both specialist encode`|`Both specialists encode`|
|第 7 页首段|`four part`|`four parts`|
|第 7 页首段及小标题|`local specialists prediction`|`local-specialist prediction`，或 `prediction accuracy of local specialists`。|
|第 7 页首段|`invariants ,`|`invariants,`|
|第 7 页首段|`Three test-case configurations: ... , are summarized`|`The three benchmark configurations—...—are summarized ...`，或删除不恰当的冒号。|
|第 7 页 §4.1 首句|`We frist comparison`|`We first compare`|
|第 7 页 §4.1|`Table 1 show local specialists achieves`|`Table 1 shows that the local specialists achieve`|
|第 7 页 §4.1|`POD-Garlerkin`|`POD--Galerkin`（LaTeX 源码）|
|第 7 页末段|`local specialist accuracy :`|建议重写为 `the effectiveness of the local specialists: ...`，删除冒号前空格。|
|第 8 页 §4.2 标题|标题末尾独有句号|删除末尾句号，与其他 subsection 标题一致。|
|第 8 页七种策略说明|`each candidate Train-fixed`|`each candidate. Train-fixed`，缺少断句句号。|
|第 8 页 Table 3|caption 无末尾句号|补句号并统一所有 caption；这是统一格式建议。|
|第 9 页 §4.4|`comparison .`|`comparison.`|
|第 12 页 References 与 Appendix A 之间|单独的 `z`|删除；对应 `appendix/rom_derivation.tex` 第 1 行。|
|第 12 页 Appendix A.1|源码 `system，as`|替换中文全角逗号，建议整句写为 `yields the semi-discrete system in Eq.~(...), governing ...`。|
|第 14 页 Appendix B.2|`Substituting equation 32 into equation 40`|用自动交叉引用 `Substituting Eq.~(...) into Eq.~(...)`，避免修改公式后编号失效。|
|第 17 页 Table 6 caption|`flow past a cylinder case(%)`|`Periodic-regime velocity and pressure errors for flow past a circular cylinder (\%).`|
|第 17 页 Table 6 下方|`Table 2 shows ... velocity and pressure errors`|此处应指 **Table 6**；Table 2 展示的是联合误差。|
|第 17 页基线比较|`Lower errors than structured model`|`Lower errors than those of the Structured model`|
|第 18 页 Appendix F.1 首段|`assembly comparison in Section 3.4`|实验比较在 **Section 4.2**；3.4 是方法定义。|
|第 18 页 Table 8 caption|`K=24.Reduction`|`K=24. Reduction`，句号后补空格。|
|第 19 页 Figure 2 caption|`Effect of history-conditioned`|`Effect of history conditioning`|
|第 20 页 Figure 3 caption|`The Square case is set Re=100`|`The Square case uses Re=100` 或 `is evaluated at Re=100`。|
|第 20 页 Figure 3 caption|`96),while Pinball at`|`96), while the Pinball case uses ...`，补空格和谓语。|
|第 21 页 Figure 4 caption|`Circular case is set at`|`The Circular case is evaluated at`|
|第 21 页 Figure 4 caption|`598).Square`|`598). Square`|
|第 22 页 F.4 段落标题|`Square case :Evaluation`|`Square case: evaluation`，去掉冒号前空格、补后空格。|
|第 26 页 H.3|`On an RTX 4090 with a Xeon ... host, give a median`|`On an RTX 4090 with a Xeon ... host, the measured prediction time has a median of ...`，原句缺主语。|

其他统一性建议：`MoE/MOEs`、`ql-ROM/QL-ROM`、`Local-Specialist/Local Specialist/Proposed`、`Periodic/periodic` 应分别统一。Regime 名称作专用模型名时可以大写，描述普通动力学状态时用小写。标题 `for Regime-Adaptive Reduced-Order Model` 更自然地写为 `for Regime-Adaptive Reduced-Order Modeling`。

## 3. 参考文献核验结论

实际引用集合为 36 篇；源码的活动引用均能在 BibTeX 数据库找到定义，未发现重复 BibTeX 键导致的冲突。全部 36 篇均找到 DOI 注册记录、出版社、会议官网、JMLR 或作者论文记录，未发现无法检索的虚构文献。这里核验的是文献存在性与书目信息，不代表逐篇重读并确认每一处引用都充分支持邻近论断。

需要处理或说明的项目：

1. **Wang et al. (2025) 页码冲突不是已确认的原稿错误。** 原稿 `31498--31527` 与 [NeurIPS 官方 BibTeX](https://proceedings.neurips.cc/paper_files/paper/2025/file/2d23a9991a6f64482bf395628e279f5f-Bibtex-Conference.bib) 一致；Crossref 的相同 DOI 返回 `35566–35595`。建议以会议官方记录为准，保留现有页码。
2. 该条作者 `huanshuo dong` 的小写来自会议元数据；作者[公开代码仓库](https://github.com/haiyangxin/MoEPOT)使用 `Huanshuo Dong`，建议将 `.bib` 中 `dong, huanshuo` 规范为 `Dong, Huanshuo`。
3. Pan & Xiao (2024) 的作者、题名、519 卷、113452 文章号可检索且匹配，建议补 DOI `10.1016/j.jcp.2024.113452`，方便访问；没有 DOI 字段本身不是格式违规。[出版社记录](https://www.sciencedirect.com/science/article/abs/pii/S0021999124007009)。
4. FNO 的 ICLR 2021 年份正确，尽管 arXiv 首发于 2020；建议使用正式会议 [OpenReview 链接](https://openreview.net/forum?id=c8P9NQVtmnO)。数据库中 `li2021` 和 `li2021fourier` 描述同一论文，但当前活动引用只用后者，不是 PDF 内重复引用。
5. `li2021cluster` 的 2021 年正确：出版社卷期为 JFM 906，2021-01-10，A21；2020 是上线日期。[出版社记录](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/clusterbased-network-model/8252D04A5438ED01E624A7C41CCF81BB)。
6. Sirovich (1987) DOI 标题记录为 `Turbulence and the dynamics of coherent structures. I. Coherent structures`。现有 `Part I:` 不妨碍识别，建议与出版社标题一致。
7. 给专有名词加 BibTeX 大括号保护：`{Fourier}`、`{Galerkin}`、`{PDE}`、`{ql-ROM}` 等，避免样式自动变成 `fourier`、`galerkin`、`pde`、`ql-rom`。标题普通词采用 sentence case 是正常现象，不需要全部改成大写。
8. ICLR 2024/2026 条目中的会议卷号及较大页码来自官方 BibTeX，可以保留；不要因为看起来不像常规期刊页码就判为错误。ICLR 2026 这篇文献确有官方会议记录。

## 4. 比标点更值得优先检查的内部一致性

### 4.1 图 1 与路由定义不一致

图 1(a) 显示 Physical history 指向 Parameter router，但正文 §3.4、式 (17) 和图注称其为 parameter-only router。应确认图中连线与真实实现一致：历史描述量应进入融合门控，而不是误画成外层参数路由器的输入。

同图使用 `Feature concatenation` 标签，而式 (19) 描述的是重构物理场的加权融合。建议改为 `History-conditioned physical-space fusion`，并标清两个 specialist 的重构输出及权重 alpha；不要让审稿人误以为是在拼接 latent features。

### 4.2 正文承诺的复现细节没有完整给出

- 第 6 页称 Appendix D 提供 benchmark-specific time-stepping schemes；当前 Appendix D（15–16 页）主要给出数据、优化设置与网络描述，没有逐 specialist 列出积分器、子步数及压力更新时间。
- 第 6 页末称 Appendix C 给出 physical-field training loss；当前 Appendix C（14–15 页）给出 descriptor、门控结构和融合公式，但没有明确训练损失公式、速度/压力权重及 gate 优化配置。
- Appendix D 使用宽度 h、残差块数 B 等符号，但未完整列出所有 specialist 的具体值、组数和路由温度。需要与实际配置补齐，不能由编辑者猜填。

### 4.3 表格应能独立解释

- Table 1、2、3 应标明误差以百分数计；Table 2、3 caption 应明确为 `E_joint = E_u + E_p` 和全窗口聚合口径。
- Table 2、6 的 `±` 应明确是跨 seed 标准差、跨窗口标准差还是其他量，并给出对应样本数量。Table 6 只对部分基线说明了三次训练，不足以解释所有行。
- Table 1 的 Steady POD–Galerkin 是 `--`，需说明是未评估、无结果还是其他原因；不能默认表示发散。正文“consistently reduce”应限定在有基线数值的设置。
- 第 7 页正文写 Periodic-core，Table 1 对应行写 Periodic (K=56)，需统一术语或解释 core 的定义。
- 第 6 页称 1–24–3 为 three-layer MLP，容易与“3 个线性层”混淆。更清楚的说法是 `an MLP with one hidden layer of width 24`，总参数 123 与此结构相符。

### 4.4 排版可读性

- Table 11 浮动到第 25 页顶部、出现在 Appendix H 之前；建议使它靠近 H.1 的首次讨论位置。
- 图 3、4 的浮动页插在式 (54) 与 (55) 之间。不是模板违规，但打断推导；可在 F.3 开始前局部清理前文浮动体。
- 图 1 是位图，文字偏小；建议改用矢量 PDF。图 6 的 expert 标签也宜适当放大。
- 不应通过缩小正文、修改行距或页边距来压缩篇幅；保留官方样式。

## 5. AI 使用声明

现在的 `used solely to assist with language polishing and LaTeX formatting` 与本次对话中已发生的结论起草、参考文献检索和审阅辅助不完全相符。应按实际使用范围修改，并与 OpenReview 提交表一致。

官方区分 required 与 recommended disclosure：解释研究结果、设计/反馈方法和实验、实现方法等属于 required；起草论文、检索信息、查找文献、改善可读性和格式参考文献等属于 recommended。该声明本身是必需的。[AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)。

下面仅为当前对话可支持的范围草稿，须在作者完成实际核验后使用；若还用 AI 写实验代码、设计实验或解释结果，应如实补充，不能用此段代替完整披露。

```latex
\subsection*{AI use statement}
Generative AI tools, including ChatGPT and Codex, were used to assist
with drafting and revising manuscript text, literature search and
bibliographic checking, and language and LaTeX formatting review.
The authors reviewed and verified the AI-assisted text and references
and take responsibility for the final manuscript, scientific claims,
experimental results, and associated artifacts.
```

## 6. 提交前最后动作

先统一源码与 PDF 版本，再修正本报告的明确错误，补齐 gate loss / 积分器 / 配置说明，更新真实 AI 披露，最后重新编译并检查第 9 页正文边界和引用编号。当前 (7).pdf 的 Conclusion 与 (5) 源码不同，不能直接把二者当作一套可复现提交文件。

官方全文截止为 2026-09-25 23:59 AoE，即北京时间 **2026-09-26 19:59**。摘要截止已过；还应在 OpenReview 检查作者资料和适用的 reciprocal-reviewer 要求。[官方日期页](https://iclr.cc/Conferences/2027/Dates)、[Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers)。

本次是投稿文本、格式、引用和文献可检索性检查，未重跑实验，也未对全部数学推导作独立证明。

## 7. 逐条文献检索记录

下表仅包括实际引用的 36 篇，不把未引用的旧 BibTeX 库条目混入投稿审查范围。

|BibTeX key|题名|检索依据|结果|
|---|---|---|---|
|ICLR2024_9aeda582|Scaling physics-informed hard constraints with mixture-of-experts|[官方/论文记录](https://proceedings.iclr.cc/paper_files/paper/2024/file/9aeda582add763c41c7b39691ce19ab0-Paper-Conference.pdf)|已找到；主要书目信息匹配|
|ICLR2026_754612bd|Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning|[官方/论文记录](https://proceedings.iclr.cc/paper_files/paper/2026/file/754612bde73a8b65ad8743f1f6d8ddf6-Paper-Conference.pdf)|已找到；主要书目信息匹配|
|NEURIPS2024_b83fae17|Neural Experts: Mixture of Experts for Implicit Neural Representations|[DOI 注册记录](https://doi.org/10.52202/079017-3224)|已找到；主要书目信息匹配|
|NEURIPS2025_2d23a999|Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training|[DOI 注册记录](https://doi.org/10.52202/085713-1057)|可检索；官方 BibTeX 与 Crossref 页码不同，保留官方页码|
|amsallem2008|Interpolation method for adapting reduced-order models and application to aeroelasticity|[DOI 注册记录](https://doi.org/10.2514/1.35374)|已找到；主要书目信息匹配|
|amsallem2012|Stabilization of projection-based reduced-order models|[DOI 注册记录](https://doi.org/10.1002/nme.4274)|已找到；主要书目信息匹配|
|benner2015|A survey of projection-based model reduction methods for parametric dynamical systems|[DOI 注册记录](https://doi.org/10.1137/130932715)|已找到；主要书目信息匹配|
|berkooz1993|The proper orthogonal decomposition in the analysis of turbulent flows|[DOI 注册记录](https://doi.org/10.1146/annurev.fl.25.010193.002543)|已找到；主要书目信息匹配|
|bhat2025|Error-based efficient parameter space partitioning for mesh adaptation and local reduced order models|[DOI 注册记录](https://doi.org/10.1016/j.cma.2024.117649)|已找到；主要书目信息匹配|
|brunton2020|Machine learning for fluid mechanics|[DOI 注册记录](https://doi.org/10.1146/annurev-fluid-010719-060214)|已找到；主要书目信息匹配|
|champion2019|Data-driven discovery of coordinates and governing equations|[DOI 注册记录](https://doi.org/10.1073/pnas.1906995116)|已找到；主要书目信息匹配|
|cho2014|Learning Phrase Representations using RNN Encoder--Decoder for Statistical Machine Translation|[DOI 注册记录](https://doi.org/10.3115/v1/D14-1179)|已找到；主要书目信息匹配|
|colanera2025|Quantized local reduced-order modeling in time (ql-ROM)|[DOI 注册记录](https://doi.org/10.1016/j.cma.2025.118393)|可检索；保护 ql-ROM 大小写|
|dar2023|Artificial neural network based correction for reduced order models in computational fluid mechanics|[DOI 注册记录](https://doi.org/10.1016/j.cma.2023.116232)|已找到；主要书目信息匹配|
|diaz2024|A fast and accurate domain decomposition nonlinear manifold reduced order model|[DOI 注册记录](https://doi.org/10.1016/j.cma.2024.116943)|已找到；主要书目信息匹配|
|duraisamy2019|Turbulence modeling in the age of data|[DOI 注册记录](https://doi.org/10.1146/annurev-fluid-010518-040547)|已找到；主要书目信息匹配|
|fedus2022|Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity|[官方/论文记录](https://jmlr.org/papers/v23/21-0998.html)|已找到；主要书目信息匹配|
|fresca2022|POD-DL-ROM: Enhancing deep learning-based reduced order models for nonlinear parametrized PDEs by proper orthogonal decomposition|[DOI 注册记录](https://doi.org/10.1016/j.cma.2021.114181)|已找到；主要书目信息匹配|
|geelen2023|Operator inference for non-intrusive model reduction with quadratic manifolds|[DOI 注册记录](https://doi.org/10.1016/j.cma.2022.115717)|已找到；主要书目信息匹配|
|hesthaven2018|Non-intrusive reduced order modeling of nonlinear problems using neural networks|[DOI 注册记录](https://doi.org/10.1016/j.jcp.2018.02.037)|已找到；主要书目信息匹配|
|hochreiter1997|Long Short-Term Memory|[DOI 注册记录](https://doi.org/10.1162/neco.1997.9.8.1735)|已找到；主要书目信息匹配|
|jacobs1991|Adaptive mixtures of local experts|[DOI 注册记录](https://doi.org/10.1162/neco.1991.3.1.79)|已找到；主要书目信息匹配|
|jordan1994|Hierarchical mixtures of experts and the EM algorithm|[DOI 注册记录](https://doi.org/10.1162/neco.1994.6.2.181)|已找到；主要书目信息匹配|
|kaiser2014|Cluster-based reduced-order modelling of a mixing layer|[DOI 注册记录](https://doi.org/10.1017/jfm.2014.355)|已找到；主要书目信息匹配|
|kovachki2023|Neural operator: Learning maps between function spaces with applications to PDEs|[官方/论文记录](https://jmlr.org/papers/v24/21-1524.html)|已找到；主要书目信息匹配|
|lee2020|Model reduction of dynamical systems on nonlinear manifolds using deep convolutional autoencoders|[DOI 注册记录](https://doi.org/10.1016/j.jcp.2019.108973)|已找到；主要书目信息匹配|
|li2021cluster|Cluster-based network model|[DOI 注册记录](https://doi.org/10.1017/jfm.2020.785)|可检索；2021 卷期正确|
|li2021fourier|Fourier Neural Operator for Parametric Partial Differential Equations|[官方/论文记录](https://arxiv.org/abs/2010.08895)|可检索；ICLR 2021 正确，建议用正式会议链接|
|lu2021|Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators|[DOI 注册记录](https://doi.org/10.1038/s42256-021-00302-5)|已找到；主要书目信息匹配|
|pan2024domain|Domain decomposition for physics-data combined neural network based parametric reduced order modelling|[官方/论文记录](https://doi.org/10.1016/j.jcp.2024.113452)|可检索；建议补 DOI|
|park2024|tLaSDI: Thermodynamics-informed latent space dynamics identification|[DOI 注册记录](https://doi.org/10.1016/j.cma.2024.117144)|已找到；主要书目信息匹配|
|peherstorfer2014|Localized discrete empirical interpolation method|[DOI 注册记录](https://doi.org/10.1137/130924408)|已找到；主要书目信息匹配|
|peherstorfer2016|Data-driven operator inference for nonintrusive projection-based model reduction|[DOI 注册记录](https://doi.org/10.1016/j.cma.2016.03.025)|已找到；主要书目信息匹配|
|rahman2019|Nonintrusive reduced order modeling framework for quasigeostrophic turbulence|[DOI 注册记录](https://doi.org/10.1103/PhysRevE.100.053306)|已找到；主要书目信息匹配|
|shazeer2017|Outrageously large neural networks: The sparsely-gated mixture-of-experts layer|[官方/论文记录](https://openreview.net/forum?id=B1ckMDqlg)|已找到；主要书目信息匹配|
|sirovich1987|Turbulence and the dynamics of coherent structures. Part I: Coherent structures|[DOI 注册记录](https://doi.org/10.1090/qam/910462)|可检索；题名标点可与注册记录统一|
