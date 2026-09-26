# ICLR Introduction 与 Method 强化修改报告

日期：2026-09-09

## 1. 结果摘要

本轮已根据 `ICLR_vs_CMAME_method_intro_revision_codex.md` 完成本地论文修改、数学核对、版面检查和独立上传包编译。保留“短主文 + 完整 Appendix”结构，未回退到 CMAME 的长推导主文。

| 检查项 | 本轮结果 |
|---|---|
| Introduction 首句明确 ROM identity | Yes，使用 “Learning reduced-order dynamical models” |
| representation / dynamics bridge | 已补充 |
| complete local chart 概念 | 已补充 |
| compact pressure equation | 主文 Eq. (4) |
| compact K-step rollout loss | 主文 Eq. (5) |
| admissibility 最小定义 | 已补充，且明确不属于 E2 概率模型 |
| 方法定义是否改变 | **No** |
| 实验数值是否改变 | **No** |
| 最新流程图是否改变 | **No**，原文件及 PDF 内嵌像素均校验一致 |
| 主文页数 | **9 页**，含 Conclusion、Reproducibility 和 AI use statements |
| 全文页数 | **23 页**，含 References 与 Appendices A-J |
| 交叉引用 / 引文 / 缺失字形 | 未发现未解决问题 |
| 本轮 Overleaf 同步 | **尚未同步**；本报告及打包结果对应本地新版本 |

此处页数是当前项目模板下的编译结果，不等同于对会议最终投稿政策的独立认证。本轮未检索或修改会议政策、模板、页边距、字号和投稿声明。

## 2. 对照材料与修改范围

执行依据为用户本轮提供的 Markdown。CMAME 对照材料使用 Downloads 中的 `PDCML_FVM_Pan (7).pdf`，重点核对原稿方法相关页面与当前完整附录。Markdown 所列 UUID 文件名 `97f23fde-a916-4623-9823-45633d2927f0.pdf` 未在 Downloads 找到；当前 ICLR 基线使用本轮修改前已验证的 `build/pdf/main_user_figure_20260908.pdf`，没有假定其与缺失 UUID 文件逐字一致。

修改前的全部 31 个论文引用文件已保存到 `build/before_method_intro_20260909/`，并与上一轮上传清单逐项核验 SHA-256。

实际修改的论文文件只有以下五个：

1. `sections/introduction.tex`：首句、ROM bridge 和 Figure 1 图注压缩。
2. `sections/method.tex`：方法路线说明、局部 chart、两条核心公式、符号说明和 admissibility。
3. `sections/experiments.tex`：压缩重复数字叙述及两处图表说明，不改变实验表格和结论限定。
4. `appendix/specialist_details.tex`：压力 gate 符号统一，补清 channel-wise MoE 的职责。
5. `appendix/routing_details.tex`：history 和 T2-C correction 符号对齐。

`main.tex`、Related Work、Limitations、Conclusion、参考文献、模板、其他附录和所有图像文件均未修改。另新增本报告与专项验证脚本 `scripts/verify_method_intro_20260909.py`；这些审核文件不进入论文上传 ZIP。

## 3. Introduction：轻量增强 ROM 身份与逻辑连接

### 3.1 开头身份锚定

原首句中的 “Learning compact dynamical models” 改为：

> Learning reduced-order dynamical models is central to scientific machine learning when high-fidelity simulation is too expensive for repeated evaluation.

ROM identity 的答案是 **Yes**：现在第一句即可辨认本文研究 reduced-order dynamics，同时保留科学机器学习和 many-query 的问题背景。不是声称旧稿完全未提 ROM，而是把已有身份提前。

### 3.2 representation / dynamics bridge

在 POD 与 Galerkin 句后补充：

> A projection-based ROM couples a low-dimensional state representation with a reduced dynamical model that advances its coordinates.

随后接截断相互作用、closure 与长时滚动问题。这使前段的“两类困难”与投影 ROM 的“表示 + 演化”自然对应。

### 3.3 保留的叙事主线

保留原有问题、表示与动力学失配、已有方法的互补作用、局部化后的不相容坐标拼接问题、两级方法、三项贡献及实验范围。不恢复长 taxonomy，不新增 FOM、POD、pressure gauge 或 RK4 推导。Related Work 仍独立保留。

## 4. Section 3：主文核心接口补全

仍为四个小节，没有新增 3.5 或 3.6。

### 4.1 路线说明：三个操作，不是三级 routing

开头明确三项操作：独立的 overlapping local charts、每个 chart 上的完整 autonomous specialist、外层对独立重构物理场的条件选择与组合。

特意采用 “three operations within a two-level routing hierarchy”，避免把 representation、specialist、assembly 错读为三个 routing levels。内层仍为 chart 内的 sparse experts；外层仍为 complete specialists。正文指向 Figure 1 和 Appendices A-D。

### 4.2 Section 3.1：complete chart 与不相容坐标

保留一个 encoding / reconstruction 公式。将旧主文的 `W_u`、下标式基与均值记号对齐为附录的 `M_u`、`Phi_u^(r)`、`bar u^(r)`，并解释 `M_u` 为 finite-volume quadrature matrix。

chart 包含参数区域、仿射均值、速度与 gauge-fixed 压力基、local normalization statistics、projected operators。明确这些量分别构造，即使 reduced dimensions 相同也不意味着坐标分量对应。因此跨 regime 融合推迟到 physical space。

### 4.3 Section 3.2：完整 specialist 与压力闭合

速度仍由 projected Galerkin backbone 加 learned closure 演化。structured sparse MoE 仍由 nonlinear、affine、low-rank bilinear 分支构成；原来挤在一行的两项表达式改为对齐的两行，共用 Eq. (3) 编号，没有增加或删去分支。

补充 group router 所选 family `g*`、速度与压力各自的 Top-2 子集、所选 group 内 shared expert，以及 active weights 非负且和为一的解释。补清 `M^c` 参数化 velocity closure `rho^u` 和 data-driven pressure candidate `rho^p`；没有将压力候选改写成与物理压力直接相加的 residual，也没有把压力 gate 本身重新定义为该候选 MoE。

新增主文 Eq. (4)：

\[
\mathbf b_{n+1}^{(r)}
=\boldsymbol\gamma_n^{(r)}\odot
\mathcal Q_r^{PP}(\mathbf a_{n+1}^{(r)};\mu)
+(\mathbf1-\boldsymbol\gamma_n^{(r)})\odot
\boldsymbol\rho^p_{\theta_r}(\boldsymbol\xi_n^{(r)}).
\]

它是现有 Appendix C 的候选与混合式直接代入后的紧凑形式，不是新算法：

- physics-based pressure candidate 读取更新后的 `a_(n+1)`；
- learned pressure candidate 和 componentwise gate 读取当前 `xi_n`；
- 压力在速度 macro-step 后代数重构，不成为另一个独立积分的 recurrent ODE；
- 原 RK4 frozen-context 实现仍完整保留在 Appendix C。

### 4.4 Section 3.3：自主递推训练目标

新增主文 Eq. (5)，与 Appendix C Eq. (39) 等价：

\[
\mathcal L_r=\frac1K\sum_{k=1}^{K}\left[
\frac{\| (\widehat{\mathbf a}_{n+k}^{(r)}-\mathbf a_{n+k}^{(r)})
\oslash\boldsymbol\sigma_a^{(r)}\|_2^2}{d_u}
+\frac{\| (\widehat{\mathbf b}_{n+k}^{(r)}-\mathbf b_{n+k}^{(r)})
\oslash\boldsymbol\sigma_b^{(r)}\|_2^2}{d_p}\right].
\]

明确 `d_u,d_p` 是局部维数，scaling vectors 来自各 chart 的训练数据，`oslash` 为逐分量除法。初始化后预测窗口内不注入 reference state；后续输入来自自身预测。

没有新增 pressure 权重、换成物理场 relative error、加入 teacher forcing 或改变 curriculum。积分器与 horizon curriculum 保留在 Appendix C，数据划分继续指向 Table 5。

### 4.5 Section 3.4：admissibility 与外层组装

补充二元 mask `m_r(H_n,mu,K)`：只限制 validation-supported 参数、历史与 horizon 对应的候选集合，不改变 parameter-only E2 概率模型。全零 mask 不声称提供 ROM prediction；没有引入新置信度阈值、新 fallback 或新的稳定性保证。

保留并明确：E2 一次给出固定的轨迹级偏好；T2-C 使用 physical-history descriptor；只允许 S-H 或 H-P 相邻对；两个 specialist 从同一历史分别编码并独立 rollout；重构后在共同物理空间融合。`q` 指共同网格上的速度与 gauge-fixed 压力，标量 `alpha` 对两种场共用且窗口内固定，融合结果不反馈给任何 local rollout。

## 5. 符号统一与数学一致性

| 记号 | 统一后职责 | 核对结果 |
|---|---|---|
| 向量 `gamma_n^(r)` | specialist 压力的逐分量 gate | 主文与 Appendix C 一致；仅替换附录四处旧向量 alpha |
| 标量 `alpha` | T2-C physical-space convex weight | 保留；窗口固定，速度/压力共用 |
| `Delta_psi = c_psi(...)` | T2-C logit correction | 附录旧 `delta, c_vartheta` 对齐主文，不改变网络 |
| `M_u, Phi_u^(r), bar u^(r)` | chart 编码/重构 | 与附录加权 POD 记号一致 |
| `T_2^c` | 通道各自的两名 routed experts | 与附录一致，主文省略函数参数依赖 |
| `H_n` | 初始化时的物理历史 | 主文明确绑定；附录 mask 参数同步 |

主文 Eq. (6) 保留概念级 logit 写法；Appendix D 的 epsilon-stabilized pair logit 仍是实际数值稳定化定义，未删去 epsilon。完整 expert 索引、矩阵维度、operator tensor 条目和 descriptor 分量仍由附录给出，不在主文重复展开。

未发现新引入且未说明的关键符号冲突。所有显式正文/附录引用均可解析，未发现 undefined references、undefined citations、重复 label 或缺失字形警告。

## 6. 保留在 Appendix 的完整推导

没有删减任何附录。全部现有 displayed equations 在允许的符号改名归一化后与基线逐项一致。

- Appendix A：semi-discrete full-order problem、压力 gauge、regime cover、area-weighted POD 与编码/重构。
- Appendix B：chart-local Galerkin tensors、Pressure-Poisson map 及独立坐标说明。
- Appendix C：压力候选及混合式、feature history、group/channel routing、structured expert、完整 frozen-context RK4 stages、闭环 loss 与训练 curriculum。
- Appendix D：adjacency matrix、masked preferences、pair logit、六分量 chart-independent descriptor、T2-C gate、训练目标与二次误差展开。
- Appendices E-J：实现与 split 协议、全部 benchmark results、内部路由诊断和参数统计均保留。

`appendix/rom_derivation.tex` 与原备份字节一致。Appendix C/D 只进行前述改名及解释补充；其余附录文件全部字节一致。

## 7. 实验与 Figure 1 的保护性检查

实验仍按三个 research questions 组织，没有改回逐 benchmark 叙事。为容纳方法内容，删除了已经出现在表中的部分数字复述，缩短 Table 1 caption 和 Figure 2 caption。

验证结果：4 个 `tabular` 与 2 个 `tablenotes` 和基线逐字符一致；实验文件数字字面量的去重集合一致。删除的正文数字都仍在相应表格中。未改写任何实验结果，也未重新运行、替换或推算实验数据。

保留关键限定：Hopf POD-Galerkin 的 numerically finite 窗口统计口径及 threshold violations；N/E 原因；best single 与 convex oracle 是事后参考；DeepONet cadence 不匹配；localization 不保证逐参数更优；内部 expert 利用情况不意味着物理机制归属。

Figure 1 使用用户 2026-09-08 最新图，未重绘、裁切、压缩或重新生成。图注更短，但仍解释 history-to-assembly 是 descriptor 路径而非 E2 输入、独立重构后融合、无反馈、shared/Top-2 并行、压力辅助 feature connection 省略及 thumbnails 是示意而非实验结果。

## 8. 方法增量与最终版面

| 度量 | 修改前 | 修改后 |
|---|---:|---:|
| `method.tex` 源码行数 | 52 | 112 |
| 方法主文小节数 | 4 | 4 |
| 主文 display equation 编号数 | 4 | 6 |
| 方法所在 PDF 页 | 4-5 | 4-5 |
| 扣除穿插浮动表格后的方法版心高度当量 | 约 1.09 页 | 约 1.44 页 |
| 主文与两项声明完成页 | 9 | 9 |
| References 起始页 | 9 | 9 |
| 含 References/Appendix 总页数 | 23 | 23 |

源码净增加 60 行，包含为可维护性增加的断行，不能当作 PDF 新增 60 行。版面用模板的 648 pt 正文高度折算，并从旧稿 Section 3 跨页范围扣除穿插的 Table 1 及其间距约 157.06 pt；最终方法没有被实验浮动表格打断，净增约 **0.35 个版心页**。

因此，实际量测没有达到意见中粗估的“1.8-2.1 页方法 / 增长 0.4-0.6 页”。这不是把两条核心公式删掉来缩页，而是当前真实基线更短，新增内容紧凑排入了两张纸面的部分区域。所有要求的主文接口已补全，本轮没有为凑页数新增推导、放大间距或恢复 CMAME 长论述。该差异在此显式记录。

主文末尾 References 的纵坐标从约 561.03 pt 变为 599.45 pt，仍在第 9 页。主文与实验图片没有缩小字号或图像尺寸。模板、边距和主文件未改动。

## 9. 编译与独立验证证据

本地使用项目既有 Tectonic 0.17.0 缓存编译链编译 `main.tex`。随后将 31 个递归引用源文件/资产打包，解压到全新目录，并从上传包再次独立编译。

专项自动验证 **197 项通过**，包括：

- 基线备份与历史 manifest 匹配；只有五个范围内论文文件修改；
- 主文 pressure 时间索引检查；主文 loss 与附录 loss 规范化后相等；
- 所有附录 display equations 在符号归一化后保留；
- 所有正文交叉引用解析、主要 equation/appendix 编号核验；
- Figure 1 源 PNG 字节及 PDF 内嵌像素与用户文件一致；
- 上传包文件与本地文件逐项一致；
- 独立包编译得到相同 23 页，逐页文本、page content stream、渲染 PNG 完全一致。

已查看全文缩略页，并重点检查主文第 3-9 页及变动附录第 13-15 页，没有发现裁切、重叠、丢图或公式编号挤出正文区域。参考文献和部分附录的既有浮动留白未做无关重排。

不是“零 warning”编译：最终日志有 22 条 underfull box 排版消息和 1 条 `h` 改为 `ht` 的浮动位置警告，另有本地 Fontconfig 默认配置环境提示。编译返回成功，未见 overfull box、缺字、未解析引用或引文。上述非致命消息不等同于数学或数据错误。

## 10. 交付物、恢复与同步状态

- 可阅读稿：`build/pdf/main_method_intro_revision_20260909.pdf`
- Overleaf 源码包：`build/overleaf_upload_method_intro_20260909.zip`，31 个文件，根目录含 `main.tex`。
- 清单：`build/method_intro_20260909/upload_manifest.json`
- 检查结果：`build/method_intro_20260909/verification.json`
- 修改前备份：`build/before_method_intro_20260909/`
- 本轮报告：`ICLR_METHOD_INTRO_REVISION_REPORT.md`

PDF SHA-256：`5c630cea014c1371fac6d67fd7730e4073318394cdde92f08b78fb97baa8fc57`

ZIP SHA-256：`06369220cb596fb6d260d2531c2d5ba94047a72c61f0ceef1b83542c749ca3ab`

Figure 1 SHA-256：`5f53933dafd79441424e031e4f1cd0da72065b44383224186ac9bcbd02274eff`

本轮没有删除或覆盖任何历史备份，也没有清空线上项目。前一轮最新流程图稿已上传至项目 `6a9936085c574b84a0ab1597`；本轮 Introduction/Method 修订目前只在本地与上述源码包中，**不能把上一轮上传成功当作本轮已经同步**。

若继续同步，只需按原目录更新五个变动的 `.tex` 文件并重新编译，不需要再次清空项目或替换图片。上传包用于提供完整可编译源集合；向既有 Overleaf 项目上传 ZIP 不一定自动展开，须保持目录层级。

## 11. 审稿人首次阅读检查

| 读者需要回答的问题 | Figure 1 + Section 3 的回答位置 |
|---|---|
| 为什么 chart 不可直接混合 coefficients？ | 3.1：独立均值、基、scalers、operators；相同维数不代表相同坐标 |
| 一个 specialist 到底更新什么？ | 3.2：速度微分演化，压力代数闭合，Eq. (2)/(4) |
| routed experts 的作用是什么？ | 3.2：选定 group 内 shared + 通道 Top-2，structured map，Eq. (3) |
| 为什么不是 one-step predictor？ | 3.3：自身预测递推，K-step loss，无窗口内 reference injection |
| E2 和 T2-C 分别读什么？ | 3.4：E2 parameter-only；T2-C 读取 physical-history descriptor |
| 什么情况下组合，在哪里组合？ | 3.4：validation-supported adjacent pair；独立 rollout 后重构到 common physical mesh 再融合，无反馈 |

最终方法的核心叙事保持：chart 内条件 closure，与不相容 charts 之间完整动力学模型的条件组装。完整数学与实现验证仍由附录承担。
