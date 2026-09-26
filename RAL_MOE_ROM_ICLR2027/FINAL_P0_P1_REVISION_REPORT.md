# ICLR final P0/P1 revision report

完成日期：2026-09-06。版本：`final_p0_p1_20260906`。

## 1. 完成结果与交付入口

已根据用户提供的 `ICLR_final_P0_P1_codex_tasks.md` 和 2026-09-06 19:01:01 最新流程图完成本地修改、编译、全页视觉检查及上传包独立编译。

本轮工作目录：`C:\Users\panxy1019\Documents\CHANNEL\RAL_MOE_ROM_ICLR2027`。
上一版源文件已完整保存在 `build/before_final_p0_p1_20260906/`；上一轮正式 PDF、ZIP 没有覆盖。

| 交付 | 本目录下的文件 |
|---|---|
| 最终匿名论文，23 页 | `build/pdf/main_final_p0_p1_20260906.pdf` |
| 可独立编译的 Overleaf 上传包，31 个引用文件 | `build/overleaf_upload_final_p0_p1_20260906.zip` |
| 可编辑 Figure 1 矢量绘图源 | `figures/ral_moe_rom_overview.py` |
| 最终 Figure 1，原生 vector PDF | `figures/ral_moe_rom_overview.pdf` |
| 正文更新 | `sections/introduction.tex`, `sections/related_work.tex`, `sections/experiments.tex`, `main.tex` |
| 附录更新 | `appendix/fluidic_pinball.tex`, `appendix/internal_routing.tex` |
| 新增参数附录 J / Table 11 | `appendix/parameter_counts.tex` |
| 文献库 | `references.bib` |
| 参数统计脚本 | `scripts/count_trainable_parameters.py` |
| 参数统计完整记录 / 简表 | `build/parameter_counts.json`, `build/parameter_counts.csv` |
| 文献来源核验记录 | `build/recent_literature_evidence.md` |
| 最终自动检查记录 | `build/final_p0_p1_verification.json` |
| ZIP 文件清单及散列 | `build/final_p0_p1_upload_manifest.json` |

科学正文与 Conclusion 截至第 9 页；AI 声明和 References 在第 9 页开始；数学附录 A 在第 11 页开始；附录 H/I 位于第 21–22 页；参数附录 J 单独位于第 23 页。Figure 1 / Figure 3 分别在第 3 / 8 页。

本轮没有修改在线 Overleaf，也没有同步历史 E 盘副本。请以本报告列出的日期版本为最新交付。

## 2. P0-1/P0-2：Figure 1 的结构和压力语义

### 2.1 最新参考与矢量实现

唯一视觉设计参考为用户最新文件：

`C:\Users\panxy1019\Downloads\ChatGPT Image Sep 6, 2026, 07_01_01 PM.png`

参考图 SHA256：`577f152dd6a5ea28733ca9477455faf1513202226966320e45b124367c1eaf39`。

按照任务文档的 vector 优先要求，最终图不是把该 PNG 再嵌入 PDF，也不是旧图重新裁切，而是使用 ReportLab 原生文字、路径、箭头和形状重建的矢量图。保留最新参考的上下两层布局和紫色路由、蓝色物理历史、橙色 specialist、绿色物理输出的视觉分工；删除装饰性场图以保持全文宽度下的可读性。图中所有文字、框和箭头均可在源脚本中独立编辑。PDF 无嵌入栅格图像。

重画后修正了文字溢出、分支标签碰撞和数学下标的基线位置，并加入文字宽度/高度检查，避免靠缩小全部字体掩盖溢出。

### 2.2 图本身的语义检查

| 检查项 | 结果与实现 |
|---|---|
| E2 的输入只有参数 | `Parameter μ → E2 router (parameter-only) → Preference π` |
| π 是 E2 输出 | 图中列出 S/H/P 三个 preference 分量 |
| history 不进入 E2 | 独立下方历史信息线路，没有通往 E2 的箭头或前置汇合点 |
| history 初始化 native charts | 单独的 `Initialize selected native chart(s)` 分支 |
| descriptor 来自物理历史 | `Physical history → Chart-independent descriptor` |
| descriptor 用于 T2-C | descriptor 接入 T2-C weight/correction，而非 E2 |
| T2-C 仍使用参数 | correction 明示 `Δψ(μ,d_n)`；pair preference 单独标注 |
| 仅允许相邻 Top-2 | 明示 `S-H and H-P pairs only` |
| 各 specialist 独立 rollout | A/B 各有独立 rollout 节点；Top-1 节点标明所选 S/H/P |
| 先重构后融合 | A/B 分别经过 `Physical reconstruction` 后才进入 T2-C |
| 共同物理空间融合 | 明示 `Fusion after reconstruction in common physical space` |
| 无融合状态反馈 | 所有 fusion 箭头仅通向输出，无回到局部状态的箭头 |
| 只积分速度 | `Velocity time integration → Velocity a_{n+1}` |
| 压力为代数重构 | 独立 `Algebraic pressure reconstruction (local features) → Pressure b_{n+1}` |
| 最终局部预测 | a/b 输出合并为 updated local prediction |
| Level-1 稀疏结构 | group router 后的 shared expert 与 Top-2 experts 并行加权求和，不是顺序串联 |
| 专家结构 | nonlinear + affine + low-rank bilinear 保留可见 |
| 禁用措辞 | 无 `key innovation` 或 `single latent chart`；使用 local reduced chart |

下半图展开速度修正路径，压力闭合细节按任务要求保持简洁；它不是声称压力通道没有 MoE。完整压力定义仍由原方法和数学附录给出，未修改。

### 2.3 Caption

Figure 1 caption 现在直接说明两层结构、parameter-only E2、同一物理历史初始化、native reduced coordinates 独立演化以及重构后的 T2-C。已经删除上一版用 caption 解释错误 history-to-E2 连线的文字；图本身不再含该错误连线。

## 3. P0-3/P0-4：Table 2 与 Periodic core

Table 2 使用任务推荐的短 caption：Periodic core 是 periodic regime 内 `K=56` 的 longer-horizon stride-four matched final-test evaluation，并引用 Appendix H。

Appendix H 补齐：每 0.25 时间单位保存快照，stride four 对应有效采样间隔 `Δt=1`，与 Periodic specialist 原生历史 cadence 对齐。该定义来自本轮用户指定 protocol；原始 pinball 源文档也记载快照间隔 0.25 与匹配比较的 `Δt=1`，不存在由误差值反推评估协议的做法。

Hopf 脚注改为：

> Errors are computed for all numerically finite windows; 141 of 320 Hopf windows exceed the predefined rollout-divergence threshold.

Steady 的 448 个窗口均违反预定义判据的说明保留。正文对应的 `finite windows` 也明确为 `numerically finite windows`。这区分了数值有限与通过 rollout-divergence 判据；没有删除 threshold-violating 窗口或重新计算误差。

**Table 2 全部性能数值、K 和原有 divergent-window 数量未改。**

## 4. P0-5：Table 3 的 Best single

列名改为 `Best single*`，脚注明确其为按对应 overlap 的 aggregate evaluation error 事后选出的固定 specialist reference，不是预测时可用的路由策略。S-H 对应 Hopf、H-P 对应 Periodic；不是逐窗口择优。

Convex oracle 继续标注为预测时不可用的 a posteriori convex reference。加粗规则仍只强调最佳可部署 T2-C；Best single 与 Oracle 不加粗。

**Table 3 每一个误差值以及加粗数值均与上一版相同。**

## 5. P0-6：AI use statement

使用当前官方模板的 `\subsection*{AI use statement}` 标题，删除 TODO。最终正文文本为：

> Generative AI tools, including ChatGPT and Codex, assisted with language editing, LaTeX and code refactoring, analysis scripting and interpretation, literature checks, and schematic figure preparation. The authors take full responsibility for reviewing and verifying all scientific content, mathematical formulations, experimental configurations, numerical results, and conclusions, and for the final manuscript and all reported results.

本轮不代替作者宣称他们已经完成全部人工复核，因此采用责任表述，而不是未经确认的 “all results were reviewed by the authors” 完成式声明。源文件中留有非排版注释，提醒作者提交前按最终实际工作流程确认。

已核验 [ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)、[Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) 与 [官方模板](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)：论文需披露 AI 使用，且投稿表单也需披露；具体 boilerplate 不要求逐字一致。当前声明不到一页。未修改会议 style 文件。

**作者提交前仍需人工确认声明覆盖实际 AI 工作范围，并填写投稿系统对应披露字段。** 这不是遗留 TODO，也不是自动代替作者完成确认。

## 6. P1-7：POD–DeepONet 主文结果

Section 4.1 的 Table 2 解释段之后、Section 4.2 之前新增简短对照，保留原 Appendix H / pinball 源文档的四个事实：4,228 held-out one-step pairs，`E_u=1.5921%`，`E_p=23.3447%`，只有 1/28 held-out trajectories 在 recursive rollout 中保持有限。

正文和附录均明示 cadence 与 matched Periodic specialist evaluation 不同。结果仅用于 one-step fitting 与 recursive stability 的对照，不作为严格匹配的长时精度基线；没有计算 relative improvement。

## 7. P1-8：四篇近期 Scientific-ML / PDE MoE 文献

| 工作 | 官方记录 | BibTeX key |
|---|---|---|
| Chalapathi, Du, Krishnapriyan, ICLR 2024, Scaling physics-informed hard constraints with mixture-of-experts | [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9aeda582add763c41c7b39691ce19ab0-Abstract-Conference.html) | `ICLR2024_9aeda582` |
| Ben-Shabat et al., NeurIPS 2024, Neural Experts | [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b83fae17d73b079b1b98ab200276db9f-Abstract-Conference.html) | `NEURIPS2024_b83fae17` |
| Wang et al., NeurIPS 2025, MoE-POT | [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2d23a9991a6f64482bf395628e279f5f-Abstract-Conference.html) | `NEURIPS2025_2d23a999` |
| Li et al., ICLR 2026, Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning / iMOOE | [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html) | `ICLR2026_754612bd` |

四条记录均使用官方 BibTeX 下载内容，重新获取并比较后 4/4 一致；不是根据题目或作者猜写。官方 MoE-POT export 中的小写作者名拼写也原样保留。下载 URL、原始内容 SHA256、原文对应说明见 `build/recent_literature_evidence.md`。

Related Work 以一段综合描述覆盖约束分解、局部隐式表示、共享/路由专家的算子预训练以及参数条件化的 operator fusion。区别落在“完整自主 local ROM specialists 的外层选择、独立坐标与重构后物理融合”，不声称所有已有 MoE 都共享同一 latent space。原有文献条目未删除或改写，全部引用已解析。

## 8. P1-9：准确参数统计与解释

### 8.1 数据与执行边界

本地没有这两个训练检查点的二进制权重，但保留了实际模型源代码、原始 checkpoint 参数配置、完整逐张量 shape/dtype inventory，以及前轮路由复现记录中的 source/checkpoint SHA256。使用这些真实定义与配置在 CPU 实例化相同模型，逐项验证 Hopf 的 592 个状态张量和 Periodic 的 1,414 个状态张量，且源码散列与原始路由审计一致。

这满足“从实际 model definitions/config 统计”，不是从旧论文误差值估算。**本轮没有加载训练权重重放 held-out rollout，也没有重新训练或更改数据。** 合成输入只用于执行路径计数和测试；不产生论文性能值。运行环境为 Python 3.12.14 / CPU PyTorch 2.14.0，依赖仅安装到本项目 `build/parameter_count_runtime/`。

### 8.2 计算定义

- Total：训练配置中 `requires_grad=True` 的模型 parameter objects 按对象身份去重后求元素数。
- Active：一次 batch-size-one Level-1 forward 实际参与调用的 trainable parameter objects 去重求和。不是整个 batch、轨迹或四个 RK4 stage 的参数并集。
- 包括 encoder、refinement、实际执行的所有 routers、选中组 shared experts、u/p Top-2 experts、pressure correction head，以及专家函数直接使用的低秩参数与 residual scales。
- 不包括固定 POD/Galerkin/Poisson 算子、scalers、buffers、冻结参数。对象级去重防止跨通道共享或重复调用被多次相加。
- 实现会计算所有组的 channel router，即使对应组专家未被执行，这些 router 仍计入 active。

### 8.3 最终准确结果

| Circular specialist | Total trainable | 原生 macro-stage1 实际 active | 原生比例 | 关闭诊断栈的 prediction-only active | 纯预测比例 |
|---|---:|---:|---:|---:|---:|
| Hopf | 31,810,392 | 22,953,160 | 72.16% | 14,165,816 | 44.53% |
| Periodic | 48,617,547 | 8,308,959 | 17.09% | 8,308,959 | 17.09% |

Hopf 架构中的单组 outer group router 有 193,755 个参数，但训练定义显式冻结它们，因此不计入 trainable。独立的 training-loss-only normal-form head 有 10 个 trainable scalars、2 个 buffer elements；不属于部署 specialist 的 forward，其 10 个参数在附录另行披露，不混入表中总量。

### 8.4 为什么 Hopf 需要两个 active 数

Hopf 原生首阶段 forward 请求 expert-diversity stack，因此额外执行全部六个 routed velocity experts；压力仍只执行 Top-2。将原生 actual active 直接写成纯预测的 14,165,816 会低估实际执行量。Periodic 的原生预测 evaluator 已关闭该 stack，两个口径相同。

合成 forward 中确认关闭诊断 stack 后，非零速度/压力修正与 pressure gate 输出逐位相同；仅该项测试关闭自己的合成调用选项，**没有修改实际 evaluator 或训练 checkpoint**。

枚举全部 225 个 Hopf、675 个 Periodic group/u-pair/p-pair 组合，证明各架构内 prediction-only count 不随具体 pair 变化；实际 hooks 执行 15 个 Hopf 与 45 个 Periodic 稀疏决策，覆盖两个通道的每个 Top-2 pair 与每组，并额外执行诊断分支。完整参数名称集合和组件分解保存在 JSON。

### 8.5 不作出的推论

Vanilla-FNN-MoE、DataOnly-MoE、Global MoE 缺少完整、模型专属的 checkpoint/config inventory，本轮没有为它们估算参数。可用的独立 Steady S4 检查点不能确认是正文 Steady 模型，也没有混入同一比较。Appendix J 没有空行或虚构比较项，并明确不声称 parameter-matched。

Active fraction 是参数对象占比，**不是 FLOPs、耗时、速度提升或预测性能**；不能从 17.09% 推出相同比例的实际耗时。正文仅用一句话指向附录 J，没有额外计算效率 claim。

## 9. P1-10：Appendix I 精度

百分比保留两位小数：99.9344% → 99.93%，84.1630% → 84.16%，固定 pair 的 100 → 100.00；原已为 16.27 / 31.25 的数值保留。Entropy 统一三位：0.3869 → 0.387，0.9292 → 0.929，0.7838 → 0.784。

重复的六位 routing-mass 与 Reynolds-number 列表改为“secondary pressure expert 的平均 mass 随三个 held-out Reynolds numbers 显著增加”的定性说明。原始统计 CSV/JSON 和主文 Figure 3 数据、主文 routing 段均未改。这里是显示精度整理，不是新实验或重算。

## 10. 回归与全稿 QA

最终 `scripts/verify_final_p0_p1.py` **246 项检查 PASS**。

- Title、Abstract、Introduction（Figure 1 区域以外）逐字一致。
- Section 3、Limitations、Conclusion、数学推导、原有实现附录和 ICLR style 逐文件散列一致。
- 原始 9 个实验/协议表的 tabular 数据与 bolding 全部一致；唯一允许的表头变更为 Best single 的星号。
- Figure 2、Figure 3 与其他原有 figure assets 未修改；Figure 3 caption 和整段主文 internal-routing result 逐字保留。
- 新参数数值与实际执行脚本一致，模型源码和 inventory 散列核验通过。
- 主文和当前附录无旧图引用、错误 history caption、过时 finite-window / Periodic-core 文字、AI TODO 或六位 routing diagnostics。
- 无 undefined references、missing citations、duplicate labels、overfull boxes、undefined font shapes。
- 所有 23 页均渲染并逐页检查，重点检查新 Figure 1、Table 2/3、四篇文献、AI 声明、附录 I/J。参数附录单独起页，避免说明段落被表格拆断。
- 编译仍有 underfull-box 提示，主要来自两端对齐和官方 BibTeX 导出的长 URL；另有本机 Fontconfig 配置提示。字体正常嵌入且已检查实际渲染，没有把这些环境提示表述成零 warnings。
- ZIP 在独立解压目录中重新编译，23 页文字、PDF 页面内容流、90 dpi 页面 PNG 均与工作稿逐页一致。ZIP 包含实际引用的 31 个文件，main.tex 在根目录；不包含 build、内部报告、训练资料或审计脚本。

**是否修改任何原始 experimental performance number：No。** 本轮新增的是已有 DeepONet 结果的主文引用、架构参数统计和附录显示精度整理，不是性能数据更改。

## 11. 复现命令与散列

在论文工作目录执行；Python 使用带 ReportLab/pypdf 的运行时，参数脚本会自动读取项目局部 torch runtime：

```powershell
$pythonRuntime = 'C:\Users\panxy1019\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $pythonRuntime figures/ral_moe_rom_overview.py
& $pythonRuntime scripts/count_trainable_parameters.py
& .\build\tools\tectonic-0.17.0\tectonic.exe --only-cached --keep-logs --keep-intermediates --outdir build/final_p0_p1_compile main.tex
& $pythonRuntime scripts/verify_final_p0_p1.py --baseline build/before_final_p0_p1_20260906 --pdf build/pdf/main_final_p0_p1_20260906.pdf --compile-dir build/final_p0_p1_compile --package build/overleaf_upload_final_p0_p1_20260906.zip --package-pdf build/final_p0_p1_package_compile/main.pdf --render-dir build/final_p0_p1_render
```

| 文件 | SHA256 |
|---|---|
| 最终论文 PDF | `d30961b9442157ebdc60849c1003ef70d6155b1cb2aebd8005f7715a4ae0a288` |
| Overleaf ZIP | `2e1ddc1e4d979df3cafbf6655ee64204b6ab6ce60d45675d05f2a6bbfe288211` |
| Figure 1 vector PDF | `f06612d3b7a6a8185d691be48888a5e6df41c9c3a86a56f92e81062e91ada7e1` |
| 参数统计 JSON | `b63658ec6b4bd0dd24bb152a0979518a16e9d65bc636a9d0f234c68ecd130e68` |

最终待作者完成的动作仅是人工确认 AI 披露/论文内容并在投稿系统填写对应字段；在线上传和正式投稿未代为执行。
