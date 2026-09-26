# Internal expert-routing 论文整合报告

版本标识：`routing_20260905`。继续完成交付日期：2026-09-06。

## 1. 完成结果

已按 `ICLR_integrate_expert_routing_into_paper_codex.md` 重新核验和完成整合，并将 Figure 1 替换为用户本次提供的完整流程图。

进入本轮时，工作目录已有部分路由整合内容，但 README、正式交付 PDF 和上传包仍指向上一轮版本。因此本轮没有重复插入同一段落，而是完成了原始统计复核、图形重建、流程图替换、浮动顺序调整、编译检查、独立上传包编译及正式交付。

当前工作目录：

```text
C:\Users\panxy1019\Documents\CHANNEL\RAL_MOE_ROM_ICLR2027
```

主要交付：

| 项目 | 相对于工作目录的文件 |
|---|---|
| 最终论文 PDF | `build/pdf/main_routing_integrated_20260905.pdf` |
| Overleaf 上传 ZIP | `build/overleaf_upload_routing_20260905.zip` |
| 本报告 | `ROUTING_INTEGRATION_REPORT.md` |
| 更新后的 Section 4.3 | `sections/experiments.tex` |
| 新增路由附录 I | `appendix/internal_routing.tex` |
| 更新后的 Limitations | `sections/limitations.tex` |
| Reproducibility statement | `main.tex` |
| 主文 Periodic-only 三面板图 | `figures/periodic_internal_routing.pdf` |
| 附录 Periodic rollout-position 图 | `figures/periodic_rollout_step_routing.pdf` |
| 用户指定流程图原样副本 | `figures/ral_moe_rom_framework_user.png` |

本轮只更新本地论文及可上传包，没有直接更改在线 Overleaf 项目。

## 2. Figure 1：使用本次用户图片

来源：

```text
C:\Users\panxy1019\Downloads\ChatGPT Image Sep 5, 2026, 09_33_58 PM.png
```

图片按字节复制，没有重新生成、重绘、裁掉面板或修改图中内容。移除了上一张图片专用的 `trim=0 175bp 0 155bp,clip`，改为直接使用完整图像。

源图与论文副本 SHA256 均为：

```text
648aa6fe6b1128466af266fe7712df91fab1d75121abaa550fcb00c998b7553b
```

图注依据现有方法保留了两项必要说明：

- 图中 history 连线解释为整体 assembly 对历史的使用；history 初始化 specialists 并条件化 T2-C，E2 的概率仍只依赖参数。
- 下半部分是 specialist 的示意更新，shared expert 在所选 group 中激活；压力仍采用原文的代数映射，不因流程图改变其方法定义。

原图的 history-to-E2 连线本身仍然存在，图注用于澄清它与正文方法的关系。这不是对原图内容已经被修正的宣称。整合过程中没有为了配图改变 E2 或压力更新公式。

## 3. 正文 Figure 3 的统计来源

统计根目录：

```text
C:\Users\panxy1019\Documents\CHANNEL\iclr_expert_routing_analysis
```

主图由 `scripts/build_routing_figures.py` 从已验证 CSV 重新生成，不是剪切此前 S/H/P 三列分析大图。

| 数据 | 用途 |
|---|---|
| `stats/group_usage.csv` | Periodic 四个 held-out Re 的 group Top-1 fractions |
| `stats/expert_routing_mass.csv` | Periodic u/p 的 mean routed-expert mass |
| `stats/rollout_step_mass.csv` | 附录 Periodic rollout-position heatmap |
| `stats/routing_summary.csv` | 正文和附表的专家数、pair 数、dominant-pair fraction、entropy |
| `stats/mass_concentration.csv` | Hopf 固定支持集内的权重集中程度 |

图形输入 CSV 的 SHA256、完整作图矩阵和 Re 数值保存在 `build/routing_figure_data.json`。

正文和图统一使用 `population=macro_stage1`，即每个自主宏步在第一积分阶段取一次记录，不将其误写为全部 RHS evaluations。Periodic 为 4 个 Re、每 Re 13 个固定窗口、K=48，共每通道 2496 个宏步路由决策。

### 3.1 三面板设计

Figure 3 位于 Section 4.3，内容为：

1. Panel (a)：四个 held-out Re 的 100% stacked group bars。
2. Panel (b)：18×4 velocity mean-routing-mass heatmap。
3. Panel (c)：18×4 pressure mean-routing-mass heatmap。

专家横向编号实际以 `g0:e0` 等 group-local ID 表示；全部零使用专家也保留在图中。Re 显示为 70.31、100.35、149.06、189.86。采用统一的 0–0.20 色标，实际主图最大质量为约 0.192050，没有截断颜色范围。

图中没有 `epoch 85`、`observed pairs`、`zero-based`、`fixed shared mass` 等内部工程标签，没有巨大总标题或长 debug footer。shared/routed 固定比例图未进入正文。

### 3.2 正文数据核对

| Periodic 通道 | 使用 routed experts | distinct group/pair configurations | 最大 pair 比例 | H |
|---|---:|---:|---:|---:|
| Velocity | 18/18 | 33 | 16.2660% | 0.929230 |
| Pressure | 14/18 | 16 | 31.2500% | 0.783768 |

| Held-out Re | g0 (%) | g1 (%) | g2 (%) |
|---:|---:|---:|---:|
| 70.314635 | 57.3718 | 20.8333 | 21.7949 |
| 100.352251 | 30.6090 | 35.7372 | 33.6538 |
| 149.059229 | 14.4231 | 27.2436 | 58.3333 |
| 189.862278 | 19.2308 | 48.3974 | 32.3718 |

正文采用简短的 `Internal routing behavior` 段落，强调 Periodic 的 conditional expert utilization 和 channel-dependent profile，并指向附录的 Hopf 对照。没有把 routing statistics 写成预测效果的因果解释。

在 Table 4 后添加局部浮动屏障，使最终阅读顺序为：Table 4 → 原表解释 → internal-routing paragraph → Figure 3。现有消融表的数字和原解释文字保持不变。

## 4. 附录 I 中保留的诊断

标题为 `Additional internal-routing diagnostics`，保留以下内容：

- 评价窗口数、K、宏步第一阶段的统计总体和窗口相关性说明。
- normalized activation frequency、mean routing mass 和 utilization entropy 的定义。
- group identity 包含在 pair 中，pair 不区分两个专家的顺序。
- shared/routed scaling 是固定架构系数，routing mass 不是输出范数或因果贡献。
- 图中的 routed-expert ID 与正文公式 shared-expert 下标 0 的区别。
- Hopf 的固定 pair 与集中质量，以及 Periodic 的 rollout-position 图。

### 4.1 Hopf compact diagnostic

没有隐藏 Hopf 固定支持集。附录明确写出：

- 单个 group 是结构设定。
- Velocity 固定 pair 为 `(e1,e3)`；pressure 固定 pair 为 `(e2,e4)`。
- 这些 pair 在全部四个积分阶段均保持不变。
- Velocity e3 占 pooled routed mass 的 99.9344%。
- Pressure e2 占 pooled routed mass 的 84.1630%，而 e4 的 Re-wise mean mass 分别为 0.000149、0.016094、0.187376。

采用 Table 10 的四行 compact table 替代独立 Hopf figure。表中包括专家数、pair 数、dominant-pair fraction 和 H，并明确 H 不是性能分数、固定 Top-2 pair 的 H 并非零。

### 4.2 Periodic rollout-position 图

附录 Figure 6 为两通道 heatmap，保留全部 18 个 routed-expert 行。横轴为 normalized rollout position k/48，图注明确不是 physical oscillation phase。它只作 temporal routing diagnostic，不用于 vortex-shedding phase specialization 的解释。

本轮没有将 Steady S4 routing panel 纳入正文或新增附录，也没有把三模型完整诊断大图放入论文。

## 5. 正文与声明修改范围

| 文件 | 修改或保留的结果 |
|---|---|
| `sections/introduction.tex` | 替换 Figure 1 输入与图注；图外 Introduction 内容逐字保持不变 |
| `sections/experiments.tex` | Section 4.3 保留 Periodic-only 路由段落与 Figure 3，完善第一阶段说明和 Table 4 顺序 |
| `appendix/internal_routing.tex` | 附录 I：统计定义、Hopf compact diagnostic、Table 10、Periodic Figure 6 |
| `sections/limitations.tex` | 不再声称没有 internal-routing analysis；保留描述性、非因果、跨 chart 不要求同等切换的边界 |
| `main.tex` | 引入附录 I，并在 Reproducibility statement 中说明冻结模型 held-out post-hoc 统计不用于模型选择 |

Abstract 未修改。Introduction 贡献的核心内容未修改。现有方法、训练定义和既有性能结果未被本轮重新定义。

**是否改变任何原始 performance number：No。**

验证方式不仅是人工查看：`scripts/verify_routing_integration.py` 将当前稿与原始整合前备份比较，检查全部九个既有 `tabular` blocks 逐字一致，原 experiments 文字在剔除浮动控制命令后逐字一致，方法及原附录源文件字节一致。该检查证明“未改原有数字”，不等价于重新审计或重新计算所有历史结果。

## 6. 编译、页数与引用

工作稿使用项目已有 Tectonic 0.17.0 与本地缓存编译；没有修改会议样式或字体大小来压缩页数。

| 项目 | 最终位置 |
|---|---|
| Figure 1：新流程图 | 第 3 页 |
| Figure 3：Periodic internal routing | 第 8 页 |
| 科学主文结束、Conclusion | 第 9 页 |
| References 起始 | 第 9 页 |
| Appendix A 起始 | 第 11 页 |
| Appendix I 与 Table 10 | 第 20 页 |
| Figure 6：rollout-position | 第 21 页 |
| PDF 总页数 | 21 |

112 项自动检查通过，记录在 `build/routing_integration_verification.json`。没有 overfull boxes、未解析引用/文献、重复标签或缺失字体形状。存在部分 underfull-box 提示和环境 Fontconfig 提示；逐页渲染检查未发现本轮新增图表的缺字、遮挡或裁切。

最终全部 21 页已用 Poppler 渲染检查。原有实验图的旋转行标签较密，属于源图自带排版，本轮没有修改实验图像数据。新流程图完整保留两个面板；Figure 3、Table 10、Figure 6 均在版面边界内，坐标与图注完整。

## 7. 上传包独立验证

`build/overleaf_upload_routing_20260905.zip` 包含 30 个实际引用的匿名论文源文件、样式、文献和图形文件，ZIP 根目录即 `main.tex`。

上传包排除了 `source_snapshot/`、`build/`、检查脚本、内部报告、机器路径和分析日志。打包逻辑递归跟踪 `input` 和 `includegraphics`，避免漏掉新增附录或图片。

上传包已独立解压到 `build/upload_routing_20260905_extracted/` 并重新编译。工作稿与包内构建的全部 21 页满足：

- 提取文字完全一致。
- PDF page content streams 完全一致。
- 85 dpi 页面渲染 PNG 逐字节一致。

记录：`build/routing_package_verification.json`；打包清单：`build/routing_upload_manifest.json`。

```text
Final PDF SHA256:
cee2573cd793a1e6fe999503115479621a988439123ebab35519a90a7a8c3257

Upload ZIP SHA256:
3601e21725a3080c1afd04ffca9350a6c5ac8029f966e4cd78d4b3e9e9af7ece
```

本轮前状态完整备份在 `build/before_routing_reexecution_20260905_215237/`。既有数值回归使用更早的 `build/before_routing_integration_20260905_171744/`；没有将这两个不同用途的基线混为一谈。

## 8. 复现与剩余作者事项

重新生成路由图：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' `
  'C:\Users\panxy1019\Documents\CHANNEL\RAL_MOE_ROM_ICLR2027\scripts\build_routing_figures.py' `
  --analysis-dir 'C:\Users\panxy1019\Documents\CHANNEL\iclr_expert_routing_analysis'
```

编译入口为项目 `main.tex`；可直接将经过验证的 ZIP 上传到 Overleaf。本轮没有重新训练、选择 checkpoint 或重跑 GPU 模型。

保留了原稿 AI-use statement 中的作者披露 TODO，未代替作者编造声明。正式投稿前仍需作者填写准确披露；该事项不属于路由结果整合。

建议以本报告指向的日期命名 PDF/ZIP 为交付依据，不使用先前 table-cleanup 文件或旧的未完成路由编译目录作为最终稿。
