# T2-C 论文整合交付报告

完成日期：2026-09-12。状态：**修改、编译、数值回归、版面检查、完整源码打包均完成。**

本轮依据 `T2C_paper_integration_codex_guide.md`，仅整合已完成的 centered-square 描述量消融和三种子证据。没有重新训练，没有访问或改写 Overleaf；由用户自行上传。

## 1. 可交付文件

项目根目录：`C:\Users\panxy1019\Documents\CHANNEL\RAL_MOE_ROM_ICLR2027`。

- 完整 Overleaf 源码 ZIP：`output/overleaf_T2C_integrated_20260912.zip`。
- 编译后的匿名 PDF：`output/pdf/RAL_MoE_ROM_T2C_integrated_20260912.pdf`。
- 本报告：`T2C_PAPER_INTEGRATION_REPORT.md`。

ZIP 共 31 个实际引用的源码、模板、参考文献和图片文件，根目录含 `main.tex`；不是仅有改动文件的补丁包。内部实验日志、checkpoint、个人路径、审计报告和脚本目录未加入上传 ZIP。上传时保留包内目录结构，主文件为 `main.tex`。若覆盖已有项目，应先保留在线版本备份。

## 2. 修改文件

| 文件 | 改动 |
|---|---|
| `sections/experiments.tex` | Section 4.2 三段紧凑整合；Table 3 新增 μ-only；固定结果与种子统计分开 |
| `appendix/square_cylinder.tex` | K24 terminal 协议、μ-only 定义和分量、三种子 mean/SD 表、逐 seed 联合误差表、验证选择解释 |
| `appendix/routing_details.tex` | 末尾追加 centered-square oracle 目标；原描述量和方法公式保持原样 |
| `sections/limitations.tex` | 更新 gate-only seed 证据边界，保留 local-specialist capacity limitation |
| `appendix/implementation.tex` | 同步更新原来“未评估 repeated-seed”的过时句子 |
| `main.tex` | Reproducibility statement 增加 descriptor ablation 和三种子证据入口 |

仅上述六个论文文件变化。使用项目执行管理和代码修改技能约束改动范围、原值回归与包内一致性；使用 PDF 技能进行最终视觉检查。

## 3. Table 3 修改前后

原列顺序：Overlap / Best single / E2 Top-1 / Equal / E2 prob. / T2-C / Oracle。

新列顺序：Overlap / Best single / E2 Top-1 / Equal / E2 prob. / **T2-C μ-only** / T2-C / Oracle。

| Overlap | Best single | E2 Top-1 | Equal | E2 prob. | 新 μ-only | 原 T2-C | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| S–H | 1.7298 | 1.7298 | 3.0321 | 2.1847 | 2.1846 | 1.1958 | 1.1842 |
| H–P | 1.2079 | 1.8473 | 1.6146 | 1.6618 | 1.1504 | 0.8465 | 0.8384 |

单位为 %，K24 末步联合误差。原六列数值全部不变，保留 Best single 的 a posteriori 脚注，Oracle 不加粗，主方法仍称 T2-C。沿用现有标签 `tab:routing_fusion`，避免破坏交叉引用。

新 μ-only 使用预先固定的 seed 42001，不选择测试最优种子。其分量为：

| Overlap | 原始 Eu | 原始 Ep | 原始 E_joint | 附录四位小数 |
|---|---:|---:|---:|---|
| S–H | 0.7897614195 | 1.3948367477 | 2.1845981672 | 0.7898 / 1.3948 / 2.1846 |
| H–P | 0.5874797880 | 0.5628883667 | 1.1503681547 | 0.5875 / 0.5629 / 1.1504 |

原 full fixed-run 的 Eu/Ep/E_joint 分别仍为 S–H 0.4615/0.7343/1.1958，H–P 0.4509/0.3956/0.8465。没有用种子平均替换原模型结果。

## 4. 主文叙事

Section 4.2 分别说明：

- S–H：μ-only 2.1846% 与 E2 prob. 2.1847% 接近，full 1.1958% 明显降低误差。
- H–P：μ-only 1.1504% 已优于 E2 prob. 1.6618%，full 进一步降至 0.8465%。
- 两个 overlap 的共同结论仅是：在当前测试中，历史条件化优于参数量匹配的纯参数修正。

保留 full 相对 best single 的 30.9%/29.9% 和相对 E2 Top-1 的 30.9%/54.2% 改善。对 oracle 只说“接近诊断参照”，不声称实现严格最优。

主文三种子句子采用四位小数：S–H `1.1957 ± 0.0001%`；H–P `0.8500 ± 0.0035%`。紧接着说明只重训 gate，specialists、E2、candidate rollouts、partitions、normalization 和优化协议固定。

## 5. Appendix G 新增内容

最终编号：

- Table 7：现有完整分量表增加两行 μ-only；全部 42 个原分量值保留。
- Table 8：原 parameter-wise 表保持不变。
- Table 9：四组（三种子）Eu/Ep/E_joint mean ± sample SD。
- Table 10：四组的三个逐 seed E_joint，仅 4 行，PDF 本身即可查到完整种子值，无需依赖单独 CSV。

Table 9 使用的五位小数：

| Overlap | 模式 | Eu | Ep | E_joint |
|---|---|---:|---:|---:|
| S–H | μ-only | 0.79006 ± 0.00026 | 1.39498 ± 0.00012 | 2.18504 ± 0.00038 |
| S–H | Full | 0.46156 ± 0.00004 | 0.73416 ± 0.00014 | 1.19572 ± 0.00010 |
| H–P | μ-only | 0.59067 ± 0.00461 | 0.56050 ± 0.00435 | 1.15116 ± 0.00070 |
| H–P | Full | 0.45211 ± 0.00138 | 0.39793 ± 0.00215 | 0.85004 ± 0.00347 |

SD 定义：三个 gate-training seeds 的 sample SD，ddof=1；不是 across-Re SD、置信区间或整个 pipeline 的不确定性。联合误差 SD 由逐 seed 的 Eu+Ep 计算，不将两个分量 SD 相加。

### 指南示例中的两处末位舍入

本轮以已验证的全精度实验数据为准，直接一次舍入到五位小数：

- S–H full Ep SD = 0.000144880288868698 → **0.00014**，不是示例中的 0.00015。
- H–P full E_joint SD = 0.0034746150319517406 → **0.00347**，不是示例中的 0.00348。

这只修正整合示例的舍入，不改变任何原始实验数值。主文对应 H–P 四位小数仍为 0.0035。

附录保留重要异常说明：S–H μ-only 三次均在完成 8000 步训练后按验证规则选 step 1；H–P μ-only 选 step 400。不改选择规则，也不从测试集挑 checkpoint。

## 6. Oracle 与 terminal metric

Appendix D 新段明确仅针对 centered-square：每窗口在 `{0, 10^-4, ..., 1}` 网格上，最小化 24 步平均 Eu+Ep，然后用该固定权重报告 K24 terminal joint error。它不是 terminal metric 的精确最小化器，目标也不同于平方误差训练损失。

Appendix G 明确：两个 overlap 各有 16 个测试窗口，每个 held-out Re 贡献 8 个窗口；报告 k=24 的末步误差在窗口间的均值。因为窗口数均衡，此处 pooled mean 等于 per-Re window averages 的均值。没有将这个特定口径推广到其他 benchmark。

## 7. Limitations 与 Reproducibility

Limitations 更新为：gate 的 repeated-seed variability 已评估，完整层级和 local specialists 的多种子实验仍是未来工作。保留 local models 的 parameter-matched capacity comparison 限制。

主文 Reproducibility statement 增加 Appendix G 的消融与三种子入口。Appendix E 原有“repeated-seed dispersion ... not evaluated”句子同步改为 scoped statement，避免与新证据矛盾。

## 8. 保持不变的内容

| 检查项 | 结果 |
|---|---|
| descriptor 维数及原定义是否改写 | **No** |
| Table 4 内容是否修改 | **No** |
| 任何已有 baseline result 是否修改 | **No** |
| full fixed-run 数字是否被平均值替换 | **No** |
| Abstract / Introduction / Method 是否改变 | **No** |
| Figure 1/2/3 或任意图像素材是否改变 | **No** |
| 是否重新训练、改 E2、改 specialist | **No** |
| 是否写入 Overleaf | **No** |

所有新实验文字采用不指定维数的 history-channel 表述。原来暂缓的维数一致性和 Table 4 数据来源问题仍未解决，本报告不将整合完成等同于这些问题已解决。

## 9. 编译、排版与 QA

- 本地使用已有 Tectonic 0.17.0 编译成功。
- 将完整 ZIP 解压到独立目录后再次编译成功。
- 两次编译均为 **24 页**；逐页文本和 page content stream 完全一致。
- 主文至 Conclusion 在第 9 页结束；Reproducibility statement 跨第 9–10 页，AI use statement 和参考文献在第 10 页。这里不是对会议最终页数政策的认证。
- 最终 24 页缩略图已全览，重点检查了 Table 3、Limitations、Appendix D/E/G 和新表格的高清渲染。
- 139 项自动检查通过：原数值、未改文件、描述量段、图片、seed 数值、匿名模式、交叉引用、包内字节和独立编译一致性。
- 无 overfull box、缺字、未解析引用/引文或 LaTeX error。
- 不是零 warning：最终 log 有 22 条 underfull box 消息、两条 `h` 调整为 `ht` 的浮动提示，另有本地 Fontconfig 环境提示；未见由其造成的文字遮挡或截断。

初次排版因 Figure 2 后的强制浮动屏障留出大块空白；本轮只删除该屏障，使现有内容自然排版。Figure 2/3 的完整 LaTeX 图环境及图片字节均未修改，未缩小字号、页边距或图像。

核对脚本：`scripts/verify_t2c_paper_integration.py`。

核对结果：`build/t2c_integration_20260911/verification.json`。

修改前 31 文件备份：`build/t2c_integration_20260911/before/`。工作目录日期保留任务开始时的 09-11；最终交付日期为 09-12。

## 10. 文件校验

ZIP SHA-256：

`1bf61a9d38a1341235bae6706d4477ebc9f4d6874eb01139d66eb2188a307c1e`

PDF SHA-256：

`b035790e97e47ce0952504b649afd03b43c29d8a2625aa3e82e4443f1103bce3`

本次交付 ZIP 不包含本报告或实验工件，保持论文源码包干净；用户可在本地单独保留本报告作为修改记录。
