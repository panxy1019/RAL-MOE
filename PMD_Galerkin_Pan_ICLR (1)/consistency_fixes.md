# 第二轮一致性修复清单

## 执行边界

采用用户指定的最新目录及当前附录 C 的方法定义。未把旧目录恢复到这里，未按网络审计直接重写方法，未修改实验数值，未上传 Overleaf。按 project-execution-manager 和 karpathy-guidelines 分批执行、保留快照并检查差异；按 PDF 技能完成编译、渲染和版面核验。

## P0

| 项目 | 结果 |
|---|---|
| P0-1 structured expert | 正文删除与附录不同的 W_o 参数化，仅保留 sparse mixture；精确定义引用附录 C；显式 branch 统一为 linear。正文 group 符号统一 m-star，并保留附录已定义的固定组/单组情形。 |
| P0-2 pressure formula | 按当前附录 C 将正文改为 gamma * PP + pressure correction；不再写 (1-gamma) * candidate。这是内部定义同步，不是所有 benchmark 实现的科学认证。 |
| P0-3 applicability | 可编辑正文/附录统一 applicable / validation-supported applicability，保留“不构成 formal stability guarantee”的限制。 |
| P0-4 advancement/curriculum refs | advancement 指 C；curricula/optimization 指 E。 |
| P0-5 Appendix ?? | `app:specialist` 改为真实标签 `app:specialists`。 |
| P0-6 affine reconstruction ref | 改引用完整附录 A，避免仅指 pressure equation。 |
| P0-7 duplicate seed paragraph | 合并为一个段落。检查发现 E 原先没有完整 gate optimizer protocol，故将 G 的已有协议移至 E，而不是直接删掉唯一的复现信息。 |
| P0-8 stability wording | 改 predefined validation-stage rollout criterion；Divergent 仍只作为 operational evaluation label，没有添加理论保证。 |

额外发现并修复：两个表格共十处 `\mathbf` 用在非数学模式中，改为 `\textbf`，数字逐字保留。

## P1

- 统一参数区域、表示 chart、自主预测 specialist 的层次，不再用 reduced spaces 覆盖 parameter domain。
- velocity RHS 使用 projected Galerkin backbone；泛指物理结构使用 projected ROM backbone。
- gauge-corrected pressure / zero-mean pressure gauge；数学 affine reconstruction 保留，affine means 改 mean fields。
- 贡献标题采用 independent local ROMs；减少不必要的 noncommensurate。
- 训练使用 closed-loop rollout training，评估采用 autonomous rollout；解释模型使用自身预测的机制时仍保留 recursively。
- one-step fitting 改 one-step prediction。

## P2

- 摘要精简方法概述；引言删除局部重复；related work 引文标点、空格和小节标题修正。
- 方法开头 coupled 改 complementary；local coordinate dimensions 改 d_u^(r)、d_p^(r)。
- E2 parameter-only、偏好在 rollout 内固定、T2-C 独立历史条件、只在输出物理场上组合、不反馈到 specialist 的规则不变。
- 实验开头及首个问题标题不再把所有 horizon 都称为 long；保留真正 longer-horizon 对比和所有 K 值。
- Global MoE 比较表述改 closest overall comparison，不扩展其科学结论。
- supplementary implementation 替代 released implementation；补充 paragraph 标题句点。
- 表头 Selected step 改 Selected，以覆盖 epoch-based run；表注说明 configured schedule 与 selected checkpoint 的区别。
- 主文 Best fixed 与附录 G 名称一致。附录 G 的 oracle 语句不再引用未提供该协议的附录 D；没有向 D 新增 benchmark-specific 协议。
- 结论及参数量限制保持原样：active fraction 不解释为 FLOPs/runtime/speedup。

## 数值与引用核验

- 14 个 tabular 块逐一比较，所有数字序列与修改前相同，包括百分比、范围、seed、预算及 mean/SD。
- 主实验及附录 F/H/I/J 的全文数字序列不变。G 的 optimizer 文字移入 E，已有 seed/selection 规则未改变。
- 两个 bibliography 文件 SHA-256 完全不变。
- descriptor 定义方程与修改前逐字相同，未从 6 改为 8；E_u/E_p 保留无 epsilon 形式。
- Appendix A--J、Table 1--14、Figure 1--6 标签均解析，编号连续；没有重复 labels 或 paragraph headings。
- 最终编译日志：undefined reference/citation 0，multiply-defined 0，overfull 0。PDF 文本中 `??` 为 0。
- 全文可编辑 tex 中清单列出的旧术语为 0；此检查不包含位图内嵌文字。

## 保持不变及原因

1. 实验行的方法身份、数值和 checkpoint 选择：不是语言问题，见 `unresolved_scientific_items.md`。
2. Table 4 与附录 F 数据：遵循用户此前暂不处理该点的限制，仅润色周围文字。
3. 六维 T2-C descriptor：用户要求当前方法草稿优先。
4. 不统一有效数字：原表有不同精度，但用户禁止改实验数值；不重新四舍五入或伪造精度。mean/SD 表原已采用相同位数。
5. AI use statement：保留作者现稿，另作明确真实性提醒；未自行判断 venue policy。
6. Figure 1 位图：保留用户提供的图，不通过语言修订静默重绘。
