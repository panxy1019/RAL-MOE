from pathlib import Path
import difflib, json

root=Path(r'C:\Users\panxy1019\Desktop\STABLEMOE\PMD_Galerkin_Pan_ICLR (2)')
backup=Path(r'C:\Users\panxy1019\Documents\CHANNEL\paper_revision_20260915\original')
out=root.parent/'Overleaf逐项修改对照说明_20260915.md'
catalog=[
('main.tex','摘要和文末声明','修正摘要大小写，并明确联合指标为 E_u+E_p；将复现声明中的附录编号改为自动引用；扩展 AI 使用声明，使其覆盖本次实际修订活动。','摘要已有的 30.9% 和 54.2% 没有修改。AI 声明需要你结合此前全部研究过程确认，不能只根据本次修订判断完整性。'),
('sections/introduction.tex','引言和图 1 图注','压缩重复文献综述；将问题限定为不同局部坐标下的独立预测与输出组合；减少过早出现的内部编号；明确初始历史、窗口固定权重和不反馈。','图 1 位图没有重画，只修改了图注。若修改图形，可显式补入 reconstruction 或 decoder 节点以及 no feedback 标注。'),
('sections/related_work.tex','相关工作和新颖性边界','补充 fl-ROM 的设计区别，并避免把物理空间融合本身写成独有创新；收窄对其他 MoE 工作的概括。','新增 fl-ROM 使用 colanera2026fuzzy，必须与 references.bib 一起更新。'),
('sections/method.tex','方法术语与计算流程','区分框架的两个功能层次与专家内部的 group-and-expert routing；将压力描述为代数更新，并说明其参与下一步递归；区分 E2 参数输入与 mask 的初始化检查。','本轮没有更改方法公式，也没有确认 applicability 阈值或 hard routing 的梯度实现。'),
('sections/experiments.tex','实验正文和主表','澄清固定 Reynolds 数和各几何分别训练；解释 Oracle 选权目标与终端报告指标的差异；区分验证排除与未报告；统一圆柱表 4 与附录范围；相应收窄性能结论。','这是本轮最需要你逐项审阅的文件。圆柱表格并非普通文字润色，详见前面的重点说明。'),
('sections/conclusion.tex','结论和适用范围','明确输出融合无反馈，并限定证据覆盖的任务范围；保留未来工作，避免未经验证的加速、跨几何迁移或方程约束保持主张。','结论在上一轮本地编译中完整结束于第 9 页；后续新增内容后应重新检查分页。'),
('appendix/circular_cylinder.tex','圆柱附录和表 7','说明主表与附录使用相同的逐轨迹最小—最大范围；解释范围不是置信区间或池化均值；同步 V/F 和 N/R 标记。','数值范围沿用原稿，不代表已用原始数据重新计算。若恢复主表均值，此处的汇总说明也要同步调整。'),
('appendix/square_cylinder.tex','方柱附录和表 8','展开解释 full-window oracle 的选权方式与 terminal metric 的关系，并统一 diagnostic oracle 名称。','没有重算 oracle，也没有改动其权重或表中误差数字。'),
('appendix/fluidic_pinball.tex','Pinball 附录','取消 oracle 数字加粗，删除“捕获几乎全部可用收益”的最优性推断；保留其事后诊断性质。','Pinball 的确切 oracle 目标及汇总协议仍需原始实现确认，不宜直接套用方柱定义。'),
('appendix/routing_details.tex','路由与融合附录','补充 mask 与 E2 的职责边界及实验的条件性；说明物理历史描述量在初始化时计算一次。','这里只增加了概念边界说明，尚未补成可执行的 applicability 判定算法。'),
('references.bib','参考文献条目','补齐五个原先缺失的引用键，并新增 fl-ROM v2 文献。','新增键为 vlachas2018、kramer2024、bhat2025、manti2025、colanera2025、colanera2026fuzzy。保留原有引用键，便于 Overleaf 解析。')
]

parts=['''# Overleaf 逐项修改对照说明

日期：2026 年 9 月 15 日  
对象：RAL-MoE-ROM 论文第二轮修订  
用途：帮助你在 Overleaf 定位、审阅和继续修改本轮改动。

本文档对照修改前的备份与当前本地源文件生成。它包含每个已修改文件的说明，以及实际发生变化的 LaTeX 行。当前论文源文件不会因生成本文档而再次改动。

## 一 使用方式

1. 如果已导入第二轮上传包，文中的“修改后”应当已经存在，不必再次粘贴。
2. 如果 Overleaf 仍是旧稿，按文件名打开对应文件，用“搜索定位词”查找所在段落，再对照修改。不要直接按旧版行号替换。
3. 下方行号基于生成说明时的本地源文件。后续插入内容会使行号移动，文件名、LaTeX 标签及搜索词更可靠。
4. 每个代码块仅包含实际变化的连续行，不一定是完整段落、表格或环境。不要把不完整块单独插入文件，也不要删除周围未展示的 `\\begin`、`\\end` 等代码。
5. 表格和说明文字需成组修改。例如圆柱主表 4、附录表 7 和结果分析应保持同一统计口径。
6. 本文档不需要作为论文内容上传；它是你继续编辑时的对照材料。

## 二 优先审阅的实质改动

### 1 Oracle 的解释改变 误差数字保留

方柱 oracle 按 full-window objective 选择权重，但主表报告 K=24 的 terminal error。这两者不是同一个优化目标。因此，本轮保留 oracle 数字，将其标为 diagnostic oracle，并删除“数值接近就意味着接近终端最优”的暗示。

继续修改时有两个可选方向：保持现在的诊断定位；或者根据原始预测重新计算 terminal-objective oracle，再同步更新表格及解释。仅改名称不能把现有 oracle 变成终端误差下界。

定位：`sections/experiments.tex` 的 `tab:routing_fusion`；`appendix/square_cylinder.tex` 的 `tab:square_fusion_full`。Pinball 仅收窄解释，没有确认其目标与方柱相同。

### 2 圆柱主表由汇总值改为已有范围

这是本轮对结果展示方式的实质调整，建议你重点审阅。修改前的主表使用单值，附录使用逐轨迹最小—最大范围，但缺少能解释两者关系的统计协议。当前 Periodic 速度单值已经是 0.7412%，不是修改意见中旧版的 0.4641%。此外，Hopf 压力的 0.1338% 也不能直接解释为范围 0.0355–0.4321% 对应的三条轨迹等权均值：在此假设下，均值至少为 (2×0.0355+0.4321)/3≈0.1670%。

本轮没有猜测正确均值，而将主表统一为附录已报告的最小—最大范围；相应说明范围不是置信区间，也不代表每条轨迹都优于基线。原单值完整保留在下方修改前代码和备份中。

若你希望保留均值主表，应先核对逐时间步、逐窗口、逐轨迹的汇总次序、样本数量和权重，再恢复相应数字与表注。当前处理只消除了展示口径冲突，没有完成对原始数值的独立复算。

定位：`sections/experiments.tex` 的 `tab:circular_full_reorg`；`appendix/circular_cylinder.tex` 的 `tab:circular_full`。

### 3 失败与未报告被区分

- V/F：验证阶段排除，未报告 held-out 测试结果，用于 Vanilla-FNN-MoE Hopf。
- N/R：结果未报告，用于缺失的 DataOnly-MoE 压力及 Pinball 基线。
- 不再仅根据破折号声称测试 rollout diverged。若原始日志证实发散，应补充对应判据并更新该项标记。

### 4 AI 声明扩大到本轮真实用途

原句仅写语言润色和 LaTeX 排版；本轮还进行了文献查找、稿件组织和论断一致性检查，因此扩展了声明。需要你核对以前是否还有 AI 辅助代码、数据或实验的用途，并据实补充。

### 5 Applicability 只澄清职责 尚未补全实现

本轮说明了 E2 提供参数条件化偏好、mask 另行检查适用性、现有重叠区结果以给定候选集为条件。判定阈值、支持域标定、时间采样与初始化检查的实际实现仍缺材料，未写入推测性算法。

## 三 文件修改索引

| 文件 | 修改重点 |
| --- | --- |
''']
for name,title,reason,note in catalog:parts.append(f'| `{name}` | {title} |\n')
parts.append('\n## 四 按文件逐项对照\n\n代码块中的反斜杠和花括号保留为 LaTeX 原文，可直接复制。\n')
count=0
for idx,(name,title,reason,note) in enumerate(catalog,1):
    old=(backup/name).read_text(encoding='utf-8').splitlines()
    new=(root/name).read_text(encoding='utf-8').splitlines()
    changes=[x for x in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes() if x[0]!='equal' and ('\n'.join(old[x[1]:x[2]]).strip() or '\n'.join(new[x[3]:x[4]]).strip())]
    parts.append(f'\n### {idx} {title}\n\n文件：`{name}`\n\n**修改理由：** {reason}\n\n**继续编辑时注意：** {note}\n')
    for seq,(tag,i,j,a,b) in enumerate(changes,1):
        count+=1
        anchor=next((line.strip() for line in new[a:b] if line.strip()),next((line.strip() for line in new[max(0,a-2):a] if line.strip()),''))
        anchor=anchor[:95].replace('`','')
        parts.append(f'\n#### {idx}.{seq} 修改位置\n\n原稿行号：{i+1}–{j if j>i else i+1}；当前行号：{a+1}–{b if b>a else a+1}。\n\n搜索定位词：`{anchor}`\n')
        parts.append('\n**修改前**\n\n'+('```latex\n'+'\n'.join(old[i:j])+'\n```\n' if j>i else '此处原来没有对应内容。\n'))
        parts.append('\n**修改后**\n\n'+('```latex\n'+'\n'.join(new[a:b])+'\n```\n' if b>a else '此处内容已删除，无需新增替代文字。\n'))

parts.append('''
## 五 继续修改时需要补充的材料

| 待办 | 建议编辑位置 | 需要核实的内容 |
| --- | --- | --- |
| 补全 applicability 协议 | `appendix/routing_details.tex` 和 `appendix/implementation.tex` | 各几何的阈值、候选区域、验证标定方法、初始化采样要求及失败处理 |
| 补全 E2 训练 | `appendix/routing_details.tex` | 标签来源、重叠区标签、损失函数及 checkpoint 选择 |
| 解释 hard group selection | `appendix/specialist_details.tex` | 真实实现中的梯度处理，不要根据 argmax 公式猜测 |
| 核对统计汇总 | `appendix/implementation.tex` 及各实验附录 | 时间、窗口、轨迹的汇总次序与权重，尤其圆柱和 Pinball |
| 补充 Oracle 精确定义 | 方柱及 Pinball 附录 | 各自优化目标、搜索方法、权重约束、报告指标 |
| 更新图 1 | `figures/ral_moe_rom_overview_user_20260908.png` 及其图注 | 各分支的重构步骤与不反馈边界 |
| 新增实验如有 | 实验正文和对应附录 | 学习常数权重、容量匹配对照、误差随时间和实测运行时间 |
| 确认 AI 声明 | `main.tex` | 覆盖整项研究真实使用情况，不限本次修改 |

上述待办没有作为已完成内容写入论文。本轮也没有修改原始数据、重新训练模型或新增实验结果。

## 六 编译与版本核对

本轮修订时的本地 Tectonic 编译结果为 27 页；结论完整结束于第 9 页。引用键、交叉引用标签和 PDF 问号检查通过；没有 overfull box。参考文献长 URL 仍有 underfull hbox 提示。

这些是本地版本的检查结果。你在 Overleaf 修改后，请以 Overleaf 的最终 PDF 为准，重点核对：正文结束页、表格与附录的统计口径、引用问号、图注是否匹配图形，以及新增文字是否带来超出实验范围的结论。

本轮没有修改 `.sty` 模板、字号、页边距、方法公式、训练代码或原始图像文件。

## 七 本地文件位置

- 修订稿目录：`C:/Users/panxy1019/Desktop/STABLEMOE/PMD_Galerkin_Pan_ICLR (2)`
- 修改前备份：`C:/Users/panxy1019/Documents/CHANNEL/paper_revision_20260915/original`
- 原始差异文件：`C:/Users/panxy1019/Documents/CHANNEL/paper_revision_20260915/revision.diff`
- 编译验证记录：`C:/Users/panxy1019/Documents/CHANNEL/paper_revision_20260915/verification.json`
- 第二轮上传包：`C:/Users/panxy1019/Desktop/STABLEMOE/PMD_Galerkin_Pan_ICLR_round2_20260915_Overleaf.zip`

新增文献的核验来源和本轮简要说明另见同目录的 `本轮修改说明_20260915.md`。
''')
out.write_text(''.join(parts),encoding='utf-8')
assert len(catalog)==11
assert out.read_text(encoding='utf-8').count('```')%2==0
print(json.dumps({'output':str(out),'files':len(catalog),'change_blocks':count,'bytes':out.stat().st_size},ensure_ascii=False))
