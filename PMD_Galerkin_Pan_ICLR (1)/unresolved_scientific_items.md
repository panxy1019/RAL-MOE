# 需要作者确认的科学与声明事项

这些事项没有被本轮语言修改解决。不要将“编译成功、全文术语一致”理解为所有方法身份、实验来源已核实。下述实现信息来自本轮重新查看的既有本地审计归档；本轮没有重新训练、重跑测试或重新下载远端权重。

## 1. Table 6：reported，不是全部 verified

已将表述降为 `reported training configurations`，保留九行原值。

| 行 | 当前证据边界 |
|---|---|
| Circular S | 已恢复 S4：主 lr 1.55e-5、3600-step 阶段预算、selected 1200；S3-B 回退版 selected 600。不能仅凭当前表值认定 Table 4 最终使用 S4。 |
| Circular H/P | 既有审计有明确冻结 checkpoint、配置与选中 step/epoch。 |
| Square S/H/P | 既有审计有明确候选 checkpoint/config；不代表各实现均完全等同正文通用公式，尤其 Square H 的最终推进路径需要作者确认。 |
| Pinball S/H/P | 找回的训练配置均来自 B1；不能将它们自动认证为 sparse-MoE 最终运行。H 的已找到验证记录不等于论文 final-test 来源。 |

Circular S 还有两项复现细节：S4 的 3600 是从 S3-B 初始化后的阶段预算，不是整条训练链总成本；shared trunk/router 初始学习率为表中主组的十分之一。原表没有展开这些细节，本轮没有用语言修改代替作者确定最终版本。

证据：[训练来源记录](<C:/Users/panxy1019/Documents/CHANNEL/RAL_MOE_ROM_ICLR2027/specialist_training_provenance_20260913.md>)。

## 2. Pinball Periodic：结果到 B1 的关联仍需作者处理

本轮复核的归档：
`C:/Users/panxy1019/Documents/CHANNEL/RAL_MOE_ROM_ICLR2027/build/network_architecture_audit_20260912/pinball_candidates.json`。

归档记录中的两个 final-test summary：

- `fluidic_pinball_periodic_v2_b1/evaluation_final_test_corrected_dt1_all_offsets_k1_k32/ROLLOUT_SUMMARY.json`
- `fluidic_pinball_periodic_v2_b1/evaluation_final_test_corrected_dt1_all_offsets_k56_core/ROLLOUT_SUMMARY.json`

两者都明确指向：
`fluidic_pinball_periodic_v2_b1/runs/FluidicPinballV2_B1_Deep_FNN_H3_seed1248/best_validation.pt`。
对应 checkpoint 归档中 variant=b1、best_step=8000；所保存 SHA-256 为 `c89a0cbf6cc0920664875f00234afaac49bc19b1c5c5f985cbba0b09e2dbdbda`。
训练器 `sources/pinball_periodic/train_b1_fluidic_pinball.py:787` 根据 b1 实例化 `DeepFNNH3`。

| 记录 | Eu (%) | Ep (%) | joint (%) | 论文显示 |
|---|---:|---:|---:|---|
| K32 | 0.2896951825 | 0.5518001349 | 0.8414953236 | 0.2897 / 0.5518 / 0.8415 |
| K56 core | 0.6221541295 | 1.0496483450 | 1.6718024255 | 0.6222 / 1.050 / 1.672 |

这恢复了 evaluation → checkpoint 路径 → b1 model class 的明确关联，但 summary 没有提供当时 checkpoint 的历史哈希，不能把当前归档哈希当作历史 byte-identity 证明。作者仍需确认这些结果在最终论文中的方法名称/角色，或提供真正对应的 MoE 评估记录。没有修改 Table 2 名称、数据或 checkpoint 路径。

Pinball H 的已有报告仅为 validation，不能用其中 joint=0.38278% 替代论文 final-test joint=2.085%；其最终结果来源仍需补齐。

## 3. 六维 descriptor 与实现差异

附录 D 仍按作者当前稿保留六个 descriptor，不添加通道、不改公式。此前 centered-square gate 审计为 8 个 descriptor + 1 个 parameter；两者不仅维数不同，部分变换/特征定义也不同。
需作者决定如何解释当前数学定义与实际评估网络的关系。本轮没有按审计强制将 6 改为 8。

## 4. 通用物理公式与 benchmark-specific 实现

正文压力公式现与当前附录 C 的 gated PP contribution + correction 一致，structured expert 参数化也统一了。这只解决稿件内部冲突。
此前审计还提示 Square H 最终 rollout 路径为直接学习 RHS/压力，而不是统一加性 Galerkin RHS；其他 baseline 也未证明只改变单一结构因素。需要作者决定是否增加 benchmark-specific implementation 说明，不应以语言润色抹去差异。

## 5. AI use statement 必须如实确认

当前 `main.tex` 仍写 AI 工具 `were used solely to assist with language polishing and LaTeX formatting`。
当前任务历史还包含 AI 辅助代码检查、实验统计/分析、图示及论文整合请求，因此仅限语言和排版的表述可能过窄。作者应核对实际采用的工作范围并给出真实声明。本轮没有自行判断投稿政策，也没有替作者确认该声明准确。

## 6. Figure 1 位图内嵌旧术语

`figures/ral_moe_rom_overview_user_20260908.png` 内仍有 `nonlinear + affine + low-rank bilinear`，而正文与附录已统一为 linear。正文 caption 已更新，但位图内文字无法通过 tex 搜索替换。需要下一步修改图源或由作者提供新版图；本轮保留原图，未静默重绘。

## 7. 投稿前还需作者核验

- Table 4 / Appendix F 的来源与均值/范围关系仍需原始逐-Re结果支撑；本轮按用户限制不改其数字。
- operational divergence criterion 的具体阈值与各实验配置仍应在补充实现中可追溯，不等于理论稳定性测试。
- supplementary implementation 是否实际随稿提供，以及 AI statement 所在页是否符合实际投稿要求，应由作者确认。本轮没有查询或推断 venue policy。

上述事项需明确证据或作者选择，不能仅为了“最终版”字样自动修正。
