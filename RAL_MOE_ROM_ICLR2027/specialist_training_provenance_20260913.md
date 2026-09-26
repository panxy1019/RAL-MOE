# 缺失训练行补全与证据说明

核验日期：2026-09-13。仅补全独立表格，没有修改论文、Table 4、模型或远端训练状态。

## 补全结果

| 训练版本 | Initial lr | Weight decay | 配置的 rollout curriculum | 配置预算 | Validation-selected |
|---|---:|---:|---|---|---:|
| Circular S：S4 | 1.5516372391099407e-5 | 1e-4 | 4,8,16 | 3600 steps | 1200 |
| Circular S：S3-B 回退版（备选，不是另一种 regime） | 1.5516372391099407e-5 | 1e-4 | 4,8,16 | 3600 steps | 600 |
| Pinball S：B1 | 1e-3 | 1e-4 | 4,8,16 | 8000 steps | 400 |
| Pinball H：B1 | 1e-3 | 1e-4 | 4,8,16,32,56 | 8000 steps | 8000 |
| Pinball P：B1 | 1e-3 | 1e-4 | 4,8,16,24,32 | 8000 steps | 8000 |

完整九行 LaTeX 见 `specialist_training_completed_20260913.tex`。原有五行保留。

## Circular S：已查明参数，但不能自动指定论文对应版本

远端根目录：`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/`。

证据：

- `FINAL_REPRODUCIBLE_STEADY_SPECIALIST.md` 明确区分 S4 原始冻结候选与 S3-B 跨硬件保守默认。
- `provenance/checkpoint_selection_manifest.json`：S4 选中 1200；实际用户停止于 3480；最后完整 validation 为 3400。不能把 3480 当 configured budget，或把它当 selected step。
- `code/train_s4_steady.py`：预算 3600；K4/K8/K16 分别配置 600/1000/2000 steps；仅从 S3-B step 600 加载模型参数，重置优化器等状态。
- 本次直接读取三个远端 checkpoint 的 step、optimizer.param_groups 与 scheduler.base_lrs。S4 和 S3-B 主组初始 lr 为 1.5516372391099407e-5，第二组为 1.5516372391099408e-6，两组 weight decay 均为 1e-4。表中仅列主组初始 lr，并以脚注说明第二组。
- S4：`checkpoint/frozen_s4_validation_step_1200.pt`；S3-B：`checkpoint/frozen_s3b_contraction.pt`。

这里的 3600 是该阶段的预算，不是从头训练整个模型的累计成本。S2-B 预训练 checkpoint step 为 8200，不能将其与 S3-B/S4 局部 step 混写。

既有 Table 4 对应关系仍未闭合；遵循用户此前“先不要管 Table 4”的要求，没有更改该表，也没有把 S4 宣称为其确定来源。

## Pinball S/H/P：参数与模型身份分开表述

远端根目录记为 R：`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/`。

| Regime | checkpoint（相对 R） | 课程来源 |
|---|---|---|
| S | `Pinball/steady_b1_rank999_20260806/runs/formal/FluidicPinballV2_Steady_rank999_B1_seed1248/best_validation.pt` | 同项目 `code/train_b1_steady.py:33` |
| H | `Pinball/hopf_b1_rank999_20260806/runs/formal/FluidicPinballV2_Hopf_rank999_B1_seed1248/best_validation.pt` | 同项目 `code/train_kda_pr_fnn_rom.py:34` |
| P | `particalMOE/fluidic_pinball_periodic_v2_b1/runs/FluidicPinballV2_B1_Deep_FNN_H3_seed1248/best_validation.pt` | 同项目 `code/train_b1_fluidic_pinball.py:39` 及 `assets/TRAINING_ASSET_MANIFEST.json` |

本地原始 checkpoint 元数据：`build/network_architecture_audit_20260912/pinball_candidates.json`。其中各自 `metadata.args` 给出 lr、weight_decay、max_steps，`metadata.best_step` 分别为 400、8000、8000。课程来自对应训练器；不能用评估 horizons 替代训练课程。

- S：K4 1600 steps，K8 2400 steps，K16 4000 steps。选中 step 400 位于 K4 阶段，因此“配置全课程”不表示此 checkpoint 已训练过 K8/K16。
- H：K4/K8/K16/K32/K56 分别为 1600/1600/2000/1600/1200 steps。远端 `reports/FLUIDIC_PINBALL_HOPF_B1_TRAINING_ROLLOUT_REPORT_20260806.md` 再次确认总步数与选中步数均为 8000。
- P：K4/K8/K16/K24/K32 分别为 1600/1600/2000/1600/1200 steps。运行 manifest 可覆盖源码默认 STAGES，已有 manifest 与这里的课程一致。

重要边界：这三个文件标记为 B1 Deep-FNN-H3，不是已经确认的 sparse-MoE 权重。故 LaTeX 中保留 B1 标识，不沿用“所有 verified regime-local specialists”这一容易误解的表述。

S/P 的已有评估汇总与论文数字有对应线索，但不等于已经完成历史 checkpoint 哈希闭环。H 的已找到评估明确为 validation-only，K56 joint 为 0.38278%，不能拿它替代论文 final-test 的 2.085%，两者是不同数据划分。当前不能据此证明或否定论文结果；缺少对应 final-test 记录与冻结权重的关联证据。

## 使用建议

若只是补齐已恢复训练记录，可使用提供的九行表与脚注。若要将表标题恢复为“论文全部最终 specialist 的确定训练设置”，须先确定 Circular S 的 S4/S3-B 版本，并闭合 Pinball 的模型身份及 Hopf final-test 来源；不能为了表格完整而省略这些区别。
