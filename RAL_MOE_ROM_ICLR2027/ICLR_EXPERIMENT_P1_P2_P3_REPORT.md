# ICLR 实验 P1 / P2 / P3：前置核对报告

日期：2026-09-11。状态：**前置来源核对未通过，按任务文档停止；不是实验完成报告。**

## 1. 结论与执行边界

本轮已读取任务文件、检查本地论文和评估代码，并通过 SSH 只读检查集群历史记录。没有启动新训练，没有替换 checkpoint，没有修改论文数值、流程图或 Overleaf。

采用项目执行管理和代码修改技能的“小步验证、保留原结果”流程。本轮在正式改表前暴露出两项需要明确处理方向的问题：

1. **Table 4 的当前数值不能由已找到的逐 Re 历史表统一重现。** 尤其 Steady 的 proposed / vanilla / global 标签与速度、压力区间对应关系不一致。历史表并非已经确认的当前论文原始输出，因此不能据此直接宣称哪一行是正确答案，也不能自行交换行列。
2. **当前 Appendix D 的 6 维描述量、7 维输入，与原始 centered-square 缓存中的 8 维描述量、9 维输入不一致。** 不仅维数不同，特征变换定义也不同。原始 gate/cache/router 文件的 SHA-256 与原报告一致，原报告数值确实对应 1.1958% / 0.8465%。不能按 7 维新网络重训后冒充原实验的公平消融。

任务第 19、26 节要求此类来源/回归问题出现时停止，不自行修正数据。因此本轮交付核对脚本、机器可读结果及本报告；尚未交付新实验、均值标准差表或新 PDF。

## 2. 文件与来源

本地论文根目录：`C:\Users\panxy1019\Documents\CHANNEL\RAL_MOE_ROM_ICLR2027`。

本地 centered-square 根目录：`C:\Users\panxy1019\Documents\CHANNEL\centeredsquare_fusion_v1`。

原运行目录：`results/E2_T2C_K24_20260730_STRICT_V3`，下文简称 `RUN`。

已检查的关键本地源码：

- `centeredsquare_fusion_v1/evaluate_heldout_rollout.py`：`errors`、`metrics`、`oracle_weights`、`evaluate_boundary`。
- `centeredsquare_fusion_v1/common.py`：`pressure_gauge`、`physical_descriptors`、`quadratic_statistics`、`ConvexGate`、`blend_relative`。
- `centeredsquare_fusion_v1/build_boundary_cache.py`：约 440–465 行物理场重建、有限性及范数阈值；约 509 行拼接特征。
- `centeredsquare_fusion_v1/train_t2c_gate.py`：标准化、训练目标及验证选择。

集群项目公共前缀：

`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/`

圆柱历史逐 Re 表：

`particalMOE/paper_experiments/revisions/revision8_detailed_per_re/REVISION8_DETAILED_ROWS.csv`

已复制到本报告旁的 `artifacts/experiment_p123_20260911/sources/REVISION8_DETAILED_ROWS.csv`，供离线复查。**这是已四舍五入的历史汇总 CSV，不是已批准用于当前论文统计的全精度原始数据。**

还检查了 revision10 目录；存在 DataOnly completion JSON 和 Global 原生评估 CSV。Global 的 `error_vs_re.csv` 包含 coefficient、energy 等字段，不能把这些列直接当物理场相对误差。revision10 的存在说明不能把 revision8 的差异全部解释为论文错误；当前任务需要先建立完整的 checkpoint → evaluator → field metric → paper cell 对应关系。

## 3. A–B：已经恢复的 centered-square 误差与聚合定义

以下精确定义仅针对已确认的 centered-square overlap evaluator，**尚不能推广到所有 benchmark**。

设两个候选物理场为 $q_1,q_2$，参考场为 $q_*$，每窗口固定权重为 $\alpha$。缓存保存：

$$
A=\|q_1-q_*\|_M^2,\quad B=\|q_2-q_*\|_M^2,
\quad C=\langle q_1-q_*,q_2-q_*\rangle_M,\quad T=\|q_*\|_M^2.
$$

逐窗口、逐步的百分比误差为：

$$
E_q(w,k)=100\sqrt{\max\left(\frac{\alpha_w^2A_{wk}+(1-\alpha_w)^2B_{wk}+2\alpha_w(1-\alpha_w)C_{wk}}{\max(T_{wk},10^{-12})},0\right)}.
$$

速度使用单元面积对两个分量加权求和；压力先对参考和两个候选分别减去面积加权空间均值。缓存参考物理场由 source 的原生参考系数及基底重建；本轮没有独立证明该重建与未投影 CFD 快照逐点等同。

注意实际分母是**平方范数截断后开根号**，不是任务示例中的 `norm + epsilon`。不同公式不能直接混写。

`E_joint = E_u + E_p`。原 JSON 存相对比例，论文乘 100 后显示百分数。

Table 3 的 K24 数值是：

$$
\overline E_q^{K24}=\frac{1}{N_w}\sum_w E_q(w,24),
$$

即所有窗口第 24 步的 pooled mean，**不是 24 步平均**。逐 Re 值也是该 Re 各窗口第 24 步的均值。S–H、H–P 各有两个 held-out Re，每个 Re 8 个窗口，共 16 个，因此这两个具体评估中 pooled mean 与 Re-balanced mean 数值相等，但代码实施的是前者。

JSON 的 `all_step_joint_mean` 是另一项统计，不能替换 Table 3 的 `by_horizon.K24.joint_mean`。

## 4. C：发散判据恢复进度

### 4.1 Centered-square overlap cache：已确认

`build_boundary_cache.py` 分开记录：

- finite：预测速度系数和压力系数在所有预测步均有限。
- divergent：任一步物理场归一化范数超过 20。

其中

$$
D_{wk}=\max\left(\frac{\|\widehat u_{wk}\|_{M_u}}{\max(\|u_{wk}\|_{M_u},10^{-8})},
\frac{\|\widehat p^\circ_{wk}\|_{M_p}}{\max(\|p^\circ_{wk}\|_{M_p},10^{-8})}\right).
$$

阈值检查为 `any(D > 20)`。缓存构建要求所有候选 finite 且没有阈值超限。不能把单独的 `divergent` 字段误说成它已经包含全部 NaN/Inf 情形。此代码两个 overlap 使用相同阈值；阈值如何在先前验证阶段决定，本轮尚未恢复记录。

### 4.2 Pinball：只完成局部源码核查

在以下文件发现的是**系数范数比阈值 10**，不是物理场范数比阈值 20：

- `Pinball/steady_b1_rank999_20260806/code/evaluate_b1_rollout.py`，约 180–184 行。
- `Pinball/hopf_b1_rank999_20260806/code/evaluate_hopf_validation_rollout.py`，约 131–136 行。

其形式为任一步 `max(||a_pred||/clamp(||a_true||), ||b_pred||/clamp(||b_true||)) > 10` 或预测系数含 NaN/Inf。所读 Steady/Hopf B1 代码先在各 Re 内对时间和窗口平均，再对 Re 平均。

**尚未完成这些 B1 文件与 Table 2 POD–Galerkin 的 448/448、141/320 原始输出的逐项关联**，也未确认 baseline 的 EPS、全部 split 与阈值来源。因此不能据上述局部代码直接完成 Table 2 的正式定义。Hopf 文件名含 validation 是需要检查的线索，不足以单独认定论文使用了错误测试集。本轮不修改对应措辞。

## 5. D：centered-square Convex Oracle 的真实目标

每窗口在 $G=\{0,10^{-4},\ldots,1\}$ 共 10001 个点上求：

$$
\alpha_w^{oracle}=\operatorname*{argmin}_{\alpha\in G}\frac1{24}\sum_{k=1}^{24}\left(E_u(w,k;\alpha)+E_p(w,k;\alpha)\right).
$$

源码使用相对比例而非百分数，不影响最优权重。网格按升序排列，`np.argmin` 同分取第一个，随后保存为 float32。它是离散网格搜索，不是已证明精确求解的连续区间最小化。

Oracle 用**所有预测步**选择权重，而论文 Table 3 报告其 **K24 末步**误差。因此不能将它宣称为 K24 指标的严格最优下界。

T2-C 训练损失则是速度、压力的**平方归一化场误差**之和，跨 batch 和所有步平均；不是 oracle 使用的开根号后误差之和。验证选择采用 `(worst_Re_mean, joint_mean)` 字典序。训练目标、验证目标和最终末步报告指标必须分别说明。

## 6. E–G：Table 4 来源回归检查

自动比较当前 TeX 表格与 revision8 候选 CSV 的结果：**24 个 model × regime × metric 检查中 11 项相符，13 项未通过**。其中 24 项包括 Vanilla Hopf 两项 N/E。

该统计只表示“这个候选历史表不能整体作为当前 Table 4 来源”，不表示已经证明当前论文有 13 个错误。失败包含数值差异、缺值或 horizon 不一致。

最直接的 Steady 对比（均为百分数）：

| 模型 / 指标 | 当前 Table 4 | revision8 按原标签重新取 min–max |
|---|---:|---:|
| Proposed / Eu | 0.0330–0.1723 | 0.0384–0.2995 |
| Proposed / Ep | 0.2500–1.3249 | 1.5226–18.5603 |
| Vanilla / Eu | 0.0384–0.2995 | 0.0330–0.1723 |
| Vanilla / Ep | 2.2163–18.5603 | 2.2163–7.4363 |
| Global / Eu | 0.0845–0.3404 | 0.0845–0.3404 |
| Global / Ep | 1.5226–7.4363 | 0.2500–1.3249 |

Proposed 历史逐 Re 值为：

| Re | Eu | Ep |
|---:|---:|---:|
| 24.630436 | 0.2995 | 18.5603 |
| 32.740068 | 0.0730 | 5.0620 |
| 39.685479 | 0.0384 | 1.5226 |
| 45.142703 | 0.0618 | 1.9665 |

其他差异包括历史 Steady DataOnly 缺第 4 个 Re 的值、Global Periodic 历史表使用 K56 而当前表使用 K48。这进一步说明不能通过自动重命名几行解决全部来源问题。

**没有计算或发布 mean ± SD_Re**；没有从端点反推；没有从已四舍五入的候选历史表制造全精度统计。待找到经过确认的全精度 per-Re 数据、重现全部旧区间后，才可使用 sample SD（ddof=1）。

## 7. H–J：T2-C 旧结果、维数和消融约束

原 JSON：`RUN/heldout_evaluation_20260730_V1/comparison/HELDOUT_ROLLOUT_COMPARISON.json`。

| Overlap | 原报告 K24 Eu (%) | Ep (%) | E_joint (%) | 四位小数 |
|---|---:|---:|---:|---:|
| S–H | 0.4615099534 | 0.7343308823 | 1.1958408357 | 1.1958 |
| H–P | 0.4509120027 | 0.3956169536 | 0.8465289563 | 0.8465 |

两组 `joint = velocity + pressure` 通过 1e-12 级比例值检查；本地 gate、router、held-out cache 的哈希全部与原报告相符。

**这是原报告和文件身份核对，不是本轮重新前向计算或重新训练复现。** `evaluation_replayed=false` 已写入机器结果。

直接读取 NPZ 的数组头，确认：

| 缓存 | features shape |
|---|---:|
| S–H development | (72, 9) |
| S–H heldout | (16, 9) |
| H–P development | (48, 9) |
| H–P heldout | (16, 9) |

`common.physical_descriptors` 返回：两个对数 RMS 指标、当前及前一步的相对速度变化率的 log1p、能量对数增长率的 tanh、增长率差的 tanh、相对压力变化率的 log1p、速度变化曲率指标的 log1p，共八项。`features` 首列为 Re，其后为这八项。

当前 Appendix D 却定义了六项统计和七维输入，故不仅是一个输入层尺寸笔误。

如用户确认以原始运行实现为准，公平的 parameter-only 实验应保持 **9→64→64→1**，在使用原训练统计完成标准化后，将八个 history 通道置零，保持标准化参数通道和 E2 pair logit 不变；不能在原始特征上置零再用原 mean/std 标准化，因为那会产生非零常量。

这只是待批准的实现约束，**尚未实现或运行**。应同时修正文稿描述，而不是静默修改方法。

已读训练脚本给出的默认设置是 seed 42001、8000 steps、每 100 steps 验证（另含 step 1 和最后一步）、AdamW lr 3e-4、weight decay 1e-4、梯度裁剪 1、有放回采样 min(256, n_train)。未来执行前仍需核对原 checkpoint/log 的实际启动参数，不能把默认值当作实际运行参数的最终证据。

## 8. 已核对的 gate 缓存划分

| Overlap | Train Re | Validation Re | Held-out Re |
|---|---|---|---|
| S–H | 94, 95, 95.05, 95.15, 95.2, 95.35, 95.4 | 94.5, 95.25 | 95.1, 95.3 |
| H–P | 98.5, 99.5, 100, 101 | 99, 101.5 | 100.5, 102 |

两组 gate 缓存内 train/validation/heldout 集合两两不交。**这不等价于已经验证所有 specialist、E2、基底构建、scaler 的全流程无泄漏**；全流程与 Appendix E 的 split 对应仍未完成。

## 9. K–L：新训练状态

- Parameter-only T2-C：未启动。
- Full / parameter-only 三种子实验：未启动。
- Circular Periodic specialist 三种子实验：未启动。
- 没有任何新增均值、seed SD 或 descriptor 有效性结论。
- Limitations 中尚不能删除 repeated-seed uncertainty 的限制说明。

## 10. M–N：交付与不变性验证

新增：

1. `scripts/verify_iclr_experiment_tables.py`：前置核对版本，标准库即可运行；读取当前 Table 4、候选 CSV、原 T2-C JSON、NPZ 元数据和划分，核验哈希、误差加和及原论文文件不变性。
2. `artifacts/iclr_experiment_summary.json`：明确 `BLOCKED_PROVENANCE`、未完成项和 `evaluation_replayed=false`。
3. `artifacts/experiment_p123_20260911/table4_candidate_regression.csv`：逐项比较，不是正式 Table 4 数据。
4. `artifacts/experiment_p123_20260911/sources/REVISION8_DETAILED_ROWS.csv`：只读来源副本。
5. 本报告。

执行命令：

```powershell
python C:\Users\panxy1019\Documents\CHANNEL\RAL_MOE_ROM_ICLR2027\scripts\verify_iclr_experiment_tables.py
```

脚本**有意以非零退出状态停止下游流程**。目前它不实现正式均值表和 seed 表生成器，因为原始数据身份尚未通过；不能把它当作全实验验证已通过。

对 `build/method_intro_20260909/upload_manifest.json` 中 **31 个论文源文件/素材逐一进行 SHA-256 核对，全部保持一致**。没有更改 main.tex、Appendix D/E/F/G、Table 2/3/4、流程图或已有 PDF；没有上传 Overleaf。

没有生成 `table4_mean_std.csv`、`t2c_descriptor_ablation.csv`、`t2c_seed_summary.csv` 或新版匿名 PDF，避免用空表或历史数值冒充已完成实验。

## 11. 建议的下一步，需要用户确定方向

建议先授权一个独立的“来源纠错”阶段：以原始运行代码、checkpoint 与全精度输出为依据，追溯 Table 4 当前数值的来源；如确认为排表或版本混用错误，逐项列出旧值、新值和证据，经确认后修正。不能预先承诺旧 min–max 一定能保留。

同时确认是否把 Appendix D 改成原始运行的八维描述量，并据此进行 `[mu, descriptor8]` 与 `[mu, zero8]` 的参数量匹配消融。若必须保留六维方法定义，则是另一套需要重新训练、重新评估的方法版本，不能与旧结果混称同一 fixed run。

完成上述来源一致性后，再依次做 full T2-C 前向复现、descriptor 消融、三种子统计、完整协议文字、编译与上传。
