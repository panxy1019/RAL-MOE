# Global MoE 同口径单检查点复评报告

日期：2026-09-26  
状态：**已完成并通过独立重算**  
执行边界：复评一个既有 Global MoE 检查点；未重训；未修改论文或旧结果

## 1. 结论先行

在冻结的共同窗口、共同物理空间参考和相同聚合协议下，Global MoE 的单检查点结果为：

| Regime | K | Eu (%) | Ep (%) | Eu+Ep (%) | 接受／尝试检查点 | 完成／预定窗口 |
|---|---:|---:|---:|---:|---:|---:|
| Hopf | 24 | 0.0197 | 0.1117 | **0.1314** | 1/1 | 42/42 |
| Hopf | 48 | 0.0225 | 0.1074 | **0.1299** | 1/1 | 42/42 |
| Periodic | 24 | 0.5389 | 2.1746 | **2.7135** | 1/1 | 32/32 |
| Periodic | 48 | 0.8062 | 3.0927 | **3.8990** | 1/1 | 32/32 |

全部 3552 个逐步记录均为有限值，没有删除或替换失败窗口。由于只有一个训练检查点，这些数值**不带误差棒**，不能把窗口间变化或重复推理当成训练种子标准差。

与当前 Circular 对照结果进行描述性比较时：

- Hopf：Global MoE 的联合误差高于 repaired Proposed 和 OpInf，但与部分 original/Structured 对照处于相近量级；不同模型的接受种子数不一致，因此不能作统计优越性判断。
- Periodic：Global MoE 优于 original Proposed 和 Structured，但仍逊于 active-matched Dense（K24：1.3251%，K48：2.3998%）。
- 因此结果不支持“Global 一定优于 local”或“local 一定优于 Global”的统一结论；它补充的是既有模型在相同窗口和参考下的复评。

## 2. 冻结评价协议

窗口来自：

```text
external_rnn_fno_20260923/results/split_manifest.json
SHA-256 c6e4c795f1920e586c59e2936efce1b1ed24882c04c8c50d9568612931b75c20
```

窗口组成：

| Regime | Re | 窗口数 | 每窗口步数 |
|---|---:|---:|---:|
| Hopf | 47.0813545644 | 14 | 48 |
| Hopf | 49.0223566571 | 14 | 48 |
| Hopf | 51.7864496836 | 14 | 48 |
| Periodic | 70.3146353337 | 8 | 48 |
| Periodic | 100.352251335 | 8 | 48 |
| Periodic | 149.059229449 | 8 | 48 |
| Periodic | 189.86227838 | 8 | 48 |

每个窗口只读取三帧初始历史，此后自主推进 48 步；K24 是同一次 K48 rollout 的前 24 步。预测与参考均在共同物理空间比较，参考为相应 Hopf/Periodic local POD 重构场。预测压力与参考压力分别去除面积加权均值。评价代码显式检查速度和压力参考范数为正，没有添加分母 epsilon。

聚合顺序严格为：步内物理场误差 → 窗口内前 K 步平均 → 同一 Re 内窗口平均 → 不同 Re 等权平均。联合误差在每个检查点层面定义为聚合后的 Eu+Ep。

## 3. 模型身份

检查点：

```text
/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/
paper_experiments/revisions/revision5_supplemental_evaluation_20260725/
recovery/global_k56_valid_history_windows_v5/
V16_1_SteadyPressureAnchor32_ru32_rp32_Re_24p630436_checkpoint.pt
```

```text
checkpoint SHA-256:
9d7e2b861d8c501eff87a940b93f7fe64ca72fd976afc0c6fc1621ff8f0b284f

source evaluator SHA-256:
eb157fde33fd6d7236444a482cef67c5a42e60526b797336bc1677929b941a0f
```

核实信息：

- 模型类：`OperatorSpaceMoEROM`；
- 单一 Global POD 空间，速度/压力维数均为 32；
- 历史长度 3，phase harmonics 4，RK4 推进；
- checkpoint best epoch 225，best validation score 0.4820905955；
- 总参数量与可训练参数量均为 48,617,547；激活参数量未做可靠执行路径统计，因此不填；
- 七个目标 Re 均列在 checkpoint 的 `holdout_Re` 中；
- 源码默认 seed 为 1234，但 checkpoint 不保存实际启动 seed，也没有找到相应启动记录，因此只能标为 `single checkpoint`，不能把 1234 宣称为已验证训练种子。

模型使用全局 POD、按 Re 均值场以及 period/phase 特征。其训练数据、辅助输入、归一化和参数预算与 local specialists 不完全相同。因此，统一评价口径不等于严格的单因素 global-vs-local 消融。

## 4. 逐 Re 结果

| Regime | Re | K | Eu (%) | Ep (%) | Joint (%) |
|---|---:|---:|---:|---:|---:|
| Hopf | 47.0814 | 24 | 0.0272 | 0.1369 | 0.1641 |
| Hopf | 49.0224 | 24 | 0.0250 | 0.1676 | 0.1926 |
| Hopf | 51.7864 | 24 | 0.0068 | 0.0307 | 0.0375 |
| Hopf | 47.0814 | 48 | 0.0315 | 0.1277 | 0.1592 |
| Hopf | 49.0224 | 48 | 0.0259 | 0.1549 | 0.1808 |
| Hopf | 51.7864 | 48 | 0.0101 | 0.0395 | 0.0496 |
| Periodic | 70.3146 | 24 | 0.8834 | 4.3624 | 5.2458 |
| Periodic | 100.3523 | 24 | 0.3264 | 1.2960 | 1.6223 |
| Periodic | 149.0592 | 24 | 0.3414 | 1.1307 | 1.4721 |
| Periodic | 189.8623 | 24 | 0.6043 | 1.9095 | 2.5138 |
| Periodic | 70.3146 | 48 | 1.4936 | 6.2969 | 7.7905 |
| Periodic | 100.3523 | 48 | 0.4619 | 1.8373 | 2.2992 |
| Periodic | 149.0592 | 48 | 0.4681 | 1.6218 | 2.0899 |
| Periodic | 189.8623 | 48 | 0.8014 | 2.6149 | 3.4163 |

Periodic 的主要困难集中在 Re=70.3146，尤其是压力误差；总平均按 Re 等权，因此该 Re 与其他三个 Re 权重相同。

## 5. 独立校验

主聚合之外，`independent_verify.py` 从逐步 CSV 独立重算：

- 目标键与实际键均为 3552，完全一致；
- 74 个窗口均完整，3552/3552 行为有限值；
- 独立结果与主聚合最大绝对差为 `8.8818e-16`；
- 抽查 12 个“跨基 Gram vs 直接物理解码”误差，最大差为 `1.3329e-11` 个百分点；
- 校验结果 `passed=true`。

这说明窗口映射、跨基物理场误差和聚合实现相互一致，但不消除模型训练资料和辅助输入差异带来的公平性限制。

## 6. 与旧 Global 数值的关系

旧报告的 Periodic Global 数值为 12 个窗口、第 48 步末步误差：Eu=1.6072%、Ep=5.8646%、Joint=7.4719%。它既不是本次 32 个窗口，也不是全窗口平均，不能直接填入当前表。

用旧 12 窗口逐步记录做聚合器烟雾测试时，得到 K24 joint=3.5008%、K48 joint=4.4435%。这些数值只用于验证“全窗口平均”和“末步误差”不同，不作为本次论文结果。

## 7. 论文可用 LaTeX

联合误差表：

```latex
Hopf & Global MoE (single checkpoint)
& 0.1314 & 0.1299 & 1/1 \\
Periodic & Global MoE (single checkpoint)
& 2.7135 & 3.8990 & 1/1 \\
```

Periodic 分量误差表：

```latex
Global MoE (single checkpoint)
& 0.5389 & 2.1746 & 0.8062 & 3.0927 \\
```

建议在 caption 或正文注明：结果来自一个既有检查点，不报告训练种子标准差；它是同窗口复评，而非严格参数、训练资料和辅助输入匹配的单因素消融。

## 8. 交付文件

- `single_checkpoint_eval/per_step_errors.csv`：逐步 Eu/Ep、窗口键和 finite 标志；
- `single_checkpoint_eval/window_mapping.csv`：sealed-test 与 Global 数组映射；
- `single_checkpoint_eval/model_identity.json`：检查点与模型身份；
- `single_checkpoint_eval/cross_basis_direct_decode_audit.json`：跨基直接解码抽查；
- `final_aggregate/per_window.csv`、`per_re.csv`、`per_seed.csv`、`final_summary.csv`：各层聚合；其中 `per_seed.csv` 的 `seed` 字段在本任务中标识单个 checkpoint；
- `final_aggregate/recomputation_audit.json`：主聚合完整性审计；
- `independent_verification.json`：独立重算；
- `evaluate_global_single.py`、`aggregate_global_results.py`、`independent_verify.py`：复现代码；
- `target_window_steps.csv`、`target_manifest_audit.json`：冻结目标键；
- `paper_rows.tex`：可直接复制的表格行。
