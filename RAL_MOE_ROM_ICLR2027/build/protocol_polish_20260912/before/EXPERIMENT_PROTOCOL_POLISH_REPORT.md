# Experiment protocol polish — P0 核查阻塞报告

日期：2026-09-12。状态：**未完成；按指南第 3.2 节 STOP 条件停止论文修改。**

本轮只进行了本地与集群只读核查，并新增本报告。没有修改论文 LaTeX、实验数值、训练代码或原有 PDF/ZIP，也没有重新训练。以下已确认事实不能替代尚缺失的 Table 2 原始评估证据。

## A. E_u / E_p 公式

尚未建立适用于全文的、逐 benchmark 核验完毕的最终公式，因此未写入 Appendix E。Centered-square 融合 evaluator 已确认的单状态计算为：

$$E_u=100\sqrt{\max\left(\frac{\|\widehat{\mathbf u}-\mathbf u\|_{M_u}^2}{\max(\|\mathbf u\|_{M_u}^2,10^{-12})},0\right)},$$
$$E_p=100\sqrt{\max\left(\frac{\|\widehat p^\circ-p^\circ\|_{M_p}^2}{\max(\|p^\circ\|_{M_p}^2,10^{-12})},0\right)}.$$

速度范数同时包含两个分量并以单元面积加权；压力预测和真值均先作共同 gauge 修正。融合通过误差二次型计算，上式对应其数学含义。联合误差为分量误差之和；不得用独立舍入后的显示分量重新生成已有 joint 数字。

## B. epsilon 与来源

Centered-square 使用**参考场平方范数的下限 1e-12**，不是范数分母加 epsilon。源码：项目父目录下 `centeredsquare_fusion_v1/evaluate_heldout_rollout.py:18–25`；物理场、面积权重与 gauge 构建见 `centeredsquare_fusion_v1/build_boundary_cache.py:440–451`。不得把指南示例中的 `norm + epsilon` 直接贴入论文。

## C. benchmark-specific aggregation

- Centered-square Table 3：K=24 终点误差，每个 overlap 16 个 held-out 窗口，两个 Re 各 8 个窗口。保留现有口径。
- 找到的 Pinball Steady/Hopf **B1** evaluator：先在各 Re 内对窗口和预测时间平均，再对 Re 平均。Steady 报告同时列出不同的终点统计，不能混用。
- 上述 B1 源码尚未与 Table 2 的 POD–Galerkin 行逐项关联；Periodic 及 circular-cylinder 的全部发表行也尚未完成本轮逐项闭环核验。不能宣布全文统一协议已经核验完成。

## D–F. divergence indicator、阈值及来源

Centered-square 候选缓存定义：

$$D_{wk}=\max\left(\frac{\|\widehat{\mathbf u}_{wk}\|_{M_u}}{\max(\|\mathbf u_{wk}\|_{M_u},10^{-8})},\frac{\|\widehat p^\circ_{wk}\|_{M_p}}{\max(\|p^\circ_{wk}\|_{M_p},10^{-8})}\right).$$

任一步 `D > 20` 标记 threshold violation。finite 字段另外检查全部预测速度/压力系数是否有限；preflight 要求所有候选 finite 且没有阈值违反。单独的 divergent 字段**不等同于**自动包含 NaN/Inf 的合并判定。

来源：`centeredsquare_fusion_v1/build_boundary_cache.py:452–465,499–505`。这里用于 norm-ratio 的 `1e-8` 与误差平方范数分母下限 `1e-12` 是不同用途。

集群已发现的 B1 evaluator 使用**系数范数比 > 10，或系数非有限**的规则：

- `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/Pinball/steady_b1_rank999_20260806/code/evaluate_b1_rollout.py`
- `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/Pinball/hopf_b1_rank999_20260806/code/evaluate_hopf_validation_rollout.py`

**没有找到足以证明这些 B1 规则就是 Table 2 POD–Galerkin 448/448、141/320 统计规则的原始输出及调用链。** 因而不能填入该基线的 exact indicator、threshold、有限但超阈值窗口的误差处理或 N/E 触发规则。不得把 10 或 20 推广成全文统一阈值。

补充只读检索检查了 Steady 的 `pinball_steady_base.py`、训练/评估代码与报告、Hopf 记录，以及远端相关 baseline/Galerkin 文件名；找到 Galerkin 算子不等于找到产生 Table 2 计数的 evaluator。此缺口也见现有 `ICLR_EXPERIMENT_P1_P2_P3_REPORT.md` 第 4.2 节。

## G. Section 4.1 wording

未修改。待恢复实际判据后，将 `fails` 改为精确的 criterion violation 描述，并使 Table 2 脚注与真实 finite/error/N/E 规则一致。

## H. Section 4.3 wording

未修改。可独立实施的待办：`isolates` → `probes the contributions of`，保留模型差异定义，并明确不是 parameter-matched capacity control。

## I. Table 3 fixed-run / seed scope

未修改。本轮未用 seed 均值替换 fixed-run 数值。待办：caption 明示 fixed-run，三种子 gate-training mean/sample SD 单独描述；不扩大为全流水线不确定性。

## J–K. Appendix J 精简及表格

因 STOP 未实施，现有 Appendix J 原样保留。待删除的是内部 hash/audit/synthetic-forward/native-stage-1/diagnostic-stack 语言；待呈现的 prediction-only 表格如下（本表是修改目标，不代表已写入论文）：

| Specialist | Total trainable | Active prediction | Active fraction |
|---|---:|---:|---:|
| Hopf | 31.81M | 14.17M | 44.53% |
| Periodic | 48.62M | 8.31M | 17.09% |

必须保留 unique trainable、batch-size-one prediction 的计数定义及排除固定算子/基/统计/buffer 的范围；不能由参数比例推导 FLOPs、速度或 runtime，也不能声明容量匹配。

## L. 是否改变任何实验数值

**No。** 没有修改实验记录或论文性能数值。

## M. 是否改变 Table 4

**No。** 没有修改 Table 4、min–max 格式或 descriptor 维数。

## N. unresolved issues 与恢复条件

阻塞项：缺少能对应 Table 2 POD–Galerkin 448/448 和 141/320 的原始 evaluator、冻结配置及评估输出。需要它们来确认 indicator、threshold、任一步/终点规则、有限超阈值窗口是否纳入误差，以及 N/E 条件。

恢复方式：提供这组原始记录的路径，继续证据闭环后再完成 P0；或者明确允许先完成独立的 P1/P2 文案与 Appendix J 精简，P0 保留未解决状态。不能在证据缺失时制造新的判据。

本轮没有生成“已完成修订”的匿名 PDF 或上传 ZIP，以免旧产物被误认为已通过 P0 核验。
