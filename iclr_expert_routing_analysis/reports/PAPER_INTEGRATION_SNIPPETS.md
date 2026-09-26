# 论文整合草案：必须保留模型身份与统计口径

## 1. 推荐主文版本：仅讨论已复现 headline 的 Hopf/Periodic

以下段落可以用于原计划的 internal-routing paragraph，但图的面板范围和编号须与最终论文一致。当前完整图包含独立 S4 列，不应直接把三列全部称为论文中已确认的 Proposed specialists。

```latex
\paragraph{Internal routing behavior.}
We analyze the fixed Hopf and Periodic checkpoints post hoc on their held-out
rollout windows, sampling the router once at the first stage of each native
integration step. The Periodic model uses multiple group--expert configurations:
the velocity and pressure channels visit 33 and 16 distinct selected pairs,
respectively, with group identity included in each pair. In contrast, the Hopf
model, which has a single group by construction, selects one fixed pair per
channel throughout the evaluated windows. Its velocity routing mass is strongly
concentrated in one routed expert, whereas the pressure weights vary within a
fixed selected pair. These findings characterize chart-dependent use of a
structured sparse expert family, rather than establishing strong conditional
expert specialization in every chart. Expert identities are not assigned
unique physical meanings.
```

解释：这里 33/16 是 Periodic 宏步 stage-1 口径；如果改用全部四阶段速度 forward，速度 pair 数必须改为 36，并相应修改采样说明。Hopf 的“fixed pair”在全部四个阶段都成立。

## 2. S4 只作为单独、明确标记的诊断

```latex
\paragraph{Independent Steady checkpoint diagnostic.}
We additionally inspect the specified Steady S4 checkpoint as an independent
diagnostic. This checkpoint is not identified with the Steady reference used
in the existing manuscript tables. Its pooled routing visits multiple groups
and expert pairs, but the usage is not uniformly distributed across Reynolds
numbers: group 1 accounts for 95.83\% of the macro-step decisions at
$\mathrm{Re}\approx45.142703$. These statistics characterize S4 only and must
not be attributed to a different Steady checkpoint.
```

不要删除第二句后将此段混入论文 Proposed Steady 结论。若未来明确改用 S4，仍需统一场误差表与严格验证口径；本轮未完成这种论文模型替换。

## 3. 当前完整三列诊断图的 caption

```latex
\caption{
Post-hoc internal routing statistics for the specified circular-cylinder
checkpoints. Columns show an independently diagnosed Steady S4 checkpoint,
Hopf H4, and the Periodic epoch-85 checkpoint; rows show velocity and pressure.
Bars report mean routed-expert mixing mass over held-out rollout windows,
with colors indicating Reynolds numbers. One routing sample is taken per
macro-step at stage 1; all zero-use experts are retained on the horizontal
axis. The upper annotations give pooled group-selection fractions, and the
panel annotations give the fixed shared mixing coefficient and the number
of observed group--pair configurations. Hopf has one group by construction
and fixed selected pairs. The S4 column is not a reproduction of the current
manuscript's Steady reference. Identities are local to each chart, channel,
and group, and mixing mass is not an attribution of physical output magnitude.
}
\label{fig:expert_routing_diagnostic}
```

## 4. Appendix 中建议给出的统计定义

```latex
For $N$ routing decisions, we report normalized selection counts
$f_e=\sum_n I_{n,e}/(2N)$ and mean routed mass
$m_e=N^{-1}\sum_n\omega_{n,e}I_{n,e}$.
Expert indices include group identity, and zero-use experts are retained.
The utilization entropy is $-\sum_e f_e\log f_e/\log N_E$.
We separately report the largest selected-pair fraction because a fixed
Top-2 pair has nonzero utilization entropy. Shared coefficients are fixed
by the architecture and are not interpreted as learned shared-expert
importance. All-forward velocity statistics include all four RK4 stages;
the pressure update uses only stage-1 pressure outputs.
```

## 5. 可替换的 limitations 句子

仅可将“完全未提供 post-hoc routing utilization”的绝对说法，改成与实际分析范围一致的表述，不可据此删掉其他实验限制：

```latex
Our post-hoc routing analysis is limited to the specified checkpoints and
finite held-out rollout windows. It reveals fixed-pair and mass concentration
in the Hopf model, and does not establish a physical interpretation or a causal
performance contribution for individual experts. Multi-seed robustness,
parameter-matched capacity controls, and interventions on routing remain
outside the scope of this diagnostic.
```

如果主文使用本轮 S4 列，还必须另外解释它与当前 Steady reference 的关系。当前三列图不能证明“三个论文 specialists 均已完成一致 checkpoint 的内部路由诊断”。

## 6. 不应沿用的旧模板句

- `The routed-expert utilization is neither uniform nor collapsed to a single configuration across all evaluated states.` 不应作为涵盖三个模型的统一结论，Hopf 固定 pair 与该表达冲突。
- `Shared experts dominate the prediction.` 本轮只测了 mixing coefficients，没有测专家输出的范数、方向或因果贡献。
- `The router identifies vortex shedding / dissipation / Hopf growth.` 本轮没有做相应机制认证。
- `All three checkpoints reproduce the reported headline ranges.` Steady 归属冲突尚在，不能这样写。
- `All three models use RK4.` 指定 S4 原生速度推进为 Euler。

以上只是待整合文本，本轮没有修改 Overleaf 或本地论文源文件。
