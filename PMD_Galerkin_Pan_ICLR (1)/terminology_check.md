# 第三轮术语检查

范围为 main.tex、sections/*.tex、appendix/*.tex；不搜索旧版快照或报告，位图内文字另列待确认。

| 原表述 | 本轮表述/结果 |
|---|---|
| bilinear responses | quadratic responses；旧短语含跨行搜索为 0，数学公式未动。 |
| Reynolds-number trajectories | parameter trajectories；残留 0。 |
| regime cover | regime coverage；prose 残留 0，内部 eq:app_regime_cover 标签保留。 |
| data partitions | data splits；残留 0。 |
| fixed-run | validation-selected；残留 0。 |
| Periodic core | Periodic-core；残留 0。 |
| Equal / equal weighting | 方法名统一 Equal blend。 |
| E2 prob. / E2-probability weighting / blending | 方法名统一 E2-probability blend。Table 3 仅作表头换行，Table 8/12 同名。 |
| Oracle（表头） | Convex oracle；普通解释文字可写 diagnostic convex oracle。 |
| convex correction（场组合） | convex fusion；旧短语残留 0。 |
| correction（Delta_psi） | logit correction，仍保留网络校正语义。 |
| physics backbone | 不新增；保留 projected Galerkin backbone 与 projected ROM backbone 两层术语。 |
| macro-step | 使用 reduced time step；旧词残留 0。 |

Table 13 表头现在自解释：Active experts / Distinct Top-2 pairs / Dominant-pair share (%) / H。Caption 使用动态 equation reference 指向 Eq. (58)，不手写易失效的编号。

标题源码 RAL-MoE-ROM 原已正确；PDF 模板的大小写变换保持不变。

Figure 1 的用户位图内嵌文字未改，本轮纯 tex 术语检查不能宣称覆盖图内英文；详见 unresolved_items.md。
