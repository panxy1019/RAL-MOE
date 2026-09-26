# Routing diagnostics

E2 is a trajectory-level Re-only `temporal-consistent regime router`; it is not evidence of learned physical S→H→P transitions. Its frozen test classification has balanced accuracy=1.0, macro-F1=1.0, NLL=0.183816, Brier=0.093280, ECE-10=0.150093, and minimum Top-1 margin=0.004595.

| Re | alpha_S mean | min | max | alpha_H mean | T2-C | E2 | gap to matched oracle |
|---|---|---|---|---|---|---|---|
| 42.359071 | 0.998749 | 0.995923 | 0.999986 | 0.00125061 | 0.0170192 | 0.017103 | 0.00101225 |
| 43.5 | 0.0861078 | 0.000459023 | 0.411899 | 0.913892 | 0.0219714 | 0.0185409 | 0.00455084 |
| 43.9 | 0.189098 | 8.87217e-10 | 0.945491 | 0.810902 | 0.00753485 | 0.0194379 | -0.0102174 |

T2-C uses Top-2 on all S–H cache windows. It beats E2 on 2/3 test Re values, not all three. RiskPrediction and LookAhead produced the same final predictions as their shared convex gate; removing their critic/lookahead modules did not change output error, so they provide no demonstrated incremental value here.

H–P Top-2 is disabled by the admissibility mask. This is a method outcome, not a missing favorable experiment.

Sources: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/trajectory_router_e1_e2_e3_20260722/E1_E2_E3_TOP1_REPORT.md`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/final_test_evaluation/EVALUATION_REPORT.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/supplements/per_re_matched_oracle_v4/PER_RE_MATCHED_ORACLE_REPORT.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_hp_20260723_v1/H_P_RECOVERY_REPORT_20260723.md`.
