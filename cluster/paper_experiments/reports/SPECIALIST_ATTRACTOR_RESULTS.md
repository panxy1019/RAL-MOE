# Attractor and asymptotic-dynamics analysis

## Table B1 — specialist attractor results

| Regime | Method | Strict attractor result | Notes |
|---|---|---|---|
| Steady | Vanilla-FNN-MoE | No preregistered strict binary count | K56 finite; early-stopped checkpoint |
| Steady | DataOnly-MoE | Generic tail diagnostics only | center=0.0101356, amp=0.0094164 |
| Steady | Proposed Specialist MoE | No preregistered strict binary count | K56 finite; fixed-point metrics available |
| Hopf | Vanilla-FNN-MoE | Not evaluated on test | validation K16 non-finite; fail closed |
| Hopf | DataOnly-MoE | No certified Hopf strict conjunction | center=0.0791413, amp=0.0115456; divergent windows present |
| Hopf | Proposed Specialist MoE | 3/29 train; 0/2 validation; 0/3 heldout | strict train passes: Re=49.3, 49.6, 50.0 |
| Periodic | Vanilla-FNN-MoE | 1/4 heldout | native K48 cycle contract |
| Periodic | DataOnly-MoE | No certified phase/frequency conjunction | center=0.033325, amp=0.0106319 |
| Periodic | Proposed Specialist MoE | 3/4 heldout | native K48 cycle contract |

The three strict Hopf train cases are Re=49.3, 49.6, and 50.0. They are in-sample diagnostics because these Re values participated in training; they must not be described as held-out generalization. The frozen Hopf specialist is finite at every audited native K56 trajectory, but strict attractor preservation is 3/29 train, 0/2 validation, and 0/3 heldout.

## Table B2 — S–H boundary

| Re | Method | Center | Amplitude abs. | Orbit | Tail inc. | Terminal inc. | alpha_S | alpha_H |
|---|---|---|---|---|---|---|---|---|
| 42.35907 | E2 Top-1 | 0.000513327 | 1.84453e-05 | 0.0193748 | 4.55771e-05 | 5.41423e-05 | NA | NA |
| 42.35907 | T2-C | 0.000511037 | 1.8464e-05 | 0.0193919 | 4.55195e-05 | 5.40254e-05 | 0.998749 | 0.00125061 |
| 43.50000 | E2 Top-1 | 0.000551498 | 9.35374e-06 | 0.0161981 | 6.35304e-05 | 8.51417e-05 | NA | NA |
| 43.50000 | T2-C | 0.000393571 | 1.86926e-06 | 0.00245798 | 2.00473e-05 | 1.71103e-05 | 0.0861078 | 0.913892 |
| 43.90000 | E2 Top-1 | 0.000573552 | 8.93236e-06 | 0.0161353 | 6.29081e-05 | 7.21409e-05 | NA | NA |
| 43.90000 | T2-C | 0.000215164 | 4.90723e-06 | 0.00520216 | 1.48479e-05 | 1.64473e-05 | 0.189098 | 0.810902 |

At Re=43.50, T2-C has worse full-window physical-field error than E2 (0.0219714 vs 0.0185409) while improving all reported tail-attractor diagnostics. This is a genuine transient–attractor discrepancy and is retained as a counterexample. At Re=43.90, T2-C strongly improves both the full-window error and the tail diagnostics. Phase/frequency are not reported for this S-native cache because no certified complete-cycle observable exists.

Sources: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/native_attractor_audit_20260723_v1/results/HOPF_NATIVE_ATTRACTOR_AUDIT.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/periodic_v4/periodic/vanilla-fnn/evaluation/test/native/periodic_r32_multihorizon_evaluation.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/supplements/t2c_attractor_v2/T2C_ATTRACTOR_ANALYSIS.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/supplements/per_re_matched_oracle_v4/PER_RE_MATCHED_ORACLE_REPORT.json`.
