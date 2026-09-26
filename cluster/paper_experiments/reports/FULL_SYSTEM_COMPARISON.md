# Full-system comparison

## Table A2

| System | Region | Metric contract | one-step u | one-step p | long/rollout | worst | Interpretation |
|---|---|---|---|---|---|---|---|
| Global MoE | all available heldout Re | modal, not unified physical K56 | 0.0399394 | 0.122906 | 0.225702 | 0.296686 | Not numerically commensurate with S-H cache |
| E2 Top-1 | S-H boundary | area-weighted physical joint | NA | NA | 0.0467098 | 0.156584 | E2 selects S on this S-native cache |
| Multi-chart + T2-C | S-H boundary | area-weighted physical joint | NA | NA | 0.03045 | 0.145023 | window-conditioned convex field blend |
| Multi-chart admissibility | H-P boundary | validation admissibility | NA | NA | NA | NA | H inadmissible; forced P-only / hard routing |

The table is deliberately partial. The available Global MoE artifact reports modal one-step and rollout errors, whereas the S–H sealed cache reports area-weighted physical-field errors at fixed horizons. These numbers are not placed in a single ranking. A fresh common-population physical-field cache would be required for a defensible all-region numerical ranking.

Within the sealed S–H cache, T2-C reduces K56 mean joint error from 0.0467098 (E2/S-only) to 0.03045, but the worst-window statistic is 0.145023 and T2-C is not better at every Re. At H–P, the Hopf expert failed the development K56 admissibility gate (0/6 validation trajectories finite), so the unified system correctly collapses to P-only/native-time hard routing.

Sources: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/V16_1_SteadyPressureAnchor32/reproduction/heldout_eval_frozen_checkpoint_20260723/V16_1_SteadyPressureAnchor32_ru32_rp32_reproduced_metrics.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/final_test_evaluation/EVALUATION_REPORT.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/top2_boundary_experiments_20260722/one_sided_recovery_hp_20260723_v1/H_P_RECOVERY_REPORT_20260723.md`.
