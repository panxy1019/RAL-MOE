# Specialist ablation objective report

## Table A1 — transient physical-field prediction

All errors are relative physical-field errors. Values are means over the Re-level summaries exposed by each frozen native evaluator. `long` is K56 except Periodic Proposed, whose frozen native result ends at K48. A finite fraction of 1 does not imply non-divergence: the DataOnly evaluator separately flags trajectories exceeding its 10× norm rule.

| Regime | Method | u K1 | p K1 | u K16 | p K16 | Long | u long | p long | joint long | worst long | finite | divergent | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Steady | Vanilla-FNN-MoE | 0.000219165 | 0.0320706 | 0.000556418 | 0.0244566 | k56 | 0.000842663 | 0.0409119 | 0.0208773 | 0.038043 | 1.00000 | 0 | EARLY_STOPPED_STEP_6200 |
| Steady | DataOnly-MoE | 0.000898074 | 34.38869 | 0.00127414 | 96.10022 | k56 | 0.00256885 | 143.689 | 71.84574 | 136.380 | 1.00000 | 3.00000 | DONE |
| Steady | Proposed Specialist MoE | 0.000219165 | 0.014307 | 0.000550593 | 0.0256445 | k56 | 0.00124462 | 0.0703616 | 0.0358031 | 0.0948226 | 1.00000 | 0 | DONE |
| Hopf | Vanilla-FNN-MoE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | BLOCKED_VALIDATION_NONFINITE |
| Hopf | DataOnly-MoE | 6.11401e-05 | 0.94632 | 0.000724979 | 4.99736 | k56 | 0.00220375 | 14.08848 | 7.04534 | 9.45309 | 1.00000 | 39.00000 | DONE_WITH_DIVERGENT_WINDOWS |
| Hopf | Proposed Specialist MoE | 5.22382e-06 | 0.000661924 | 3.65463e-05 | 0.00172596 | k56 | 0.000126969 | 0.00169688 | 0.000911926 | 0.00230461 | 1.00000 | 0 | DONE |
| Periodic | Vanilla-FNN-MoE | 0.00406129 | 0.0226127 | 0.00900079 | 0.0478907 | k56 | 0.0252646 | 0.105334 | 0.065299 | 0.0908335 | 1.00000 | 0 | DONE |
| Periodic | DataOnly-MoE | 0.00272869 | 0.0107529 | 0.00683409 | 0.0260322 | k56 | 0.0109615 | 0.0461192 | 0.0285403 | 0.105976 | 1.00000 | 0 | DONE |
| Periodic | Proposed Specialist MoE | 0.00186043 | 0.010377 | 0.0033029 | 0.0164071 | k48 | 0.00813182 | 0.0344454 | 0.0212886 | 0.0365367 | 1.00000 | 0 | DONE |

## Objective interpretation

- Steady: the early-stopped Vanilla checkpoint has lower K56 mean joint error than the frozen Proposed Steady checkpoint (0.0208773 vs 0.0358031). This ablation therefore does not support a blanket claim that the proposed expert architecture is superior in Steady long-horizon field error. DataOnly has finite arithmetic outputs but severe pressure error and three divergent K56 windows.
- Hopf: Proposed has low physical-field error and zero divergence on the native held-out split. DataOnly is finite in the NaN/Inf sense but has 39 divergent K56 windows and large pressure error. Vanilla cannot be compared on test because validation K16 contains non-finite trajectories.
- Periodic: Proposed improves over Vanilla at K1/K16/K48 and preserves 3/4 held-out attractors versus 1/4 for Vanilla. DataOnly has competitive short-horizon field error but lacks the certified phase/frequency/orbit contract.

These statements are single-seed observations, not uncertainty estimates.

Primary sources: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/steady/vanilla-fnn/frozen_user_stop_step6200_20260724/evaluation/test/metrics.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/steady/data-only/evaluation/test/metrics.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/portability_runs/best_checkpoint_evaluation_retry/heldout_metrics.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/hopf/data-only/evaluation/test/metrics.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/heldout_metrics.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/periodic_v4/periodic/vanilla-fnn/evaluation/test/native/periodic_r32_multihorizon_evaluation.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/periodic/data-only/evaluation/test/metrics.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json`.
