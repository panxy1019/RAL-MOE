# Final experiment status

Generated mechanically from frozen artifacts. No model selection or threshold was changed.

| Regime | Method | Status | Source key | Result path |
|---|---|---|---|---|
| Steady | Vanilla-FNN-MoE | EARLY_STOPPED_STEP_6200 | steady_vanilla | /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/steady/vanilla-fnn/frozen_user_stop_step6200_20260724/evaluation/test/metrics.json |
| Steady | DataOnly-MoE | DONE | steady_data | /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/steady/data-only/evaluation/test/metrics.json |
| Steady | Proposed Specialist MoE | DONE | steady_proposed | /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/portability_runs/best_checkpoint_evaluation_retry/heldout_metrics.json |
| Hopf | Vanilla-FNN-MoE | BLOCKED_VALIDATION_NONFINITE | hopf_vanilla_validation | /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/hopf_v6/hopf/vanilla-fnn/evaluation/validation/metrics.json |
| Hopf | DataOnly-MoE | DONE_WITH_DIVERGENT_WINDOWS | hopf_data | /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/hopf/data-only/evaluation/test/metrics.json |
| Hopf | Proposed Specialist MoE | DONE | hopf_proposed | /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/heldout_metrics.json |
| Periodic | Vanilla-FNN-MoE | DONE | periodic_vanilla | /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/periodic_v4/periodic/vanilla-fnn/evaluation/test/native/periodic_r32_multihorizon_evaluation.json |
| Periodic | DataOnly-MoE | DONE | periodic_data | /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/periodic/data-only/evaluation/test/metrics.json |
| Periodic | Proposed Specialist MoE | DONE | periodic_proposed | /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json |

Important execution facts:

- Exactly one fixed seed was used per new ablation; no mean±std across seeds exists.
- Steady Vanilla is a user-requested early stop at optimizer step 6200 and is not a strict qualified checkpoint.
- Hopf Vanilla failed the validation long-rollout contract; test access was therefore blocked.
- Periodic Vanilla and all three DataOnly jobs reached terminal evaluation states.
- Proposed specialist results are the previously frozen native evaluations.

Source: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/steady/vanilla-fnn/FROZEN_SELECTION.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/hopf_v6/hopf/vanilla-fnn/evaluation/validation/metrics.json`; `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/periodic_v4/periodic/vanilla-fnn/FROZEN_SELECTION.json`.
