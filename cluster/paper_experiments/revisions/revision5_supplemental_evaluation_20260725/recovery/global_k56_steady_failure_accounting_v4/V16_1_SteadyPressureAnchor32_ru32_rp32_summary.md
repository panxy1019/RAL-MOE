# AttractorMoE-ROM V16 Physics-Generalizable Attractor Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.85; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

V16 pressure input mode: `pressure_only`. V16 keeps the V14/V15 best `[a_t,b_t]` pressure state while changing only attractor losses or the lightweight attractor conditioning layer.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop multi-step rollout, energy consistency, trajectory consistency, pressure closure, relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime supervision.

## Dense Training Split

Test selection: `regime_default`, time stride=1, Re stride=1.

- Train Re count: 89
- Test Re count: 11
- Excluded Re count from Re sparsity: 0
- Dense train samples before time sparsity: 10970
- Kept train samples: 10970
- Validation samples: 1547
- Test samples: 1350
- Compression vs dense train: 1
- Compression vs all non-test candidates: 0.876408

| Re | role | total | dense train | kept train | val | test |
|---:|---|---:|---:|---:|---:|---:|
| 20 | train | 61 | 51 | 51 | 10 | 0 |
| 22.5357 | train | 61 | 51 | 51 | 10 | 0 |
| 24.6304 | test | 61 | 0 | 0 | 0 | 61 |
| 26.6673 | train | 61 | 51 | 51 | 10 | 0 |
| 28.6951 | train | 61 | 51 | 51 | 10 | 0 |
| 30.7204 | train | 61 | 51 | 51 | 10 | 0 |
| 32.7401 | test | 61 | 0 | 0 | 0 | 61 |
| 34.7376 | train | 61 | 51 | 51 | 10 | 0 |
| 36.6578 | train | 60 | 50 | 50 | 10 | 0 |
| 38.3572 | train | 61 | 51 | 51 | 10 | 0 |
| 39.6855 | test | 61 | 0 | 0 | 0 | 61 |
| 40.7115 | train | 61 | 51 | 51 | 10 | 0 |
| 41.5766 | train | 60 | 50 | 50 | 10 | 0 |
| 42.3591 | train | 61 | 51 | 51 | 10 | 0 |
| 43.0939 | train | 61 | 51 | 51 | 10 | 0 |
| 43.7974 | train | 60 | 50 | 50 | 10 | 0 |
| 44.4783 | train | 61 | 51 | 51 | 10 | 0 |
| 45.1427 | test | 61 | 0 | 0 | 0 | 61 |
| 45.7952 | train | 61 | 51 | 51 | 10 | 0 |
| 46.4401 | train | 61 | 51 | 51 | 10 | 0 |
| 47.0814 | test | 158 | 0 | 0 | 0 | 158 |
| 47.7229 | train | 158 | 139 | 139 | 19 | 0 |
| 48.3687 | train | 158 | 139 | 139 | 19 | 0 |
| 49.0224 | test | 158 | 0 | 0 | 0 | 158 |
| 49.6876 | train | 159 | 140 | 140 | 19 | 0 |
| 50.3681 | train | 158 | 139 | 139 | 19 | 0 |
| 51.0668 | train | 158 | 139 | 139 | 19 | 0 |
| 51.7864 | test | 158 | 0 | 0 | 0 | 158 |
| 52.5288 | train | 159 | 140 | 140 | 19 | 0 |
| 53.2942 | train | 158 | 139 | 139 | 19 | 0 |
| 54.0815 | train | 158 | 139 | 139 | 19 | 0 |
| 54.888 | train | 158 | 139 | 139 | 19 | 0 |
| 55.7096 | train | 158 | 139 | 139 | 19 | 0 |
| 56.5433 | train | 158 | 139 | 139 | 19 | 0 |
| 57.39 | train | 159 | 140 | 140 | 19 | 0 |
| 58.2626 | train | 158 | 139 | 139 | 19 | 0 |
| 59.2014 | train | 158 | 139 | 139 | 19 | 0 |
| 60.3077 | train | 159 | 140 | 140 | 19 | 0 |
| 61.756 | train | 158 | 139 | 139 | 19 | 0 |
| 63.4998 | train | 158 | 139 | 139 | 19 | 0 |
| 65.2598 | train | 158 | 139 | 139 | 19 | 0 |
| 66.9701 | train | 158 | 139 | 139 | 19 | 0 |
| 68.6497 | train | 158 | 139 | 139 | 19 | 0 |
| 70.3146 | test | 158 | 0 | 0 | 0 | 158 |
| 71.9729 | train | 158 | 139 | 139 | 19 | 0 |
| 73.6283 | train | 158 | 139 | 139 | 19 | 0 |
| 75.2823 | train | 158 | 139 | 139 | 19 | 0 |
| 76.9358 | train | 158 | 139 | 139 | 19 | 0 |
| 78.589 | train | 158 | 139 | 139 | 19 | 0 |
| 80.242 | train | 158 | 139 | 139 | 19 | 0 |
| 81.8947 | train | 158 | 139 | 139 | 19 | 0 |
| 83.5472 | train | 158 | 139 | 139 | 19 | 0 |
| 85.1991 | train | 158 | 139 | 139 | 19 | 0 |
| 86.8502 | train | 158 | 139 | 139 | 19 | 0 |
| 88.4998 | train | 158 | 139 | 139 | 19 | 0 |
| 90.1473 | train | 158 | 139 | 139 | 19 | 0 |
| 91.7922 | train | 158 | 139 | 139 | 19 | 0 |
| 93.4352 | train | 158 | 139 | 139 | 19 | 0 |
| 95.0817 | train | 158 | 139 | 139 | 19 | 0 |
| 96.7493 | train | 158 | 139 | 139 | 19 | 0 |
| 98.4804 | train | 158 | 139 | 139 | 19 | 0 |
| 100.352 | test | 158 | 0 | 0 | 0 | 158 |
| 102.44 | train | 158 | 139 | 139 | 19 | 0 |
| 104.711 | train | 158 | 139 | 139 | 19 | 0 |
| 107.051 | train | 158 | 139 | 139 | 19 | 0 |
| 109.396 | train | 158 | 139 | 139 | 19 | 0 |
| 111.734 | train | 158 | 139 | 139 | 19 | 0 |
| 114.066 | train | 158 | 139 | 139 | 19 | 0 |
| 116.395 | train | 158 | 139 | 139 | 19 | 0 |
| 118.723 | train | 158 | 139 | 139 | 19 | 0 |
| 121.05 | train | 158 | 139 | 139 | 19 | 0 |
| 123.377 | train | 158 | 139 | 139 | 19 | 0 |
| 125.703 | train | 159 | 140 | 140 | 19 | 0 |
| 128.029 | train | 158 | 139 | 139 | 19 | 0 |
| 130.355 | train | 158 | 139 | 139 | 19 | 0 |
| 132.68 | train | 158 | 139 | 139 | 19 | 0 |
| 135.004 | train | 158 | 139 | 139 | 19 | 0 |
| 137.325 | train | 159 | 140 | 140 | 19 | 0 |
| 139.642 | train | 159 | 140 | 140 | 19 | 0 |
| 141.956 | train | 158 | 139 | 139 | 19 | 0 |
| 144.273 | train | 158 | 139 | 139 | 19 | 0 |
| 146.62 | train | 158 | 139 | 139 | 19 | 0 |
| 149.059 | test | 158 | 0 | 0 | 0 | 158 |
| 151.686 | train | 158 | 139 | 139 | 19 | 0 |
| 154.521 | train | 158 | 139 | 139 | 19 | 0 |
| 157.46 | train | 158 | 139 | 139 | 19 | 0 |
| 160.415 | train | 158 | 139 | 139 | 19 | 0 |
| 163.364 | train | 158 | 139 | 139 | 19 | 0 |
| 166.306 | train | 158 | 139 | 139 | 19 | 0 |
| 169.245 | train | 158 | 139 | 139 | 19 | 0 |
| 172.182 | train | 158 | 139 | 139 | 19 | 0 |
| 175.118 | train | 158 | 139 | 139 | 19 | 0 |
| 178.054 | train | 158 | 139 | 139 | 19 | 0 |
| 180.992 | train | 158 | 139 | 139 | 19 | 0 |
| 183.933 | train | 158 | 139 | 139 | 19 | 0 |
| 186.885 | train | 159 | 140 | 140 | 19 | 0 |
| 189.862 | test | 158 | 0 | 0 | 0 | 158 |
| 192.912 | train | 159 | 140 | 140 | 19 | 0 |
| 196.161 | train | 158 | 139 | 139 | 19 | 0 |
| 200 | train | 159 | 140 | 140 | 19 | 0 |

## Aggregate Held-out Metrics

| Metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| rhs_l2 | 0.252926 | 0.0652013 | 0.195454 | 0.427212 |
| pressure_head_l2 | 0.123199 | 0.0598969 | 0.0903901 | 0.28935 |
| one_step_a_l2 | 0.014115 | 0.0033966 | 0.0109953 | 0.023167 |
| one_step_b_l2 | 0.124158 | 0.0590743 | 0.091519 | 0.287446 |
| rollout_a_l2 | nan | nan | nan | nan |
| rollout_b_l2 | nan | nan | nan | nan |
| one_step_pressure_energy_error | 0.0685355 | 0.0639257 | 0.00059763 | 0.220577 |
| rollout_pressure_energy_error | nan | nan | nan | nan |

Error curve CSV: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/revisions/revision5_supplemental_evaluation_20260725/recovery/global_k56_steady_failure_accounting_v4/V16_1_SteadyPressureAnchor32_ru32_rp32_error_vs_re.csv`

Error curve SVG: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/revisions/revision5_supplemental_evaluation_20260725/recovery/global_k56_steady_failure_accounting_v4/V16_1_SteadyPressureAnchor32_ru32_rp32_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.497643 | 47.4738 | - | 0.0292339 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 0.427212 | 47.4738 | 0.28935 | 0.0231625 | 0.023167 | 0.287446 | nan | nan | nan | 3 | 2.71154 | 1.00833 | 18 |
| 32.74006652832031 | Galerkin only | 0.422785 | 56.3403 | - | 0.0247627 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.27126 | 56.3403 | 0.126234 | 0.0150619 | 0.015064 | 0.129478 | nan | nan | nan | 3 | 2.71154 | 1.00833 | 18 |
| 34.73756790161133 | Galerkin only | 0.410547 | 54.8969 | - | 0.0240435 | - | - | - | - | - | - | - | - | - |
| 34.73756790161133 | HPRS-MoE | 0.254821 | 54.8969 | 0.117651 | 0.0141908 | 0.0141925 | 0.120779 | nan | nan | nan | 3 | 2.71154 | 1.00833 | 18 |
| 36.65776824951172 | Galerkin only | 0.395785 | 53.4976 | - | 0.0235459 | - | - | - | - | - | - | - | - | - |
| 36.65776824951172 | HPRS-MoE | 0.240662 | 53.4976 | 0.107415 | 0.0136703 | 0.0136723 | 0.107507 | nan | nan | nan | 3 | 2.71154 | 1.00833 | 18 |
| 38.35725021362305 | Galerkin only | 0.391414 | 51.3175 | - | 0.0229332 | - | - | - | - | - | - | - | - | - |
| 38.35725021362305 | HPRS-MoE | 0.231713 | 51.3175 | 0.0962646 | 0.0129747 | 0.0129766 | 0.0964862 | nan | nan | nan | 3 | 2.71154 | 1.00833 | 18 |
| 39.68547821044922 | Galerkin only | 0.385127 | 49.4623 | - | 0.0225734 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.224675 | 49.4623 | 0.0922865 | 0.0126038 | 0.0126059 | 0.0947541 | nan | nan | nan | 3 | 2.71154 | 1.00833 | 18 |
| 40.711524963378906 | Galerkin only | 0.378878 | 49.1589 | - | 0.022214 | - | - | - | - | - | - | - | - | - |
| 40.711524963378906 | HPRS-MoE | 0.21724 | 49.1589 | 0.0903901 | 0.0121887 | 0.0121909 | 0.091519 | nan | nan | nan | 3 | 2.71154 | 1.00833 | 18 |
| 41.576576232910156 | Galerkin only | 0.370459 | 48.7373 | - | 0.0220642 | - | - | - | - | - | - | - | - | - |
| 41.576576232910156 | HPRS-MoE | 0.213296 | 48.7373 | 0.0949546 | 0.0121681 | 0.0121703 | 0.0947839 | nan | nan | nan | 3 | 2.71154 | 1.00833 | 18 |
| 45.142704010009766 | Galerkin only | 0.356736 | 45.695 | - | 0.0209462 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.195454 | 45.695 | 0.0942448 | 0.0109929 | 0.0109953 | 0.0946691 | nan | nan | nan | 3 | 2.71154 | 1.00833 | 18 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 32.74006652832031 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 34.73756790161133 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 36.65776824951172 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 38.35725021362305 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 39.68547821044922 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 40.711524963378906 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 41.576576232910156 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 45.142704010009766 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.592129 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 32.74006652832031 | 0.590804 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 34.73756790161133 | 0.590417 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 36.65776824951172 | 0.590466 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 38.35725021362305 | 0.590014 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 39.68547821044922 | 0.589994 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 40.711524963378906 | 0.589808 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 41.576576232910156 | 0.589987 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 45.142704010009766 | 0.58948 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |

Runtime: 122.51 s.
