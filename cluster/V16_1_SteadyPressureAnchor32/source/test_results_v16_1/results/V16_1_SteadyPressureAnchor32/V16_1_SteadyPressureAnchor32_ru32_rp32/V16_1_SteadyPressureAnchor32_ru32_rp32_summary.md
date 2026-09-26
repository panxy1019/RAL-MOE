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
| rhs_l2 | 0.310536 | 0.142192 | 0.185206 | 0.640404 |
| pressure_head_l2 | 0.115868 | 0.0898427 | 0.0180906 | 0.287885 |
| one_step_a_l2 | 0.0399394 | 0.0330447 | 0.0109953 | 0.131729 |
| one_step_b_l2 | 0.122907 | 0.0850955 | 0.0324594 | 0.288167 |
| rollout_a_l2 | 0.225702 | 0.198285 | 0.0254706 | 0.606338 |
| rollout_b_l2 | 0.296692 | 0.215596 | 0.0323469 | 0.660263 |
| one_step_pressure_energy_error | 0.0844876 | 0.0830162 | 0.000594273 | 0.220579 |
| rollout_pressure_energy_error | 0.314341 | 0.388212 | 0.00142337 | 1.27048 |

Error curve CSV: `/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_SteadyPressureAnchor32/V16_1_SteadyPressureAnchor32_ru32_rp32/V16_1_SteadyPressureAnchor32_ru32_rp32_error_vs_re.csv`

Error curve SVG: `/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_SteadyPressureAnchor32/V16_1_SteadyPressureAnchor32_ru32_rp32/V16_1_SteadyPressureAnchor32_ru32_rp32_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.497642 | 47.4738 | - | 0.0292339 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 0.427402 | 47.4738 | 0.287885 | 0.0231625 | 0.023167 | 0.287447 | 0.306066 | 0.306106 | 0.627432 | 3 | 2.71154 | 1.00833 | 18 |
| 32.74006652832031 | Galerkin only | 0.422785 | 56.3403 | - | 0.0247627 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.271224 | 56.3403 | 0.129217 | 0.0150618 | 0.015064 | 0.129479 | 0.206761 | 0.206771 | 0.467988 | 3 | 2.71154 | 1.00833 | 18 |
| 39.68547821044922 | Galerkin only | 0.385127 | 49.4623 | - | 0.0225734 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.224628 | 49.4623 | 0.094275 | 0.0126038 | 0.0126059 | 0.0947519 | 0.160865 | 0.160867 | 0.303405 | 3 | 2.71154 | 1.00833 | 18 |
| 45.142704010009766 | Galerkin only | 0.356736 | 45.695 | - | 0.0209462 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.195353 | 45.695 | 0.0942794 | 0.0109929 | 0.0109953 | 0.0946697 | 0.125607 | 0.125622 | 0.229509 | 3 | 2.71154 | 1.00833 | 18 |
| 47.081356048583984 | Galerkin only | 2.65753 | 67.4831 | - | 0.329896 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 0.640404 | 67.4831 | 0.122706 | 0.0558937 | 0.0561294 | 0.123395 | 0.59152 | 0.592475 | 0.355033 | 3 | 2.71154 | 1.00833 | 18 |
| 49.02235794067383 | Galerkin only | 2.12087 | 54.6819 | - | 0.25518 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 0.500991 | 54.6819 | 0.135438 | 0.0417147 | 0.0419877 | 0.136034 | 0.286887 | 0.287203 | 0.372855 | 3 | 2.71154 | 1.00833 | 18 |
| 51.78644943237305 | Galerkin only | 3.13795 | 588.069 | - | 1.86061 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 0.315426 | 588.069 | 0.282153 | 0.131617 | 0.131729 | 0.288167 | 0.603018 | 0.606338 | 0.660263 | 3 | 2.71154 | 1.00833 | 18 |
| 70.31463623046875 | Galerkin only | 0.21882 | 1.42014 | - | 0.33963 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.208119 | 1.42014 | 0.0705494 | 0.135533 | 0.0610801 | 0.0866429 | 0.429399 | 0.101916 | 0.127152 | 4.39 | 1.74105 | 1.15272 | 8 |
| 100.35224914550781 | Galerkin only | 0.270851 | 1.01569 | - | 0.410135 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.23131 | 1.01569 | 0.0180906 | 0.13543 | 0.0290145 | 0.0324594 | 0.25102 | 0.0283432 | 0.0366359 | 4.62 | 2.475 | 1.11518 | 14 |
| 149.05923461914062 | Galerkin only | 0.295979 | 0.830144 | - | 0.443647 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.215833 | 0.830144 | 0.0186972 | 0.0980024 | 0.0256517 | 0.034868 | 0.198613 | 0.0254706 | 0.0323469 | 4.32 | 1.9146 | 1.08356 | 13 |
| 189.86227416992188 | Galerkin only | 0.304632 | 0.714768 | - | 0.45917 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.185206 | 0.714768 | 0.0212541 | 0.103366 | 0.0319093 | 0.0440603 | 0.26676 | 0.0416079 | 0.0509873 | 4.3 | 1.80027 | 1.05303 | 11 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 32.74006652832031 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 39.68547821044922 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 45.142704010009766 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 47.081356048583984 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 49.02235794067383 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 51.78644943237305 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 70.31463623046875 | True | [0.728, 0.082, 0.190] | [0.728, 0.082, 0.190] | 0 |
| 100.35224914550781 | True | [0.000, 0.981, 0.019] | [0.000, 0.981, 0.019] | 0 |
| 149.05923461914062 | True | [0.032, 0.747, 0.222] | [0.032, 0.747, 0.222] | 0 |
| 189.86227416992188 | True | [0.025, 0.297, 0.677] | [0.025, 0.297, 0.677] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.59211 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 32.74006652832031 | 0.590798 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 39.68547821044922 | 0.58996 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 45.142704010009766 | 0.58944 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 47.081356048583984 | 0.594799 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 49.02235794067383 | 0.594557 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 51.78644943237305 | 0.594503 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 70.31463623046875 | 0.470006 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 100.35224914550781 | 0.63405 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 149.05923461914062 | 0.748063 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 189.86227416992188 | 0.777608 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |

Runtime: 41547.31 s.
