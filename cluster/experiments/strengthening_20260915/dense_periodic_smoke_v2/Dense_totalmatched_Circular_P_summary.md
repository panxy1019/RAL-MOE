# AttractorMoE-ROM V16 Physics-Generalizable Attractor Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.75; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

V16 pressure input mode: `pressure_only`. V16 keeps the V14/V15 best `[a_t,b_t]` pressure state while changing only attractor losses or the lightweight attractor conditioning layer.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop multi-step rollout, energy consistency, trajectory consistency, pressure closure, relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime supervision.

## Dense Training Split

Test selection: `values`, time stride=1, Re stride=1.

- Train Re count: 53
- Test Re count: 4
- Excluded Re count from Re sparsity: 0
- Dense train samples before time sparsity: 8380
- Kept train samples: 8380
- Validation samples: 949
- Test samples: 632
- Compression vs dense train: 1
- Compression vs all non-test candidates: 1

| Re | role | total | dense train | kept train | val | test |
|---:|---|---:|---:|---:|---:|---:|
| 60.3077 | train | 159 | 159 | 159 | 0 | 0 |
| 61.756 | train | 158 | 158 | 158 | 0 | 0 |
| 63.4998 | train | 158 | 158 | 158 | 0 | 0 |
| 65.2598 | train | 158 | 158 | 158 | 0 | 0 |
| 66.9701 | validation | 158 | 0 | 0 | 158 | 0 |
| 68.6497 | train | 158 | 158 | 158 | 0 | 0 |
| 70.3146 | test | 158 | 0 | 0 | 0 | 158 |
| 71.9729 | train | 158 | 158 | 158 | 0 | 0 |
| 73.6283 | train | 158 | 158 | 158 | 0 | 0 |
| 75.2823 | train | 158 | 158 | 158 | 0 | 0 |
| 76.9358 | train | 158 | 158 | 158 | 0 | 0 |
| 78.589 | train | 158 | 158 | 158 | 0 | 0 |
| 80.242 | train | 158 | 158 | 158 | 0 | 0 |
| 81.8947 | train | 158 | 158 | 158 | 0 | 0 |
| 83.5472 | train | 158 | 158 | 158 | 0 | 0 |
| 85.1991 | train | 158 | 158 | 158 | 0 | 0 |
| 86.8502 | train | 158 | 158 | 158 | 0 | 0 |
| 88.4998 | train | 158 | 158 | 158 | 0 | 0 |
| 90.1473 | train | 158 | 158 | 158 | 0 | 0 |
| 91.7922 | validation | 158 | 0 | 0 | 158 | 0 |
| 93.4352 | train | 158 | 158 | 158 | 0 | 0 |
| 95.0817 | train | 158 | 158 | 158 | 0 | 0 |
| 96.7493 | train | 158 | 158 | 158 | 0 | 0 |
| 98.4804 | train | 158 | 158 | 158 | 0 | 0 |
| 100.352 | test | 158 | 0 | 0 | 0 | 158 |
| 102.44 | train | 158 | 158 | 158 | 0 | 0 |
| 104.711 | train | 158 | 158 | 158 | 0 | 0 |
| 107.051 | train | 158 | 158 | 158 | 0 | 0 |
| 109.396 | train | 158 | 158 | 158 | 0 | 0 |
| 111.734 | train | 158 | 158 | 158 | 0 | 0 |
| 114.066 | train | 158 | 158 | 158 | 0 | 0 |
| 116.395 | train | 158 | 158 | 158 | 0 | 0 |
| 118.723 | train | 158 | 158 | 158 | 0 | 0 |
| 121.05 | validation | 158 | 0 | 0 | 158 | 0 |
| 123.377 | train | 158 | 158 | 158 | 0 | 0 |
| 125.703 | train | 159 | 159 | 159 | 0 | 0 |
| 128.029 | train | 158 | 158 | 158 | 0 | 0 |
| 130.355 | train | 158 | 158 | 158 | 0 | 0 |
| 132.68 | train | 158 | 158 | 158 | 0 | 0 |
| 135.004 | train | 158 | 158 | 158 | 0 | 0 |
| 137.325 | train | 159 | 159 | 159 | 0 | 0 |
| 139.642 | validation | 159 | 0 | 0 | 159 | 0 |
| 141.956 | train | 158 | 158 | 158 | 0 | 0 |
| 144.273 | train | 158 | 158 | 158 | 0 | 0 |
| 146.62 | train | 158 | 158 | 158 | 0 | 0 |
| 149.059 | test | 158 | 0 | 0 | 0 | 158 |
| 151.686 | train | 158 | 158 | 158 | 0 | 0 |
| 154.521 | train | 158 | 158 | 158 | 0 | 0 |
| 157.46 | train | 158 | 158 | 158 | 0 | 0 |
| 160.415 | train | 158 | 158 | 158 | 0 | 0 |
| 163.364 | train | 158 | 158 | 158 | 0 | 0 |
| 166.306 | train | 158 | 158 | 158 | 0 | 0 |
| 169.245 | validation | 158 | 0 | 0 | 158 | 0 |
| 172.182 | train | 158 | 158 | 158 | 0 | 0 |
| 175.118 | train | 158 | 158 | 158 | 0 | 0 |
| 178.054 | train | 158 | 158 | 158 | 0 | 0 |
| 180.992 | train | 158 | 158 | 158 | 0 | 0 |
| 183.933 | train | 158 | 158 | 158 | 0 | 0 |
| 186.885 | train | 159 | 159 | 159 | 0 | 0 |
| 189.862 | test | 158 | 0 | 0 | 0 | 158 |
| 192.912 | train | 159 | 159 | 159 | 0 | 0 |
| 196.161 | validation | 158 | 0 | 0 | 158 | 0 |
| 200 | train | 159 | 159 | 159 | 0 | 0 |

## Aggregate Held-out Metrics

| Metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| rhs_l2 | 0.241717 | 0.0390598 | 0.192196 | 0.299913 |
| pressure_head_l2 | 0.039089 | 0.011378 | 0.0205417 | 0.0508748 |
| one_step_a_l2 | 0.0406221 | 0.0164524 | 0.0143741 | 0.057339 |
| one_step_b_l2 | 0.0884376 | 0.0448274 | 0.0255521 | 0.141144 |
| rollout_a_l2 | 0.139878 | 0.0336807 | 0.0865994 | 0.176802 |
| rollout_b_l2 | 0.295383 | 0.16293 | 0.116807 | 0.478724 |
| one_step_pressure_energy_error | 0.0650355 | 0.0422711 | 0.00856783 | 0.114754 |
| rollout_pressure_energy_error | 0.218614 | 0.140236 | 0.0679277 | 0.372628 |

Error curve CSV: `experiments/strengthening_20260915/dense_periodic_smoke_v2/Dense_totalmatched_Circular_P_error_vs_re.csv`

Error curve SVG: `experiments/strengthening_20260915/dense_periodic_smoke_v2/Dense_totalmatched_Circular_P_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70.31463623046875 | Galerkin only | 0.21123 | 0.0955416 | - | 0.0364167 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.299913 | 0.0955416 | 0.0205417 | 0.0152542 | 0.0143741 | 0.0255521 | 0.0696339 | 0.0865994 | 0.116807 | 1 | 0 | 0 | 0 |
| 100.35224914550781 | Galerkin only | 0.270196 | 0.281613 | - | 0.187136 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.248007 | 0.281613 | 0.0449264 | 0.0587227 | 0.0512569 | 0.0684908 | 0.174787 | 0.138009 | 0.1504 | 1 | 0 | 0 | 0 |
| 149.05923461914062 | Galerkin only | 0.295542 | 0.372729 | - | 0.216541 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.22675 | 0.372729 | 0.0508748 | 0.051167 | 0.057339 | 0.141144 | 0.138469 | 0.176802 | 0.478724 | 1 | 0 | 0 | 0 |
| 189.86227416992188 | Galerkin only | 0.304112 | 0.216227 | - | 0.141195 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.192196 | 0.216227 | 0.0400131 | 0.0344017 | 0.0395182 | 0.118563 | 0.157376 | 0.158101 | 0.435602 | 1 | 0 | 0 | 0 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 70.31463623046875 | False | - | - | nan |
| 100.35224914550781 | False | - | - | nan |
| 149.05923461914062 | False | - | - | nan |
| 189.86227416992188 | False | - | - | nan |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 70.31463623046875 | nan | False | low_Re_lt_80: e0; mid_Re_80_160: e0; high_Re_ge_160: e0 |
| 100.35224914550781 | nan | False | low_Re_lt_80: e0; mid_Re_80_160: e0; high_Re_ge_160: e0 |
| 149.05923461914062 | nan | False | low_Re_lt_80: e0; mid_Re_80_160: e0; high_Re_ge_160: e0 |
| 189.86227416992188 | nan | False | low_Re_lt_80: e0; mid_Re_80_160: e0; high_Re_ge_160: e0 |

Runtime: 43.58 s.
