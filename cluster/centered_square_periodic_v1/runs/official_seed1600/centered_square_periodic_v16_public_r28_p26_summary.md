# AttractorMoE-ROM V16 Physics-Generalizable Attractor Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.75; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

V16 pressure input mode: `pressure_only`. V16 keeps the V14/V15 best `[a_t,b_t]` pressure state while changing only attractor losses or the lightweight attractor conditioning layer.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop multi-step rollout, energy consistency, trajectory consistency, pressure closure, relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime supervision.

## Dense Training Split

Test selection: `values`, time stride=1, Re stride=1.

- Train Re count: 27
- Test Re count: 4
- Excluded Re count from Re sparsity: 0
- Dense train samples before time sparsity: 3321
- Kept train samples: 3321
- Validation samples: 738
- Test samples: 492
- Compression vs dense train: 1
- Compression vs all non-test candidates: 1

| Re | role | total | dense train | kept train | val | test |
|---:|---|---:|---:|---:|---:|---:|
| 98.5 | train | 123 | 123 | 123 | 0 | 0 |
| 99 | validation | 123 | 0 | 0 | 123 | 0 |
| 99.5 | train | 123 | 123 | 123 | 0 | 0 |
| 100 | train | 123 | 123 | 123 | 0 | 0 |
| 100.5 | test | 123 | 0 | 0 | 0 | 123 |
| 101 | train | 123 | 123 | 123 | 0 | 0 |
| 101.5 | validation | 123 | 0 | 0 | 123 | 0 |
| 101.724 | train | 123 | 123 | 123 | 0 | 0 |
| 102 | test | 123 | 0 | 0 | 0 | 123 |
| 103.448 | train | 123 | 123 | 123 | 0 | 0 |
| 105.172 | train | 123 | 123 | 123 | 0 | 0 |
| 106.897 | train | 123 | 123 | 123 | 0 | 0 |
| 108.621 | train | 123 | 123 | 123 | 0 | 0 |
| 110.345 | validation | 123 | 0 | 0 | 123 | 0 |
| 112.069 | train | 123 | 123 | 123 | 0 | 0 |
| 113.793 | train | 123 | 123 | 123 | 0 | 0 |
| 115.517 | train | 123 | 123 | 123 | 0 | 0 |
| 117.241 | train | 123 | 123 | 123 | 0 | 0 |
| 118.966 | train | 123 | 123 | 123 | 0 | 0 |
| 120.69 | test | 123 | 0 | 0 | 0 | 123 |
| 122.414 | train | 123 | 123 | 123 | 0 | 0 |
| 124.138 | train | 123 | 123 | 123 | 0 | 0 |
| 125.862 | validation | 123 | 0 | 0 | 123 | 0 |
| 127.586 | train | 123 | 123 | 123 | 0 | 0 |
| 129.31 | train | 123 | 123 | 123 | 0 | 0 |
| 131.034 | train | 123 | 123 | 123 | 0 | 0 |
| 132.759 | train | 123 | 123 | 123 | 0 | 0 |
| 134.483 | train | 123 | 123 | 123 | 0 | 0 |
| 136.207 | train | 123 | 123 | 123 | 0 | 0 |
| 137.931 | train | 123 | 123 | 123 | 0 | 0 |
| 139.655 | train | 123 | 123 | 123 | 0 | 0 |
| 141.379 | validation | 123 | 0 | 0 | 123 | 0 |
| 143.103 | train | 123 | 123 | 123 | 0 | 0 |
| 144.828 | test | 123 | 0 | 0 | 0 | 123 |
| 146.552 | train | 123 | 123 | 123 | 0 | 0 |
| 148.276 | train | 123 | 123 | 123 | 0 | 0 |
| 150 | validation | 123 | 0 | 0 | 123 | 0 |

## Aggregate Held-out Metrics

| Metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| rhs_l2 | 0.949333 | 0.040219 | 0.893929 | 0.988976 |
| pressure_head_l2 | 0.0168188 | 0.0164319 | 0.00394195 | 0.0450388 |
| one_step_a_l2 | 0.948019 | 0.254904 | 0.618301 | 1.21968 |
| one_step_b_l2 | 0.957949 | 0.297665 | 0.583862 | 1.30038 |
| rollout_a_l2 | 0.791961 | 0.272969 | 0.427834 | 1.14903 |
| rollout_b_l2 | 0.763207 | 0.319189 | 0.343456 | 1.15558 |
| one_step_pressure_energy_error | 0.054387 | 0.0403954 | 0.00368321 | 0.0947688 |
| rollout_pressure_energy_error | 0.170769 | 0.128738 | 0.0763142 | 0.390717 |

Error curve CSV: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/centered_square_periodic_v1/runs/official_seed1600/centered_square_periodic_v16_public_r28_p26_error_vs_re.csv`

Error curve SVG: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/centered_square_periodic_v1/runs/official_seed1600/centered_square_periodic_v16_public_r28_p26_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100.5 | Galerkin only | 5.73399 | 4.96621 | - | 3.51984 | - | - | - | - | - | - | - | - | - |
| 100.5 | HPRS-MoE | 0.893929 | 4.96621 | 0.00888527 | 0.881502 | 0.618301 | 0.583862 | 0.785839 | 0.427834 | 0.343456 | 3.91 | 1.52023 | 1.01916 | 10 |
| 102.0 | Galerkin only | 5.79803 | 4.8133 | - | 4.12585 | - | - | - | - | - | - | - | - | - |
| 102.0 | HPRS-MoE | 0.928018 | 4.8133 | 0.0094091 | 1.0155 | 0.782489 | 0.753871 | 0.977684 | 0.657768 | 0.580674 | 3.86 | 1.40796 | 1.02558 | 8 |
| 120.68965148925781 | Galerkin only | 6.0904 | 3.05858 | - | 5.56289 | - | - | - | - | - | - | - | - | - |
| 120.68965148925781 | HPRS-MoE | 0.98641 | 3.05858 | 0.0450388 | 0.869174 | 1.21968 | 1.30038 | 1.21552 | 0.93321 | 0.973121 | 4.15 | 1.38767 | 1.05125 | 6 |
| 144.8275909423828 | Galerkin only | 6.1138 | 1.8967 | - | 5.48241 | - | - | - | - | - | - | - | - | - |
| 144.8275909423828 | HPRS-MoE | 0.988976 | 1.8967 | 0.00394195 | 0.975779 | 1.17161 | 1.19368 | 0.959324 | 1.14903 | 1.15558 | 4.04 | 1.44106 | 1.05274 | 7 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 100.5 | True | [0.171, 0.488, 0.341] | [0.171, 0.488, 0.341] | 0 |
| 102.0 | True | [0.333, 0.244, 0.423] | [0.333, 0.244, 0.423] | 0 |
| 120.68965148925781 | True | [0.390, 0.407, 0.203] | [0.390, 0.407, 0.203] | 0 |
| 144.8275909423828 | True | [0.350, 0.480, 0.171] | [0.350, 0.480, 0.171] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 100.5 | 0.913048 | False | mid_Re_80_160: e0 |
| 102.0 | 0.903405 | False | mid_Re_80_160: e0 |
| 120.68965148925781 | 0.675843 | False | mid_Re_80_160: e0 |
| 144.8275909423828 | 0.69 | False | mid_Re_80_160: e0 |

Runtime: 11528.91 s.
