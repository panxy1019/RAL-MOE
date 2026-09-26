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
| rhs_l2 | 0.941805 | 0.0375443 | 0.886801 | 0.977393 |
| pressure_head_l2 | 0.0183549 | 0.0172506 | 0.00648914 | 0.0481506 |
| one_step_a_l2 | 0.0797908 | 0.00968838 | 0.067712 | 0.089739 |
| one_step_b_l2 | 0.0772692 | 0.00365585 | 0.0709891 | 0.0797474 |
| rollout_a_l2 | 0.057603 | 0.0122823 | 0.0385562 | 0.0718341 |
| rollout_b_l2 | 0.0576391 | 0.00899647 | 0.0479195 | 0.0723601 |
| one_step_pressure_energy_error | 0.0141549 | 0.014768 | 0.00491282 | 0.0396969 |
| rollout_pressure_energy_error | 0.00535952 | 0.00328021 | 0.00138604 | 0.00976231 |

Error curve CSV: `runs/optimized_long_seed1600/centered_square_periodic_optimized_r28_p26_error_vs_re.csv`

Error curve SVG: `runs/optimized_long_seed1600/centered_square_periodic_optimized_r28_p26_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100.5 | Galerkin only | 5.73399 | 4.96621 | - | 3.51984 | - | - | - | - | - | - | - | - | - |
| 100.5 | HPRS-MoE | 0.886801 | 4.96621 | 0.00868054 | 0.973129 | 0.089739 | 0.0785983 | 0.884582 | 0.063652 | 0.0553142 | 4.08 | 2.65975 | 1.04129 | 14 |
| 102.0 | Galerkin only | 5.79803 | 4.8133 | - | 4.12585 | - | - | - | - | - | - | - | - | - |
| 102.0 | HPRS-MoE | 0.927452 | 4.8133 | 0.0100992 | 1.12808 | 0.0888663 | 0.0797474 | 1.11113 | 0.0563696 | 0.0479195 | 4.15 | 2.63332 | 1.04798 | 14 |
| 120.68965148925781 | Galerkin only | 6.0904 | 3.05858 | - | 5.56289 | - | - | - | - | - | - | - | - | - |
| 120.68965148925781 | HPRS-MoE | 0.975574 | 3.05858 | 0.0481506 | 1.13043 | 0.067712 | 0.0797419 | 1.31902 | 0.0385562 | 0.0549625 | 4.07 | 2.61518 | 1.05864 | 14 |
| 144.8275909423828 | Galerkin only | 6.1138 | 1.8967 | - | 5.48241 | - | - | - | - | - | - | - | - | - |
| 144.8275909423828 | HPRS-MoE | 0.977393 | 1.8967 | 0.00648914 | 0.991178 | 0.0728458 | 0.0709891 | 1.05407 | 0.0718341 | 0.0723601 | 4.17 | 2.60673 | 1.04841 | 14 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 100.5 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 102.0 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 120.68965148925781 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 144.8275909423828 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 100.5 | 0.651499 | False | mid_Re_80_160: e7 |
| 102.0 | 0.649587 | False | mid_Re_80_160: e7 |
| 120.68965148925781 | 0.329317 | False | mid_Re_80_160: e7 |
| 144.8275909423828 | 0.292727 | False | mid_Re_80_160: e7 |

Runtime: 34403.22 s.
