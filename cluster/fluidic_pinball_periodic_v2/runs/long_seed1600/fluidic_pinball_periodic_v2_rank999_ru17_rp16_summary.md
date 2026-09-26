# AttractorMoE-ROM V16 Physics-Generalizable Attractor Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.75; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

V16 pressure input mode: `pressure_only`. V16 keeps the V14/V15 best `[a_t,b_t]` pressure state while changing only attractor losses or the lightweight attractor conditioning layer.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop multi-step rollout, energy consistency, trajectory consistency, pressure closure, relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime supervision.

## Dense Training Split

Test selection: `values`, time stride=1, Re stride=1.

- Train Re count: 47
- Test Re count: 9
- Excluded Re count from Re sparsity: 0
- Dense train samples before time sparsity: 2291
- Kept train samples: 2291
- Validation samples: 2053
- Test samples: 2053
- Compression vs dense train: 1
- Compression vs all non-test candidates: 1

| Re | role | total | dense train | kept train | val | test |
|---:|---|---:|---:|---:|---:|---:|
| 21.25 | train | 61 | 61 | 61 | 0 | 0 |
| 21.5 | validation | 141 | 0 | 0 | 141 | 0 |
| 21.75 | train | 61 | 61 | 61 | 0 | 0 |
| 22 | train | 61 | 61 | 61 | 0 | 0 |
| 22.25 | test | 141 | 0 | 0 | 0 | 141 |
| 22.5 | train | 61 | 61 | 61 | 0 | 0 |
| 22.75 | train | 61 | 61 | 61 | 0 | 0 |
| 23 | train | 61 | 61 | 61 | 0 | 0 |
| 23.25 | validation | 141 | 0 | 0 | 141 | 0 |
| 23.5 | train | 61 | 61 | 61 | 0 | 0 |
| 23.75 | train | 61 | 61 | 61 | 0 | 0 |
| 24 | test | 141 | 0 | 0 | 0 | 141 |
| 24.25 | train | 61 | 61 | 61 | 0 | 0 |
| 24.5 | train | 61 | 61 | 61 | 0 | 0 |
| 25 | train | 61 | 61 | 61 | 0 | 0 |
| 25.5 | train | 45 | 45 | 45 | 0 | 0 |
| 26 | validation | 253 | 0 | 0 | 253 | 0 |
| 26.5 | train | 45 | 45 | 45 | 0 | 0 |
| 27 | train | 45 | 45 | 45 | 0 | 0 |
| 27.5 | test | 253 | 0 | 0 | 0 | 253 |
| 28 | train | 45 | 45 | 45 | 0 | 0 |
| 28.5 | train | 45 | 45 | 45 | 0 | 0 |
| 29 | train | 45 | 45 | 45 | 0 | 0 |
| 29.5 | validation | 253 | 0 | 0 | 253 | 0 |
| 30 | train | 45 | 45 | 45 | 0 | 0 |
| 30.5 | train | 45 | 45 | 45 | 0 | 0 |
| 31 | test | 253 | 0 | 0 | 0 | 253 |
| 31.5 | train | 45 | 45 | 45 | 0 | 0 |
| 32 | train | 45 | 45 | 45 | 0 | 0 |
| 32.5 | train | 45 | 45 | 45 | 0 | 0 |
| 33 | validation | 253 | 0 | 0 | 253 | 0 |
| 33.5 | train | 45 | 45 | 45 | 0 | 0 |
| 34 | train | 45 | 45 | 45 | 0 | 0 |
| 34.5 | test | 253 | 0 | 0 | 0 | 253 |
| 35 | train | 45 | 45 | 45 | 0 | 0 |
| 35.5 | train | 45 | 45 | 45 | 0 | 0 |
| 36 | train | 45 | 45 | 45 | 0 | 0 |
| 36.5 | validation | 253 | 0 | 0 | 253 | 0 |
| 37 | train | 45 | 45 | 45 | 0 | 0 |
| 37.5 | train | 45 | 45 | 45 | 0 | 0 |
| 38 | test | 253 | 0 | 0 | 0 | 253 |
| 38.5 | train | 45 | 45 | 45 | 0 | 0 |
| 39 | train | 45 | 45 | 45 | 0 | 0 |
| 39.5 | train | 45 | 45 | 45 | 0 | 0 |
| 40 | validation | 253 | 0 | 0 | 253 | 0 |
| 41 | train | 45 | 45 | 45 | 0 | 0 |
| 42 | train | 45 | 45 | 45 | 0 | 0 |
| 43 | test | 253 | 0 | 0 | 0 | 253 |
| 44 | train | 45 | 45 | 45 | 0 | 0 |
| 45 | train | 45 | 45 | 45 | 0 | 0 |
| 46 | train | 45 | 45 | 45 | 0 | 0 |
| 47 | validation | 253 | 0 | 0 | 253 | 0 |
| 48 | train | 45 | 45 | 45 | 0 | 0 |
| 49 | train | 45 | 45 | 45 | 0 | 0 |
| 50 | test | 253 | 0 | 0 | 0 | 253 |
| 51 | train | 45 | 45 | 45 | 0 | 0 |
| 52 | train | 45 | 45 | 45 | 0 | 0 |
| 53 | train | 45 | 45 | 45 | 0 | 0 |
| 54 | validation | 253 | 0 | 0 | 253 | 0 |
| 55 | train | 45 | 45 | 45 | 0 | 0 |
| 56 | train | 45 | 45 | 45 | 0 | 0 |
| 57 | test | 253 | 0 | 0 | 0 | 253 |
| 58 | train | 45 | 45 | 45 | 0 | 0 |
| 59 | train | 45 | 45 | 45 | 0 | 0 |
| 60 | train | 45 | 45 | 45 | 0 | 0 |

## Aggregate Held-out Metrics

| Metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| rhs_l2 | 3.97992 | 8.55371 | 0.0203508 | 27.0402 |
| pressure_head_l2 | 0.0478545 | 0.0146108 | 0.0150407 | 0.0632461 |
| one_step_a_l2 | 0.0032486 | 0.00197325 | 0.000598768 | 0.00620641 |
| one_step_b_l2 | 0.0477098 | 0.0145565 | 0.0151033 | 0.0635686 |
| rollout_a_l2 | 0.025267 | 0.0138535 | 0.00540792 | 0.0500174 |
| rollout_b_l2 | 0.0542087 | 0.020541 | 0.0150392 | 0.0863219 |
| one_step_pressure_energy_error | 0.0172626 | 0.00630127 | 0.00609684 | 0.0301969 |
| rollout_pressure_energy_error | 0.0270905 | 0.0165184 | 0.00558017 | 0.0591901 |

Error curve CSV: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/fluidic_pinball_periodic_v2/runs/long_seed1600/fluidic_pinball_periodic_v2_rank999_ru17_rp16_error_vs_re.csv`

Error curve SVG: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/fluidic_pinball_periodic_v2/runs/long_seed1600/fluidic_pinball_periodic_v2_rank999_ru17_rp16_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 22.25 | Galerkin only | 372.149 | 2.00896 | - | 0.00947168 | - | - | - | - | - | - | - | - | - |
| 22.25 | HPRS-MoE | 27.0402 | 2.00896 | 0.0381287 | 0.000673428 | 0.000673512 | 0.0380615 | 0.005484 | 0.00540792 | 0.0249269 | 4 | 2.79022 | 0.98043 | 18 |
| 24.0 | Galerkin only | 91.6945 | 1.88338 | - | 0.00672635 | - | - | - | - | - | - | - | - | - |
| 24.0 | HPRS-MoE | 8.35503 | 1.88338 | 0.0349076 | 0.000598718 | 0.000598768 | 0.0349741 | 0.0047439 | 0.0121656 | 0.0863219 | 5 | 2.75903 | 1.05686 | 17 |
| 27.5 | Galerkin only | 1.1881 | 1.63401 | - | 0.00450025 | - | - | - | - | - | - | - | - | - |
| 27.5 | HPRS-MoE | 0.254819 | 1.63401 | 0.0150407 | 0.000941591 | 0.000941719 | 0.0151033 | 0.00665232 | 0.00694388 | 0.0150392 | 5 | 2.68409 | 1.18355 | 16 |
| 31.0 | Galerkin only | 0.0424677 | 1.37301 | - | 0.0135474 | - | - | - | - | - | - | - | - | - |
| 31.0 | HPRS-MoE | 0.0412561 | 1.37301 | 0.0632461 | 0.0057767 | 0.005675 | 0.0635686 | 0.0385157 | 0.0373567 | 0.0674397 | 4.06 | 2.69215 | 1.06355 | 17 |
| 34.5 | Galerkin only | 0.0307343 | 1.24419 | - | 0.0141349 | - | - | - | - | - | - | - | - | - |
| 34.5 | HPRS-MoE | 0.0279979 | 1.24419 | 0.0570216 | 0.00419694 | 0.00417573 | 0.0571822 | 0.0271699 | 0.0271972 | 0.0589306 | 4.1 | 2.69166 | 1.07156 | 17 |
| 38.0 | Galerkin only | 0.0274277 | 1.14453 | - | 0.0149809 | - | - | - | - | - | - | - | - | - |
| 38.0 | HPRS-MoE | 0.0223738 | 1.14453 | 0.0568702 | 0.00353432 | 0.00359679 | 0.0568369 | 0.0270744 | 0.0279057 | 0.0620668 | 4.08 | 2.69576 | 1.07875 | 17 |
| 43.0 | Galerkin only | 0.0270248 | 1.04367 | - | 0.0159693 | - | - | - | - | - | - | - | - | - |
| 43.0 | HPRS-MoE | 0.0203508 | 1.04367 | 0.0587535 | 0.00340918 | 0.00349525 | 0.058468 | 0.0267254 | 0.0282896 | 0.0620181 | 4.07 | 2.68942 | 1.07695 | 17 |
| 50.0 | Galerkin only | 0.0263631 | 0.959174 | - | 0.0168707 | - | - | - | - | - | - | - | - | - |
| 50.0 | HPRS-MoE | 0.0207281 | 0.959174 | 0.0527819 | 0.00380323 | 0.00387424 | 0.0520931 | 0.031265 | 0.0321188 | 0.0505336 | 4.14 | 2.67867 | 1.06798 | 17 |
| 57.0 | Galerkin only | 0.039421 | 0.909422 | - | 0.0179636 | - | - | - | - | - | - | - | - | - |
| 57.0 | HPRS-MoE | 0.0364641 | 0.909422 | 0.0539405 | 0.00617472 | 0.00620641 | 0.0531006 | 0.0501926 | 0.0500174 | 0.0606013 | 4.07 | 2.67501 | 1.06391 | 17 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 22.25 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 24.0 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 27.5 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 31.0 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 34.5 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 38.0 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 43.0 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 50.0 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 57.0 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 22.25 | 0.98747 | True | low_Re_lt_80: e7 |
| 24.0 | 0.993512 | True | low_Re_lt_80: e7 |
| 27.5 | 0.996557 | True | low_Re_lt_80: e7 |
| 31.0 | 0.988505 | True | low_Re_lt_80: e7 |
| 34.5 | 0.990977 | True | low_Re_lt_80: e7 |
| 38.0 | 0.989832 | True | low_Re_lt_80: e7 |
| 43.0 | 0.989817 | True | low_Re_lt_80: e7 |
| 50.0 | 0.991223 | True | low_Re_lt_80: e7 |
| 57.0 | 0.990894 | True | low_Re_lt_80: e7 |

Runtime: 46347.36 s.
