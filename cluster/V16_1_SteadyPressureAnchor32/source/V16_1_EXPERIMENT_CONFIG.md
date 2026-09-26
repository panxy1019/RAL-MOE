# V16_1 TrainStableAttractorMoE Experiment Config

Baseline: `V16_AttractorMoEROM_FullRegimeLoss32`.

All V16_1 experiments keep the Re=20-200 Physics-Generalizable Attractor Database, train/validation/held-out split, ru=32, rp=32, HPRS-MoE topology, Shared Encoder, routers, experts, Galerkin ROM, RK4, Pressure Poisson Surrogate, modal AdaptiveGate, optimizer, learning rate, batch size, epoch schedule, rollout curriculum `4,8,12,16`, seed, and evaluation protocol unchanged.

The only changes are closed-loop attractor-stability losses.

## Diagnostic Basis

V16 FullRegimeLoss32 has strong one-step behavior, but training-Re autonomous rollout is still unstable.

- Hopf near-onset worst train point is around Re=47.7229. Diagnostics show `r_true ~ 1.55e-6`, `r_pred ~ 2.50e-5`, standard relative velocity error around `7.21`, floor-normalized error around `1.96`, frequency error around `2.78e-2`, predicted growth positive while true growth is negative. This points to near-onset false oscillation / false growth, not primarily frequency error.
- Low-Re steady pressure has `alpha ~ 0.063`, raw Poisson base relative error around `80-107`, base contribution around `0.18-0.32`, and residual contribution around `1.0`. This points to closed-loop pressure residual drift as the main target, not direct Poisson-base injection.

## Cases

| Case | Added Losses |
|---|---|
| `V16_1_HopfOnsetGrowthLoss32` | Hopf growth consistency, Hopf false-growth penalty, Hopf floor-normalized rollout; Hopf radius weight lowered to `0.01`, overshoot stays `0.02`. |
| `V16_1_SteadyPressureAnchor32` | Steady pressure state anchor, pressure mean anchor, pressure delta damping, pressure residual damping, pressure energy consistency. |
| `V16_1_TrainStableCombined32` | Union of Hopf and steady losses with 30-epoch linear warm-up. |

Outputs are written under `test_results_v16_1/results/<case>/<case>_ru32_rp32/`.
