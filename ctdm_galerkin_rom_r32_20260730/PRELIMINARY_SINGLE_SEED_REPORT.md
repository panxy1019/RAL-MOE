# CTDM-Galerkin-ROM final comparative report

## Scope

This experiment isolates continuous-time delta memory. POD/ROM assets, r32 ranks, Reynolds splits, pressure gauge, native query times and the physical-field evaluator are frozen. B3 and B4 have exactly matched parameters; their only architectural difference is discrete macro-step versus continuous joint-RK4 memory evolution.

The read-only audit found that the frozen B0 checkpoint consumes 493, not 560, inputs (`encoder.net.0.weight=[256,493]`). The implemented 67-dimensional Hopf augmentation was loss-only. Accordingly, B1 uses the verified 493-dimensional history contract and B2-B4 use the true 109-dimensional current-only contract.

## Method

### Frozen data and ROM contract

The Reynolds-number split contains 29 training conditions, validation conditions Re=46.7 and 56.543246, and sealed held-out conditions Re=47.081355, 49.022357 and 51.786450. Velocity and pressure POD means, bases and scalers are fitted from training data only; both retained ranks are 32. Galerkin velocity operators, Pressure--Poisson assets, pressure gauge, native nonuniform query times, adaptive pressure closure and the physical-field evaluator are shared by every model.

At every autonomous stage, the 109-dimensional current feature vector is reconstructed from the trial velocity state, fixed macro-step pressure state, Galerkin RHS and frozen physical descriptors. B1 adds two verified 192-dimensional historical feature blocks, giving 493 inputs. No held-out tensor participates in fitting, checkpoint selection or validation decisions.

### Compared closures

- B0 is the frozen HPRS-MoE-ROM reference.
- B1 is a plain deep FNN with the original three-state/493-dimensional history contract.
- B2 is the same current-state closure family without history or memory.
- B3 is Discrete-KDA-FNN.
- B4 is CTDM-Galerkin-ROM. B3 and B4 have identical named parameter tensors and exactly 2,238,718 trainable parameters.

B1--B4 remove MoE routers, routed/shared experts, route loss, attention, oscillatory rotation, attractor/radial/normal-form losses, phase or frequency supervision and energy projection. Each closure uses ordinary SiLU MLPs; residual output layers are zero initialized. The pressure gate starts at sigmoid probability 0.99.

### Matrix memory

The current encoder is LayerNorm--256--256--128 with SiLU activations. Four heads each store S_h in R^(16x16), so each trajectory carries 1024 float32 memory scalars. Normalized key and separate normalized velocity and pressure queries read 64-dimensional m_u and m_p vectors. Values use bounded tanh, eta uses a bounded sigmoid and gamma uses gamma_min+softplus to guarantee positive decay.

The implemented continuous equation is dS/dt=-Gamma S+eta k(v-S^T k)^T. The plus write sign is forced by the declared discrete update: expanding D=exp(-dt Gamma) and beta=1-exp(-eta dt) gives this equation to first order. The originally stated negative-write form is not that limit and failed the independent convergence test.

B3 performs one matched macro-step update with D and beta. B4 jointly integrates Y=[a,vec(S_1),...,vec(S_4)] using RK4. Every RK4 stage recomputes the Galerkin RHS, current features, token, memory parameters, queries, reads and both derivatives from the trial a and S while keeping b_n fixed. Pressure b_(n+1) is then obtained from the unchanged Pressure--Poisson and adaptive closure. There is no extra discrete write and no future truth.

### Warm-up, optimization and evaluation

The main K56 evaluation uses exactly three true initial states. Memory starts at zero at t_(n-2) and is warmed over two observed intervals using linear interpolation of a and b; from t_n onward the rollout is fully autonomous. Warm-up lengths 8 and 16 are capacity diagnostics only.

Training uses contiguous trajectory fragments and the curriculum K4->K8->K16->K32->K56 over 8000 AdamW steps. Effective batch size is 16, gradient clipping is 1.0, and memory arithmetic remains float32. The loss weights are Ea=1, Eb=0.55, modal Eu=0.38, modal Ep=0.095 and memory-norm regularization 1e-6. Validation reports physical-field Eu/Ep, modal Ea/Eb, time-averaged and terminal K56 errors, worst-window error, pressure drift, finite fraction, divergence, time-step consistency and memory spectral diagnostics.

## Mathematical and numerical gates

- All pre-training numerical tests passed: `True`.
- Discrete-to-continuous observed orders: 1.0021, 1.0010, 1.0005.
- Continuous-memory RK4 observed orders: 4.0114, 4.0049, 4.0022.
- Long bounded-input maximum memory norm: `0.993638`.

## Parameter counts

| Model | Trainable parameters | Runtime memory state |
|---|---:|---:|
| B0 frozen HPRS-MoE-ROM | 32,004,147 | 0 |
| B1 Deep-FNN-H3 | 2,379,284 | 0 |
| B2 Deep-FNN-current | 1,984,532 | 0 |
| B3 Discrete-KDA-FNN | 2,238,718 | 1,024 |
| B4 CTDM-Galerkin-ROM | 2,238,718 | 1,024 |

## Validation summary

| Model | Seeds | Train hours | Eu | Ep | Ea | Eb | Terminal joint | Worst-window joint | Pressure drift | Divergent windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 frozen HPRS-MoE-ROM | 1 | 1.979 | 0.0365% | 0.2649% | 0.233713 | 0.289706 | 0.5753% | 0.8787% | 1.1412% | 0 |
| B1 Deep-FNN-H3 | 1 | 1.207 | 0.0207% | 0.0725% | 0.0389277 | 0.120971 | 0.1263% | 0.1670% | 0.5263% | 0 |
| B2 Deep-FNN-current | 1 | 1.016 | 0.0625% | 0.1179% | 0.0413242 | 0.123985 | 0.3216% | 0.3612% | 0.9992% | 0 |
| B3 Discrete-KDA-FNN | 1 | 1.976 | 0.0766% | 0.4351% | 0.0781569 | 0.13532 | 0.7490% | 0.6852% | 4.2741% | 0 |
| B4 CTDM-Galerkin-ROM | 1 | 2.043 | 0.0920% | 0.4585% | 0.109642 | 0.195263 | 0.9288% | 1.1234% | 3.9676% | 0 |

## Gated held-out summary

No new held-out tensor was opened. The validation gate either has not completed or did not authorize test access.

## Required questions

1. **Does discrete KDA converge to the implemented continuous equation?** Yes; the measured order is approximately `1.0012`. The sign-reversed equation did not converge.

2. **Does joint RK4 have the expected time-step behavior?** The manufactured memory equation is fourth order. Model-level C_u/C_p/C_S values are recorded in `TIMESTEP_CONSISTENCY.csv`.

3. **Is memory bounded?** The independent long-time test passed. Learned-trajectory norms and singular values are recorded in `MEMORY_DIAGNOSTICS.csv`.

4. **Is memory necessary relative to current-only FNN?** On validation, degraded by 210.99% (0.0112344 vs 0.00361241) in worst-window joint error.

5. **Does KDA outperform fixed three-state history?** On validation, degraded by 310.44% (0.00685243 vs 0.00166954) in worst-window joint error.

6. **Does continuous KDA outperform matched discrete KDA?** On validation, degraded by 63.95% (0.0112344 vs 0.00685243) in worst-window joint error.

7. **Where does any improvement come from?** Compare Eu, Ep, pressure drift and time-step consistency in the accompanying CSV files. For B4 versus B3, Eu degraded by 20.17% (0.000920194 vs 0.000765728), Ep degraded by 5.36% (0.00458476 vs 0.00435146), and pressure drift improved by 7.17% (0.0396758 vs 0.0427413). Mean dt-to-dt/2 C_u is 1.83621e-05 for B4 versus 0.000112533 for B3; C_p is 0.000115668 versus 0.000502795.

8. **Learned physical memory scale.** Mean validation time scale `1/gamma=130.568` and half-life `90.5031` in physical time units.

9. **Was memory bypassed or degenerate?** Mean eta is `0.0246252` and mean effective rank is `1.36552`. The final decision also checks nonzero memory norm, eta and effective rank.

10. **Should continuous memory proceed to oscillatory-generator work?** Not yet. The pre-registered validation gate did not authorize that claim.

## Test-access decision

No frozen validation decision was found.
