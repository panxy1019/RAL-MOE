# CTDM-Galerkin-ROM final comparative report

## Scope

This experiment isolates continuous-time delta memory. POD/ROM assets, r32 ranks, Reynolds splits, pressure gauge, native query times and the physical-field evaluator are frozen. B3 and B4 have exactly matched parameters; their only architectural difference is discrete macro-step versus continuous joint-RK4 memory evolution.

The read-only audit found that the frozen B0 checkpoint consumes 493, not 560, inputs (`encoder.net.0.weight=[256,493]`). The implemented 67-dimensional Hopf augmentation was loss-only. Accordingly, B1 uses the verified 493-dimensional history contract and B2-B4 use the true 109-dimensional current-only contract.

## Mathematical and numerical gates

- All pre-training numerical tests passed: `True`.
- Discrete-to-continuous observed orders: 1.0021, 1.0010, 1.0005.
- Continuous-memory RK4 observed orders: 4.0114, 4.0049, 4.0022.
- Long bounded-input maximum memory norm: `0.993638`.

## Parameter counts

| Model | Trainable parameters | Runtime memory state |
|---|---:|---:|
| B0 frozen HPRS-MoE-ROM | 32,004,147 | 0 |

## Validation summary

| Model | Seeds | Train hours | Eu | Ep | Ea | Eb | Terminal joint | Worst-window joint | Pressure drift | Divergent windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 frozen HPRS-MoE-ROM | 1 | 1.979 | 0.0365% | 0.2649% | 0.233713 | 0.289706 | 0.5753% | 0.8787% | 1.1412% | 0 |

## Gated held-out summary

No new held-out tensor was opened. The validation gate either has not completed or did not authorize test access.

## Required questions

1. **Does discrete KDA converge to the implemented continuous equation?** Yes; the measured order is approximately `1.0012`. The sign-reversed equation did not converge.

2. **Does joint RK4 have the expected time-step behavior?** The manufactured memory equation is fourth order. Model-level C_u/C_p/C_S values are recorded in `TIMESTEP_CONSISTENCY.csv`.

3. **Is memory bounded?** The independent long-time test passed. Learned-trajectory norms and singular values are recorded in `MEMORY_DIAGNOSTICS.csv`.

4. **Is memory necessary relative to current-only FNN?** On validation, insufficient completed runs in worst-window joint error.

5. **Does KDA outperform fixed three-state history?** On validation, insufficient completed runs in worst-window joint error.

6. **Does continuous KDA outperform matched discrete KDA?** On validation, insufficient completed runs in worst-window joint error.

7. **Where does any improvement come from?** Compare Eu, Ep, pressure drift and time-step consistency in the accompanying CSV files; no attractor, radial, phase or frequency loss is present.

8. **Learned physical memory scale.** Mean validation time scale `1/gamma=nan` and half-life `nan` in physical time units.

9. **Was memory bypassed or degenerate?** Mean eta is `nan` and mean effective rank is `nan`. The final decision also checks nonzero memory norm, eta and effective rank.

10. **Should continuous memory proceed to oscillatory-generator work?** Not yet. The pre-registered validation gate did not authorize that claim.

## Test-access decision

No frozen validation decision was found.

## Frozen B0 note

The frozen B0 checkpoint was re-evaluated at K56 with the same validation physical-field metrics and is included in the table above. Its earlier published K48 held-out summary is retained only as historical context and is not substituted for a gated K56 test.
