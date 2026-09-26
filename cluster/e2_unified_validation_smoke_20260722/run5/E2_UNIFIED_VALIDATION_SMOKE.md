# E2 Unified Validation Top-1 Smoke

- Decision: `PASS`
- Frozen E2 SHA256: `f0286a9974f16b513bb42cd94232c397123dcd3e0a01eab475ff1593f7881dcd`
- Validation routes matching oracle: 10/10
- Specialists actually invoked: Steady, Hopf, Periodic
- Contract: one trajectory-level Top-1 decision; selected native specialist retains its own scaler, feature/history, integrator and pressure closure.

| Specialist | Native entrypoint | Validation Re | Finite | Divergent | PASS |
|---|---|---:|---:|---:|---:|
| Steady | `finalize_s4_one_time.evaluate_horizon(exp, 'validation', 4)` | 2 | 1.000000 | 0 | True |
| Hopf | `train_h4_expanded.validate(...)` | 2 | 1.000000 | 0 | True |
| Periodic | `train_periodic_moe.integrate_autonomous_step_np(...), K4` | 6 | 1.000000 | 0 | True |

## Scope

This is a validation smoke, not a new scientific benchmark. It proves that the frozen E2 route can dispatch each validation trajectory class to the corresponding native rollout implementation and receive finite, structurally valid outputs without substituting E0 hashes for execution.

No Top-2 path was executed.
