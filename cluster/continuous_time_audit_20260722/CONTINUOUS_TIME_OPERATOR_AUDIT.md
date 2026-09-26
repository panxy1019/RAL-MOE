# Continuous-time operator audit

Audit date: 2026-07-22  
Mode: read-only source audit plus minimal checkpoint execution  
Formal A/B/C training: **not started**  
Specialist parameters/assets: **not modified**

## Decision

**`COMMON_TIME_RETRAIN_REQUIRED`**

The frozen checkpoints cannot currently support the requested full unified continuous-RHS fusion contract. All three specialists expose a meaningful continuous velocity derivative, but none exposes a pressure derivative: pressure is produced by a macro-step algebraic next-state closure. The final certified Steady wrapper additionally uses Euler rather than RK4. Periodic's four phase harmonics depend on a database trajectory endpoint, H–P has no sampled Reynolds overlap, and the native history feature contract uses raw differences rather than differences divided by elapsed physical time.

The safe no-retraining fallback is therefore **native-time hard routing only**. Achieving the requested common-time soft RHS fusion requires warm-start re-certification or minimal specialist fine-tuning; it does not require random initialization and no such training was launched in this round.

## Contract audit by specialist

| Question | Steady | Hopf | Periodic |
|---|---|---|---|
| Network/Galerkin velocity output | Continuous `da/dt` | Continuous `da/dt` | Continuous `da/dt` |
| Pressure output | Algebraic surrogate plus residual, producing `b_next` | Algebraic base plus adaptive-gate residual, producing `b_next` | Algebraic base plus adaptive-gate residual, producing `b_next` |
| Full `(da/dt, db/dt)` available | No | No | No |
| Native certified integrator | Euler in final S2/S3/S4 wrapper | RK4 for velocity | RK4 for velocity |
| Role of `dt` | External Euler multiplier | External RK4 parameter | External RK4 parameter |
| `dt` is a network input | No | No | No |
| Derivative training target divides by actual interval | Yes | Yes | Yes |
| History `da/db/drhs` divides by elapsed time | No; raw differences | No; raw differences | No; raw differences |
| Phase/time features | 4 harmonics from database-normalized trajectory progress | Phase-free, 0 harmonics | 4 harmonics from database-normalized trajectory progress |
| RHS callable from current/history/Re/autonomous clock with no future truth | No certified contract because phase normalization needs the trajectory endpoint; velocity otherwise is callable | Velocity: yes; full state: no | No certified contract because phase normalization needs the trajectory endpoint; full state: no |

Source evidence:

- The Periodic loader uses an actual `phase` field if present, otherwise a supplied period, otherwise `(t-t_first)/(t_last-t_first)`: [train_periodic_moe.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/periodic_moe_3090/train_periodic_moe.py:615). Neither Steady nor Periodic snapshot CSV contains a `phase`, `period`, or `estimated_period` column.
- The Periodic centered finite-difference target divides by the actual two-sided time interval: [train_periodic_moe.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/periodic_moe_3090/train_periodic_moe.py:676).
- Native history features append raw `a_current-a_hist`, `b_current-b_hist`, and `rhs_current-rhs_hist`, with no timestamp argument: [train_periodic_moe.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/periodic_moe_3090/train_periodic_moe.py:1273).
- Periodic `model_outputs_from_states_torch` constructs Galerkin plus learned residual RHS before `dt` appears: [train_periodic_moe.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/periodic_moe_3090/train_periodic_moe.py:3378). Its integrator applies RK4 only to velocity and then computes `b_next` algebraically: [train_periodic_moe.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/periodic_moe_3090/train_periodic_moe.py:3418).
- Hopf is phase-free by construction and its training residual uses `(a_next-a)/dt`; it likewise performs velocity RK4 followed by one algebraic pressure closure: [train_hopf_moe_expanded.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/hopf_h4_expanded/train_hopf_moe_expanded.py:378), [train_hopf_moe_expanded.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/hopf_h4_expanded/train_hopf_moe_expanded.py:469).
- The frozen Steady S4 checkpoint inherits the S2/S3/S4 rollout contract `a_next=a+dt*rhs`, then evaluates the pressure surrogate and residual: [train_s2b_3090.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/v17_s2b_3090/train_s2b_3090.py:251). The vendor RK4 implementation is not the final wrapper's executed integration path.

## `rhs_only` wrapper

The audit-only wrapper is implemented in [rhs_only_wrappers.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/continuous_time_audit/rhs_only_wrappers.py). It:

- calls each native feature/scaler/Galerkin/MoE/pressure-closure forward unchanged;
- verifies supplied modules are frozen and executes under `torch.no_grad()`;
- exposes `velocity_rhs`, `pressure_closure_output`, `galerkin_rhs`, and diagnostics;
- deliberately raises `FullStateRHSUnavailable` if code requests `pressure_rhs` or a 64-dimensional full-state RHS.

Strict restore and specialist-specific calls are in [run_step_halving.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/continuous_time_audit/run_step_halving.py). All three checkpoint restores passed with `strict=True`:

- Steady: frozen S4 validation step 1200.
- Hopf: H4 expanded best step 7200.
- Periodic: best epoch 85.

## Step-halving test

For one validation state per specialist, the test compared one native step of `dt` with two native steps of `dt/2`. Five `dt` values were taken from each specialist's native interval distribution. History was held fixed inside the compared macro-step, matching the native feature contract. Errors below compare the two predictions, not prediction versus ground truth.

| Specialist | `dt` range tested | Velocity physical-field relative L2 | Pressure physical-field relative L2 | Velocity error trend | Pressure error trend |
|---|---:|---:|---:|---:|---:|
| Steady | 4.10144–4.96594 | 8.19e-6–1.11e-5 | 0.1106–0.1156 | observed order 1.56–1.72 | order -0.24 to -0.20; non-vanishing closure discrepancy |
| Hopf | 4.06133–21.8731 | 2.03e-7–1.90e-6 | 5.12e-4–6.07e-4 | positive, order up to 3.19 | nearly step-independent, negative order |
| Periodic | 3.50146–7.03835 | 6.17e-4–4.15e-3 | 0.01924–0.02834 | positive, order 2.10–2.94 | positive but only 0.23–0.85 and not a pressure-RHS certification |

Interpretation: velocity behaves like a pre-integrator derivative. Pressure does not satisfy the same flow contract because applying the algebraic next-state closure twice is not equivalent to applying it once at the end of a macro-step. These results do not justify mapping or averaging pressure closure outputs as `db/dt`.

Complete state/modal/field values and observed orders are in [steady_step_halving.json](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/artifacts/continuous_time_operator_audit_20260722/raw/steady_step_halving.json), [hopf_step_halving.json](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/artifacts/continuous_time_operator_audit_20260722/raw/hopf_step_halving.json), and [periodic_step_halving.json](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/artifacts/continuous_time_operator_audit_20260722/raw/periodic_step_halving.json).

## Periodic phase audit

The checkpoint has four harmonics, yielding eight sine/cosine inputs. They are not derived from the modal state and are not driven by a certified Re-conditioned frequency. The CSV has no phase or period field, so the loader normalizes each entire database trajectory to one nominal cycle using its final timestamp. That final timestamp is future information in an autonomous rollout.

As requested, a small autonomous diagnostic estimator was tested:

1. fit a two-dimensional principal observation plane using training velocity coefficients only;
2. compute `atan2` from the current predicted modal state;
3. select orientation and a single circular offset using training data only;
4. unwrap sequentially using only the prior estimated phase.

It reads neither future states nor the trajectory endpoint at runtime, but it is not a drop-in replacement for the checkpoint feature:

| Split | Trajectories | Phase circular MAE | Relative frequency error | Mean long phase drift |
|---|---:|---:|---:|---:|
| Validation | 6 | 1.5668 rad | 0.9783 | 6.1466 rad |
| Held-out | 4 | 1.5682 rad | 0.9589 | 6.0250 rad |

The poor agreement is expected: the database feature describes normalized trajectory progress, whereas an `atan2` observation measures physical shedding cycles. Replacing the input changes the frozen feature distribution and requires re-certification or a small warm-start fine-tune. The estimator and all per-Re results are in [phase_projection_audit.json](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/artifacts/continuous_time_operator_audit_20260722/raw/phase_projection_audit.json).

## Analytic adjacent-chart adapters

All maps use the original area-weighted POD inner product and identical point/cell ordering already certified in the first audit. For state conversion,

`a_j = c_(j<-i) + T_(j<-i) a_i`, where `T_(j<-i)=Phi_j M Phi_i^T` and `c_(j<-i)=Phi_j M(mu_i-mu_j)`.

For any future RHS conversion, only the linear part is valid: `f_j=T_(j<-i) f_i`. The affine mean offset must never be applied to an RHS.

The analytic adapter field error equals the target POD best-projection floor by construction. Table cells are mean/worst relative errors on the stated support.

| Pair/direction/field | Projection floor = field error | Fluctuation error | Energy error | Roundtrip modal error | `sigma_min` | shared directions `cos(angle)>=0.9` |
|---|---:|---:|---:|---:|---:|---:|
| S→H velocity | 2.71e-4 / 3.43e-4 | 2.26e-3 / 2.35e-3 | 5.12e-6 / 5.53e-6 | 5.00e-5 / 1.00e-4 | 1.19e-5 | 10/32 |
| H→S velocity | 1.26e-4 / 1.33e-4 | 1.27e-3 / 1.39e-3 | 1.62e-6 / 1.94e-6 | 7.95e-6 / 2.24e-5 | 1.19e-5 | 10/32 |
| S→H pressure | 1.72e-5 / 5.75e-4 | 2.77e-3 / 2.94e-3 | 7.68e-6 / 8.67e-6 | 7.15e-5 / 3.84e-4 | 5.71e-5 | 10/32 |
| H→S pressure | 6.12e-7 / 8.89e-5 | 2.93e-3 / 3.28e-3 | 8.63e-6 / 1.08e-5 | 2.22e-5 / 1.12e-4 | 5.71e-5 | 10/32 |
| H→P velocity | 3.27e-3 / 8.04e-3 | 3.42e-2 / 5.68e-2 | 1.19e-3 / 3.23e-3 | 2.32e-2 / 4.58e-2 | 3.75e-5 | 9/32 |
| P→H velocity | 2.21e-2 / 7.72e-2 | 4.28e-2 / 4.49e-2 | 1.84e-3 / 2.02e-3 | 2.23e-3 / 1.09e-2 | 3.75e-5 | 9/32 |
| H→P pressure | 7.72e-3 / 9.87e-3 | 9.23e-2 / 1.48e-1 | 8.63e-3 / 2.19e-2 | 1.20e-1 / 1.48e-1 | 3.70e-4 | 9/32 |
| P→H pressure | 5.70e-2 / 1.46e-1 | 2.07e-1 / 2.11e-1 | 4.29e-2 / 4.44e-2 | 3.01e-3 / 1.27e-2 | 3.70e-4 | 9/32 |

S–H uses the exact shared Re support 45.5–46.440071584. H–P has **no exact overlap snapshots**: Hopf ends at Re 59.2014322659 and Periodic begins at Re 60.3077454666. Its numbers therefore use the nearest three Re values on each side as a boundary-support proxy, never as a pseudo-transition trajectory.

Condition number is reported but is not used as a sole veto; online mapping does not invert `T`. The singular spectra instead show that only 9–10 of 32 directions have `cos(principal angle)>=0.9`, with maximum angles near 90 degrees. A principal-angle-aware overlap mask is recommended: soft blend only the shared singular directions and evolve the anchor chart's private complement with the anchor specialist. Full singular values, principal angles, condition numbers, and low-singular-direction data-energy fractions are in the raw JSON.

## Fully autonomous history-switch contract

The correct runtime state is the latest three **predicted physical modal states** plus their physical timestamps, not a cached feature tensor.

On a permitted S↔H or H↔P switch:

1. map all three velocity and pressure modal states with the target chart's analytic affine state adapter;
2. recompute Galerkin RHS history in the target chart from the mapped states and Re;
3. call the target specialist's original feature builder to reconstruct absolute history states, raw `da/db/drhs` differences, Re features, and pressure-closure inputs;
4. construct phase only from a separately certified autonomous clock/phase contract;
5. keep all timestamps for routing and future re-certification, but do not divide history differences by elapsed time for the current frozen checkpoints—their scalers were fitted to raw differences;
6. never copy a source 493/501-dimensional feature tensor and never read a future true state, phase, regime, or intermediate state.

This design is semantically correct, but it is **not certified with the present frozen S/P phase inputs**. Changing native raw history differences to time-normalized differences would also change checkpoint/scaler semantics and requires re-certification.

## Conditional RHS-fusion design after re-certification

If the failed gates are repaired, the requested design should be implemented as a semi-explicit continuous velocity system with an explicitly certified pressure treatment:

- use a shared nondimensional convective clock `tau=t U_ref/L_ref` with one `global_dtau` certified by step-halving and long rollouts across all charts; do not resample old training snapshots merely because their intervals differ;
- map the current and latest three autonomous physical states into each adjacent candidate chart and invoke that chart's native feature builder;
- map candidate velocity RHS values with the linear `T_(anchor<-candidate)` only;
- blend shared directions as `a_dot_anchor=sum(pi_r T f_r)` and leave anchor-private directions to the anchor specialist;
- hold Router weights fixed over one RK4 macro-step in version 1;
- apply affine mean offsets only on state/chart switches;
- define and train/re-certify either a continuous pressure RHS or a consistent algebraic DAE projection evaluated under a documented stage/macro-step policy. Do not reinterpret the current pressure closure as `db/dt`.

No `global_dtau` is selected in this audit because the prerequisite operator and phase contracts failed.

## Route comparison

### Route 1: native-time hard routing

This is the only path available without modifying the specialists. Switch only at native macro-step boundaries, obey S↔H↔P topology, map complete state/history affinely, rebuild native features, and use the selected specialist's own integrator and pressure closure at its certified native interval. It may be called only a **temporal-consistent regime router**. It cannot claim continuous soft vector-field fusion or validated physical S→H→P migration.

### Route 2: common-time warm-start re-certification

This is required for the requested soft RHS framework. Preserve the current checkpoints and warm-start; do not randomly initialize. Build common nondimensional-time evaluation data from genuine trajectories, certify a pressure DAE/RHS policy, replace or re-certify S/P endpoint-normalized phase inputs, add H–P overlap or transition support, and minimally fine-tune only the affected inputs/heads if zero-shot re-certification fails. Keep complete-Re and complete-trajectory train/validation/test isolation.

To strengthen later dynamic-transition claims, collect genuine startup trajectories and/or slowly varying `Re(t)` parameter-ramp trajectories spanning S–H and H–P. Do not concatenate independent fixed-Re snapshots into pseudo-time sequences.

## Artifacts

- Machine-readable decision: [CONTINUOUS_TIME_OPERATOR_AUDIT.json](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/artifacts/continuous_time_operator_audit_20260722/CONTINUOUS_TIME_OPERATOR_AUDIT.json)
- Audit wrapper: [rhs_only_wrappers.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/continuous_time_audit/rhs_only_wrappers.py)
- Step-halving runner: [run_step_halving.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/continuous_time_audit/run_step_halving.py)
- Phase/projection runner: [run_phase_projection_audit.py](/C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/continuous_time_audit/run_phase_projection_audit.py)

Final status: **`COMMON_TIME_RETRAIN_REQUIRED`**
