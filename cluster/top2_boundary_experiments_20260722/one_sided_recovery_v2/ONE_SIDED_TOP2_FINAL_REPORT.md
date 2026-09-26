# One-sided native-supported adjacent Top-2 recovery report

## Decision

The recovery completed for the **S–H boundary only**. `G_SH_native` passed with a mechanically test-sealed Steady train/validation runtime. `G_HP_native` was independently blocked because the available Periodic modal archives couple train, validation, and heldout coefficients and no separate Periodic train/validation modal bundle exists.

The validation-frozen promotion is:

`T2-C_LearnedConvexCorrection_FieldBlend`

All three S–H routes were finite with zero divergent windows and their validation scores were numerically indistinguishable. The preregistered lower-complexity tie-break therefore selected T2-C before final metric evaluation. RiskPrediction's marginally lower heldout mean does not alter that decision.

## Contract actually tested

- S–H boundary uses only Steady database-native trajectories.
- Steady uses its exact indexed database phase and native feature/history/integrator/pressure-closure contract.
- Hopf is phase-free and starts from the same three-step physical history after analytic S→H projection.
- Both frozen specialists run independently at the same physical query times.
- Fusion is output-only, convex, trajectory-constant, and never fed back.
- No S–P, Top-3, continuous-RHS fusion, per-step switching, or cross-model state feedback is present.

## Independent preflight

`G_SH_native`: PASS. Four train/validation preflight windows had finite fraction 1.0 and zero divergence. Initial S→H relative projection errors were approximately `8.2e-6–2.3e-5` for velocity and `1.5e-5–8.3e-5` for pressure.

`G_HP_native`: `BLOCKED_TEST_SEAL_ASSET`. The metadata-only audit did not load Periodic coefficient arrays. The Periodic index contains 8,539 train, 967 validation, and 644 heldout snapshots, but the coefficients are stored in combined archives.

## Test-seal correction

An early S–H wrapper indexed only train/validation rows but its upstream Steady `Experiment` constructor loaded the combined coefficient archive into memory. Those preflight/cache/training attempts are retained and explicitly invalidated. A clean runtime was then materialized from `steady_trainval_modal_r32.npz` (894 train and 127 validation snapshots), with exact indexed phase rows and a reindexed train/validation perturbation bank. All canonical preflight, cache, and route training results descend from that sealed runtime.

## Shared development cache

- Complete-Re train boundary: 44.478355, 45.795193, 46.440067.
- Complete-Re validation boundary: 43.797394.
- Full K56 windows: 18 train and 5 validation.
- Requested `32 windows/Re` is an upper bound; a roughly 64-snapshot S trajectory permits only 5–6 nonduplicated K56 windows after history requirements. No window was duplicated to inflate sample count.
- Cache SHA256: `f02e7c6c5ac2a6cefb9e9729dde1ae10d8c5bfdfbc20b7f7113ec0f23a3c51e6`.

## Validation selection

All routes: finite fraction 1.0, zero divergence, Top-2 usage 1.0. Approximate joint field metrics:

| Method | all-step mean | all-step worst | K56 mean | K56 worst |
|---|---:|---:|---:|---:|
| E2 pair Top-1 | 0.019290 | 0.160211 | 0.071029 | 0.160211 |
| Fixed 0.5 | 0.011113 | 0.076533 | 0.035102 | 0.076533 |
| E2 probability blend | 0.012709 | 0.093844 | 0.042440 | 0.093844 |
| T2-C | 0.006088 | 0.015940 | 0.007888 | 0.008355 |
| RiskPrediction | 0.006088 | 0.015940 | 0.007888 | 0.008355 |
| LookAhead | 0.006088 | 0.015940 | 0.007888 | 0.008355 |

RiskPrediction and LookAhead validation thresholds were both frozen at `0.05`.

## Final heldout evaluation

Final fields were accessed only after all three checkpoints, thresholds, and the validation decision were frozen. The frozen boundary rule selected Steady heldout `Re=45.142704`, yielding six complete K56 windows. This is not claimed as a historically blind test because legacy test results were known before the recovery; no heldout field or metric was used for this recovery's tuning or selection.

| Method | all-step mean | all-step worst | K56 mean | K56 worst | fixed-point drift mean | Top-2 use |
|---|---:|---:|---:|---:|---:|---:|
| E2 pair Top-1 | 0.021864 | 0.193413 | 0.061307 | 0.170087 | 2.102e-4 | 0% |
| Fixed 0.5 | 0.011022 | 0.096078 | 0.029786 | 0.084354 | 1.860e-4 | 100% |
| E2 probability blend | 0.011071 | 0.096528 | 0.029931 | 0.084751 | 1.861e-4 | 100% |
| T2-C | 0.002298 | 0.020505 | 0.003120 | 0.003533 | 1.796e-4 | 100% |
| RiskPrediction | 0.002291 | 0.021041 | 0.003089 | 0.003517 | 1.796e-4 | 100% |
| LookAhead | 0.002294 | 0.020750 | 0.003097 | 0.003496 | 1.796e-4 | 100% |

The metric-matched trajectory-constant convex oracle mean is `0.002284`. Gaps are `1.48e-5` (T2-C), `7.08e-6` (RiskPrediction), and `1.07e-5` (LookAhead). Risk ranking accuracy is 1.0 for RiskPrediction and 0.333 for LookAhead on six heldout windows. T2-C uses mean S weight `0.00983` (range `0.00177–0.04607`), so this boundary strongly favors the mapped Hopf prediction while retaining a small learned Steady correction.

## Frozen hashes

- T2-C best: `1287f9c1071f2830add29a72a1dca89cac03d603ac89f3cca79b7245863d1225`
- RiskPrediction best: `32882e37c754dfae23ce338e3c646360db0bbf38b4126f8b2dd9616da157a1b1`
- LookAhead best: `fae7626721eba9030454a1fd23d3540ebe975856077768366b09160316c63a5a`
- Frozen decision: `a6b1668bd3dbb271c286eba62c8ede06b6232561ffc4bc3df85c89e23f686212`
- Final test cache: `390863add9ca6798ada08628696eef81c60b9f2895410c293b26a77e06484c9b`

## Limitations

This result certifies one-sided S-native S–H output blending, not symmetric cross-source execution and not true temporal S→H regime migration. Only one validation Re and one heldout Re lie in the tested S–H boundary, with five and six K56 windows respectively. H–P remains unevaluated. The reported Steady boundary attractor diagnostic is fixed-point drift; Hopf/Periodic amplitude, frequency, phase, and orbit metrics are not applicable to this S-native heldout target and must not be inferred from this experiment. Both native specialists must run, so Top-2 specialist cost is approximately twice Top-1; the small gate timing does not include native rollout cost.
