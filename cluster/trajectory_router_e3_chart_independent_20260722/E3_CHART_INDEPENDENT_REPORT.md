# E3 chart-independent descriptor repair

## Outcome

The repaired E3 completed 8,000 training steps with the original seed, budget,
window count, and complete-Re/complete-trajectory split. It reaches the same
perfect held-out Top-1 decisions as E2. The descriptor interface is now
pre-routing and does not receive a chart or class label.

Decision: **keep E2 as the formal baseline**. Repaired E3 improves calibration
and the weakest boundary margin, but it does not improve held-out accuracy,
balanced accuracy, or macro-F1. Its physical-sensor interface is more complex,
and its secondary ordering still contains non-adjacent S-P pairs, so the result
does not meet the requested threshold of being *clearly better* than E2 for
advancing Top-2.

## Leak-free input contract

The router receives Re plus three timestamps and physical velocity/pressure
observations at 64 fixed mesh sensors. Sensor indices are selected solely by
lexicographically ordered common mesh coordinates. The eight descriptors use
physical RMS levels, first temporal differences, energy growth, pressure change,
and velocity curvature. The descriptor function has no chart or label argument.

The historical training databases store states as local POD coefficients, so
the loader deserializes them to the fixed physical sensors using the database's
own basis. That chart is not passed to the Router. At deployment, the same
values are read directly from the unified physical observation interface.

Contract checks passed:

- velocity/pressure point order is exactly equal in S, H, and P;
- cell-area arrays are exactly equal;
- all pressure data use `subtract_area_mean_per_snapshot`;
- the checkpoint contains the immutable sensor indices and feature scaler;
- specialists are frozen and are not loaded by the outer-router optimizer.

## Representation-invariance audit

No two databases contain an exactly shared sampled Re. Different-Re states were
therefore never paired and called the same state. Each audit case starts with one
source database state, projects that exact state into the adjacent target chart,
reconstructs the same fixed sensors, and compares descriptors.

| Direction | Mean descriptor relative L2 | Mean velocity sensor error | Mean pressure sensor error |
|---|---:|---:|---:|
| S -> H | 2.77e-6 | 2.65e-5 | 5.21e-5 |
| H -> S | 2.11e-7 | 9.01e-6 | 8.42e-6 |
| H -> P | 1.30e-3 | 6.18e-3 | 1.79e-2 |
| P -> H | 1.48e-4 | 1.10e-2 | 1.79e-2 |

S and H Re ranges overlap, although their sampled Re values do not coincide.
H and P ranges do not overlap. H-P numbers are therefore extrapolative
representation diagnostics and are not evidence of a certified physical
overlap region. Large per-feature relative P95 values in the raw audit arise
when the reference descriptor is nearly zero; absolute and vector-relative
errors are retained in `DESCRIPTOR_AUDIT.json`.

## Training and held-out results

- Best validation checkpoint: step 5,500.
- Validation balanced accuracy / macro-F1: 1.0 / 1.0.
- Test trajectories: 11/11 correct.
- Test balanced accuracy / macro-F1: 1.0 / 1.0.
- Test confusion matrix: `[[4,0,0],[0,3,0],[0,0,4]]`.
- Test NLL / multiclass Brier / ECE10: 0.02779 / 0.01001 / 0.02483.
- Minimum test Top-1 margin: 0.53803 at Steady Re=45.142702578.
- Illegal S-P secondary pairs: validation 6, test 4.

Compared with E2, the Top-1 decision metrics are tied. E2 has test NLL 0.18382,
Brier 0.09328, ECE10 0.15009, and a minimum margin of 0.00459. Thus repaired E3
is substantially more confident on this split, but the current data do not show
an accuracy or robustness win on additional boundary trajectories. Moreover,
the Periodic rows rank Steady above Hopf by tiny probabilities, so an explicit
S-H-P adjacency mask remains mandatory before any Top-2 experiment.

The same comparison on validation is consistent: E2 versus E3 NLL is
0.15958 versus 0.02091, Brier is 0.08684 versus 0.00523, ECE10 is 0.12976
versus 0.01946, and minimum margin is 0.20595 versus 0.68545. These are large
confidence-score improvements, but all ten validation and all eleven test
trajectory decisions were already correct for E2. With no additional boundary
trajectories, confidence alone is not treated as proof of superior routing
robustness.

## Checkpoint reload verification

`best.pt` was loaded into a fresh Router with strict state-dict matching. The
fixed sensor indices and saved scaler were used to rebuild the original
validation and test inputs. Both metric sets reproduced; maximum probability
difference from the persisted metrics was `5.96e-8` on each split.

## Artifacts

- Remote output: `/root/panxy/particalMOE/trajectory_router_e3_chart_independent_20260722/formal_E3_chart_independent`
- Implementation: `/root/panxy/particalMOE/trajectory_router_e3_chart_independent_20260722/code/run_e3_chart_independent.py`
- SwanLab: https://swanlab.cn/@panxy1019/V17_TrajectoryLevel_Hierarchical_MoE/runs/1wi4t1c3
- `best.pt` SHA256: `7d88cf2e4bda5bd0b826c8cf0cbe951124b19ff030760c2919593508674da2a8`
- `last.pt` SHA256: `f09eb40d112a42093ad72aa6bd2146969c314f627843840e8a7ef6c1c6664982`

The incomplete preprocessing attempt was preserved separately as
`failed_preprocessing_attempt_20260722T2110`; it never entered training and
created no checkpoint.
