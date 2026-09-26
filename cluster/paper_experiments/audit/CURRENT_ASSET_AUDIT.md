# Current paper-experiment asset audit

Date: 2026-07-23  
Status: **BLOCKED_BY_PREREGISTERED_HARD_STOP**  
New Vanilla-FNN/DataOnly training started: **No**

## Outcome

The proposed specialist expert is not a vanilla FNN. It contains residual FFN
blocks, an explicit linear state branch, and a low-rank quadratic state branch.
The requested Vanilla-FNN replacement is therefore a meaningful controlled
ablation. The supplied DataOnly definition is also sufficiently explicit to
implement.

SwanLab 0.9.0 is installed in the actual training environment.

However, the audit found a blocking conflict with the supplied experiment
contract before any new training was started.

## Blocking finding 1: the uploaded Global MoE uses an all-Re global POD

The uploaded Global MoE uses:

```text
V16_1_SteadyPressureAnchor32/assets/common_global_data/
Global_POD_AreaWeighted_L2
```

Its POD metadata records 14,167 snapshots spanning all 100 Reynolds numbers
from Re=20 through Re=200. The accompanying project split contract explicitly
discloses that the frozen global POD and Re-specific means were fit using all
100 Re cases. The Global model itself subsequently divides these cases into 89
training and 11 held-out Reynolds numbers.

Consequently, the 11 model-held-out cases were already used to construct the
global representation. This triggers the attachment's mandatory stop clause:

> Stop if POD used validation or test data.

The existing frozen-checkpoint transient rollout completed successfully in
538.706 seconds, but it cannot be presented as a leakage-free held-out Global
MoE result under the supplied paper contract. Its unified attractor evaluation
has not yet been run.

## Blocking finding 2: the requested Hopf “test” cases are training cases

Re=49.3, 49.6, and 50.0 all belong to the expanded Hopf training set. They may
be retained exactly as requested as named **in-sample strict-K56
attractor-preservation case studies**. They cannot simultaneously be labeled an
independent test set under the complete-Re isolation rule.

The scientifically safe comparison is:

- retain the actual disjoint Hopf held-out population for test/generalization;
- additionally report Re=49.3, 49.6, and 50.0 as training-point preservation
  examples.

## Frozen checkpoint inventory

| Model | Frozen artifact | SHA256 | Selection |
|---|---|---|---|
| Steady | `frozen_s4_validation_step_1200.pt` | `bcab5661af39e3262103ed887ce9846415467452b453f3dcc14e133f0f8bd80d` | step 1200 |
| Hopf | `best_validation.pt` | `02148741ed8bc9fbec88709f69b10492e263514b2edcee76652485b399962235` | optimizer step 7200 |
| Periodic | `FINAL_PERIODIC_SPECIALIST.pt` | `b2052a746fdaab60f83e65d1b91d038a9dded8255ce74f41e3f6fca4c9e717c5` | epoch 85 |
| Global MoE | `V16_1_..._checkpoint.pt` | `9d7e2b861d8c501eff87a940b93f7fe64ca72fd976afc0c6fc1621ff8f0b284f` | epoch 225 |

All four use the r32/rp32 control configuration.

## Global POD evidence

| Asset | Shape | SHA256 |
|---|---|---|
| Velocity POD | phi `(80,194736)`, coeff `(14167,80)` | `11633d29b3c7db8cf73a0d813504521829689f32e6e9d2d233d90062becf76c5` |
| Pressure POD | phi `(80,97368)`, coeff `(14167,80)` | `6103c45d77da830aa5cb29f081047440ad561ec27bfad36594b381e2770abcc8` |
| Snapshot index | all 100 Re | `bde29f13e59c84335c1810f53172f701a8633c5eaa704802a9b4a72f1d884aee` |

## Required decision before implementation resumes

There are two defensible Global-MoE routes:

1. Build a train-only global POD, mean, scaler, and ROM, then warm-start or
   retrain the Global MoE for a leakage-free comparison.
2. Explicitly revise the paper contract and label the uploaded model a
   **transductive all-Re representation baseline**, not a held-out POD
   generalization baseline.

The specialist ablations can then use the frozen specialist-local assets and
SwanLab. No new ablation process was launched while the governing contract was
in a hard-stop state.

Machine-readable evidence is in `CURRENT_ASSET_AUDIT.json`; raw checkpoint
metadata and source hashes are in `checkpoint_contracts.json`.
