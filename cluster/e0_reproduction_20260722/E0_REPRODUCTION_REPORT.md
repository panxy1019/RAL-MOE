# E0 native specialist reproduction

Status: **PASS**. No Router, physical-field mixture, specialist training, or checkpoint modification was performed.

Each specialist was evaluated only through its own frozen native evaluator on the new RTX 4090. New outputs were compared against the corresponding new-cluster reference artifacts, field by field.

| Specialist | Frozen checkpoint | Native result | Numeric comparison |
|---|---|---|---|
| Periodic | epoch 85 | 3/4 held-out preserved; all K1–K48 windows finite; zero divergence | 318 common numeric fields; max absolute difference `0.0` |
| Hopf | H4 step 7200 | 3/3 K48 finite; zero divergence; strict attractor preserved 0/3 | 254 common numeric fields; max absolute difference `0.0` |
| Steady | S4 step 1200 | clean-rollout and fixed-point gates pass; contraction protection fails, so native decision is `S4_FAILED` | 118,974 common numeric fields; max absolute difference `0.0` |

Steady note: the first fresh finalizer attempt completed all rollout groups but lacked `validation.json` during report aggregation. That attempt is retained as `steady_attempt1_missing_validation` in the remote E0 directory. The second attempt used the existing checkpoint-selection-only validation artifact, then reproduced the complete prior 4090 raw metrics exactly. This is an evaluation-harness dependency, not a model, data, or checkpoint mismatch.

Remote outputs: `/root/panxy/particalMOE/e0_reproduction_20260722`

Raw comparison inputs are stored beside this report in `raw/`.
