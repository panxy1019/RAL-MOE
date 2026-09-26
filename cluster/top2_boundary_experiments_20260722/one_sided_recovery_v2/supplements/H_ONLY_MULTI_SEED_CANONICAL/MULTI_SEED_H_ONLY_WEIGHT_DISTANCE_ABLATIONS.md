# Multi-seed H-only and module-ablation supplement

Three frozen route-module seeds per method were evaluated on the same native caches. Specialists were never retrained.

## validation

H-only joint mean/worst: `0.00608807243` / `0.0159398634`.

| Method | seeds | a_H mean of seed means | a_H all-window range | error mean across seeds | d_H mean | ablation max | decisions changed |
|---|---|---:|---:|---:|---:|---:|---:|
| LookAheadShortRolloutRouter | 42003,42103,42203 | 0.999996 | [0.999986,0.999999] | 0.00608807476 | 1.85e-09 | 0 | 0 |
| RiskPredictionRouter | 42002,42102,42202 | 0.999997 | [0.999992,0.999999] | 0.00608807523 | 1.33e-09 | 0 | 0 |
| T2-C_LearnedConvexCorrection_FieldBlend | 42001,42101,42201 | 0.999996 | [0.999984,0.999999] | 0.00608807523 | 1.86e-09 | 0 | 0 |

## heldout

H-only joint mean/worst: `0.00228635943` / `0.0219203737`.

| Method | seeds | a_H mean of seed means | a_H all-window range | error mean across seeds | d_H mean | ablation max | decisions changed |
|---|---|---:|---:|---:|---:|---:|---:|
| LookAheadShortRolloutRouter | 42003,42103,42203 | 0.991306 | [0.947364,0.999444] | 0.00229548492 | 4.8e-06 | 0 | 0 |
| RiskPredictionRouter | 42002,42102,42202 | 0.990538 | [0.926939,0.999207] | 0.00229955814 | 5.24e-06 | 0 | 0 |
| T2-C_LearnedConvexCorrection_FieldBlend | 42001,42101,42201 | 0.991219 | [0.948898,0.998954] | 0.00229628958 | 4.85e-06 | 0 | 0 |

## Scientific interpretation

Across seeds, validation predictions are numerically H-only. Heldout gates retain only a small Steady contribution and do not consistently improve H-only mean error. Removing the Risk critic or LookAhead scorer changes no routed output and no reported error. The current evidence therefore supports H-only as the source of nearly all observed gain; a small convex correction can alter worst-case metrics, but the attached Risk/LookAhead mechanisms have no demonstrated incremental value on this boundary cache.
