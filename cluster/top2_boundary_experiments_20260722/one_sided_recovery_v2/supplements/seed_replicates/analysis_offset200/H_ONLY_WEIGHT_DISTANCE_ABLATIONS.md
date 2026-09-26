# H-only and attached-module ablations

All results use frozen native-output caches and frozen route checkpoints; no specialist was retrained.

## validation

H-only joint all-step mean/worst: `0.00608807243` / `0.0159398634`.

| Method | a_H mean [min,max] | error mean | delta vs H-only | d_H mean | gate-only ablation delta | decisions changed |
|---|---:|---:|---:|---:|---:|---:|
| T2-C_LearnedConvexCorrection_FieldBlend | 0.999993 [0.999984,0.999997] | 0.00608807569 | +3.26e-09 | 2.89e-09 | +0 | 0 |
| RiskPredictionRouter | 0.999997 [0.999995,0.999998] | 0.00608807523 | +2.79e-09 | 1.14e-09 | +0 | 0 |
| LookAheadShortRolloutRouter | 0.999997 [0.999990,0.999999] | 0.0060880729 | +4.66e-10 | 1.42e-09 | +0 | 0 |

## heldout

H-only joint all-step mean/worst: `0.00228635943` / `0.0219203737`.

| Method | a_H mean [min,max] | error mean | delta vs H-only | d_H mean | gate-only ablation delta | decisions changed |
|---|---:|---:|---:|---:|---:|---:|
| T2-C_LearnedConvexCorrection_FieldBlend | 0.989175 [0.948898,0.998482] | 0.00230154209 | +1.52e-05 | 6.01e-06 | +0 | 0 |
| RiskPredictionRouter | 0.992814 [0.969611,0.998503] | 0.00229153037 | +5.17e-06 | 3.93e-06 | +0 | 0 |
| LookAheadShortRolloutRouter | 0.989293 [0.947364,0.998509] | 0.0023016443 | +1.53e-05 | 5.98e-06 | +0 | 0 |

## Seed scope

This file reports the seed embedded in each frozen checkpoint. Cross-seed aggregation, when available, is reported separately and is not used for model selection.
