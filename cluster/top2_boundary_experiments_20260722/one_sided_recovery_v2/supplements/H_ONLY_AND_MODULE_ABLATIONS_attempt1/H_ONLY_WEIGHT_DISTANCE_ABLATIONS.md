# H-only and attached-module ablations

All results use frozen native-output caches and frozen route checkpoints; no specialist was retrained.

## validation

H-only joint all-step mean/worst: `0.00608807243` / `0.0159398634`.

| Method | a_H mean [min,max] | error mean | delta vs H-only | d_H mean | gate-only ablation delta | decisions changed |
|---|---:|---:|---:|---:|---:|---:|
| T2-C_LearnedConvexCorrection_FieldBlend | 0.999995 [0.999989,0.999997] | 0.00608807569 | +3.26e-09 | 2.12e-09 | +0 | 0 |
| RiskPredictionRouter | 0.999996 [0.999992,0.999999] | 0.00608807476 | +2.33e-09 | 1.68e-09 | +0 | 0 |
| LookAheadShortRolloutRouter | 0.999995 [0.999991,0.999999] | 0.00608807988 | +7.45e-09 | 2.24e-09 | +0 | 0 |

## heldout

H-only joint all-step mean/worst: `0.00228635943` / `0.0219203737`.

| Method | a_H mean [min,max] | error mean | delta vs H-only | d_H mean | gate-only ablation delta | decisions changed |
|---|---:|---:|---:|---:|---:|---:|
| T2-C_LearnedConvexCorrection_FieldBlend | 0.990173 [0.953934,0.998234] | 0.0022984317 | +1.21e-05 | 5.45e-06 | +0 | 0 |
| RiskPredictionRouter | 0.993082 [0.971403,0.998903] | 0.00229073199 | +4.37e-06 | 3.77e-06 | +0 | 0 |
| LookAheadShortRolloutRouter | 0.991366 [0.961936,0.999444] | 0.00229435368 | +7.99e-06 | 4.75e-06 | +0 | 0 |

## Seed scope

The frozen preregistration contains one seed per method (42001, 42002, 42003). This supplement reports each seed explicitly but does not claim within-method multi-seed robustness.
