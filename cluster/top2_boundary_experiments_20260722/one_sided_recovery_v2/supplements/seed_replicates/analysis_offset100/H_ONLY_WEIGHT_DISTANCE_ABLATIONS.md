# H-only and attached-module ablations

All results use frozen native-output caches and frozen route checkpoints; no specialist was retrained.

## validation

H-only joint all-step mean/worst: `0.00608807243` / `0.0159398634`.

| Method | a_H mean [min,max] | error mean | delta vs H-only | d_H mean | gate-only ablation delta | decisions changed |
|---|---:|---:|---:|---:|---:|---:|
| T2-C_LearnedConvexCorrection_FieldBlend | 0.999999 [0.999998,0.999999] | 0.0060880743 | +1.86e-09 | 5.87e-10 | +0 | 0 |
| RiskPredictionRouter | 0.999997 [0.999996,0.999999] | 0.00608807569 | +3.26e-09 | 1.17e-09 | +0 | 0 |
| LookAheadShortRolloutRouter | 0.999996 [0.999986,0.999999] | 0.0060880715 | -9.31e-10 | 1.9e-09 | +0 | 0 |

## heldout

H-only joint all-step mean/worst: `0.00228635943` / `0.0219203737`.

| Method | a_H mean [min,max] | error mean | delta vs H-only | d_H mean | gate-only ablation delta | decisions changed |
|---|---:|---:|---:|---:|---:|---:|
| T2-C_LearnedConvexCorrection_FieldBlend | 0.994307 [0.977790,0.998954] | 0.00228889496 | +2.54e-06 | 3.08e-06 | +0 | 0 |
| RiskPredictionRouter | 0.985718 [0.926939,0.999207] | 0.00231641205 | +3.01e-05 | 8.04e-06 | +0 | 0 |
| LookAheadShortRolloutRouter | 0.993259 [0.971536,0.998677] | 0.00229045679 | +4.1e-06 | 3.68e-06 | +0 | 0 |

## Seed scope

This file reports the seed embedded in each frozen checkpoint. Cross-seed aggregation, when available, is reported separately and is not used for model selection.
