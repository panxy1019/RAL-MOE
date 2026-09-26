# S–H Top-2 Re-resplit comparison

All three routes used the same frozen-specialist cache, seed 42001, and 8000 optimizer steps.

| Method | Val mean | Val worst | Test mean | Test worst | Test K56 mean | Top-2 usage |
|---|---:|---:|---:|---:|---:|---:|
| T2-C_LearnedConvexCorrection_FieldBlend | 0.01320503 | 0.10878930 | 0.01550846 | 0.14502297 | 0.03045001 | 1.000 |
| RiskPredictionRouter | 0.01320503 | 0.10878930 | 0.01550846 | 0.14502297 | 0.03045001 | 1.000 |
| LookAheadShortRolloutRouter | 0.01320503 | 0.10878930 | 0.01550846 | 0.14502297 | 0.03045001 | 1.000 |

## Heldout baselines

| Baseline | Mean joint error | Worst | K56 mean |
|---|---:|---:|---:|
| S_only | 0.01836060 | 0.15658388 | 0.04670979 |
| H_only | 0.08053322 | 0.31622419 | 0.05606842 |
| E2_pair_Top1 | 0.01836060 | 0.15658388 | 0.04670979 |
| fixed_0.5 | 0.04355361 | 0.15544908 | 0.04474644 |
| E2_pair_probability_blend | 0.03054046 | 0.10782462 | 0.04175646 |
| per_window_convex_oracle | 0.01110288 | 0.12744576 | 0.02481132 |

## Decision

Validation-only promotion: `T2-C_LearnedConvexCorrection_FieldBlend`.

RiskPrediction and LookAhead produced exactly the same final predictions as the common convex gate. Their Top-2 usage was 1.0, and their gate-only ablations had unchanged error; the extra scorer modules therefore added no output-level value in this split.

This rerun is not a blind test. Heldout values were excluded from optimization, checkpoint selection, threshold selection, and promotion.
