# Independent POD build: periodic

Status: **PASS**

- Centering: `single_train_only_regime_mean`
- Pressure gauge: `subtract_area_mean_per_snapshot`
- Fit population: 53 training Re / 8539 snapshots
- Reporting-only population: 6 validation Re, 4 held-out Re
- Retained rank: 32; fixed fair-control rank: 32
- No validation or held-out field contributed to the mean, basis, spectrum, rank selection, or normalization.

## Held-out full-field relative L2

| Variable | local r32 mean | frozen global r32 mean | local r80 mean |
|---|---:|---:|---:|
| velocity | 0.0034207598 | 0.0030143166 | 0.0034207598 |
| pressure (gauge-fixed) | 0.015239387 | 0.014191324 | 0.015239387 |

The frozen global control is the existing transductive asset: it used all Re-specific means and all 100 Re cases. Its comparison is retained for continuity and is not a leakage-free control.
