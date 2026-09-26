# Independent POD build: hopf

Status: **PASS**

- Centering: `single_train_only_regime_mean`
- Pressure gauge: `subtract_area_mean_per_snapshot`
- Fit population: 29 training Re / 4092 snapshots
- Reporting-only population: 2 validation Re, 3 held-out Re
- Retained rank: 80; fixed fair-control rank: 32
- No validation or held-out field contributed to the mean, basis, spectrum, rank selection, or normalization.

## Held-out full-field relative L2

| Variable | local r32 mean | frozen global r32 mean | local r80 mean |
|---|---:|---:|---:|
| velocity | 2.4602172e-06 | 0.00018582597 | 2.4584124e-07 |
| pressure (gauge-fixed) | 8.5768202e-06 | 0.0013495404 | 1.5938141e-06 |

The frozen global control is the existing transductive asset: it used all Re-specific means and all 100 Re cases. Its comparison is retained for continuity and is not a leakage-free control.
