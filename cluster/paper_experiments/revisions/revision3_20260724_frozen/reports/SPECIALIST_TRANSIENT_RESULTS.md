# Specialist transient results

| Regime | Method | u K1 | p K1 | u K16 | p K16 | Long | u long | p long | joint long | worst long | finite | divergent | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Steady | Vanilla-FNN-MoE | 0.000219165 | 0.0320706 | 0.000556418 | 0.0244566 | k56 | 0.000842663 | 0.0409119 | 0.0208773 | 0.038043 | 1.00000 | 0 | EARLY_STOPPED_STEP_6200 |
| Steady | DataOnly-MoE | 0.000898074 | 34.38869 | 0.00127414 | 96.10022 | k56 | 0.00256885 | 143.689 | 71.84574 | 136.380 | 1.00000 | 3.00000 | DONE |
| Steady | Proposed Specialist MoE | 0.000219165 | 0.014307 | 0.000550593 | 0.0256445 | k56 | 0.00124462 | 0.0703616 | 0.0358031 | 0.0948226 | 1.00000 | 0 | DONE |
| Hopf | Vanilla-FNN-MoE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | BLOCKED_VALIDATION_NONFINITE |
| Hopf | DataOnly-MoE | 6.11401e-05 | 0.94632 | 0.000724979 | 4.99736 | k56 | 0.00220375 | 14.08848 | 7.04534 | 9.45309 | 1.00000 | 39.00000 | DONE_WITH_DIVERGENT_WINDOWS |
| Hopf | Proposed Specialist MoE | 5.22382e-06 | 0.000661924 | 3.65463e-05 | 0.00172596 | k56 | 0.000126969 | 0.00169688 | 0.000911926 | 0.00230461 | 1.00000 | 0 | DONE |
| Periodic | Vanilla-FNN-MoE | 0.00406129 | 0.0226127 | 0.00900079 | 0.0478907 | k56 | 0.0252646 | 0.105334 | 0.065299 | 0.0908335 | 1.00000 | 0 | DONE |
| Periodic | DataOnly-MoE | 0.00272869 | 0.0107529 | 0.00683409 | 0.0260322 | k56 | 0.0109615 | 0.0461192 | 0.0285403 | 0.105976 | 1.00000 | 0 | DONE |
| Periodic | Proposed Specialist MoE | 0.00186043 | 0.010377 | 0.0033029 | 0.0164071 | k48 | 0.00813182 | 0.0344454 | 0.0212886 | 0.0365367 | 1.00000 | 0 | DONE |

See `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/paper_experiments/reports/SPECIALIST_ABLATION_REPORT.md` for interpretation and source paths.
