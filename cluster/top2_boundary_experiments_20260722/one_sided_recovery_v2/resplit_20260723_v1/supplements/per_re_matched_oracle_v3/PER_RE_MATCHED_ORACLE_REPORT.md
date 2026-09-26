# Per-Re heldout and matched-oracle supplement

| Re | S-only | H-only | E2 Top-1 | T2-C | Best expert overall/K56 | αS mean [min,max] | αH mean [min,max] | Shared-Re oracle | T2-C − oracle | Earliest-gate shared |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 42.359071 | 0.01710303 | 0.17855300 | 0.01710303 | 0.01701915 | S/S | 0.9987 [0.9959,1.0000] | 0.0013 [0.0000,0.0041] | 0.01600690 (αS=0.9709) | +0.00101225 | 0.01682477 (αS=0.9959) |
| 43.500000 | 0.01854086 | 0.03345450 | 0.01854086 | 0.02197138 | S/S | 0.0861 [0.0005,0.4119] | 0.9139 [0.5881,0.9995] | 0.01742054 (αS=0.9015) | +0.00455084 | 0.02390996 (αS=0.4119) |
| 43.900000 | 0.01943789 | 0.02959213 | 0.01943789 | 0.00753485 | S/H | 0.1891 [0.0000,0.9455] | 0.8109 [0.0545,1.0000] | 0.01775225 (αS=0.8607) | -0.01021739 | 0.01829196 (αS=0.9455) |

T2-C beats E2 on 2/3 heldout Re values.

The shared-Re oracle uses one constant weight across all five starts from the same Re trajectory. The previously reported per-window oracle remains listed separately as a more optimistic hindsight bound.
