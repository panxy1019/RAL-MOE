# Circular Periodic supplemental experiments — completed

Accuracy: 12 aligned windows, K48 terminal mean; common local POD-reconstructed physical truth; pressure area-mean removed. Not FOM snapshot error.

| Method | Eu (%) | Ep (%) | Ejoint (%) |
|---|---:|---:|---:|
| Dense correction | 5.459022 | 20.115291 | 25.574313 |
| RAL local specialist | 1.935168 | 7.427516 | 9.362685 |
| POD-Galerkin | 9.121724 | 36.247335 | 45.369059 |
| Global MoE | 1.607222 | 5.864642 | 7.471863 |

## Native implementation runtime

Same host; neural correction on GPU, native NumPy Galerkin on CPU. Modal rollout and pressure reconstruction only; excludes field decoding and loading. This is not a full Top-2 fusion speed comparison.

| Method | K48 median seconds | Peak GPU allocated MiB |
|---|---:|---:|
| dense | 0.576120 | 194.80 |
| full | 2.126448 | 194.78 |
| global | 4.329513 | 194.78 |
| galerkin | 0.374510 | 0.00 |

Scope: one Circular Periodic total-parameter-matched Dense seed. No claim of full multi-benchmark, multi-regime, multi-seed completion. Preserve unfavourable comparisons; do not conclude universal superiority.

See per-run protocol.json, architecture.json, checkpoint hashes and runtime.json for exact provenance.
