# Steady expanded validation dataset (2026-07-23)

This package adds seven same-configuration OpenFOAM simulations for evaluating
the frozen `steady_specialist_v1`. It does **not** refit the Steady POD basis or
normalization.

Requested Reynolds-number split:

- Train: 43.20, 43.40, 43.60
- Validation: 43.30, 43.70
- Final test (`heldout` in the existing evaluator): 43.50, 43.90

`project_frozen_steady_pod.py` validates mesh/field integrity, applies the
existing per-snapshot pressure gauge, projects onto the frozen rank-32 Steady
POD, and writes an evaluator-friendly modal archive. The raw CFD archives,
metadata, logs, exact split contract, projection report, and checksums are kept
alongside it.
