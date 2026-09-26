# CenteredSquare Hopf frozen-code audit

## Scope and immutable references

- Only the Hopf specialist is in scope.
- Frozen checkpoint: `selected_validation_step7200.pt`, SHA-256
  `07af06ec5506e5b6a12d9d00f557a4e48fa722be3c2a183df752f0883ad40beb`.
- POD and pressure ranks are both 11.
- Train/validation/held-out split is 23/6/5 complete Reynolds-number cases.
- Held-out Re are hard-disabled in the training program.

## Runtime feature contract

The attachment's 560/192/67 dimensions describe a different rank/configuration.
The frozen CenteredSquare runtime probe is authoritative:

| Component | Dimension |
|---|---:|
| normalized Re and inverse Re | 2 |
| velocity coefficients `a` | 11 |
| pressure coefficients `b` | 11 |
| Galerkin RHS `g` | 11 |
| current energy/norm descriptors | 11 |
| current-only block | 46 |
| each explicit history block: `(a,b,g,da,db,dg)` | 66 |
| two history blocks | 132 |
| frozen H3/H4 network input | 178 |

The source contains an unused helper that would append 25 fluctuation features
(`a'`, `b'`, two critical-plane coordinates, and radius). It is not called by
the frozen trainer or evaluator; the checkpoint encoder weight has shape
`[256,178]`. KDA therefore uses the actual 46-dimensional current-only contract,
not a guessed 176-dimensional input.

## Numerical backbone deviation

The attachment requests direct Galerkin-residual RK4. The frozen square-cylinder
trainer explicitly records that direct Galerkin RK4 is unstable at the native
snapshot interval. Its deployed velocity update learns the finite-difference
derivative while retaining Galerkin as an operator feature. This experiment
keeps that frozen velocity contract for a controlled architecture comparison.
Pressure uses the existing train-only Pressure-Poisson operator with an adaptive
algebraic residual and gate.

## Frozen training facts

- Optimizer: AdamW, cosine schedule, AMP/TF32, gradient clipping 1.0.
- Frozen H4 budget: 8000 optimizer steps.
- Frozen model state: 31,318,329 parameters plus 12 normal-form parameters.
- The normal-form head was diagnostic and did not participate in inference.
- Frozen validation uses Re `94.5, 95.25, 95.5, 97.5, 99, 101.5`.
- Held-out Re `95.1, 95.3, 96.5, 100.5, 102` remain sealed.

## Isolation

All new code, logs, checkpoints and reports are written below
`KDA_PR_FNN_ROM_20260730`. No frozen source, asset, checkpoint or result is
modified.
