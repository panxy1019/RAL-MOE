# CTDM-Galerkin-ROM r32: frozen Hopf code audit

## Scope and immutable inputs

- Baseline: `HopfExpanded34_H4_NormalFormRadial_r32`.
- Train Re: 29 complete trajectories.
- Validation Re: `46.7`, `56.543246`.
- Sealed evaluation Re: `47.081355`, `49.022357`, `51.786450`.
- Velocity and pressure POD ranks: `ru=rp=32`.
- New code, logs, checkpoints and reports live only below
  `CTDM_GALERKIN_ROM_R32_20260730`.
- Existing source, assets, checkpoints and results are read-only references.

## Runtime-verified input contract

The dimensions were checked from source, runtime feature tensors and the frozen
checkpoint. The checkpoint's first encoder matrix has shape `256 x 493`; this is
the authoritative model-input contract.

| Component | Dimension |
|---|---:|
| normalized Re and inverse Re | 2 |
| current velocity coefficients `a` | 32 |
| current pressure coefficients `b` | 32 |
| current Galerkin RHS `g` | 32 |
| current norm/energy descriptors | 11 |
| current base block | 109 |
| one history block `(a,b,g,da,db,dg)` | 192 |
| two history blocks | 384 |
| raw current plus history | 493 |
| implemented Hopf augmentation `(a',b',z1,z2,rho)` | 67 |
| intended dimension if that augmentation were connected | 560 |
| frozen B0 checkpoint input | 493 |
| actual current-only input retained by B2-B4 | 109 |

The 67-dimensional augmentation consists of 32 normalized velocity fluctuations,
32 normalized pressure fluctuations, two critical-plane coordinates and one
normalized radius. Although `build_state` implements this augmentation, the
formal H4 training path binds `state=B.state_features`, not
`state=build_state(...)`; the saved first-layer tensor independently confirms that
the augmentation was loss-only and was not part of the frozen model input.

The new comparison follows the verified checkpoint contract. B1 therefore uses
the original 493-dimensional three-state input and B2-B4 use the true
109-dimensional current-only input. The 67 loss-only descriptors are not added to
the new FNNs because radial/normal-form losses are explicitly removed in this
experiment.

## Numerical backbone

- Velocity: additive Galerkin RHS plus learned standardized residual.
- The baseline evaluates four explicit RK4 velocity stages. Pressure is fixed
  within a macro-step.
- Pressure: train-only Pressure-Poisson base followed by an adaptive algebraic
  residual/gate closure after the accepted velocity step.
- Pressure gauge, POD means/bases, scalers, native query times and physical-field
  evaluator remain unchanged.

## Native time intervals

The sanitized train+validation coefficient view contains nonuniform,
trajectory-dependent native intervals. Typical median intervals range from about
`4.06` to `23.63` physical time units. Some trajectories contain a shorter
residual interval at one boundary. All rollouts therefore use the stored adjacent
time difference, never a global assumed time step.

## Frozen training and baseline result

- Optimizer: AdamW, cosine schedule, gradient clipping `1.0`.
- Formal H4 budget: 8000 optimizer steps; AMP and TF32 enabled.
- Frozen checkpoint: validation-selected step 7200.
- Frozen model state parameter count: `32,004,147`.
- Frozen checkpoint first encoder weight: `encoder.net.0.weight = [256,493]`.
- Training time: 1.979 hours.
- Validation score: `0.878767498`, hard gate passed.
- Historical K48 evaluation: all three evaluation Re were finite with zero
  divergent windows. Velocity field errors were below `0.017%`; pressure errors
  were below `0.279%`. Strict attractor preservation was `0/3`.

The new experiment will not read evaluation tensors until its validation decision
has been frozen. Historical aggregate facts above come from the already published
baseline report, not from opening the sealed tensor view.
