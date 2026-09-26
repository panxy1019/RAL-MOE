# CenteredSquare Hopf CDM-GROM Implementation

This directory implements the deterministic, operator-based CDM-GROM theory
for the frozen CenteredSquare Hopf split. It is independent of the existing
HPRS-MoE-ROM and neural CTDM experiments.

The implementation contract is:

- train-only volume-weighted POD is recomputed and stores 64 velocity and 64
  pressure modes;
- modes 1--11 are resolved and modes 12--64 form a 53-state unresolved
  realization;
- the linearization reference is the last-32-snapshot mean of each training
  trajectory, and its nonzero affine residual is retained explicitly;
- Pressure--Poisson is algebraic;
- no neural network, MoE, attention, phase input, or learned router is used;
- validation and held-out cases are never used to fit POD means, bases, ROM
  tensors, equilibrium references, or stability corrections;
- held-out evaluation is forbidden until a validation report and frozen
  method manifest exist.

See `CONFIG.json` for the frozen split and numerical defaults.
