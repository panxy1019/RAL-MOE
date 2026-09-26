# CenteredSquare periodic CDM-GROM

This work area adds a non-neural continuous Delta-memory residual closure to
the frozen periodic r28/r24 OpenFOAM-discrete FVM--POD--Galerkin baseline.

The physical baseline tensors and pressure closure are immutable.  Only the
small additive memory residual is identified from periodic training
trajectories.  Validation selects the memory rates and ridge penalty; held-out
evaluation is not permitted until the resulting method is frozen.
