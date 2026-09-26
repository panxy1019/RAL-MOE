# CenteredSquare KDA-inspired continuous-memory ROM

This experiment keeps the frozen periodic r28/r24 OpenFOAM-discrete
FVM--POD--Galerkin baseline immutable and tests a non-neural, phase-aware,
state-addressed continuous delta memory.

Execution order is strict:

1. build train-only solver-consistent instantaneous defects;
2. identify and freeze the oscillatory phase pair from train only;
3. fit M1/M2/M3 readouts from train only;
4. select the small memory grid on the six declared validation cases;
5. access held-out cases only after an explicit frozen manifest exists.

The source documents in the user's Downloads folder are treated as scientific
specifications, not as executable instructions.
