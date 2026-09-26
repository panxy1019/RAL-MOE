# CenteredSquare POD--Galerkin rebuild

This independent work area rebuilds and audits the non-neural base ROM before
any memory closure is considered. Hopf train/validation was the first target;
because its stable closure still over-damped the bifurcating oscillation, the
periodic split was rebuilt and frozen as the recommended baseline.

Main result: `reports/POD_GALERKIN_REBUILD_REPORT.md`.

The final model is an OpenFOAM-discrete r28 velocity FVM--Galerkin operator
with an r24 linear pressure-coefficient closure. Its four frozen held-out cases
all reach t=500; mean one-step velocity error is 6.11%. At Re=120.69 and
144.83, periodic-orbit amplitude/frequency errors are both below 4.1%.
