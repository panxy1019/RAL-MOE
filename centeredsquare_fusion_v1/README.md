# CenteredSquare E2 + T2-C fusion reproduction

This directory migrates the frozen CenteredSquare Steady, Hopf, and Periodic
specialists to the final hierarchical fusion contract:

- one Re-only E2 decision per trajectory/query;
- only adjacent S-H and H-P pairs;
- pair-specific zero-initialized convex correction gates;
- independent native rollouts from a common three-frame physical history;
- output-field fusion only, with no feedback into either specialist;
- train-only normalization and validation-only checkpoint selection;
- development cache contains only train/validation trajectories.

The common horizon is K24 because all three delivered specialists have a
validated finite K24 autonomous-rollout contract. Candidate models are loaded
sequentially while building each cache to stay within a partially occupied
24 GiB GPU.

`build_boundary_cache.py` is the dataset-specific adapter and preflight.
`train_e2_router.py` and `train_t2c_gate.py` retain the reference router/gate
math while removing old dataset paths and ranks.

## Plotting and migration

The current SH/HP boundary figure entry points, frozen inputs, output locations,
scientific contracts, and migration checklist are documented in
[`PLOTTING_MIGRATION_GUIDE.md`](PLOTTING_MIGRATION_GUIDE.md). Exact SHA256
values and the current local output-name aliases are recorded in
[`PLOTTING_FILE_MANIFEST.json`](PLOTTING_FILE_MANIFEST.json).
