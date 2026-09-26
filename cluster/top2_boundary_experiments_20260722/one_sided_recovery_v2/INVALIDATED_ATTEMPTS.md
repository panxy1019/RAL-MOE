# Invalidated recovery attempts

The following artifacts are retained only as audit evidence and must not be used for scientific comparison or final selection:

- `preflight/G_SH_native_attempt2`: the wrapper indexed only train/validation windows, but the upstream Steady `Experiment` constructor loaded the combined POD coefficient archive, including heldout coefficients, into memory.
- `cache/G_SH_native_attempt1`: derived from that non-sealed runtime.
- `training/S_H_routes_attempt1`: failed before training because of an unavailable Torch convenience API.
- `training/S_H_routes_attempt2`: completed validation optimization on the invalid cache and is therefore scientifically invalid despite not indexing heldout rows during loss computation.

No heldout row or heldout metric was used in the invalid cache or route loss, but memory loading alone violates the stricter recovery test-seal contract. The canonical clean line begins at `sealed_runtime/steady_trainval_attempt3`, `preflight/G_SH_native_attempt4_test_sealed`, and their descendants.
