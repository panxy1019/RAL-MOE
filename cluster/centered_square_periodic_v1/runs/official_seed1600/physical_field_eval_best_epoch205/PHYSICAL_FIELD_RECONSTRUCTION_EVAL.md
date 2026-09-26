# CenteredSquare Periodic Specialist — Physical-field reconstruction evaluation

Checkpoint: `best_validation.pt` (epoch 205). This is an independent post-training evaluation; it does not modify the trained model.

## Definition

- Four held-out Reynolds numbers: 100.5, 102.0, 120.690, 144.828.
- Autonomous rollout with external snapshot step `dt=4`; RK4 uses eight internal substeps (`dt=0.5`).
- Coefficients are reconstructed using the global POD mean and retained modes (`r_u=28`, `r_p=26`) and compared directly to the original full CFD snapshots on 9,400 cells.
- Velocity and pressure errors are aggregate volume-weighted relative L2 over all selected windows and all field components. Pressure is volume-mean removed independently per frame, so the pressure gauge is consistent.
- Rollout starts use the checkpoint's evaluation stride (24); K1/K4 use 20 windows and K8/K16/K24 use 16 windows. No window became non-finite.
- `POD floor` is the error from reconstructing the reference coefficient itself. It isolates representation truncation from model rollout error.

| Horizon | Physical time | Velocity field L2 | Pressure field L2 | Velocity POD floor | Pressure POD floor | Windows |
|---:|---:|---:|---:|---:|---:|---:|
| K1 | 4 | 15.06% | 17.79% | 0.407% | 0.457% | 20 |
| K4 | 16 | 13.86% | 16.24% | 0.398% | 0.442% | 20 |
| K8 | 32 | 12.80% | 15.03% | 0.397% | 0.428% | 16 |
| K16 | 64 | 12.68% | 15.09% | 0.397% | 0.432% | 16 |
| K24 | 96 | 13.03% | 15.48% | 0.402% | 0.435% | 16 |

K24 per-Re field errors (velocity / pressure): 100.5: 4.19% / 3.12%; 102.0: 5.11% / 4.16%; 120.690: 12.84% / 14.31%; 144.828: 21.36% / 27.62%.

Machine-readable measurements: `physical_field_eval_best_epoch205/physical_field_multihorizon.json`.
