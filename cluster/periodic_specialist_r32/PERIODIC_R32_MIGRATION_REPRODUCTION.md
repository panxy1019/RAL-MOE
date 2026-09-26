# Periodic r32/rp32 migration reproduction

## Result

The transferred Periodic specialist is reusable on the target cluster. The
native Periodic POD coordinates, snapshot index, Galerkin tensors, pressure
surrogate, finalized checkpoint, training code, and evaluation code form a
complete runtime package.

Target package:

`/root/panxy/particalMOE/periodic_specialist_r32`

Runtime environment:

- GPU: NVIDIA GeForce RTX 4090, 24 GB
- Python: `/root/miniconda3/envs/pt_env/bin/python`
- PyTorch: `2.13.0+cu126`
- CUDA available: yes

## Native data contract

- Total snapshots: 10,150
- Valid history-3 samples: 9,961
- Velocity rank: 32
- Pressure rank: 32
- Train: 53 Reynolds numbers, 8,380 samples
- Validation: 6 Reynolds numbers, 949 samples
- Heldout: 4 Reynolds numbers, 632 samples
- Train/validation/heldout intersections: all zero

Heldout Reynolds numbers:

`70.314635, 100.352251, 149.059229, 189.862278`

## One-epoch from-scratch smoke

Output:

`reproduction/one_epoch_from_scratch_seed1600`

The smoke run reused the finalized checkpoint's scientific configuration but
did not load its model, optimizer, scheduler, or RNG state. It started from
seed 1600 and random initialization, with only these operational overrides:
local runtime paths, one epoch, validation every epoch, isolated output, and
SwanLab disabled.

- Runtime: 137.477 seconds
- Train loss: 3.9811626564
- Validation score: 0.3957520019
- Validation RHS relative L2: 0.2417745739
- Validation coefficient relative L2: 0.1295826435
- Validation pressure relative L2: 0.0696993843
- OOM/NaN: none

The original epoch-1 log recorded train loss 3.9633279352 and validation score
0.3911216082. The small numerical difference is expected across GPU and
PyTorch/CUDA kernels; it does not indicate a data-contract mismatch.

## Frozen best-checkpoint heldout reproduction

Checkpoint:

`checkpoint/FINAL_PERIODIC_SPECIALIST.pt`

Checkpoint SHA256:

`b2052a746fdaab60f83e65d1b91d038a9dded8255ce74f41e3f6fca4c9e717c5`

Output:

`reproduction/best_epoch85_heldout_eval_v2`

The evaluation used the frozen epoch-85 weights and the native local r32 POD
data, not the incompatible unified heldout POD coordinates.

- Preserved count: 3/4, identical to the source result
- Finite fraction: 1.0 at every Re and horizon
- Divergent windows: 0
- Common numeric leaves compared: 318
- Maximum absolute difference versus source JSON: 3.8185e-6
- Maximum relative difference versus source JSON: 0.0447%

K48 area-weighted physical-field relative L2 errors:

| Re | Velocity | Pressure | Preserved |
|---:|---:|---:|:---:|
| 70.314635 | 1.347254% | 5.960086% | No |
| 100.352251 | 0.397196% | 1.708284% | Yes |
| 149.059229 | 0.380851% | 1.682540% | Yes |
| 189.862278 | 1.127427% | 4.427241% | Yes |

## Portable entry points

- `code/run_periodic_migration_smoke.py`: reuses the finalized scientific
  configuration for a one-epoch, from-scratch migration smoke.
- `code/evaluate_periodic_r32_portable.py`: adds explicit local overrides for
  the POD root, velocity ROM, and pressure-surrogate paths while leaving the
  model and metric definitions unchanged.
- `MIGRATION_RUNTIME.sha256`: hashes the checkpoint and all runtime-critical
  data/code files.

The failed first evaluation attempt is retained at
`reproduction/best_epoch85_heldout_eval`; it stopped before rollout because the
source training metrics dependency had not yet been transferred. The valid
reproduction is the `_v2` directory above.
