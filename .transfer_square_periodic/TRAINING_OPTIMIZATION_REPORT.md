# CenteredSquare periodic specialist training optimization

## Outcome

The original 240-epoch schedule assigned only 60 epochs to each of K4/K8/K12/K16. Since the selected best checkpoint was epoch 205, it had seen only 25 K16 epochs when selected.

The new run uses 640 epochs:

| Stage | Epochs | Range |
|---|---:|---:|
| K4 | 120 | 1–120 |
| K8 | 160 | 121–280 |
| K12 | 160 | 281–440 |
| K16 | 200 | 441–640 |

## Performance changes

- Removed CUDA-to-CPU synchronization from per-group and per-expert routing.
- Replaced six independent routed experts in each active group with batched tensor contractions. Evaluation-mode equivalence was checked; maximum absolute output difference was `2.38e-7`.
- Pinned this homogeneous periodic-only dataset to supervised regime group 1, while retaining straight-through gradients for the group router.
- Reused the known eight RK4 substeps instead of reading the fixed snapshot `dt` back from CUDA.
- Enabled BF16 autocast on model-heavy training paths.
- Increased one-step batch from 256 to 1024.
- Increased rollout batch from 24 to 56 and fixed rollout-bearing updates to two per epoch, so changing the one-step batch does not silently reduce rollout learning.
- Deferred detailed batch-metric synchronization to evaluation epochs.
- Added explicit epoch timing, peak allocated memory, and optimization configuration to logs and SwanLab.

## Benchmarks

All K16 benchmarks used two rollout updates per epoch:

| Configuration | Rollout trajectories/epoch | K16 epoch time |
|---|---:|---:|
| Sync-free sparse experts, batch 768, rollout batch 48 | 96 | 176.5 s |
| Batched experts, batch 1024, rollout batch 48 | 96 | 103.5–108.3 s |
| Batched experts + fixed periodic group, batch 1024, rollout batch 64 | 128 | 87.5–91.2 s |

The final stable run uses rollout batch 56 to retain memory headroom. During a 20-second sample, SM utilization was mostly 98–99% (brief minimum 77%), and framebuffer use was approximately 21.1/24.6 GB.

## Active run

- Output: `runs/optimized_long_seed1600`
- PID at launch: `105686`
- SwanLab: https://swanlab.cn/@panxy1019/V17SquarePeriodicMOE/runs/unw6gnxm
- Trainer: `code/train_square_periodic_moe_optimized.py`
- Launcher: `code/run_square_periodic_training_optimized.py`

## Resume after the K8-to-K12 memory boundary

The initial long run completed epoch 280 and then exhausted GPU memory at the
first K12 rollout. The valid recovery checkpoint is
`runs/optimized_long_seed1600/latest.pt` (epoch 280); the best checkpoint at
that point is epoch 270 with validation score `1.225052708853036`.

Training was resumed with the following stage-aware rollout batches:

| Stage | Epoch range | Rollout batch |
|---|---:|---:|
| K4 | 1–120 | 56 |
| K8 | 121–280 | 56 |
| K12 | 281–440 | 40 |
| K16 | 441–720 | 28 |

K16 was extended from 200 to 280 epochs. The scheduler endpoint was changed
from 640 to 720 while preserving the optimizer and scheduler state at epoch
280. Resume checkpoint tensors are explicitly released after loading, and the
best state is retained on CPU to avoid wasting GPU memory.

Active resume log:
`runs/optimized_long_seed1600/train_resume_epoch280_v3.log`.

Resume SwanLab:
https://swanlab.cn/@panxy1019/V17SquarePeriodicMOE/runs/hw6bwkpk
