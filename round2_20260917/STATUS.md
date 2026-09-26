# Approved second-round execution

## 2026-09-18 superseding update

Read 第二轮实验实证进展_20260918.md for current verified status. Earlier launch snapshots below are historical.

- P legacy nine training runs and nine matched-window evaluations completed.
- H proposed: 3 training processes finished, but seed1600 was validation-rejected; only two accepted checkpoints evaluated.
- H Dense: seed1248 failed in FP16 and again in FP32; FP32 remaining fixed seeds continue with failures preserved. H structured legacy training continues.
- OpInf H v2 (corrected blocked velocity weights) and P trained/validated/evaluated; reference is POD, NOT FOM. Synthetic algebra/RK4 tests passed.
- Full/mu-only gate 12-checkpoint reevaluation and exact fusion energy-deficit diagnostic completed.
- Archived low-rank quadratic factors verified identically zero. User approved independent corrected-initialization H/P proposed/structured 12-run campaign. Native random right factor retained, left zeroed; nonzero gradients and actual smoke-checkpoint updates verified.
- Corrected training and subsequent evaluation queues launched. Isolated cost queue waits for compute to be idle.
- FOM evaluation blocked on complete raw data: H source NPZ unreadable (BadZipFile); P raw snapshots not found. User asked for complete source location. Do not replace FOM with reconstructed POD truth silently.
- Final paper integration/diff and full physical-history-to-fusion timing remain incomplete.

Scope: manuscript corrections; frozen-model reevaluation; Circular H/P proposed, dense and structured-nonrouted three-seed controls; regularized OpInf.

## Checkpoints

- [x] Cluster reachable; RTX 4090 initially idle.
- [x] Back up current paper to paper_before/.
- [x] Replace centered-square six-channel descriptor with exact eight-channel equations.
- [x] Numerically test descriptor formulas against original implementation: 200 cases, exact float32 agreement; gate checkpoint input width verified as nine.
- [ ] Remove duplicate Circular range table; migrate unique overlap information and references.
- [x] Circular Periodic PPE metadata/report traced; 63-node effective-polynomial replay passed (maximum absolute difference 1.74e-14). Mesh reassembly and other specialist provenance remain pending.
- [ ] Freeze dataset, model identities, FOM evaluation references and protocol.
- [x] Cached SH/HP full-window, terminal, window-maximum and macro-Re metrics for nine frozen/scalar controls; saved in fusion_reevaluation/.
- [ ] Reevaluate remaining models, mu-only, supported physical diagnostics and end-to-end costs; verify FOM lineage.
- [x] Native H stage-1 eight-update smoke passed; proposed H seeds 1248/1600/2026 launched sequentially in isolated directories.
- [ ] H nonrouted/NN adapters, all P arms, matched active parameter validation and formal heldout reevaluation.
- [x] H quadratic OpInf development pilot fitted on sealed train/validation asset, 13 ridge values, fixed RK4 substeps; checkpoint selected without heldout access.
- [ ] OpInf synthetic regression/integration unit tests, P adapter, full FOM-reference heldout evaluation and final baseline protocol verification.
- [ ] Complete evaluation, paper integration and compiled review diff only.

H proposed training is running remotely; local status JSON files are snapshots, not live status. On the user's parallel-execution request, P proposed and P controls lanes were launched alongside H. P proposed, active-matched Dense and nonrouted structured each passed a one-epoch train/validation/checkpoint smoke. P proposed lane runs three seeds; P controls lane runs Dense and structured sequentially for each of three seeds. Each lane fails closed on errors; requires 8 GiB free before launch. No heldout evaluation during training. Historical runs are evidence candidates, not automatically accepted as new-protocol results. Full source backup and old metrics are preserved. No normal main.pdf compilation; updated review diff remains pending final integration.

Outstanding architecture caveat: native Periodic physical-zero initialization zeros both low-rank quadratic factors. Structured control preserves that policy for fidelity; effective quadratic gradients must be audited before interpreting a nonlinear/linear/quadratic mechanism comparison. No silent change to the native method was made.
# 2026-09-20：H 修正版 FP32 修复与 P 三种子汇总

- H 修正版的 AMP 全覆盖预检在 `Re=51.066785, K=8` 出现非有限前向滚动；同一初始化和数据的 FP32 预检通过全部 116 个 `Re × K` 检查。
- proposed 与 structured 两种架构的 FP32 冒烟均通过。新的隔离目录为 `quadratic_fixed_fp32/H`，旧的六次失败记录保留。
- 正式队列已启动，按 proposed/structured × 1248/1600/2026 顺序执行；独立评价队列会在每次训练完成后检查原验证准入并评价通过的检查点。
- 2026-09-20 本次快照：proposed seed 1248 已到 optimizer step 720，K=1，未出现非有限值。完整训练尚未完成。
- Circular P 修正版 6/6 训练和评价完成。K24 三种子：修正版 proposed `Eu+Ep=3.0667±1.8373%`，原版 proposed `2.8791±1.7269%`，Dense `1.3251±0.2268%`；修正版 structured `2.8598±1.3081%`，原版 structured `2.8666±1.4342%`。
- P 的详细逐种子结果和解释见 `Circular_P_修正版三种子对比_20260920.md`。所有误差当前仍以 POD 重构场为参考。
