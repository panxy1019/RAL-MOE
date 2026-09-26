# Experimental strengthening, 2026-09-15

Scope: user-supplied ICLR_Experimental_Strengthening_and_Ablation_Plan.md.
Preserve original checkpoints, data splits and manuscript. All new outputs isolated.

1. Constant-alpha: centered-square SH/HP, original K24 frozen candidate caches. Fit one scalar on TRAIN only using the existing full-horizon squared normalized field loss. Analytic convex optimum, clipped to [0,1]. A validation-selected grid scalar is a separately labelled sensitivity comparison; no held-out selection. Match original candidate order (SH=S/H, HP=P/H). Compare fixed original T2-C, equal, E2 and future-informed oracle. Save per-step errors and router/alpha window data.
2. Dense correction: inspect native trainers and checkpoint architecture; replace sparse closure with a single dense correction, retain physical backbone/input/curriculum. Count total and active parameters explicitly. Smoke-test before training. Do not silently bypass preflight safety gates.
3. Native POD-Galerkin / Global MoE / local model rollout curves: reconstruct from compatible held-out trajectories or rerun evaluators, not from terminal values.
4. Same-hardware synchronized rollout and peak-memory measurement, warm-up and repetitions, specify included operations.
5. Router/fusion plots using available held-out overlap values; no invented interpolation or new data.
6. Report experimental outcome without assuming improvement. Supply bounded claims and oracle diagnostic wording.

Status: COMPLETE within the defined scope: centered-square SH/HP constant-alpha and sensitivity controls; Circular Periodic single-seed Dense training (native early stopping at epoch 180, selected epoch 20), aligned 12-window K48 comparisons and native runtime/GPU memory for Dense, Full local, Global and Galerkin. Zombie-process scheduler bug fixed and four tests passed. Global replay verified. Final report and paper-ready inserts disclose projected truth, pressure centering, auxiliary phase/mean-field contracts and native timing exclusions. This is not a full multi-regime/multi-seed or end-to-end Top-2 benchmark. See FINAL_REPORT_CN.md.
