# Centered-square T2-C descriptor ablation / three-seed study

User authorized proceeding on 2026-09-11 while deferring Table 4 provenance and manuscript descriptor edits. No manuscript sources are changed by these scripts.

## Fixed contract

- Original run: `centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3` on the cluster.
- Boundaries: SH and HP; seeds: 42001, 42002, 42003, chosen before new training.
- Modes: full `[standardized Re, standardized history8]`; mu_only `[standardized Re, zero8]`.
- Architecture: 9 → 64 → 64 → 1, SiLU, 4865 stored trainable parameters in both modes. Inactive history weights still exist in mu_only; this matches allocated architecture, not functional information capacity.
- Original training procedure, 8000 steps, validation every 100 steps plus step 1, validation-only checkpoint selection. No test-based tuning or seed selection.
- Fixed source specialists, E2, development/test caches, candidate order, training normalization and K24 endpoint evaluation.
- No new online experiment-tracking uploads (`--swanlab-mode disabled`).

## Reproduce

Use a CUDA PyTorch environment with NumPy. The completed environment is recorded in `results_v1/preregistration.json`.

```text
python code/run_study.py --run-root <original-run-absolute-path> --output <new-empty-output-path>
python code/verify_checkpoints.py --run-root <original-run-absolute-path> --study <study-output-path>
```

The first command refuses an existing output directory. It first replays the original heldout evaluator and checks 1.1958% / 0.8465%, then runs all twelve fixed-seed jobs. Original assets are never overwritten.

After copying `results_v1` to this local directory:

```text
python ../../scripts/verify_iclr_experiment_tables.py --t2c-study-only
```

The default invocation without `--t2c-study-only` remains the historical Table4 precondition audit and intentionally fails closed. The study-only invocation produces separate `t2c_*` artifacts and does not assert Table4 is resolved.

## Source preservation

`common.py` and `evaluate_heldout_rollout.py` are unchanged copies of the original local source. `train_t2c_gate.py` adds only an input mode, history masking after standardization, run configuration logging, and checkpoint mode metadata. The original training script outside this study is not edited.

Each run stores config, best checkpoint, validation history/metrics, runtime status, and heldout metrics with window weights and per-Re values. Statistics report sample SD over seeds (`ddof=1`), never across-Re SD or confidence intervals.
