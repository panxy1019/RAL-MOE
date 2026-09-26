# Complete execution commands

All commands run in the cluster `pt_env`. The scripts fail closed if another CUDA
compute process is present.

```bash
ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/CTDM_GALERKIN_ROM_R32_20260730
PY=/root/miniconda3/envs/pt_env/bin/python

# Mathematical gate
"$PY" "$ROOT/code/test_memory_numerics.py" \
  --output "$ROOT/numerical_tests/results.json"

# Low-priority CPU shape/gradient/checkpoint gate
bash "$ROOT/code/run_cpu_smoke_all.sh"

# CUDA bfloat16 shape, forward, backward and checkpoint gate
bash "$ROOT/code/run_smoke_all.sh"

# B4 K56 batch-size benchmark
bash "$ROOT/code/run_b4_batch_benchmark.sh"

# Deterministic batch selection, B1--B4 matched-budget training and validation
bash "$ROOT/code/run_single_seed_pipeline.sh"

# Confirmatory seeds 2248 and 3248 for every seed-1248 candidate, followed by
# the frozen validation decision, gated test (only if authorized), environment
# capture and final report generation.
bash "$ROOT/code/run_remaining_seeds_pipeline.sh"

# Idempotent post-training entry point if training was launched separately.
bash "$ROOT/code/finalize_evaluation.sh"
```

The frozen B0 K56 validation baseline is evaluated independently:

```bash
"$PY" "$ROOT/code/analyze_b0_checkpoint.py" \
  --split validation \
  --checkpoint <FROZEN_B0_CHECKPOINT> \
  --baseline-trainer <FROZEN_BASE_TRAINER> \
  --analysis-helper "$ROOT/code/analyze_ctdm_checkpoint.py" \
  --coefficient-view <SANITIZED_TRAINVAL_VIEW> \
  --galerkin-path <TRAIN_ONLY_GALERKIN> \
  --pressure-path <TRAIN_ONLY_PRESSURE_POISSON> \
  --training-throughput <FROZEN_B0_TRAINING_THROUGHPUT_JSON> \
  --output-dir "$ROOT/validation/B0_seed1248" \
  --windows-per-re 16
```

Validation analysis is run against each `best_validation.pt`:

```bash
"$PY" "$ROOT/code/analyze_ctdm_checkpoint.py" \
  --split validation \
  --checkpoint <BEST_VALIDATION_CHECKPOINT> \
  --trainer "$ROOT/code/train_ctdm_comparison.py" \
  --baseline-trainer <FROZEN_BASE_TRAINER> \
  --h4-trainer <FROZEN_H4_TRAINER> \
  --coefficient-view <SANITIZED_TRAINVAL_VIEW> \
  --galerkin-path <TRAIN_ONLY_GALERKIN> \
  --pressure-path <TRAIN_ONLY_PRESSURE_POISSON> \
  --asset-manifest <TRAINING_ASSET_MANIFEST> \
  --fluctuation-contract <TRAIN_ONLY_FLUCTUATION_CONTRACT> \
  --output-dir <VALIDATION_ANALYSIS_DIR>
```

The same analysis program can open held-out tensors only with `--split test` and
a `VALIDATION_FREEZE_MANIFEST.json` that authorizes the exact checkpoint SHA-256.
Without that manifest, or if any registered gate is false, the program exits
before opening the held-out POD files.

The validation freeze command explicitly lists the seed-1248 screening candidates:

```bash
"$PY" "$ROOT/code/freeze_validation_decision.py" \
  --analysis <B1_TO_B4_VALIDATION_JSON> \
  --checkpoint <B0_AND_B1_TO_B4_FROZEN_CHECKPOINTS> \
  --candidate-variant <FINITE_ZERO_DIVERGENCE_SCREENING_CANDIDATE> \
  --output-dir "$ROOT/validation_freeze"
```

Only variants that were finite with zero divergent windows in the seed-1248
screening result receive seeds 2248 and 3248. The freeze program verifies that the
declared candidate list exactly matches the recorded seed-1248 analyses.
