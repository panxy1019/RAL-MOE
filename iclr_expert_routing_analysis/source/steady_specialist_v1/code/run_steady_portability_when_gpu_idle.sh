#!/usr/bin/env bash
set -euo pipefail

root=/root/panxy/particalMOE/steady_specialist_v1
python=/root/miniconda3/envs/pt_env/bin/python
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$busy" ]]; then
  echo "REFUSED_GPU_BUSY pids=$busy" >&2
  exit 75
fi

"$python" "$root/code/reproduce_s4_one_epoch.py" --root "$root"

output="$root/portability_runs/best_checkpoint_evaluation"
mkdir -p "$output"
"$python" "$root/code/finalize_s4_one_time.py" \
  --trainer "$root/code/train_s2b_3090.py" \
  --finalizer "$root/code/finalize_s2b_3090.py" \
  --s3-trainer "$root/code/train_s3.py" \
  --audit-script "$root/code/audit_paired_gain_metrics.py" \
  --heldout-evaluator "$root/code/evaluate_s3_heldout.py" \
  --config "$root/code/training_s2b_portable.json" \
  --source-checkpoint "$root/checkpoint/frozen_s2b_validation.pt" \
  --training-bank "$root/data/perturbation_bank_train_validation.npz" \
  --heldout-bank /root/panxy/particalMOE/unified_rollout_eval_heldout_v1/steady/heldout_perturbation_bank.npz \
  --selected-checkpoint "$root/checkpoint/frozen_s4_validation_step_1200.pt" \
  --selection-manifest "$root/provenance/checkpoint_selection_manifest.json" \
  --s3-heldout-metrics "$root/evaluation_inputs/s3_heldout_metrics.json" \
  --s3-audit-metrics "$root/evaluation_inputs/s3_paired_gain_metric_audit_raw.json" \
  --output-dir "$output" \
  --learning-rate 0.000015516372391099407
