#!/usr/bin/env python3
"""Create a compact validation/held-out summary figure from frozen metrics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evaluation-root", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
ART = args.evaluation_root or ROOT / "artifacts" / "periodic" / "evaluation"
OUT = args.output or ROOT / "reports" / "periodic_fvm_grom_summary.png"

validation_dir = ART / "validation_raw_fvm_linear_pressure" if (ART / "validation_raw_fvm_linear_pressure").is_dir() else ART / "validation"
validation = json.loads((validation_dir / "VALIDATION_METRICS.json").read_text())
heldout = json.loads((ART / "heldout" / "HELDOUT_METRICS.json").read_text())
validation_quality = json.loads((validation_dir / "PERIODIC_QUALITY.json").read_text())
heldout_quality = json.loads((ART / "heldout" / "HELDOUT_PERIODIC_QUALITY.json").read_text())

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
colors = {"validation": "#2878b5", "held-out": "#d95f02"}

for label, payload in [("validation", validation), ("held-out", heldout)]:
    re_values = [case["Re"] for case in payload["cases"]]
    axes[0].plot(re_values, [case["one_step_velocity_relative_l2"] for case in payload["cases"]], "o-", label=label, color=colors[label])
    axes[1].plot(re_values, [case["velocity_rollout_relative_l2"] for case in payload["cases"]], "o-", label=label, color=colors[label])

for label, payload in [("validation", validation_quality), ("held-out", heldout_quality)]:
    re_values = np.asarray([case["Re"] for case in payload["cases"]])
    amplitude = [case["amplitude_relative_error"] for case in payload["cases"]]
    frequency = [case["dominant_frequency_relative_error"] for case in payload["cases"]]
    axes[2].plot(re_values, amplitude, "o-", color=colors[label], label=f"{label}: amplitude")
    axes[2].plot(re_values, frequency, "s--", color=colors[label], alpha=0.75, label=f"{label}: frequency")

axes[0].set(title="One-step velocity error", xlabel="Re", ylabel="relative L2")
axes[1].set(title="Full rollout pointwise error", xlabel="Re", ylabel="relative L2")
axes[2].set(title="Periodic-orbit diagnostics", xlabel="Re", ylabel="relative error")
for axis in axes:
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180)
print(OUT)
