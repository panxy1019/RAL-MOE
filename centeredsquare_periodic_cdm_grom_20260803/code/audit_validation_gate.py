#!/usr/bin/env python3
"""Apply the periodic-orbit validation gate to CDM candidates."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    baseline_metrics = json.loads(args.baseline_metrics.read_text())
    baseline_quality = json.loads(args.baseline_quality.read_text())
    baseline = {
        "velocity_rollout_relative_l2": baseline_metrics["aggregate"]["velocity_rollout_relative_l2"],
        "amplitude_relative_error": baseline_quality["aggregate"]["mean_amplitude_relative_error"],
        "dominant_frequency_relative_error": baseline_quality["aggregate"]["mean_dominant_frequency_relative_error"],
    }
    reports = []
    for candidate in selection["candidates"]:
        aggregate = candidate["aggregate"]
        gates = {
            "stable": candidate["stable"],
            "rollout_improves": aggregate["velocity_rollout_relative_l2"] < baseline["velocity_rollout_relative_l2"],
            "amplitude_not_degraded_10pct": aggregate["amplitude_relative_error"] <= 1.1 * baseline["amplitude_relative_error"],
            "frequency_not_degraded_10pct": aggregate["dominant_frequency_relative_error"] <= 1.1 * baseline["dominant_frequency_relative_error"],
        }
        reports.append({
            "rates": candidate["rates"], "ridge": candidate["ridge"],
            "memory_state_dimension": candidate["memory_state_dimension"],
            "aggregate": aggregate, "gates": gates, "passes": all(gates.values()),
        })
    payload = {
        "schema_version": 1,
        "status": "PASS" if any(item["passes"] and item["memory_state_dimension"] > 0 for item in reports) else "REJECT_CDM",
        "baseline": baseline,
        "gate": "stable; lower rollout L2; amplitude and frequency degradation <=10%",
        "passing_candidates": [item for item in reports if item["passes"]],
        "best_true_memory_by_rollout": min(
            (item for item in reports if item["memory_state_dimension"] > 0),
            key=lambda item: item["aggregate"]["velocity_rollout_relative_l2"],
        ),
        "heldout_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
