#!/usr/bin/env python3
"""Compact revision-9 Global evaluator outputs without recomputing metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compact(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_loaded = next(
        (
            event["checkpoint"]
            for event in payload["results"][0].get("history_tail", [])
            if event.get("event") == "eval_only_checkpoint_loaded"
        ),
        None,
    )
    if checkpoint_loaded is None:
        raise RuntimeError(f"{path}: missing eval_only_checkpoint_loaded provenance")
    rows = []
    for item in payload["results"]:
        r = item["rollout_autonomous_pressure"]
        rows.append(
            {
                "Re": item["test_Re"],
                "Re_label": item["test_Re_label"],
                "regime": item["test_regime"],
                "horizon": payload["settings"]["rollout_steps"],
                "velocity_physical_relative_l2": r[
                    "velocity_area_weighted_physical_relative_l2_mean"
                ],
                "pressure_physical_relative_l2": r[
                    "pressure_area_weighted_physical_relative_l2_mean"
                ],
                "velocity_coefficient_relative_l2": r["a_relative_l2_mean"],
                "pressure_coefficient_relative_l2": r["b_relative_l2_mean"],
                "finite_fraction": r["finite_fraction"],
                "divergent_windows": r["divergent_windows"],
                "attempted_windows": r["attempted_windows"],
                "complete_windows": r["num_windows"],
                "incomplete_windows": r["incomplete_windows"],
            }
        )
    return {
        "source": str(path),
        "source_sha256": sha256(path),
        "checkpoint": checkpoint_loaded,
        "evaluation_output_checkpoint": payload["results"][0]["checkpoint_path"],
        "checkpoint_best_epoch": payload["results"][0]["best_epoch"],
        "checkpoint_best_val_score": payload["results"][0]["best_val_score"],
        "settings": {
            "r_u": payload["settings"]["r_u"],
            "r_p": payload["settings"]["r_p"],
            "rollout_steps": payload["settings"]["rollout_steps"],
            "history_len": payload["settings"]["history_len"],
            "integrator": payload["settings"]["integrator"],
            "eval_re_values": payload["settings"]["eval_re_values"],
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periodic-k48", type=Path, required=True)
    parser.add_argument("--sh-k56", type=Path, required=True)
    parser.add_argument("--prepare-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare = json.loads(args.prepare_manifest.read_text(encoding="utf-8"))
    output = {
        "schema": "global_revision9_compact/v1",
        "metric_policy": "direct copy from frozen evaluator; no aggregation or recomputation",
        "periodic_K48": compact(args.periodic_k48),
        "S_H_same_window_K56": compact(args.sh_k56),
        "external_asset_preparation": {
            "source": str(args.prepare_manifest),
            "source_sha256": sha256(args.prepare_manifest),
            "checkpoint_modified": prepare["checkpoint_modified"],
            "global_pod_basis_modified": prepare["global_pod_basis_modified"],
            "global_scaler_modified": prepare["global_scaler_modified"],
            "representation": prepare["representation"],
            "external_nodes": prepare["external_nodes"],
        },
    }
    checkpoint = Path(output["periodic_K48"]["checkpoint"])
    output["checkpoint_sha256"] = sha256(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
