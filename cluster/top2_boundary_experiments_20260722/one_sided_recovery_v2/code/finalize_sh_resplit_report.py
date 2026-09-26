"""Finalize the structured and human-readable Re-resplit comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


METHODS = (
    "T2-C_LearnedConvexCorrection_FieldBlend",
    "RiskPredictionRouter",
    "LookAheadShortRolloutRouter",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_dir / "final_report"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prereg = load(args.run_dir / "preregistration/PREREGISTRATION.json")
    development_cache = load(args.run_dir / "cache_development/CACHE_MANIFEST.json")
    test_cache = load(args.run_dir / "cache_final_test/CACHE_MANIFEST.json")
    comparison = load(args.run_dir / "training/VALIDATION_COMPARISON.json")
    decision = load(args.run_dir / "FROZEN_DECISION.json")
    validation = load(args.run_dir / "validation_evaluation/EVALUATION_REPORT.json")
    heldout = load(args.run_dir / "final_test_evaluation/EVALUATION_REPORT.json")
    checkpoints = {}
    for method in METHODS:
        method_dir = args.run_dir / "training" / method
        freeze = load(method_dir / "FREEZE_MANIFEST.json")
        config = load(method_dir / "CONFIG_EFFECTIVE.json")
        checkpoints[method] = {
            "seed": int(config["seed"]),
            "validation_selected_step": int(config["validation_selected_step"]),
            "threshold": config["validation_selected_threshold"],
            "best_checkpoint": str(method_dir / "best.pt"),
            "best_checkpoint_sha256": sha256(method_dir / "best.pt"),
            "last_checkpoint": str(method_dir / "last.pt"),
            "last_checkpoint_sha256": sha256(method_dir / "last.pt"),
            "config_sha256": sha256(method_dir / "CONFIG_EFFECTIVE.json"),
            "freeze_manifest_sha256": sha256(method_dir / "FREEZE_MANIFEST.json"),
            "elapsed_seconds": float(freeze["elapsed_seconds"]),
        }
    route_comparison = {}
    for method in METHODS:
        val = validation["routes"][method]
        test = heldout["routes"][method]
        route_comparison[method] = {
            "validation": {
                "joint_all_mean": val["joint_all_mean"],
                "joint_all_worst": val["joint_all_worst"],
                "K56": val["horizons"]["K56"],
                "per_Re": val["per_Re"],
                "top2_usage": val["top2_usage"],
                "weights": val["weights"],
                "risk": val["risk"],
                "gate_only_ablation_joint_all_mean": val["gate_only_ablation"][
                    "joint_all_mean"
                ],
                "oracle_gap": val["relative_oracle_gap"],
            },
            "heldout": {
                "joint_all_mean": test["joint_all_mean"],
                "joint_all_worst": test["joint_all_worst"],
                "K56": test["horizons"]["K56"],
                "per_Re": test["per_Re"],
                "top2_usage": test["top2_usage"],
                "weights": test["weights"],
                "risk": test["risk"],
                "gate_only_ablation_joint_all_mean": test["gate_only_ablation"][
                    "joint_all_mean"
                ],
                "oracle_gap": test["relative_oracle_gap"],
            },
        }
    report = {
        "status": "RESPLIT_20260723_COMPARISON_COMPLETE",
        "protocol": prereg["protocol_version"],
        "complete_Re_split": prereg["complete_Re_split"],
        "windows_per_Re": prereg["windows_per_Re"],
        "development_cache": {
            "sha256": development_cache["cache_sha256"],
            "split_counts": development_cache["split_counts"],
        },
        "heldout_cache": {
            "sha256": test_cache["cache_sha256"],
            "split_counts": test_cache["split_counts"],
        },
        "specialist_checkpoint_sha256": {
            "Steady": development_cache["steady_checkpoint_sha256"],
            "Hopf": development_cache["hopf_checkpoint_sha256"],
        },
        "route_checkpoints": checkpoints,
        "validation_decision": decision,
        "routes": route_comparison,
        "validation_baselines": validation["baselines"],
        "heldout_baselines": heldout["baselines"],
        "scientific_conclusion": {
            "promoted_method": decision["promoted_method"],
            "all_three_predictions_identical": len(
                {
                    float(heldout["routes"][method]["joint_all_mean"])
                    for method in METHODS
                }
            )
            == 1,
            "risk_module_added_output_value": False,
            "lookahead_module_added_output_value": False,
            "basis": "Risk and LookAhead selected Top-2 for every validation and heldout window; deleting each scorer and retaining the common convex gate leaves the reported error unchanged.",
        },
        "limitations": [
            "This rerun is not a blind test because these Re values and earlier results were previously visible.",
            "No specialist was retrained; new Re use the frozen Steady nearest-ROM-node compatibility contract.",
            "Only one-sided S-native S-H output-level fusion is evaluated; no feedback, S-P, Top-3, H-P, or continuous-RHS fusion is used.",
            "Each Re supplies five K56 windows; K128 is unavailable for the 64-snapshot expanded trajectories.",
        ],
        "test_used_for_training_threshold_checkpoint_or_promotion": False,
        "source_validation_comparison_sha256": sha256(
            args.run_dir / "training/VALIDATION_COMPARISON.json"
        ),
    }
    atomic_json(output / "METHOD_COMPARISON.json", report)
    lines = [
        "# S–H Top-2 Re-resplit comparison",
        "",
        "All three routes used the same frozen-specialist cache, seed 42001, and 8000 optimizer steps.",
        "",
        "| Method | Val mean | Val worst | Test mean | Test worst | Test K56 mean | Top-2 usage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        val = route_comparison[method]["validation"]
        test = route_comparison[method]["heldout"]
        lines.append(
            f"| {method} | {val['joint_all_mean']:.8f} | {val['joint_all_worst']:.8f} | "
            f"{test['joint_all_mean']:.8f} | {test['joint_all_worst']:.8f} | "
            f"{test['K56']['joint_mean']:.8f} | {test['top2_usage']:.3f} |"
        )
    lines += [
        "",
        "## Heldout baselines",
        "",
        "| Baseline | Mean joint error | Worst | K56 mean |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "S_only",
        "H_only",
        "E2_pair_Top1",
        "fixed_0.5",
        "E2_pair_probability_blend",
        "per_window_convex_oracle",
    ):
        metric = heldout["baselines"][name]
        lines.append(
            f"| {name} | {metric['joint_all_mean']:.8f} | {metric['joint_all_worst']:.8f} | "
            f"{metric['horizons']['K56']['joint_mean']:.8f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"Validation-only promotion: `{decision['promoted_method']}`.",
        "",
        "RiskPrediction and LookAhead produced exactly the same final predictions as the common convex gate. "
        "Their Top-2 usage was 1.0, and their gate-only ablations had unchanged error; the extra scorer modules therefore added no output-level value in this split.",
        "",
        "This rerun is not a blind test. Heldout values were excluded from optimization, checkpoint selection, threshold selection, and promotion.",
    ]
    (output / "METHOD_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    critical_files = {
        "preregistration": args.run_dir / "preregistration/PREREGISTRATION.json",
        "development_cache_manifest": args.run_dir / "cache_development/CACHE_MANIFEST.json",
        "validation_comparison": args.run_dir / "training/VALIDATION_COMPARISON.json",
        "validation_decision": args.run_dir / "FROZEN_DECISION.json",
        "validation_report": args.run_dir / "validation_evaluation/EVALUATION_REPORT.json",
        "heldout_cache_manifest": args.run_dir / "cache_final_test/CACHE_MANIFEST.json",
        "heldout_report": args.run_dir / "final_test_evaluation/EVALUATION_REPORT.json",
        "heldout_per_window": args.run_dir / "final_test_evaluation/PER_WINDOW_RESULTS.csv",
        "method_comparison_json": output / "METHOD_COMPARISON.json",
        "method_comparison_md": output / "METHOD_COMPARISON.md",
    }
    manifest = {
        "status": "FINAL_FROZEN",
        "critical_file_sha256": {
            name: sha256(path) for name, path in critical_files.items()
        },
        "route_checkpoint_sha256": {
            method: {
                "best": values["best_checkpoint_sha256"],
                "last": values["last_checkpoint_sha256"],
            }
            for method, values in checkpoints.items()
        },
        "specialist_checkpoint_sha256": report["specialist_checkpoint_sha256"],
    }
    atomic_json(output / "FINAL_FREEZE_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "status": report["status"],
                "promoted": decision["promoted_method"],
                "heldout_joint": {
                    method: heldout["routes"][method]["joint_all_mean"]
                    for method in METHODS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
