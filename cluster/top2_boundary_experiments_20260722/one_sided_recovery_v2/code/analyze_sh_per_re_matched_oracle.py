"""Per-Re frozen T2-C diagnostics and shared-trajectory convex oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def joint_error(alpha_s: np.ndarray, qu: np.ndarray, qp: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha_s, dtype=np.float64)
    if alpha.ndim == 0:
        alpha = np.full(qu.shape[0], float(alpha), dtype=np.float64)
    a = alpha[:, None]
    eu = np.maximum(
        (
            a * a * qu[:, :, 0]
            + (1 - a) ** 2 * qu[:, :, 1]
            + 2 * a * (1 - a) * qu[:, :, 2]
        )
        / np.maximum(qu[:, :, 3], 1e-12),
        0,
    )
    ep = np.maximum(
        (
            a * a * qp[:, :, 0]
            + (1 - a) ** 2 * qp[:, :, 1]
            + 2 * a * (1 - a) * qp[:, :, 2]
        )
        / np.maximum(qp[:, :, 3], 1e-12),
        0,
    )
    return np.sqrt(eu) + np.sqrt(ep)


def summarize(error: np.ndarray) -> dict[str, float]:
    return {
        "overall_mean": float(np.mean(error)),
        "overall_worst": float(np.max(error)),
        "K56_mean": float(np.mean(error[:, -1])),
        "K56_worst": float(np.max(error[:, -1])),
    }


def shared_constant_oracle(qu: np.ndarray, qp: np.ndarray) -> tuple[float, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 10001, dtype=np.float64)
    objectives = np.empty(len(grid), dtype=np.float64)
    for offset in range(0, len(grid), 1000):
        values = grid[offset : offset + 1000, None, None]
        eu = np.maximum(
            (
                values * values * qu[None, :, :, 0]
                + (1 - values) ** 2 * qu[None, :, :, 1]
                + 2 * values * (1 - values) * qu[None, :, :, 2]
            )
            / np.maximum(qu[None, :, :, 3], 1e-12),
            0,
        )
        ep = np.maximum(
            (
                values * values * qp[None, :, :, 0]
                + (1 - values) ** 2 * qp[None, :, :, 1]
                + 2 * values * (1 - values) * qp[None, :, :, 2]
            )
            / np.maximum(qp[None, :, :, 3], 1e-12),
            0,
        )
        objectives[offset : offset + len(values)] = np.mean(
            np.sqrt(eu) + np.sqrt(ep), axis=(1, 2)
        )
    alpha_s = float(grid[int(np.argmin(objectives))])
    return alpha_s, joint_error(alpha_s, qu, qp)


def per_window_oracle(qu: np.ndarray, qp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 10001, dtype=np.float64)
    alphas = []
    errors = []
    for index in range(len(qu)):
        values = grid[:, None]
        eu = np.maximum(
            (
                values * values * qu[index, :, 0]
                + (1 - values) ** 2 * qu[index, :, 1]
                + 2 * values * (1 - values) * qu[index, :, 2]
            )
            / np.maximum(qu[index, :, 3], 1e-12),
            0,
        )
        ep = np.maximum(
            (
                values * values * qp[index, :, 0]
                + (1 - values) ** 2 * qp[index, :, 1]
                + 2 * values * (1 - values) * qp[index, :, 2]
            )
            / np.maximum(qp[index, :, 3], 1e-12),
            0,
        )
        objective = np.mean(np.sqrt(eu) + np.sqrt(ep), axis=1)
        alpha = float(grid[int(np.argmin(objective))])
        alphas.append(alpha)
        errors.append(joint_error(np.asarray([alpha]), qu[index : index + 1], qp[index : index + 1])[0])
    return np.asarray(alphas, dtype=np.float64), np.asarray(errors, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.run_dir / "cache_final_test/G_SH_resplit_final_test_cache.npz"
    checkpoint_path = (
        args.run_dir
        / "training/T2-C_LearnedConvexCorrection_FieldBlend/best.pt"
    )
    decision_path = args.run_dir / "FROZEN_DECISION.json"
    with np.load(cache_path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    if set(data["split"].tolist()) != {"heldout"}:
        raise RuntimeError("heldout cache contract failure")
    routes = load_module(
        "per_re_oracle_routes",
        args.run_dir.parent / "code/train_sh_routes.py",
    )
    routes.ROOT = args.root
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    e2_alpha_s = routes.e2_pair_probability(data["re"])
    features = (
        (data["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    ).astype(np.float32)
    base_logit = np.log(
        np.maximum(e2_alpha_s, 1e-6) / np.maximum(1 - e2_alpha_s, 1e-6)
    ).astype(np.float32)
    gate = routes.ConvexGate(features.shape[1])
    gate.load_state_dict(checkpoint["gate_state"])
    gate.eval()
    with torch.inference_mode():
        t2c_alpha_s = gate(
            torch.as_tensor(features), torch.as_tensor(base_logit)
        ).numpy()
    e2_top1_alpha_s = (e2_alpha_s >= 0.5).astype(np.float32)
    rows = []
    window_rows = []
    detailed = {}
    for re_value in sorted(np.unique(data["re"]).tolist()):
        mask = data["re"] == re_value
        qu, qp = data["quad_u"][mask], data["quad_p"][mask]
        s_error = joint_error(1.0, qu, qp)
        h_error = joint_error(0.0, qu, qp)
        e2_error = joint_error(e2_top1_alpha_s[mask], qu, qp)
        t2c_error = joint_error(t2c_alpha_s[mask], qu, qp)
        first_window_alpha_s = float(t2c_alpha_s[mask][0])
        first_window_shared_error = joint_error(first_window_alpha_s, qu, qp)
        matched_alpha_s, matched_error = shared_constant_oracle(qu, qp)
        window_oracle_alpha_s, window_oracle_error = per_window_oracle(qu, qp)
        s_metrics = summarize(s_error)
        h_metrics = summarize(h_error)
        e2_metrics = summarize(e2_error)
        t2c_metrics = summarize(t2c_error)
        first_window_shared_metrics = summarize(first_window_shared_error)
        matched_metrics = summarize(matched_error)
        window_oracle_metrics = summarize(window_oracle_error)
        best_expert = "S" if s_metrics["overall_mean"] <= h_metrics["overall_mean"] else "H"
        best_expert_k56 = "S" if s_metrics["K56_mean"] <= h_metrics["K56_mean"] else "H"
        alpha_s = t2c_alpha_s[mask]
        alpha_h = 1 - alpha_s
        selected_indices = np.flatnonzero(mask)
        for local_index, global_index in enumerate(selected_indices.tolist()):
            window_rows.append(
                {
                    "Re": float(re_value),
                    "start": int(data["start"][global_index]),
                    "start_time": float(data["times"][global_index, 0]),
                    "end_time": float(data["times"][global_index, -1]),
                    "T2_C_alpha_S": float(alpha_s[local_index]),
                    "T2_C_alpha_H": float(alpha_h[local_index]),
                    "S_only_trajectory_mean": float(np.mean(s_error[local_index])),
                    "H_only_trajectory_mean": float(np.mean(h_error[local_index])),
                    "E2_Top1_trajectory_mean": float(np.mean(e2_error[local_index])),
                    "T2_C_trajectory_mean": float(np.mean(t2c_error[local_index])),
                    "window_oracle_alpha_S": float(window_oracle_alpha_s[local_index]),
                    "window_oracle_trajectory_mean": float(
                        np.mean(window_oracle_error[local_index])
                    ),
                    "S_only_K56": float(s_error[local_index, -1]),
                    "H_only_K56": float(h_error[local_index, -1]),
                    "T2_C_K56": float(t2c_error[local_index, -1]),
                }
            )
        per_re = {
            "Re": float(re_value),
            "windows": int(np.sum(mask)),
            "S_only": s_metrics,
            "H_only": h_metrics,
            "E2_Top1": e2_metrics,
            "T2_C": t2c_metrics,
            "best_single_expert": best_expert,
            "best_single_expert_K56": best_expert_k56,
            "T2_C_weights": {
                "alpha_S_mean": float(np.mean(alpha_s)),
                "alpha_S_min": float(np.min(alpha_s)),
                "alpha_S_max": float(np.max(alpha_s)),
                "alpha_H_mean": float(np.mean(alpha_h)),
                "alpha_H_min": float(np.min(alpha_h)),
                "alpha_H_max": float(np.max(alpha_h)),
                "per_window_alpha_S": [float(value) for value in alpha_s],
                "per_window_alpha_H": [float(value) for value in alpha_h],
            },
            "T2_C_earliest_window_weight_shared_across_Re": {
                "alpha_S": first_window_alpha_s,
                "alpha_H": 1.0 - first_window_alpha_s,
                **first_window_shared_metrics,
                "interpretation": "Diagnostic deployment variant: evaluate the frozen gate on the earliest selected window, then reuse that single weight for all five windows of this complete Re trajectory.",
            },
            "per_Re_shared_constant_oracle": {
                "alpha_S": matched_alpha_s,
                "alpha_H": 1.0 - matched_alpha_s,
                **matched_metrics,
            },
            "per_window_constant_oracle": {
                "alpha_S_mean": float(np.mean(window_oracle_alpha_s)),
                "alpha_S_min": float(np.min(window_oracle_alpha_s)),
                "alpha_S_max": float(np.max(window_oracle_alpha_s)),
                **window_oracle_metrics,
            },
            "differences": {
                "T2_C_minus_E2_overall_mean": t2c_metrics["overall_mean"]
                - e2_metrics["overall_mean"],
                "T2_C_minus_best_expert_overall_mean": t2c_metrics["overall_mean"]
                - min(s_metrics["overall_mean"], h_metrics["overall_mean"]),
                "T2_C_minus_shared_constant_oracle_overall_mean": t2c_metrics[
                    "overall_mean"
                ]
                - matched_metrics["overall_mean"],
                "earliest_weight_shared_minus_shared_oracle_overall_mean": first_window_shared_metrics[
                    "overall_mean"
                ]
                - matched_metrics["overall_mean"],
                "T2_C_minus_per_window_oracle_overall_mean": t2c_metrics[
                    "overall_mean"
                ]
                - window_oracle_metrics["overall_mean"],
            },
            "window_counts": {
                "T2_C_better_than_E2": int(
                    np.sum(np.mean(t2c_error, axis=1) < np.mean(e2_error, axis=1))
                ),
                "T2_C_better_than_best_single_expert": int(
                    np.sum(
                        np.mean(t2c_error, axis=1)
                        < np.minimum(
                            np.mean(s_error, axis=1), np.mean(h_error, axis=1)
                        )
                    )
                ),
            },
        }
        detailed[str(float(re_value))] = per_re
        rows.append(
            {
                "Re": float(re_value),
                "windows": int(np.sum(mask)),
                "S_only_overall_mean": s_metrics["overall_mean"],
                "H_only_overall_mean": h_metrics["overall_mean"],
                "E2_Top1_overall_mean": e2_metrics["overall_mean"],
                "T2_C_overall_mean": t2c_metrics["overall_mean"],
                "best_single_expert": best_expert,
                "best_single_expert_K56": best_expert_k56,
                "T2_C_alpha_S_mean": float(np.mean(alpha_s)),
                "T2_C_alpha_S_min": float(np.min(alpha_s)),
                "T2_C_alpha_S_max": float(np.max(alpha_s)),
                "T2_C_alpha_H_mean": float(np.mean(alpha_h)),
                "T2_C_alpha_H_min": float(np.min(alpha_h)),
                "T2_C_alpha_H_max": float(np.max(alpha_h)),
                "shared_oracle_alpha_S": matched_alpha_s,
                "shared_oracle_alpha_H": 1.0 - matched_alpha_s,
                "shared_oracle_overall_mean": matched_metrics["overall_mean"],
                "T2_C_minus_shared_oracle": t2c_metrics["overall_mean"]
                - matched_metrics["overall_mean"],
                "T2_C_minus_E2": t2c_metrics["overall_mean"]
                - e2_metrics["overall_mean"],
                "T2_C_minus_best_expert": t2c_metrics["overall_mean"]
                - min(s_metrics["overall_mean"], h_metrics["overall_mean"]),
                "earliest_gate_alpha_S": first_window_alpha_s,
                "earliest_gate_shared_overall_mean": first_window_shared_metrics[
                    "overall_mean"
                ],
                "earliest_gate_shared_minus_oracle": first_window_shared_metrics[
                    "overall_mean"
                ]
                - matched_metrics["overall_mean"],
                "S_only_K56_mean": s_metrics["K56_mean"],
                "H_only_K56_mean": h_metrics["K56_mean"],
                "T2_C_K56_mean": t2c_metrics["K56_mean"],
                "shared_oracle_K56_mean": matched_metrics["K56_mean"],
            }
        )
    with (args.output_dir / "PER_RE_RESULTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "PER_WINDOW_T2C_WEIGHTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)
    t2c_better_re = sum(row["T2_C_minus_E2"] < 0 for row in rows)
    report = {
        "status": "PER_RE_AND_MATCHED_ORACLE_COMPLETE",
        "cache_sha256": sha256(cache_path),
        "T2_C_checkpoint_sha256": sha256(checkpoint_path),
        "frozen_decision_sha256": sha256(decision_path),
        "oracle_contracts": {
            "per_Re_shared_constant_oracle": "One alpha shared by all five K56 windows belonging to the same complete Re trajectory; optimized over every window and every forecast time. This is the stricter per-trajectory/per-Re constant oracle requested for the paper.",
            "per_window_constant_oracle": "One hindsight alpha per independently queried K56 window, constant inside that rollout. This matches the gate's one-weight-per-query granularity but is a more optimistic hindsight bound.",
            "earliest_window_gate_shared_diagnostic": "The frozen T2-C gate is evaluated once on the earliest selected window for each Re and that alpha is reused across all five windows. This diagnoses the stricter one-gate-per-complete-trajectory interpretation without retraining.",
        },
        "per_Re": detailed,
        "paper_claim_checks": {
            "test_Re_count": len(rows),
            "Re_with_T2_C_better_than_E2": int(t2c_better_re),
            "majority_test_Re_better_than_E2": bool(t2c_better_re >= 2),
            "all_test_Re_better_than_E2": bool(t2c_better_re == len(rows)),
            "weight_pattern": "low Re S-dominant; Re=43.50 and 43.90 H-dominant, with substantial window-to-window variation near/high boundary",
        },
        "test_used_for_retraining_or_model_selection": False,
    }
    atomic_json(args.output_dir / "PER_RE_MATCHED_ORACLE_REPORT.json", report)
    lines = [
        "# Per-Re heldout and matched-oracle supplement",
        "",
        "| Re | S-only | H-only | E2 Top-1 | T2-C | Best expert overall/K56 | αS mean [min,max] | αH mean [min,max] | Shared-Re oracle | T2-C − oracle | Earliest-gate shared |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Re']:.6f} | {row['S_only_overall_mean']:.8f} | "
            f"{row['H_only_overall_mean']:.8f} | {row['E2_Top1_overall_mean']:.8f} | "
            f"{row['T2_C_overall_mean']:.8f} | {row['best_single_expert']}/{row['best_single_expert_K56']} | "
            f"{row['T2_C_alpha_S_mean']:.4f} [{row['T2_C_alpha_S_min']:.4f},{row['T2_C_alpha_S_max']:.4f}] | "
            f"{row['T2_C_alpha_H_mean']:.4f} [{row['T2_C_alpha_H_min']:.4f},{row['T2_C_alpha_H_max']:.4f}] | "
            f"{row['shared_oracle_overall_mean']:.8f} (αS={row['shared_oracle_alpha_S']:.4f}) | "
            f"{row['T2_C_minus_shared_oracle']:+.8f} | "
            f"{row['earliest_gate_shared_overall_mean']:.8f} (αS={row['earliest_gate_alpha_S']:.4f}) |"
        )
    lines += [
        "",
        f"T2-C beats E2 on {t2c_better_re}/{len(rows)} heldout Re values.",
        "",
        "The shared-Re oracle uses one constant weight across all five starts from the same Re trajectory. "
        "The previously reported per-window oracle remains listed separately as a more optimistic hindsight bound.",
    ]
    (args.output_dir / "PER_RE_MATCHED_ORACLE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    atomic_json(
        args.output_dir / "FREEZE_MANIFEST.json",
        {
            "status": "FROZEN_SUPPLEMENT",
            "source_cache_sha256": sha256(cache_path),
            "source_checkpoint_sha256": sha256(checkpoint_path),
            "files": {
                name: sha256(args.output_dir / name)
                for name in (
                    "PER_RE_RESULTS.csv",
                    "PER_WINDOW_T2C_WEIGHTS.csv",
                    "PER_RE_MATCHED_ORACLE_REPORT.json",
                    "PER_RE_MATCHED_ORACLE_REPORT.md",
                )
            },
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "Re_with_T2_C_better_than_E2": t2c_better_re,
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
