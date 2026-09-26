"""Evaluate frozen Re-resplit routes and emit per-window blend diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

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


def fixed_point(alpha: np.ndarray, data: dict[str, np.ndarray], root: Path) -> dict[str, float]:
    steady_root = root / "steady_specialist_v1/source_artifacts/steady"
    hopf_root = root / "Hopf/artifacts/hopf"
    values = []
    with np.load(steady_root / "velocity_pod_steady.npz") as su, np.load(
        hopf_root / "velocity_pod_hopf.npz"
    ) as hu:
        weights = np.r_[su["point_areas"], su["point_areas"]].astype(np.float64)
        phi_s = su["phi_uv"][:32].astype(np.float64)
        phi_h = hu["phi_uv"][:32].astype(np.float64)
        gs = (phi_s * weights) @ phi_s.T
        gh = (phi_h * weights) @ phi_h.T
        cross = (phi_s * weights) @ phi_h.T
        mean_s = su["mean_uv_regime"].astype(np.float64)
        psms = (phi_s * weights) @ mean_s
        mean_s_sq = float((mean_s * weights) @ mean_s)
        for index, weight_s in enumerate(alpha):
            ds = data["s_a"][index, -1].astype(float) - data["s_a"][index, -2].astype(float)
            dh = data["h_a"][index, -1].astype(float) - data["h_a"][index, -2].astype(float)
            numerator = (
                weight_s**2 * (ds @ gs @ ds)
                + (1 - weight_s) ** 2 * (dh @ gh @ dh)
                + 2 * weight_s * (1 - weight_s) * (ds @ cross @ dh)
            )
            truth = data["true_a"][index, -1].astype(float)
            denominator = mean_s_sq + 2 * truth @ psms + truth @ gs @ truth
            values.append(float(np.sqrt(max(numerator, 0) / max(denominator, 1e-12))))
    return {
        "mean": float(np.mean(values)),
        "worst": float(np.max(values)),
        "definition": "area-weighted final-step field drift relative to final true field norm",
    }


def weight_summary(alpha_s: np.ndarray, re_values: np.ndarray) -> dict[str, object]:
    return {
        "alpha_S": {
            "mean": float(np.mean(alpha_s)),
            "min": float(np.min(alpha_s)),
            "max": float(np.max(alpha_s)),
        },
        "alpha_H": {
            "mean": float(np.mean(1 - alpha_s)),
            "min": float(np.min(1 - alpha_s)),
            "max": float(np.max(1 - alpha_s)),
        },
        "per_Re": {
            str(float(re_value)): {
                "alpha_S_mean": float(np.mean(alpha_s[re_values == re_value])),
                "alpha_S_min": float(np.min(alpha_s[re_values == re_value])),
                "alpha_S_max": float(np.max(alpha_s[re_values == re_value])),
                "alpha_H_mean": float(np.mean(1 - alpha_s[re_values == re_value])),
            }
            for re_value in np.unique(re_values)
        },
    }


def joint_per_window(alpha: np.ndarray, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a = alpha[:, None]
    qu, qp = data["quad_u"], data["quad_p"]
    eu = np.maximum(
        (a * a * qu[:, :, 0] + (1 - a) ** 2 * qu[:, :, 1] + 2 * a * (1 - a) * qu[:, :, 2])
        / np.maximum(qu[:, :, 3], 1e-12),
        0,
    )
    ep = np.maximum(
        (a * a * qp[:, :, 0] + (1 - a) ** 2 * qp[:, :, 1] + 2 * a * (1 - a) * qp[:, :, 2])
        / np.maximum(qp[:, :, 3], 1e-12),
        0,
    )
    joint = np.sqrt(eu) + np.sqrt(ep)
    return np.mean(joint, axis=1), joint[:, -1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "heldout"), required=True)
    parser.add_argument("--decision", type=Path)
    args = parser.parse_args()
    marker = args.training_dir / "ALL_ROUTES_VALIDATION_FROZEN.json"
    if not marker.is_file():
        raise RuntimeError("routes are not validation-frozen")
    decision = None
    if args.split == "heldout":
        if args.decision is None or not args.decision.is_file():
            raise RuntimeError("heldout evaluation requires a frozen validation-only decision")
        decision = json.loads(args.decision.read_text(encoding="utf-8"))
        if decision["test_used_for_selection"] is not False:
            raise RuntimeError("decision contract failure")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    routes = load_module(
        "resplit_evaluation_routes",
        Path(__file__).with_name("train_sh_routes.py"),
    )
    routes.ROOT = args.root
    with np.load(args.cache, allow_pickle=False) as archive:
        full = {key: np.asarray(archive[key]) for key in archive.files}
    mask = full["split"] == args.split
    if not np.any(mask):
        raise RuntimeError(f"cache has no {args.split} rows")
    data = {key: value[mask] for key, value in full.items()}
    if set(data["split"].tolist()) != {args.split}:
        raise RuntimeError("split isolation failure")
    e2_alpha_s = routes.e2_pair_probability(data["re"])
    results = {}
    alphas = {}
    active_flags = {}
    inference = {}
    for method in routes.METHODS:
        checkpoint_path = args.training_dir / method / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        x = ((data["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]).astype(
            np.float32
        )
        tx = torch.as_tensor(x)
        base = np.log(
            np.maximum(e2_alpha_s, 1e-6) / np.maximum(1 - e2_alpha_s, 1e-6)
        ).astype(np.float32)
        gate = routes.ConvexGate(x.shape[1])
        gate.load_state_dict(checkpoint["gate_state"])
        gate.eval()
        scorer = None
        scorer_x = tx
        if method == "RiskPredictionRouter":
            scorer = routes.MLP(x.shape[1], 2)
        elif method == "LookAheadShortRolloutRouter":
            lookahead = routes.lookahead_features(data)
            xla = np.concatenate(
                (
                    x,
                    (lookahead - checkpoint["lookahead_mean"]) / checkpoint["lookahead_std"],
                ),
                axis=1,
            ).astype(np.float32)
            scorer_x = torch.as_tensor(xla)
            scorer = routes.MLP(xla.shape[1], 2)
        if scorer is not None:
            scorer.load_state_dict(checkpoint["scorer_state"])
            scorer.eval()
        started = time.perf_counter()
        with torch.inference_mode():
            gate_alpha = gate(tx, torch.as_tensor(base)).numpy()
            predicted_risk = scorer(scorer_x).numpy() if scorer is not None else None
        elapsed = time.perf_counter() - started
        alpha = gate_alpha.copy()
        active = np.ones(len(alpha), dtype=bool)
        risk_metrics = {}
        if predicted_risk is not None:
            margin = np.abs(predicted_risk[:, 0] - predicted_risk[:, 1])
            active = margin <= float(checkpoint["threshold"])
            top = np.argmin(predicted_risk, axis=1)
            alpha[~active] = (top[~active] == 0).astype(np.float32)
            true_risk = routes.risk_targets(data["quad_u"], data["quad_p"])
            risk_metrics = {
                "ranking_accuracy": float(
                    np.mean(np.argmin(predicted_risk, axis=1) == np.argmin(true_risk, axis=1))
                ),
                "risk_mse": float(np.mean((predicted_risk - true_risk) ** 2)),
            }
        metric = routes.metrics(
            alpha, data["quad_u"], data["quad_p"], data["re"], active
        )
        metric["steady_fixed_point"] = fixed_point(alpha, data, args.root)
        metric["risk"] = risk_metrics
        metric["weights"] = weight_summary(alpha, data["re"])
        metric["gate_only_ablation"] = routes.metrics(
            gate_alpha,
            data["quad_u"],
            data["quad_p"],
            data["re"],
            np.ones(len(gate_alpha), dtype=bool),
        )
        results[method] = metric
        alphas[method] = alpha
        active_flags[method] = active
        inference[method] = {
            "routing_seconds_per_trajectory": elapsed / len(alpha),
            "relative_native_rollout_calls": 2,
        }
    baselines = {}
    baseline_alpha = {
        "S_only": np.ones(len(e2_alpha_s), dtype=np.float32),
        "H_only": np.zeros(len(e2_alpha_s), dtype=np.float32),
        "E2_pair_Top1": (e2_alpha_s >= 0.5).astype(np.float32),
        "fixed_0.5": np.full(len(e2_alpha_s), 0.5, dtype=np.float32),
        "E2_pair_probability_blend": e2_alpha_s.astype(np.float32),
    }
    for name, alpha in baseline_alpha.items():
        active = (
            np.ones(len(alpha), dtype=bool)
            if name in {"fixed_0.5", "E2_pair_probability_blend"}
            else np.zeros(len(alpha), dtype=bool)
        )
        baselines[name] = routes.metrics(
            alpha, data["quad_u"], data["quad_p"], data["re"], active
        )
        baselines[name]["steady_fixed_point"] = fixed_point(alpha, data, args.root)
    grid = np.linspace(0, 1, 10001, dtype=np.float64)[:, None]
    oracle = []
    for index in range(len(data["re"])):
        qu = data["quad_u"][index]
        qp = data["quad_p"][index]
        eu = (
            grid * grid * qu[:, 0]
            + (1 - grid) ** 2 * qu[:, 1]
            + 2 * grid * (1 - grid) * qu[:, 2]
        ) / np.maximum(qu[:, 3], 1e-12)
        ep = (
            grid * grid * qp[:, 0]
            + (1 - grid) ** 2 * qp[:, 1]
            + 2 * grid * (1 - grid) * qp[:, 2]
        ) / np.maximum(qp[:, 3], 1e-12)
        objective = np.mean(
            np.sqrt(np.maximum(eu, 0)) + np.sqrt(np.maximum(ep, 0)), axis=1
        )
        oracle.append(float(grid[np.argmin(objective), 0]))
    oracle = np.asarray(oracle, dtype=np.float32)
    baselines["per_window_convex_oracle"] = routes.metrics(
        oracle, data["quad_u"], data["quad_p"], data["re"]
    )
    oracle_score = float(baselines["per_window_convex_oracle"]["joint_all_mean"])
    for metric in results.values():
        metric["relative_oracle_gap"] = float(metric["joint_all_mean"] - oracle_score)
    window_rows = []
    window_metrics = {
        method: joint_per_window(alpha, data) for method, alpha in alphas.items()
    }
    for index in range(len(data["re"])):
        row = {
            "split": args.split,
            "Re": float(data["re"][index]),
            "start": int(data["start"][index]),
            "start_time": float(data["times"][index, 0]),
            "end_time": float(data["times"][index, -1]),
        }
        for method in routes.METHODS:
            row[f"{method}_alpha_S"] = float(alphas[method][index])
            row[f"{method}_alpha_H"] = float(1 - alphas[method][index])
            row[f"{method}_top2_active"] = bool(active_flags[method][index])
            row[f"{method}_joint_trajectory_mean"] = float(window_metrics[method][0][index])
            row[f"{method}_joint_K56"] = float(window_metrics[method][1][index])
        window_rows.append(row)
    with (args.output_dir / "PER_WINDOW_RESULTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)
    report = {
        "status": f"{args.split.upper()}_EVALUATION_COMPLETE",
        "protocol": "one-sided S-native S-H output-only Top-2 Re-resplit 20260723",
        "split": args.split,
        "complete_Re": sorted(float(value) for value in np.unique(data["re"])),
        "windows": len(data["re"]),
        "cache_sha256": sha256(args.cache),
        "training_freeze_marker_sha256": sha256(marker),
        "decision_sha256": sha256(args.decision) if args.decision else None,
        "routes": results,
        "baselines": baselines,
        "inference": inference,
        "promoted_method_validation_frozen": decision["promoted_method"] if decision else None,
        "test_used_for_selection": False,
        "known_prior_test_disclosure": True,
    }
    atomic_json(args.output_dir / "EVALUATION_REPORT.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "routes_joint_all_mean": {
                    method: value["joint_all_mean"] for method, value in results.items()
                },
                "baselines_joint_all_mean": {
                    method: value["joint_all_mean"] for method, value in baselines.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
