"""Frozen-cache H-only, weight, H-distance, and router-module ablations."""

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


ROOT = Path("/root/panxy/particalMOE")
HORIZONS = (1, 4, 8, 16, 56)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def select_split(data: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    mask = data["split"] == split
    if not np.any(mask):
        raise RuntimeError(f"cache has no {split} rows")
    return {key: value[mask] if value.ndim and len(value) == len(mask) else value for key, value in data.items()}


def h_norm(coeff: np.ndarray, pod: object, vector: bool) -> np.ndarray:
    areas = np.asarray(pod["point_areas"], dtype=np.float64)
    weights = np.concatenate((areas, areas)) if vector else areas
    phi_key, mean_key = ("phi_uv", "mean_uv_regime") if vector else ("phi_p", "mean_p_regime")
    phi = np.asarray(pod[phi_key][:32], dtype=np.float64)
    mean = np.asarray(pod[mean_key], dtype=np.float64)
    gram = (phi * weights) @ phi.T
    cross = (phi * weights) @ mean
    mean_norm = float(np.dot(mean * weights, mean))
    c = coeff.astype(np.float64)
    return mean_norm + 2 * np.einsum("nti,i->nt", c, cross) + np.einsum("nti,ij,ntj->nt", c, gram, c)


def distance_to_h(alpha_s: np.ndarray, data: dict[str, np.ndarray], hnorm_u: np.ndarray, hnorm_p: np.ndarray) -> dict[str, object]:
    qdu = np.maximum(data["quad_u"][:, :, 0] + data["quad_u"][:, :, 1] - 2 * data["quad_u"][:, :, 2], 0)
    qdp = np.maximum(data["quad_p"][:, :, 0] + data["quad_p"][:, :, 1] - 2 * data["quad_p"][:, :, 2], 0)
    scale = alpha_s[:, None]
    du = scale * np.sqrt(qdu / np.maximum(hnorm_u, 1e-12))
    dp = scale * np.sqrt(qdp / np.maximum(hnorm_p, 1e-12))
    dj = scale * np.sqrt((qdu + qdp) / np.maximum(hnorm_u + hnorm_p, 1e-12))
    by_horizon = {}
    for horizon in HORIZONS:
        by_horizon[f"K{horizon}"] = {
            "velocity_mean": float(np.mean(du[:, horizon - 1])),
            "velocity_worst": float(np.max(du[:, horizon - 1])),
            "pressure_mean": float(np.mean(dp[:, horizon - 1])),
            "pressure_worst": float(np.max(dp[:, horizon - 1])),
            "joint_mean": float(np.mean(dj[:, horizon - 1])),
            "joint_worst": float(np.max(dj[:, horizon - 1])),
        }
    return {
        "definition": "||x_method-x_H||/(||x_H||+epsilon), exact area-weighted physical reconstruction norm",
        "all_steps": {"joint_mean": float(np.mean(dj)), "joint_worst": float(np.max(dj))},
        "horizons": by_horizon,
        "joint_per_window_all_step_mean": np.mean(dj, axis=1).tolist(),
    }


def summarize_weights(alpha_h: np.ndarray, re_values: np.ndarray, seed: int) -> dict[str, object]:
    by_re = {}
    for re_value in np.unique(re_values):
        values = alpha_h[re_values == re_value]
        by_re[str(float(re_value))] = {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()), "count": int(len(values))}
    return {
        "seed": seed,
        "overall": {"mean": float(alpha_h.mean()), "min": float(alpha_h.min()), "max": float(alpha_h.max()), "count": int(len(alpha_h))},
        "by_Re": by_re,
        "cross_seed_note": "This frozen experiment has one preregistered seed for this method; no additional-seed gate retraining was performed.",
    }


def route_outputs(routes, method: str, checkpoint: dict[str, object], data: dict[str, np.ndarray], e2_alpha_s: np.ndarray):
    x = ((data["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]).astype(np.float32)
    base = np.log(np.maximum(e2_alpha_s, 1e-6) / np.maximum(1 - e2_alpha_s, 1e-6)).astype(np.float32)
    gate = routes.ConvexGate(x.shape[1])
    gate.load_state_dict(checkpoint["gate_state"])
    gate.eval()
    with torch.inference_mode():
        gate_alpha_s = gate(torch.as_tensor(x), torch.as_tensor(base)).numpy()
    full_alpha_s = gate_alpha_s.copy()
    active = np.ones(len(x), dtype=bool)
    scorer_values = None
    if method == "RiskPredictionRouter":
        scorer = routes.MLP(x.shape[1], 2)
        scorer.load_state_dict(checkpoint["scorer_state"])
        scorer.eval()
        with torch.inference_mode():
            scorer_values = scorer(torch.as_tensor(x)).numpy()
    elif method == "LookAheadShortRolloutRouter":
        lookahead = routes.lookahead_features(data)
        scorer_x = np.concatenate((x, (lookahead - checkpoint["lookahead_mean"]) / checkpoint["lookahead_std"]), axis=1).astype(np.float32)
        scorer = routes.MLP(scorer_x.shape[1], 2)
        scorer.load_state_dict(checkpoint["scorer_state"])
        scorer.eval()
        with torch.inference_mode():
            scorer_values = scorer(torch.as_tensor(scorer_x)).numpy()
    if scorer_values is not None:
        margin = np.abs(scorer_values[:, 0] - scorer_values[:, 1])
        active = margin <= float(checkpoint["threshold"])
        selected = np.argmin(scorer_values, axis=1)
        full_alpha_s[~active] = (selected[~active] == 0).astype(np.float32)
    return full_alpha_s, gate_alpha_s, active, scorer_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--heldout-cache", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not (args.training_dir / "ALL_ROUTES_VALIDATION_FROZEN.json").exists():
        raise RuntimeError("route checkpoints are not validation-frozen")
    routes = load_module("h_only_ablation_routes", Path(__file__).with_name("train_sh_routes.py"))
    data_by_split = {
        "validation": select_split(load_cache(args.validation_cache), "validation"),
        "heldout": select_split(load_cache(args.heldout_cache), "heldout"),
    }
    with np.load(ROOT / "Hopf/artifacts/hopf/velocity_pod_hopf.npz") as hu, np.load(ROOT / "Hopf/artifacts/hopf/pressure_pod_hopf.npz") as hp:
        hnorm = {split: (h_norm(data["h_a"], hu, True), h_norm(data["h_b"], hp, False)) for split, data in data_by_split.items()}
    output = {
        "status": "H_ONLY_AND_MODULE_ABLATIONS_COMPLETE",
        "specialists_retrained": False,
        "specialist_checkpoints_modified": False,
        "cache_sha256": {"validation": sha256(args.validation_cache), "heldout": sha256(args.heldout_cache)},
        "splits": {},
    }
    rows = []
    for split, data in data_by_split.items():
        e2_alpha_s = routes.e2_pair_probability(data["re"])
        h_only = routes.metrics(np.zeros(len(data["re"]), dtype=np.float32), data["quad_u"], data["quad_p"], data["re"], np.zeros(len(data["re"]), dtype=bool))
        split_result = {"H_only": h_only, "methods": {}}
        for method, _registered_seed in routes.METHODS.items():
            checkpoint_path = args.training_dir / method / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            seed = int(checkpoint["seed"])
            full_alpha_s, gate_alpha_s, active, scorer_values = route_outputs(routes, method, checkpoint, data, e2_alpha_s)
            full_metrics = routes.metrics(full_alpha_s, data["quad_u"], data["quad_p"], data["re"], active)
            gate_metrics = routes.metrics(gate_alpha_s, data["quad_u"], data["quad_p"], data["re"], np.ones(len(active), dtype=bool))
            ablation_name = "not_applicable_T2C_is_gate_only" if method.startswith("T2-C") else ("without_risk_critic_gate_only" if method == "RiskPredictionRouter" else "without_short_rollout_gate_only")
            method_result = {
                "checkpoint_sha256": sha256(checkpoint_path),
                "seed": seed,
                "full": full_metrics,
                "H_only_error_delta": {"joint_all_mean": full_metrics["joint_all_mean"] - h_only["joint_all_mean"], "joint_all_worst": full_metrics["joint_all_worst"] - h_only["joint_all_worst"]},
                "alpha_H": summarize_weights(1 - full_alpha_s, data["re"], seed),
                "distance_to_H_only": distance_to_h(full_alpha_s, data, *hnorm[split]),
                "ablation": {
                    "name": ablation_name,
                    "metrics": gate_metrics,
                    "delta_full_minus_gate_only": {"joint_all_mean": full_metrics["joint_all_mean"] - gate_metrics["joint_all_mean"], "joint_all_worst": full_metrics["joint_all_worst"] - gate_metrics["joint_all_worst"]},
                    "routing_decisions_changed": int(np.sum(np.abs(full_alpha_s - gate_alpha_s) > 1e-8)),
                },
            }
            split_result["methods"][method] = method_result
            distance_rows = method_result["distance_to_H_only"]["joint_per_window_all_step_mean"]
            for index in range(len(data["re"])):
                rows.append({
                    "split": split, "method": method, "seed": seed, "window_index": index,
                    "Re": float(data["re"][index]), "start_snapshot": int(data["start"][index]),
                    "start_time": float(data["times"][index, 0]), "end_time": float(data["times"][index, -1]),
                    "alpha_H_full": float(1 - full_alpha_s[index]), "alpha_H_gate_only": float(1 - gate_alpha_s[index]),
                    "top2_active_full": bool(active[index]), "distance_to_H_joint_all_step_mean": float(distance_rows[index]),
                })
        output["splits"][split] = split_result
    atomic_json(args.output_dir / "H_ONLY_WEIGHT_DISTANCE_ABLATIONS.json", output)
    with (args.output_dir / "PER_WINDOW_ALPHA_H_AND_DISTANCE.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# H-only and attached-module ablations", "", "All results use frozen native-output caches and frozen route checkpoints; no specialist was retrained.", ""]
    for split, split_result in output["splits"].items():
        lines += [f"## {split}", "", f"H-only joint all-step mean/worst: `{split_result['H_only']['joint_all_mean']:.9g}` / `{split_result['H_only']['joint_all_worst']:.9g}`.", "", "| Method | a_H mean [min,max] | error mean | delta vs H-only | d_H mean | gate-only ablation delta | decisions changed |", "|---|---:|---:|---:|---:|---:|---:|"]
        for method, value in split_result["methods"].items():
            weight=value["alpha_H"]["overall"]; full=value["full"]; dist=value["distance_to_H_only"]["all_steps"]; ab=value["ablation"]
            lines.append(f"| {method} | {weight['mean']:.6f} [{weight['min']:.6f},{weight['max']:.6f}] | {full['joint_all_mean']:.9g} | {value['H_only_error_delta']['joint_all_mean']:+.3g} | {dist['joint_mean']:.3g} | {ab['delta_full_minus_gate_only']['joint_all_mean']:+.3g} | {ab['routing_decisions_changed']} |")
        lines.append("")
    lines += ["## Seed scope", "", "This file reports the seed embedded in each frozen checkpoint. Cross-seed aggregation, when available, is reported separately and is not used for model selection.", ""]
    (args.output_dir / "H_ONLY_WEIGHT_DISTANCE_ABLATIONS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": output["status"], "rows": len(rows), "summary": {split: {"H_only": value["H_only"]["joint_all_mean"], **{method: route["full"]["joint_all_mean"] for method, route in value["methods"].items()}} for split, value in output["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
