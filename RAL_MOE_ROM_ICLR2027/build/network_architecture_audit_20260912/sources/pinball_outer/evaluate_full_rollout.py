#!/usr/bin/env python3
"""Frozen multi-horizon rollout audit for the Pinball E2 + T2-C fusion."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from build_boundary_cache import (
    field_statistics,
    load_runtime,
    release,
    run_rollout,
    specs,
    target_data,
    velocity_field,
)
from common import (
    ConvexGate,
    atomic_json,
    evenly_spaced,
    history_matrix,
    physical_descriptors,
    reconstruct,
    router_probabilities,
    sha256,
)

EPS = 1.0e-12
BOUNDARY_RANGES = {"SH": (16.6, 19.0), "HP": (21.25, 25.0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", default="1,8,16,24,32,56")
    parser.add_argument("--windows-per-re", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_final_test(runtime, root: Path) -> tuple[dict, dict]:
    expert = runtime.spec.name.lower()
    directory = root / "fluidicPinball_v2/eval_pod_coefficients_v2" / expert / "final_test"
    parts = {key: [] for key in ("a", "b", "re", "time")}
    hashes, observed, mesh_hash = {}, [], None
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(directory)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data["split"].item()) != "final_test" or str(data["expert"].item()) != expert:
                raise RuntimeError(f"sealed evaluation contract mismatch: {path}")
            current_mesh = str(data["mesh_hash"].item())
            mesh_hash = current_mesh if mesh_hash is None else mesh_hash
            if current_mesh != mesh_hash:
                raise RuntimeError(f"mesh mismatch: {path}")
            re_value = float(data["Re"])
            observed.append(re_value)
            count = len(data["times"])
            parts["a"].append(np.asarray(data["a_velocity_rank999"], dtype=np.float32))
            parts["b"].append(np.asarray(data["b_pressure_rank999"], dtype=np.float32))
            parts["re"].append(np.full(count, re_value, dtype=np.float32))
            parts["time"].append(np.asarray(data["times"], dtype=np.float32))
        hashes[str(path.relative_to(root))] = sha256(path)
    a, b, reynolds, times = (np.concatenate(parts[key]) for key in ("a", "b", "re", "time"))
    if a.shape[1] != runtime.data["a"].shape[1] or b.shape[1] != runtime.data["b"].shape[1]:
        raise RuntimeError(f"{runtime.spec.name} final-test POD rank mismatch")
    order = np.lexsort((times, reynolds))
    a, b, reynolds, times = a[order], b[order], reynolds[order], times[order]
    nxt = np.full(len(reynolds), -1, dtype=np.int64)
    prev = np.full(len(reynolds), -1, dtype=np.int64)
    for value in np.unique(reynolds):
        ids = np.flatnonzero(np.isclose(reynolds, value, atol=5e-7))
        if np.any(np.diff(times[ids]) <= 0):
            raise RuntimeError(f"non-increasing final-test time: {runtime.spec.name}, Re={value}")
        nxt[ids[:-1]], prev[ids[1:]] = ids[1:], ids[:-1]
    hist = history_matrix(prev, 3)
    valid = np.flatnonzero((nxt >= 0) & np.all(hist >= 0, axis=1))
    data = {
        "a": a, "b": b, "re": reynolds, "time": times,
        "next": nxt, "prev": prev, "hist": hist, "valid": valid,
        "train_ids": np.empty(0, dtype=np.int64), "val_ids": valid,
        "phi_u": runtime.data["phi_u"], "phi_p": runtime.data["phi_p"],
        "mean_u": runtime.data["mean_u"], "mean_p": runtime.data["mean_p"],
        "areas": runtime.data["areas"],
    }
    audit = {
        "expert": runtime.spec.name,
        "directory": str(directory),
        "reynolds": sorted(observed),
        "snapshots": int(len(reynolds)),
        "mesh_hash": mesh_hash,
        "file_sha256": hashes,
    }
    return data, audit


def boundary_re(data: dict, boundary: str) -> np.ndarray:
    lo, hi = BOUNDARY_RANGES[boundary]
    values = np.unique(data["re"][data["val_ids"]])
    return values[(values >= lo - 5e-7) & (values <= hi + 5e-7)]


def select_common_starts(runtime, boundary: str, horizon: int, windows_per_re: int) -> tuple[np.ndarray, np.ndarray]:
    starts, labels = [], []
    for value in boundary_re(runtime.data, boundary):
        ids = runtime.data["val_ids"][np.isclose(runtime.data["re"][runtime.data["val_ids"]], value, atol=5e-7)]
        legal = runtime.trainer.legal_starts(runtime.baseline, runtime.data, ids, horizon)
        chosen = evenly_spaced(legal, windows_per_re)
        if len(chosen) != windows_per_re:
            raise RuntimeError(f"only {len(chosen)} common K{horizon} windows: {boundary}, Re={value}")
        starts.extend(chosen.tolist())
        labels.extend([float(value)] * len(chosen))
    if not starts:
        raise RuntimeError(f"no boundary data: {boundary}")
    return np.asarray(starts, dtype=np.int64), np.asarray(labels, dtype=np.float64)


def descriptors(runtime, starts: np.ndarray) -> np.ndarray:
    rows = []
    for start in starts:
        history = np.asarray(runtime.data["hist"][start], dtype=np.int64)[::-1]
        velocity = velocity_field(runtime.data["a"][history], runtime)
        pressure = reconstruct(runtime.data["b"][history], runtime.data["phi_p"], runtime.data["mean_p"])
        rows.append(physical_descriptors(velocity, pressure, runtime.data["time"][history], runtime.data["areas"]))
    return np.stack(rows)


def method_weights(run_root: Path, boundary: str, features: np.ndarray, reynolds: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    router_path = run_root / "e2_router/best.pt"
    gate_path = run_root / f"gate_{boundary.lower()}/best.pt"
    checkpoint = torch.load(gate_path, map_location="cpu", weights_only=False)
    if checkpoint["boundary"] != boundary:
        raise RuntimeError(f"gate boundary mismatch: {gate_path}")
    if checkpoint["router_checkpoint_sha256"] != sha256(router_path):
        raise RuntimeError(f"router hash mismatch in {gate_path}")
    probabilities = router_probabilities(router_path, reynolds)
    pair = probabilities[:, np.asarray(checkpoint["pair_indices"], dtype=int)]
    pair /= pair.sum(axis=1, keepdims=True)
    base = np.clip(pair[:, 0], 1e-6, 1 - 1e-6).astype(np.float32)
    logit = np.log(base / (1 - base)).astype(np.float32)
    normalized = ((features - checkpoint["feature_mean"]) / checkpoint["feature_std"]).astype(np.float32)
    gate = ConvexGate(normalized.shape[1])
    gate.load_state_dict(checkpoint["gate_state"], strict=True)
    gate.eval()
    with torch.inference_mode():
        learned = gate(torch.as_tensor(normalized), torch.as_tensor(logit)).numpy()
    weights = {
        "source_only": np.ones(len(reynolds), dtype=np.float32),
        "hopf_only": np.zeros(len(reynolds), dtype=np.float32),
        "fixed_0.5": np.full(len(reynolds), 0.5, dtype=np.float32),
        "E2_pair_blend": base,
        "T2-C": learned,
    }
    audit = {
        "router_checkpoint": str(router_path), "router_sha256": sha256(router_path),
        "gate_checkpoint": str(gate_path), "gate_sha256": sha256(gate_path),
        "gate_best_step": int(checkpoint["best_step"]),
        "gate_training_horizon": 24,
    }
    return weights, audit


def errors(alpha: np.ndarray, quad_u: np.ndarray, quad_p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = np.asarray(alpha, dtype=np.float64)[:, None]
    def component(quad: np.ndarray) -> np.ndarray:
        numerator = (weight * weight * quad[:, :, 0] + (1 - weight) ** 2 * quad[:, :, 1]
                     + 2 * weight * (1 - weight) * quad[:, :, 2])
        return np.sqrt(np.maximum(numerator / np.maximum(quad[:, :, 3], EPS), 0.0))
    velocity, pressure = component(quad_u), component(quad_p)
    return velocity, pressure, velocity + pressure


def oracle_weights(quad_u: np.ndarray, quad_p: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 2001, dtype=np.float64)
    output = np.empty(len(quad_u), dtype=np.float32)
    for index, (velocity, pressure) in enumerate(zip(quad_u, quad_p)):
        weight = grid[:, None]
        vu = (weight * weight * velocity[None, :, 0] + (1 - weight) ** 2 * velocity[None, :, 1]
              + 2 * weight * (1 - weight) * velocity[None, :, 2]) / np.maximum(velocity[None, :, 3], EPS)
        pp = (weight * weight * pressure[None, :, 0] + (1 - weight) ** 2 * pressure[None, :, 1]
              + 2 * weight * (1 - weight) * pressure[None, :, 2]) / np.maximum(pressure[None, :, 3], EPS)
        score = np.mean(np.sqrt(np.maximum(vu, 0)) + np.sqrt(np.maximum(pp, 0)), axis=1)
        output[index] = grid[int(np.argmin(score))]
    return output


def summarize(method: str, boundary: str, split: str, horizon: int, re_value: float | None,
              alpha: np.ndarray, quad_u: np.ndarray, quad_p: np.ndarray) -> dict:
    velocity, pressure, joint = errors(alpha, quad_u, quad_p)
    finite = np.isfinite(joint).all(axis=1)
    divergent = (~finite) | np.any((velocity > 20.0) | (pressure > 20.0), axis=1)
    return {
        "split": split, "boundary": boundary, "horizon": horizon, "method": method,
        "Re": "ALL" if re_value is None else f"{re_value:.6g}",
        "windows": int(len(alpha)),
        "velocity_mean": float(np.mean(velocity)),
        "pressure_mean": float(np.mean(pressure)),
        "joint_mean": float(np.mean(joint)),
        "joint_terminal": float(np.mean(joint[:, -1])),
        "joint_worst_point": float(np.max(joint)),
        "joint_worst_window_mean": float(np.max(np.mean(joint, axis=1))),
        "finite_fraction": float(np.mean(finite)),
        "divergent_windows": int(np.sum(divergent)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
    }


def plot_results(aggregate: list[dict], weights: list[dict], growth: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("source_only", "hopf_only", "E2_pair_blend", "T2-C", "per_window_oracle")
    for split in ("validation", "final_test"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=False)
        for ax, boundary in zip(axes, ("SH", "HP")):
            for method in methods:
                rows = [r for r in aggregate if r["split"] == split and r["boundary"] == boundary and r["method"] == method]
                rows.sort(key=lambda r: r["horizon"])
                ax.plot([r["horizon"] for r in rows], [100 * r["joint_mean"] for r in rows], marker="o", label=method)
            ax.set(xlabel="Autonomous horizon K", ylabel="Mean joint relative error (%)", title=f"{split}: {boundary}")
            ax.grid(alpha=.25)
        axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(output / f"{split}_error_by_horizon.png", dpi=190); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, boundary in zip(axes, ("SH", "HP")):
        for split, marker in (("validation", "o"), ("final_test", "s")):
            rows = [r for r in weights if r["boundary"] == boundary and r["split"] == split and r["method"] == "T2-C"]
            rows.sort(key=lambda r: r["Re"])
            ax.errorbar([r["Re"] for r in rows], [r["alpha_mean"] for r in rows],
                        yerr=[[r["alpha_mean"] - r["alpha_min"] for r in rows], [r["alpha_max"] - r["alpha_mean"] for r in rows]],
                        marker=marker, capsize=3, label=split)
        ax.set(xlabel="Re", ylabel="Source expert weight alpha", title=boundary, ylim=(-.03, 1.03)); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output / "t2c_weight_vs_re.png", dpi=190); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row_ax, split in zip(axes, ("validation", "final_test")):
        for ax, boundary in zip(row_ax, ("SH", "HP")):
            for method in ("source_only", "E2_pair_blend", "T2-C", "per_window_oracle"):
                rows = [r for r in growth if r["split"] == split and r["boundary"] == boundary and r["horizon"] == 56 and r["method"] == method]
                rows.sort(key=lambda r: r["step"])
                ax.plot([r["step"] for r in rows], [100 * r["joint_mean"] for r in rows], label=method)
            ax.set(title=f"{split}: {boundary}", ylabel="Joint relative error (%)"); ax.grid(alpha=.25)
    axes[1, 0].set_xlabel("Autonomous step"); axes[1, 1].set_xlabel("Autonomous step"); axes[0, 1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "K56_error_growth.png", dpi=190); plt.close(fig)


def main() -> None:
    args = parse_args()
    horizons = tuple(dict.fromkeys(int(value) for value in args.horizons.split(",")))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must be positive")
    max_horizon = max(horizons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_specs = specs(args.root)
    checkpoint_inventory = {
        "Steady": sha256(all_specs["Steady"].checkpoint),
        "Hopf": sha256(all_specs["Hopf"].checkpoint),
        "Periodic": sha256(all_specs["Periodic"].checkpoint),
        "E2": sha256(args.run_root / "e2_router/best.pt"),
        "T2C_SH": sha256(args.run_root / "gate_sh/best.pt"),
        "T2C_HP": sha256(args.run_root / "gate_hp/best.pt"),
    }
    seal = {
        "status": "FINAL_TEST_UNSEAL_STARTED", "started_unix": time.time(),
        "selection_frozen_before_unseal": True, "training_after_unseal": False,
        "checkpoint_sha256": checkpoint_inventory,
    }
    atomic_json(args.output_dir / "FINAL_TEST_SEAL.json", seal)
    detail_rows, aggregate_rows, growth_rows, weight_rows = [], [], [], []
    asset_audit, gate_audit, stability = {}, {}, {}
    started_at = time.time()
    for boundary in ("SH", "HP"):
        source_name = "Steady" if boundary == "SH" else "Periodic"
        for split in ("validation", "final_test"):
            source = load_runtime(all_specs[source_name], device)
            split_audit = None
            if split == "final_test":
                source.data, split_audit = load_final_test(source, args.root)
                asset_audit[f"{boundary}_{split}"] = split_audit
            starts, reynolds = select_common_starts(source, boundary, max_horizon, args.windows_per_re)
            feature_values = np.concatenate((reynolds[:, None].astype(np.float32), descriptors(source, starts)), axis=1)
            weights, current_gate_audit = method_weights(args.run_root, boundary, feature_values, reynolds)
            gate_audit[boundary] = current_gate_audit
            source_a, source_b, truth_a, truth_b = run_rollout(source, starts, max_horizon)
            target = load_runtime(all_specs["Hopf"], device)
            if not np.allclose(source.data["areas"], target.data["areas"], atol=1e-8, rtol=1e-6):
                raise RuntimeError("common-grid area mismatch")
            original_target_data = target.data
            target.data = target_data(source, target, starts)
            target_a, target_b, _, _ = run_rollout(target, starts, max_horizon)
            target.data = original_target_data
            quad_u, quad_p, source_ratio, target_ratio = field_statistics(
                source_a, source_b, target_a, target_b, truth_a, truth_b, source, target
            )
            stability[f"{boundary}_{split}"] = {
                "source_expert": source_name, "target_expert": "Hopf",
                "windows": int(len(starts)), "horizon": max_horizon,
                "source_finite_fraction": float(np.mean(np.isfinite(source_a).all((1, 2)) & np.isfinite(source_b).all((1, 2)))),
                "target_finite_fraction": float(np.mean(np.isfinite(target_a).all((1, 2)) & np.isfinite(target_b).all((1, 2)))),
                "source_max_norm_ratio": float(np.max(source_ratio)), "target_max_norm_ratio": float(np.max(target_ratio)),
                "source_divergent_windows": int(np.sum(np.any(source_ratio > 20, axis=1))),
                "target_divergent_windows": int(np.sum(np.any(target_ratio > 20, axis=1))),
            }
            for method in ("E2_pair_blend", "T2-C"):
                for value in np.unique(reynolds):
                    selected = reynolds == value
                    weight_rows.append({
                        "split": split, "boundary": boundary, "method": method, "Re": float(value),
                        "windows": int(np.sum(selected)), "alpha_mean": float(np.mean(weights[method][selected])),
                        "alpha_min": float(np.min(weights[method][selected])), "alpha_max": float(np.max(weights[method][selected])),
                        "alpha_std": float(np.std(weights[method][selected])),
                    })
            for horizon in horizons:
                current_u, current_p = quad_u[:, :horizon], quad_p[:, :horizon]
                current_weights = dict(weights)
                current_weights["per_window_oracle"] = oracle_weights(current_u, current_p)
                for method, alpha in current_weights.items():
                    aggregate = summarize(method, boundary, split, horizon, None, alpha, current_u, current_p)
                    aggregate_rows.append(aggregate)
                    velocity, pressure, joint = errors(alpha, current_u, current_p)
                    for step in range(horizon):
                        growth_rows.append({
                            "split": split, "boundary": boundary, "horizon": horizon, "method": method,
                            "step": step + 1, "velocity_mean": float(np.mean(velocity[:, step])),
                            "pressure_mean": float(np.mean(pressure[:, step])), "joint_mean": float(np.mean(joint[:, step])),
                        })
                    for value in np.unique(reynolds):
                        selected = reynolds == value
                        detail_rows.append(summarize(method, boundary, split, horizon, float(value),
                                                     alpha[selected], current_u[selected], current_p[selected]))
            release(source); release(target)
            print(json.dumps({"event": "case_complete", "boundary": boundary, "split": split,
                              "windows": len(starts), "horizon": max_horizon}), flush=True)
    write_csv(detail_rows, args.output_dir / "ROLLOUT_BY_RE.csv")
    write_csv(aggregate_rows, args.output_dir / "ROLLOUT_AGGREGATE.csv")
    write_csv(growth_rows, args.output_dir / "ERROR_GROWTH_CURVES.csv")
    write_csv(weight_rows, args.output_dir / "WEIGHT_DIAGNOSTICS.csv")
    plot_status = "complete"
    try:
        plot_results(aggregate_rows, weight_rows, growth_rows, args.output_dir)
    except ModuleNotFoundError as error:
        plot_status = f"skipped_optional_dependency:{error.name}"
        print(json.dumps({"event": "optional_plot_skipped", "dependency": error.name}), flush=True)
    summary = {
        "schema_version": 1, "status": "FROZEN_FULL_ROLLOUT_COMPLETE",
        "protocol": {
            "splits": ["validation", "final_test"], "boundaries": ["SH", "HP"],
            "horizons": list(horizons), "windows_per_re": args.windows_per_re,
            "window_sampling": f"64 evenly spaced legal K{max_horizon} starts per Re; shorter horizons are prefixes of the identical trajectories",
            "fusion_weight_scope": "one scalar per trajectory fixed from the initial 3-frame history",
            "fusion_feedback": False, "final_test_used_for_selection": False,
            "training_after_final_test_unseal": False, "divergence_threshold_relative_component_error": 20.0,
            "oracle_grid_spacing": 0.0005,
        },
        "checkpoint_sha256": checkpoint_inventory, "gate_audit": gate_audit,
        "sealed_asset_audit": asset_audit, "stability": stability,
        "aggregate": aggregate_rows, "plot_status": plot_status,
        "elapsed_seconds": time.time() - started_at,
    }
    atomic_json(args.output_dir / "FULL_ROLLOUT_SUMMARY.json", summary)
    seal.update({"status": "FINAL_TEST_UNSEAL_COMPLETE", "completed_unix": time.time(),
                 "training_after_unseal": False, "result_sha256": sha256(args.output_dir / "FULL_ROLLOUT_SUMMARY.json")})
    atomic_json(args.output_dir / "FINAL_TEST_SEAL.json", seal)
    print(json.dumps({"event": "full_rollout_complete", "output": str(args.output_dir),
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
