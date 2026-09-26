"""Frozen-cache attractor diagnostics for S-only, H-only, E2, and T2-C."""

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


RANK = 32
TAIL = 16
METHODS = ("S_only", "H_only", "E2_Top1", "T2_C")


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


class MixedGeometry:
    def __init__(self, steady_pod, hopf_pod):
        areas = np.asarray(steady_pod["point_areas"], dtype=np.float64)
        self.weights = np.concatenate((areas, areas))
        self.ps = np.asarray(steady_pod["phi_uv"][:RANK], dtype=np.float64)
        self.ph = np.asarray(hopf_pod["phi_uv"][:RANK], dtype=np.float64)
        self.ms = np.asarray(steady_pod["mean_uv_regime"], dtype=np.float64)
        self.mh = np.asarray(hopf_pod["mean_uv_regime"], dtype=np.float64)
        self.gss = (self.ps * self.weights) @ self.ps.T
        self.ghh = (self.ph * self.weights) @ self.ph.T
        self.csh = (self.ps * self.weights) @ self.ph.T
        self.psw = self.ps * self.weights
        self.phw = self.ph * self.weights

    def mixed_norm_sq(
        self,
        coeff_s: np.ndarray,
        coeff_h: np.ndarray,
        field_offset: np.ndarray | None = None,
    ) -> np.ndarray:
        cs = np.asarray(coeff_s, dtype=np.float64)
        ch = np.asarray(coeff_h, dtype=np.float64)
        value = (
            np.einsum("...i,ij,...j->...", cs, self.gss, cs)
            + np.einsum("...i,ij,...j->...", ch, self.ghh, ch)
            + 2 * np.einsum("...i,ij,...j->...", cs, self.csh, ch)
        )
        if field_offset is not None:
            offset = np.asarray(field_offset, dtype=np.float64)
            value = (
                value
                + float(np.dot(offset * self.weights, offset))
                + 2 * np.einsum("...i,i->...", cs, self.psw @ offset)
                + 2 * np.einsum("...i,i->...", ch, self.phw @ offset)
            )
        return np.maximum(value, 0.0)

    def truth_state_norm_sq(self, coeff_s: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(coeff_s, dtype=np.float64)
        return self.mixed_norm_sq(coeff_s, zeros, self.ms)

    def distance_matrix(
        self, coeff_s: np.ndarray, coeff_h: np.ndarray
    ) -> np.ndarray:
        cs = np.asarray(coeff_s, dtype=np.float64)
        ch = np.asarray(coeff_h, dtype=np.float64)
        gram = (
            cs @ self.gss @ cs.T
            + ch @ self.ghh @ ch.T
            + cs @ self.csh @ ch.T
            + ch @ self.csh.T @ cs.T
        )
        diagonal = np.diag(gram)
        squared = np.maximum(diagonal[:, None] + diagonal[None, :] - 2 * gram, 0.0)
        return np.sqrt(squared)


def truth_tail_amplitude(geometry: MixedGeometry, truth: np.ndarray) -> float:
    tail = np.asarray(truth[-TAIL:], dtype=np.float64)
    centered = tail - tail.mean(axis=0, keepdims=True)
    zeros = np.zeros_like(centered)
    return float(np.sqrt(np.mean(geometry.mixed_norm_sq(centered, zeros))))


def diagnostics(
    geometry: MixedGeometry,
    steady: np.ndarray,
    hopf: np.ndarray,
    truth: np.ndarray,
    alpha_s: float,
    amplitude_floor: float,
) -> dict[str, float]:
    steady = np.asarray(steady, dtype=np.float64)
    hopf = np.asarray(hopf, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    alpha_h = 1.0 - float(alpha_s)
    tail_s = steady[-TAIL:]
    tail_h = hopf[-TAIL:]
    tail_t = truth[-TAIL:]
    centered_s = alpha_s * (tail_s - tail_s.mean(axis=0, keepdims=True))
    centered_h = alpha_h * (tail_h - tail_h.mean(axis=0, keepdims=True))
    centered_t = tail_t - tail_t.mean(axis=0, keepdims=True)
    zeros = np.zeros_like(centered_t)
    pred_amplitude = float(
        np.sqrt(np.mean(geometry.mixed_norm_sq(centered_s, centered_h)))
    )
    true_amplitude = float(
        np.sqrt(np.mean(geometry.mixed_norm_sq(centered_t, zeros)))
    )
    truth_tail_state_scale = float(
        np.sqrt(np.mean(geometry.truth_state_norm_sq(tail_t)))
    )
    center_coeff_s = alpha_s * tail_s.mean(axis=0) - tail_t.mean(axis=0)
    center_coeff_h = alpha_h * tail_h.mean(axis=0)
    center_offset = alpha_h * (geometry.mh - geometry.ms)
    center_error = float(
        np.sqrt(
            geometry.mixed_norm_sq(
                center_coeff_s[None], center_coeff_h[None], center_offset
            )[0]
        )
        / max(truth_tail_state_scale, 1e-12)
    )
    amplitude_absolute_error = abs(pred_amplitude - true_amplitude) / max(
        truth_tail_state_scale, 1e-12
    )
    amplitude_relative_error = abs(pred_amplitude - true_amplitude) / max(
        true_amplitude, amplitude_floor
    )
    pred_distances = geometry.distance_matrix(centered_s, centered_h)
    truth_distances = geometry.distance_matrix(centered_t, zeros)
    orbit_geometry_error = float(
        np.linalg.norm(pred_distances - truth_distances)
        / max(np.linalg.norm(truth_distances), amplitude_floor, 1e-12)
    )
    pred_ds = alpha_s * np.diff(tail_s, axis=0)
    pred_dh = alpha_h * np.diff(tail_h, axis=0)
    truth_ds = np.diff(tail_t, axis=0)
    increment_error = float(
        np.sqrt(
            np.mean(
                geometry.mixed_norm_sq(pred_ds - truth_ds, pred_dh)
            )
        )
        / max(truth_tail_state_scale, 1e-12)
    )
    pred_increment_rms = float(
        np.sqrt(np.mean(geometry.mixed_norm_sq(pred_ds, pred_dh)))
    )
    truth_increment_rms = float(
        np.sqrt(np.mean(geometry.mixed_norm_sq(truth_ds, np.zeros_like(truth_ds))))
    )
    terminal_ds = (
        alpha_s * (steady[-1] - steady[-2]) - (truth[-1] - truth[-2])
    )[None]
    terminal_dh = (alpha_h * (hopf[-1] - hopf[-2]))[None]
    terminal_increment_error = float(
        np.sqrt(geometry.mixed_norm_sq(terminal_ds, terminal_dh)[0])
        / max(float(np.sqrt(geometry.truth_state_norm_sq(truth[-1:])[0])), 1e-12)
    )
    return {
        "tail_center_field_error": center_error,
        "truth_tail_amplitude_normalized": true_amplitude
        / max(truth_tail_state_scale, 1e-12),
        "pred_tail_amplitude_normalized": pred_amplitude
        / max(truth_tail_state_scale, 1e-12),
        "tail_amplitude_absolute_error": amplitude_absolute_error,
        "tail_amplitude_relative_error_train_floor": amplitude_relative_error,
        "tail_orbit_geometry_error": orbit_geometry_error,
        "tail_increment_field_error": increment_error,
        "terminal_increment_field_error": terminal_increment_error,
        "pred_tail_increment_rms_normalized": pred_increment_rms
        / max(truth_tail_state_scale, 1e-12),
        "truth_tail_increment_rms_normalized": truth_increment_rms
        / max(truth_tail_state_scale, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development_cache = (
        args.run_dir / "cache_development/G_SH_resplit_train_validation_cache.npz"
    )
    test_cache = args.run_dir / "cache_final_test/G_SH_resplit_final_test_cache.npz"
    checkpoint_path = (
        args.run_dir / "training/T2-C_LearnedConvexCorrection_FieldBlend/best.pt"
    )
    with np.load(development_cache, allow_pickle=False) as archive:
        development = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(test_cache, allow_pickle=False) as archive:
        test = {key: np.asarray(archive[key]) for key in archive.files}
    if set(test["split"].tolist()) != {"heldout"}:
        raise RuntimeError("test cache split failure")
    routes = load_module(
        "attractor_routes", args.run_dir.parent / "code/train_sh_routes.py"
    )
    routes.ROOT = args.root
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    e2_probability_s = routes.e2_pair_probability(test["re"])
    e2_alpha_s = (e2_probability_s >= 0.5).astype(np.float32)
    features = (
        (test["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    ).astype(np.float32)
    base_logit = np.log(
        np.maximum(e2_probability_s, 1e-6)
        / np.maximum(1 - e2_probability_s, 1e-6)
    ).astype(np.float32)
    gate = routes.ConvexGate(features.shape[1])
    gate.load_state_dict(checkpoint["gate_state"])
    gate.eval()
    with torch.inference_mode():
        t2c_alpha_s = gate(
            torch.as_tensor(features), torch.as_tensor(base_logit)
        ).numpy()
    steady_pod_path = (
        args.root / "steady_specialist_v1/source_artifacts/steady/velocity_pod_steady.npz"
    )
    hopf_pod_path = args.root / "Hopf/artifacts/hopf/velocity_pod_hopf.npz"
    with np.load(steady_pod_path) as steady_pod, np.load(hopf_pod_path) as hopf_pod:
        geometry = MixedGeometry(steady_pod, hopf_pod)
        train_mask = development["split"] == "train"
        train_amplitudes = np.asarray(
            [
                truth_tail_amplitude(geometry, truth)
                for truth in development["true_a"][train_mask]
            ],
            dtype=np.float64,
        )
        amplitude_floor = float(max(np.quantile(train_amplitudes, 0.05), 1e-12))
        rows = []
        alpha_by_method = {
            "S_only": np.ones(len(test["re"]), dtype=np.float32),
            "H_only": np.zeros(len(test["re"]), dtype=np.float32),
            "E2_Top1": e2_alpha_s,
            "T2_C": t2c_alpha_s,
        }
        for index in range(len(test["re"])):
            for method in METHODS:
                metric = diagnostics(
                    geometry,
                    test["s_a"][index],
                    test["h_a"][index],
                    test["true_a"][index],
                    float(alpha_by_method[method][index]),
                    amplitude_floor,
                )
                rows.append(
                    {
                        "method": method,
                        "Re": float(test["re"][index]),
                        "start": int(test["start"][index]),
                        "start_time": float(test["times"][index, 0]),
                        "end_time": float(test["times"][index, -1]),
                        "alpha_S": float(alpha_by_method[method][index]),
                        "alpha_H": float(1 - alpha_by_method[method][index]),
                        **metric,
                    }
                )
    with (args.output_dir / "PER_WINDOW_ATTRACTOR_METRICS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_names = [
        "tail_center_field_error",
        "tail_amplitude_absolute_error",
        "tail_amplitude_relative_error_train_floor",
        "tail_orbit_geometry_error",
        "tail_increment_field_error",
        "terminal_increment_field_error",
    ]
    per_re = {}
    for re_value in sorted(set(float(row["Re"]) for row in rows)):
        per_re[str(re_value)] = {}
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["method"] == method and float(row["Re"]) == re_value
            ]
            per_re[str(re_value)][method] = {
                name: {
                    "mean": float(np.mean([float(row[name]) for row in selected])),
                    "worst": float(np.max([float(row[name]) for row in selected])),
                }
                for name in metric_names
            }
            per_re[str(re_value)][method]["state_diagnostics"] = {
                name: {
                    "mean": float(np.mean([float(row[name]) for row in selected])),
                    "min": float(np.min([float(row[name]) for row in selected])),
                    "max": float(np.max([float(row[name]) for row in selected])),
                }
                for name in (
                    "truth_tail_amplitude_normalized",
                    "pred_tail_amplitude_normalized",
                    "truth_tail_increment_rms_normalized",
                    "pred_tail_increment_rms_normalized",
                )
            }
        comparisons = {}
        for name in metric_names:
            e2_value = per_re[str(re_value)]["E2_Top1"][name]["mean"]
            t2c_value = per_re[str(re_value)]["T2_C"][name]["mean"]
            comparisons[name] = {
                "T2_C_minus_E2": t2c_value - e2_value,
                "T2_C_relative_change": (
                    (t2c_value - e2_value) / e2_value if e2_value > 0 else None
                ),
                "improved": bool(t2c_value < e2_value),
            }
        per_re[str(re_value)]["T2_C_vs_E2"] = comparisons
    aggregate = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            name: {
                "mean": float(np.mean([float(row[name]) for row in selected])),
                "worst": float(np.max([float(row[name]) for row in selected])),
            }
            for name in metric_names
        }
    dominance = {}
    for re_value, values in per_re.items():
        comparisons = values["T2_C_vs_E2"]
        improved_count = sum(item["improved"] for item in comparisons.values())
        dominance[re_value] = {
            "improved_metric_count": int(improved_count),
            "metric_count": len(metric_names),
            "improves_all_primary_attractor_metrics": bool(
                improved_count == len(metric_names)
            ),
            "degrades_any_primary_attractor_metric": bool(
                improved_count < len(metric_names)
            ),
        }
    report = {
        "status": "T2C_ATTRACTOR_ANALYSIS_COMPLETE",
        "source_sha256": {
            "development_cache": sha256(development_cache),
            "test_cache": sha256(test_cache),
            "T2_C_checkpoint": sha256(checkpoint_path),
            "steady_velocity_POD": sha256(steady_pod_path),
            "hopf_velocity_POD": sha256(hopf_pod_path),
        },
        "metric_contract": {
            "tail_steps": TAIL,
            "physical_inner_product": "exact area-weighted velocity-field inner products using S/H POD self- and cross-Gram matrices; no coefficient-space Euclidean shortcut",
            "tail_center_field_error": "Physical error of the predicted versus true tail mean, normalized by true tail-state field norm.",
            "tail_amplitude_absolute_error": "Absolute error in tail fluctuation RMS, normalized by true tail-state field norm.",
            "tail_amplitude_relative_error_train_floor": "Tail fluctuation RMS relative error with denominator floored by the train-only 5th-percentile truth amplitude.",
            "tail_orbit_geometry_error": "Relative Frobenius error of the tail pairwise physical-distance matrix; phase-origin invariant.",
            "tail_increment_field_error": "RMS physical error of tail one-step increments, normalized by true tail-state field norm.",
            "terminal_increment_field_error": "Physical error of the final predicted increment versus the final true increment, normalized by final true-state norm.",
            "train_only_amplitude_floor": amplitude_floor,
            "phase_frequency": "NOT_REPORTED: the S-native boundary cache has no certified train-only periodic observable/complete-cycle guarantee, so phase or frequency numbers would be method-dependent and potentially misleading.",
        },
        "per_Re": per_re,
        "aggregate": aggregate,
        "dominance": dominance,
        "conclusion_rule": "T2-C is declared attractor-improving at a Re only if all six preregistered physical attractor diagnostics improve versus E2; mixed changes are reported as non-dominating.",
        "test_used_for_training_or_metric_definition": False,
    }
    atomic_json(args.output_dir / "T2C_ATTRACTOR_ANALYSIS.json", report)
    lines = [
        "# Frozen T2-C attractor analysis",
        "",
        "All diagnostics use exact area-weighted physical velocity inner products. Lower is better.",
        "",
        "| Re | Method | Center | Amplitude abs. | Orbit geometry | Tail increment | Terminal increment |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for re_value in sorted(per_re, key=float):
        for method in ("E2_Top1", "T2_C"):
            value = per_re[re_value][method]
            lines.append(
                f"| {float(re_value):.6f} | {method} | "
                f"{value['tail_center_field_error']['mean']:.8f} | "
                f"{value['tail_amplitude_absolute_error']['mean']:.8f} | "
                f"{value['tail_orbit_geometry_error']['mean']:.8f} | "
                f"{value['tail_increment_field_error']['mean']:.8f} | "
                f"{value['terminal_increment_field_error']['mean']:.8f} |"
            )
    lines += [
        "",
        "Phase/frequency are not reported because this S-native boundary cache has no certified train-only periodic observable or complete-cycle guarantee.",
    ]
    (args.output_dir / "T2C_ATTRACTOR_ANALYSIS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    atomic_json(
        args.output_dir / "FREEZE_MANIFEST.json",
        {
            "status": "FROZEN_ATTRACTOR_SUPPLEMENT",
            "source_sha256": report["source_sha256"],
            "files": {
                name: sha256(args.output_dir / name)
                for name in (
                    "PER_WINDOW_ATTRACTOR_METRICS.csv",
                    "T2C_ATTRACTOR_ANALYSIS.json",
                    "T2C_ATTRACTOR_ANALYSIS.md",
                )
            },
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "dominance": dominance,
                "aggregate_E2": aggregate["E2_Top1"],
                "aggregate_T2_C": aggregate["T2_C"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
