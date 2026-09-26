#!/usr/bin/env python3
"""Fit and validate non-neural algebraic pressure closures for FVM-GROM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--pressure-pod", type=Path, required=True)
    parser.add_argument("--canonical-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--velocity-rank", type=int, default=11)
    parser.add_argument("--pressure-rank", type=int, default=11)
    parser.add_argument("--validation-re", type=float, nargs="+", default=[94.5, 95.25, 95.5, 97.5, 99.0, 101.5])
    parser.add_argument("--force-kind", choices=["linear", "quadratic", "quadratic_nu"])
    return parser.parse_args()


def quadratic_features(state: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(state.shape[1])
    return state[:, i] * state[:, j]


def features(state: np.ndarray, viscosity: np.ndarray, kind: str) -> np.ndarray:
    columns = [np.ones((len(state), 1)), viscosity[:, None], state, viscosity[:, None] * state]
    quadratic = quadratic_features(state)
    if kind in {"quadratic", "quadratic_nu"}:
        columns.append(quadratic)
    if kind == "quadratic_nu":
        columns.append(viscosity[:, None] * quadratic)
    return np.concatenate(columns, axis=1)


def tag_to_re(tag: str) -> float:
    return float(tag[2:].replace("p", "."))


def project_case(path, u_mean, u_weights, u_modes, p_mean, p_weights, p_modes):
    with np.load(path, allow_pickle=False) as source:
        u = np.asarray(source["U"], dtype=np.float64)
        p = np.asarray(source["p"], dtype=np.float64)
        re_value = float(source["Re"])
    u_weighted = (u - u_mean[None]).reshape(len(u), -1) * u_weights[None]
    p_weighted = (p - p_mean[None]) * p_weights[None]
    return re_value, u_weighted @ u_modes.T, p_weighted @ p_modes.T


def relative(prediction, truth):
    return float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth))


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    canonical = json.loads(args.canonical_cases.read_text())
    with np.load(args.velocity_pod) as pod:
        train_u = np.asarray(pod["coefficients"], dtype=np.float64)[:, : args.velocity_rank]
        tags = np.asarray(pod["snapshot_case_tags"]).astype(str)
        u_mean = np.asarray(pod["mean"], dtype=np.float64)
        u_weights = np.asarray(pod["weights"], dtype=np.float64)
        u_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)[: args.velocity_rank]
    with np.load(args.pressure_pod) as pod:
        train_p = np.asarray(pod["coefficients"], dtype=np.float64)[:, : args.pressure_rank]
        p_tags = np.asarray(pod["snapshot_case_tags"]).astype(str)
        p_mean = np.asarray(pod["mean"], dtype=np.float64)
        p_weights = np.asarray(pod["weights"], dtype=np.float64)
        p_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)[: args.pressure_rank]
    if not np.array_equal(tags, p_tags):
        raise RuntimeError("velocity and pressure train snapshot tags differ")
    train_nu = np.asarray([1.0 / tag_to_re(tag) for tag in tags])
    validation = []
    validation_re = np.asarray(args.validation_re, dtype=np.float64)
    for value in validation_re:
        matches = [entry for entry in canonical if abs(float(entry["Re"]) - value) < 5e-7]
        if len(matches) != 1:
            raise RuntimeError(f"canonical validation case mismatch at Re={value}")
        validation.append(project_case(
            Path(matches[0]["path"]), u_mean, u_weights, u_modes,
            p_mean, p_weights, p_modes,
        ))
    candidate_reports = []
    candidate_assets = []
    kinds = [args.force_kind] if args.force_kind else ["linear", "quadratic", "quadratic_nu"]
    for kind in kinds:
        train_x_raw = features(train_u, train_nu, kind)
        scale = np.std(train_x_raw, axis=0)
        scale[scale < 1e-12] = 1.0
        scale[0] = 1.0
        train_x = train_x_raw / scale[None]
        gram = train_x.T @ train_x
        right = train_x.T @ train_p
        for ridge in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0]:
            regularizer = np.eye(train_x.shape[1]) * ridge
            regularizer[0, 0] = 0.0
            try:
                weights = np.linalg.solve(gram + regularizer, right)
            except np.linalg.LinAlgError:
                continue
            train_prediction = train_x @ weights
            case_errors = []
            for re_value, u_coeff, p_coeff in validation:
                x = features(u_coeff, np.full(len(u_coeff), 1.0 / re_value), kind) / scale[None]
                case_errors.append(relative(x @ weights, p_coeff))
            report = {
                "kind": kind,
                "ridge": ridge,
                "feature_count": train_x.shape[1],
                "train_relative_l2": relative(train_prediction, train_p),
                "validation_mean_relative_l2": float(np.mean(case_errors)),
                "validation_max_relative_l2": float(np.max(case_errors)),
                "validation_case_relative_l2": case_errors,
            }
            candidate_reports.append(report)
            candidate_assets.append((report, scale, weights))
    eligible = (
        [item for item in candidate_assets if item[0]["kind"] == args.force_kind]
        if args.force_kind else candidate_assets
    )
    best_report, best_scale, best_weights = min(
        eligible, key=lambda item: item[0]["validation_mean_relative_l2"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, kind=np.asarray(best_report["kind"]), ridge=np.asarray(best_report["ridge"]),
        feature_scale=best_scale, weights=best_weights,
        velocity_rank=np.asarray(args.velocity_rank), pressure_rank=np.asarray(args.pressure_rank),
        validation_re=validation_re,
    )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "selected": best_report,
        "candidates": sorted(candidate_reports, key=lambda item: item["validation_mean_relative_l2"]),
        "validation_loaded": True,
        "heldout_loaded": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "selected": best_report, "top5": payload["candidates"][:5]}, indent=2))


if __name__ == "__main__":
    main()
