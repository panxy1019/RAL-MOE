#!/usr/bin/env python3
"""Fit train-only KDA phase-memory readouts for every preregistered candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.continuous_delta_memory import memory_readout, memory_rhs
from memory.features import phase_key_query, phase_value


def scaled(value, statistics):
    return (value - statistics["mean"]) / statistics["scale"]


def feature_at(theta, rho, omega, rho_dot, re_value, normalization):
    rho_scaled = scaled(rho, normalization["rho"])
    key = phase_key_query(theta, rho_scaled, scaled(re_value, normalization["Re"]))
    value = phase_value(
        scaled(omega, normalization["omega_fvm"]),
        scaled(rho_dot, normalization["rho_dot_fvm"]), rho_scaled,
    )
    return key, value


def teacher_force(theta, rho, omega, rho_dot, times, re_value, normalization, gamma, eta, max_step):
    memory = np.zeros((7, 4), dtype=np.float64)
    memories = np.empty((len(theta), 7, 4), dtype=np.float64)
    keys = np.empty((len(theta), 7), dtype=np.float64)
    values = np.empty((len(theta), 4), dtype=np.float64)
    readouts = np.empty((len(theta), 4), dtype=np.float64)
    for index in range(len(theta)):
        key, value = feature_at(theta[index], rho[index], omega[index], rho_dot[index], re_value, normalization)
        memories[index], keys[index], values[index] = memory, key, value
        readouts[index] = memory_readout(memory, key)
        if index == len(theta) - 1:
            continue
        duration = float(times[index + 1] - times[index])
        count = max(1, int(np.ceil(duration / max_step)))
        step = duration / count
        for substep in range(count):
            alpha = substep / count

            def derivative(current, local_fraction):
                fraction = alpha + local_fraction / count
                interpolate = lambda array: array[index] + fraction * (array[index + 1] - array[index])
                local_key, local_value = feature_at(
                    interpolate(theta), interpolate(rho), interpolate(omega), interpolate(rho_dot),
                    re_value, normalization,
                )
                return memory_rhs(current, local_key, local_value, gamma, eta)

            k1 = derivative(memory, 0.0)
            k2 = derivative(memory + 0.5 * step * k1, 0.5)
            k3 = derivative(memory + 0.5 * step * k2, 0.5)
            k4 = derivative(memory + step * k3, 1.0)
            memory += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return memories, readouts, keys, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-defect-root", type=Path, required=True)
    parser.add_argument("--phase-target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-step", type=float, default=0.05)
    args = parser.parse_args()
    summary = json.loads((args.phase_target_root / "PHASE_TARGET_SUMMARY.json").read_text())
    pair = (int(summary["phase_mode_p"]), int(summary["phase_mode_q"]))
    normalization = summary["normalization"]
    case_data = []
    for phase_path in sorted(args.phase_target_root.glob("Re*_phase_targets.npz")):
        tag = phase_path.name.removesuffix("_phase_targets.npz")
        with np.load(phase_path, allow_pickle=False) as source:
            target = np.asarray(source["delta_omega"], dtype=np.float64)
            times = np.asarray(source["time"], dtype=np.float64)
            re_value = float(source["Re"])
            theta = np.asarray(source["theta"], dtype=np.float64)
            rho = np.asarray(source["rho"], dtype=np.float64)
            omega = np.asarray(source["omega_fvm"], dtype=np.float64)
            rho_dot = np.asarray(source["rho_dot_fvm"], dtype=np.float64)
        case_data.append((tag, re_value, times, theta, rho, omega, rho_dot, target))

    args.output_root.mkdir(parents=True, exist_ok=False)
    markov_design, markov_target = [], []
    for _, re_value, _, theta, rho, _, _, target in case_data:
        rho_scaled = (rho - normalization["rho"]["mean"]) / normalization["rho"]["scale"]
        re_scaled = scaled(re_value, normalization["Re"])
        markov_design.append(np.column_stack([
            np.ones(len(theta)), np.cos(theta), np.sin(theta), np.cos(2.0 * theta),
            np.sin(2.0 * theta), rho_scaled, np.full(len(theta), re_scaled),
        ]))
        markov_target.append(target)
    markov_design = np.concatenate(markov_design)
    markov_target = np.concatenate(markov_target)
    markov_coefficients, markov_reports = [], []
    for ridge in summary["ridge_candidates"]:
        coefficient = np.linalg.solve(
            markov_design.T @ markov_design + ridge * np.eye(markov_design.shape[1]),
            markov_design.T @ markov_target,
        )
        prediction = markov_design @ coefficient
        markov_coefficients.append(coefficient)
        markov_reports.append({
            "ridge": ridge, "train_rmse": float(np.sqrt(np.mean((prediction - markov_target) ** 2))),
            "coefficient_norm": float(np.linalg.norm(coefficient)),
        })
    np.savez_compressed(
        args.output_root / "markov_phase_fit.npz",
        ridge_candidates=np.asarray(summary["ridge_candidates"]),
        readout_coefficients=np.asarray(markov_coefficients), train_reports=json.dumps(markov_reports),
    )
    candidate_reports = []
    for gamma in summary["gamma_candidates"]:
        for eta in summary["eta_candidates"]:
            all_memory, all_readout, all_key, all_value, all_target, all_tag = [], [], [], [], [], []
            for tag, re_value, times, theta, rho, omega, rho_dot, target in case_data:
                memory, readout, key, value = teacher_force(
                    theta, rho, omega, rho_dot, times, re_value, normalization, gamma, eta, args.max_step,
                )
                all_memory.append(memory); all_readout.append(readout); all_key.append(key); all_value.append(value)
                all_target.append(target); all_tag.extend([tag] * len(target))
            memories = np.concatenate(all_memory)
            design = np.concatenate(all_readout)
            keys = np.concatenate(all_key)
            values = np.concatenate(all_value)
            target = np.concatenate(all_target)
            readout_coefficients, ridge_reports = [], []
            for ridge in summary["ridge_candidates"]:
                coefficient = np.linalg.solve(design.T @ design + ridge * np.eye(4), design.T @ target)
                prediction = design @ coefficient
                readout_coefficients.append(coefficient)
                ridge_reports.append({
                    "ridge": ridge, "train_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
                    "coefficient_norm": float(np.linalg.norm(coefficient)),
                })
            name = f"gamma_{gamma:.8g}_eta_{eta:.8g}".replace(".", "p")
            np.savez_compressed(
                args.output_root / f"{name}.npz", gamma=gamma, eta=eta,
                ridge_candidates=np.asarray(summary["ridge_candidates"]),
                readout_coefficients=np.asarray(readout_coefficients), memory_state_S=memories,
                memory_readout_m=design, key=keys, query=keys, value=values,
                delta_omega_target=target, case_tags=np.asarray(all_tag),
            )
            candidate_reports.append({"gamma": gamma, "eta": eta, "asset": f"{name}.npz", "ridge_reports": ridge_reports})
    payload = {
        "schema_version": 1, "selection_scope": "periodic_train_only", "phase_pair": list(pair),
        "candidate_count": len(candidate_reports), "candidates": candidate_reports,
        "markov_phase_ridge_reports": markov_reports,
        "initialization": "zero_at_each_train_trajectory_start",
        "validation_loaded": False, "heldout_loaded": False,
    }
    (args.output_root / "TEACHER_MEMORY_FIT.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
