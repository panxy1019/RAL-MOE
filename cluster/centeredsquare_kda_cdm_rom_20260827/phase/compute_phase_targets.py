#!/usr/bin/env python3
"""Compute train-only phase/amplitude defects and data-derived forgetting rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def safe_scale(values):
    value = float(np.std(values))
    return value if value > 1e-12 else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-defect-root", type=Path, required=True)
    parser.add_argument("--phase-pair", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    pair = json.loads(args.phase_pair.read_text())
    first, second = int(pair["phase_mode_p"]), int(pair["phase_mode_q"])
    cases, all_columns = [], {name: [] for name in ("rho", "omega_fvm", "rho_dot_fvm", "delta_omega", "delta_rho_dot")}
    for path in sorted(item for item in args.train_defect_root.glob("Re*") if item.is_dir()):
        states = np.load(path / "a_cfd.npy")
        adot = np.load(path / "adot_cfd_projected.npy")
        fvm = np.load(path / "fvm_rhs_tensor.npy")
        times = np.load(path / "time.npy")
        re_value = float(json.loads((path / "Re.json").read_text())["Re"])
        ap, aq = states[:, first], states[:, second]
        rho = np.sqrt(ap**2 + aq**2)
        epsilon_squared = max(1e-14, 1e-8 * float(np.median(rho**2)))
        epsilon = np.sqrt(epsilon_squared)
        omega_cfd = (ap * adot[:, second] - aq * adot[:, first]) / (rho**2 + epsilon_squared)
        omega_fvm = (ap * fvm[:, second] - aq * fvm[:, first]) / (rho**2 + epsilon_squared)
        rho_dot_cfd = (ap * adot[:, first] + aq * adot[:, second]) / (rho + epsilon)
        rho_dot_fvm = (ap * fvm[:, first] + aq * fvm[:, second]) / (rho + epsilon)
        values = {
            "rho": rho, "theta": np.unwrap(np.arctan2(aq, ap)),
            "omega_cfd": omega_cfd, "omega_fvm": omega_fvm,
            "rho_dot_cfd": rho_dot_cfd, "rho_dot_fvm": rho_dot_fvm,
            "delta_omega": omega_cfd - omega_fvm, "delta_rho_dot": rho_dot_cfd - rho_dot_fvm,
        }
        cases.append((path.name, re_value, times, values))
        for name in all_columns:
            all_columns[name].append(values[name])

    concatenated = {name: np.concatenate(parts) for name, parts in all_columns.items()}
    re_values = np.asarray([item[1] for item in cases])
    normalization = {
        name: {"mean": float(np.mean(values)), "scale": safe_scale(values)}
        for name, values in concatenated.items() if name in {"rho", "omega_fvm", "rho_dot_fvm"}
    }
    normalization["Re"] = {"mean": float(np.mean(re_values)), "scale": safe_scale(re_values)}

    autocorrelations = []
    for _, _, _, values in cases:
        signal = values["delta_omega"] - np.mean(values["delta_omega"])
        correlation = np.correlate(signal, signal, mode="full")[len(signal) - 1:]
        correlation /= max(correlation[0], 1e-30)
        autocorrelations.append(correlation)
    common_length = min(map(len, autocorrelations))
    mean_acf = np.mean([item[:common_length] for item in autocorrelations], axis=0)
    sample_step = float(np.median(np.concatenate([np.diff(item[2]) for item in cases])))
    below = np.flatnonzero(mean_acf <= np.exp(-1.0))
    tau = float((below[0] if len(below) else common_length - 1) * sample_step)
    tau = max(tau, sample_step)
    gamma_candidates = sorted(set(float(np.clip(factor / tau, 1e-4, 2.0)) for factor in (0.5, 1.0, 2.0)))

    args.output_root.mkdir(parents=True, exist_ok=False)
    for tag, re_value, times, values in cases:
        np.savez_compressed(args.output_root / f"{tag}_phase_targets.npz", Re=re_value, time=times, **values)
    np.save(args.output_root / "delta_omega_mean_acf.npy", mean_acf)
    payload = {
        "schema_version": 1, "selection_scope": "periodic_train_only",
        "phase_mode_p": first, "phase_mode_q": second, "train_case_count": len(cases),
        "delta_omega_rms": float(np.sqrt(np.mean(concatenated["delta_omega"] ** 2))),
        "delta_rho_dot_rms": float(np.sqrt(np.mean(concatenated["delta_rho_dot"] ** 2))),
        "acf_one_over_e_time": tau, "gamma_candidates": gamma_candidates,
        "eta_candidates": [0.1, 0.25, 0.5, 1.0],
        "ridge_candidates": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        "normalization": normalization, "validation_loaded": False, "heldout_loaded": False,
    }
    (args.output_root / "PHASE_TARGET_SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
