#!/usr/bin/env python3
"""Summarize absolute coefficient errors from already-frozen trajectories."""

import glob
import json
import pathlib

import numpy as np

from evaluate_cdm_grom import interpolation_weights, interpolate, pressure_from_velocity


ROOT = pathlib.Path("/home/ray/Desktop/centeredSquare/cdm_grom_hopf_r64_v1")
with np.load(
    ROOT / "cdm_operator_tailref/centeredsquare_hopf_cdm_operator_r11_u53.npz",
    allow_pickle=False,
) as source:
    raw = {key: np.asarray(source[key]) for key in source.files}

results = {}
for role, directory in {
    "validation": ROOT / "evaluation/validation_tailref",
    "heldout": ROOT / "evaluation/heldout",
}.items():
    rows = []
    for path_string in sorted(glob.glob(str(directory / "trajectory_*.npz"))):
        with np.load(path_string, allow_pickle=False) as trajectory:
            re_value = float(trajectory["Re"])
            truth_u = np.asarray(trajectory["truth_velocity"])[3:]
            truth_p = np.asarray(trajectory["truth_pressure"])[3:]
            cdm_u = np.asarray(trajectory["cdm_velocity"])[3:]
            baseline_u = np.asarray(trajectory["baseline_velocity"])[3:]
        bracket = interpolation_weights(raw["Re_nodes"], re_value)
        operator = {
            "pressure_c": interpolate(raw["pressure_c"], bracket),
            "pressure_A": interpolate(raw["pressure_A"], bracket),
            "pressure_H": raw["pressure_H"],
        }
        cdm_p = pressure_from_velocity(operator, cdm_u)
        baseline_p = pressure_from_velocity(operator, baseline_u)

        def rms(array):
            return float(np.sqrt(np.mean(array * array)))

        rows.append(
            {
                "Re": re_value,
                "truth_velocity_coefficient_rms": rms(truth_u),
                "cdm_velocity_coefficient_rmse": rms(cdm_u - truth_u),
                "baseline_velocity_coefficient_rmse": rms(baseline_u - truth_u),
                "truth_pressure_coefficient_rms": rms(truth_p),
                "cdm_pressure_coefficient_rmse": rms(cdm_p - truth_p),
                "baseline_pressure_coefficient_rmse": rms(baseline_p - truth_p),
                "truth_final_velocity_norm": float(np.linalg.norm(truth_u[-1])),
                "cdm_final_velocity_norm": float(np.linalg.norm(cdm_u[-1])),
                "baseline_final_velocity_norm": float(np.linalg.norm(baseline_u[-1])),
                "cdm_max_velocity_norm": float(np.max(np.linalg.norm(cdm_u, axis=1))),
            }
        )
    results[role] = rows
print(json.dumps(results, indent=2))
