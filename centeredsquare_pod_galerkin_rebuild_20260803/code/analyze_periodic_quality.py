#!/usr/bin/env python3
"""Analyze periodic-ROM amplitude, frequency, and phase-aligned errors."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--role", choices=["validation", "heldout"], required=True)
args = parser.parse_args()
ROOT = args.input_dir
reports = []
for path_string in sorted(glob.glob(str(ROOT / "trajectory_*.npz"))):
    with np.load(path_string) as source:
        re_value = float(source["Re"])
        times = np.asarray(source["times"], dtype=np.float64)
        truth = np.asarray(source["truth_velocity"], dtype=np.float64)
        prediction = np.asarray(source["predicted_velocity"], dtype=np.float64)
    ids = np.flatnonzero(times >= 100.0)
    truth = truth[ids]
    prediction = prediction[ids]
    truth_fluctuation = truth - truth.mean(axis=0, keepdims=True)
    prediction_fluctuation = prediction - prediction.mean(axis=0, keepdims=True)
    truth_amplitude = float(np.sqrt(np.mean(np.sum(truth_fluctuation**2, axis=1))))
    prediction_amplitude = float(np.sqrt(np.mean(np.sum(prediction_fluctuation**2, axis=1))))
    frequencies = np.fft.rfftfreq(len(ids), d=float(times[1] - times[0]))
    truth_spectrum = np.sum(np.abs(np.fft.rfft(truth_fluctuation, axis=0)) ** 2, axis=1)
    prediction_spectrum = np.sum(np.abs(np.fft.rfft(prediction_fluctuation, axis=0)) ** 2, axis=1)
    truth_index = int(np.argmax(truth_spectrum[1:]) + 1)
    prediction_index = int(np.argmax(prediction_spectrum[1:]) + 1)
    raw_error = float(np.linalg.norm(prediction_fluctuation - truth_fluctuation) / np.linalg.norm(truth_fluctuation))
    shifts = []
    for shift in range(len(ids)):
        shifted = np.roll(prediction_fluctuation, shift, axis=0)
        shifts.append(float(np.linalg.norm(shifted - truth_fluctuation) / np.linalg.norm(truth_fluctuation)))
    reports.append({
        "Re": re_value,
        "tail_interval": [float(times[ids[0]]), float(times[ids[-1]])],
        "truth_rms_amplitude": truth_amplitude,
        "prediction_rms_amplitude": prediction_amplitude,
        "amplitude_relative_error": abs(prediction_amplitude - truth_amplitude) / truth_amplitude,
        "truth_dominant_frequency": float(frequencies[truth_index]),
        "prediction_dominant_frequency": float(frequencies[prediction_index]),
        "dominant_frequency_relative_error": abs(frequencies[prediction_index] - frequencies[truth_index]) / frequencies[truth_index],
        "tail_fluctuation_relative_l2": raw_error,
        "best_integer_shift_relative_l2": min(shifts),
        "best_integer_shift_snapshots": int(np.argmin(shifts)),
    })
payload = {
    "schema_version": 1,
    "cases": reports,
    "aggregate": {
        "mean_amplitude_relative_error": float(np.mean([item["amplitude_relative_error"] for item in reports])),
        "mean_dominant_frequency_relative_error": float(np.mean([item["dominant_frequency_relative_error"] for item in reports])),
        "mean_tail_fluctuation_relative_l2": float(np.mean([item["tail_fluctuation_relative_l2"] for item in reports])),
        "mean_best_integer_shift_relative_l2": float(np.mean([item["best_integer_shift_relative_l2"] for item in reports])),
    },
    "role": args.role,
    "validation_loaded": args.role == "validation",
    "heldout_loaded": args.role == "heldout",
}
(ROOT / f"{args.role.upper()}_PERIODIC_QUALITY.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
