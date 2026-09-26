#!/usr/bin/env python3
"""Freeze a train-only oscillatory POD pair using deterministic spectral diagnostics."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def wrap(value):
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def case_diagnostic(states, times, first, second, mode_limit):
    sample_step = float(np.median(np.diff(times)))
    x = states[:, first] - np.mean(states[:, first])
    y = states[:, second] - np.mean(states[:, second])
    sx, sy = float(np.std(x)), float(np.std(y))
    if min(sx, sy) < 1e-12:
        return None
    xn, yn = x / sx, y / sy
    window = np.hanning(len(xn))
    xf, yf = np.fft.rfft(xn * window), np.fft.rfft(yn * window)
    frequencies = np.fft.rfftfreq(len(xn), sample_step)
    px, py = abs(xf) ** 2, abs(yf) ** 2
    ix, iy = int(np.argmax(px[1:]) + 1), int(np.argmax(py[1:]) + 1)
    common = ix if px[ix] + py[ix] >= px[iy] + py[iy] else iy
    phase_lag = float(wrap(np.angle(yf[common] * np.conj(xf[common]))))
    quadrature_error = float(abs(abs(phase_lag) - 0.5 * np.pi))
    frequency_mismatch = float(abs(frequencies[ix] - frequencies[iy]) / max(frequencies[ix], frequencies[iy], 1e-14))
    correlation = float(np.corrcoef(xn, yn)[0, 1])
    theta = np.unwrap(np.arctan2(yn, xn))
    direction_fraction = float(max(np.mean(np.diff(theta) > 0), np.mean(np.diff(theta) < 0)))
    purity = float(np.sqrt(
        px[ix] / max(np.sum(px[1:]), 1e-30) * py[iy] / max(np.sum(py[1:]), 1e-30)
    ))
    energy_share = float((sx * sx + sy * sy) / max(np.sum(np.var(states[:, :mode_limit], axis=0)), 1e-30))
    local_score = (
        np.exp(-(frequency_mismatch / 0.15) ** 2)
        * np.exp(-(quadrature_error / (np.pi / 6.0)) ** 2)
        * np.exp(-(correlation / 0.30) ** 2)
        * direction_fraction * purity * np.sqrt(energy_share)
    )
    return {
        "frequency_first": float(frequencies[ix]), "frequency_second": float(frequencies[iy]),
        "common_frequency": float(frequencies[common]), "phase_lag": phase_lag,
        "quadrature_error": quadrature_error, "frequency_mismatch": frequency_mismatch,
        "correlation": correlation, "direction_fraction": direction_fraction,
        "spectral_purity": purity, "energy_share": energy_share, "local_score": float(local_score),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-defect-root", type=Path, required=True)
    parser.add_argument("--max-mode", type=int, default=12)
    parser.add_argument("--min-re", type=float, default=110.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = sorted(path for path in args.train_defect_root.glob("Re*") if path.is_dir())
    if not cases:
        raise FileNotFoundError("no train defect cases")
    trajectories = []
    for path in cases:
        re_value = float(json.loads((path / "Re.json").read_text())["Re"])
        if re_value >= args.min_re:
            trajectories.append((path.name, np.load(path / "a_cfd.npy"), np.load(path / "time.npy")))
    if len(trajectories) < 3:
        raise RuntimeError("fewer than three sufficiently periodic train cases")
    reports = []
    for first, second in combinations(range(args.max_mode), 2):
        diagnostics = []
        for tag, states, times in trajectories:
            item = case_diagnostic(states, times, first, second, args.max_mode)
            if item is not None:
                diagnostics.append({"tag": tag, **item})
        frequencies = np.asarray([item["common_frequency"] for item in diagnostics])
        stability = float(np.std(frequencies) / max(np.mean(frequencies), 1e-14))
        aggregate = float(np.median([item["local_score"] for item in diagnostics]) * np.exp(-(stability / 0.10) ** 2))
        reports.append({
            "phase_mode_p": first, "phase_mode_q": second, "score": aggregate,
            "frequency_cv_across_re": stability, "case_diagnostics": diagnostics,
        })
    reports.sort(key=lambda item: item["score"], reverse=True)
    payload = {
        "schema_version": 1, "selection_scope": "periodic_train_only",
        "train_case_count": len(cases), "selection_case_count": len(trajectories),
        "selection_min_re": args.min_re, "max_mode_exclusive": args.max_mode,
        "phase_mode_p": reports[0]["phase_mode_p"], "phase_mode_q": reports[0]["phase_mode_q"],
        "winning_score": reports[0]["score"], "top_candidates": reports[:10],
        "validation_loaded": False, "heldout_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "top_candidates"}, indent=2))


if __name__ == "__main__":
    main()
