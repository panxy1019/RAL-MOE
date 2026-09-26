#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import ROOT, case_dir, parse_numeric_table


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def find_force_files(case_path: Path) -> list[Path]:
    patterns = [
        "postProcessing/forceCoeffs*/**/forceCoeffs.dat",
        "postProcessing/forceCoeffs*/**/coefficient.dat",
        "processor0/postProcessing/forceCoeffs*/**/forceCoeffs.dat",
        "processor0/postProcessing/forceCoeffs*/**/coefficient.dat",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(case_path.glob(pattern))
    return sorted(set(files))


def find_probe_files(case_path: Path) -> list[Path]:
    files = list(case_path.glob("postProcessing/probes*/**/U"))
    files.extend(case_path.glob("processor0/postProcessing/probes*/**/U"))
    return sorted(set(files))


def read_force_series(case_path: Path) -> tuple[list[str], np.ndarray]:
    columns: list[str] = []
    rows: list[list[float]] = []
    for path in find_force_files(case_path):
        cols, vals = parse_numeric_table(path)
        if cols:
            columns = cols
        rows.extend(vals)
    if not rows:
        return columns, np.empty((0, 0))
    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    _, idx = np.unique(arr[:, 0], return_index=True)
    idx.sort()
    return columns, arr[idx]


def read_probe_v_series(case_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import re

    rows: list[tuple[float, list[float]]] = []
    vector_re = re.compile(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)"
    )
    for path in find_probe_files(case_path):
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                t = float(line.split(maxsplit=1)[0])
            except (IndexError, ValueError):
                continue
            vecs = vector_re.findall(line)
            if vecs:
                rows.append((t, [float(v[1]) for v in vecs]))
    if not rows:
        return np.asarray([]), np.empty((0, 0))
    rows.sort(key=lambda item: item[0])
    times = np.asarray([item[0] for item in rows], dtype=float)
    values = np.asarray([item[1] for item in rows], dtype=float)
    _, idx = np.unique(times, return_index=True)
    idx.sort()
    return times[idx], values[idx]


def tail_mask(times: np.ndarray, fraction: float) -> np.ndarray:
    if times.size == 0:
        return np.asarray([], dtype=bool)
    start = float(np.min(times) + (1.0 - fraction) * (np.max(times) - np.min(times)))
    return times >= start


def peaks_and_period(times: np.ndarray, signal: np.ndarray) -> tuple[int, float, float, float]:
    if times.size < 5 or signal.size < 5:
        return 0, math.nan, math.nan, math.nan
    y = signal - float(np.mean(signal))
    peaks = []
    for i in range(1, y.size - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(float(times[i]))
    if len(peaks) < 3:
        return len(peaks), math.nan, math.nan, math.nan
    periods = np.diff(np.asarray(peaks[-4:], dtype=float))
    if periods.size == 0:
        return len(peaks), math.nan, math.nan, math.nan
    period = float(np.median(periods))
    cv = float(np.std(periods) / period) if period > 0 else math.nan
    freq = float(1.0 / period) if period > 0 else math.nan
    return len(peaks), period, freq, cv


def finite_ratio(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline) or abs(baseline) < 1e-15:
        return math.nan
    return value / baseline


def refine_one(row: dict[str, str], tail_fraction: float) -> dict[str, Any]:
    re_value = float(row["Re"])
    pmin = float(row.get("pimple_time_min", "nan"))
    pmax = float(row.get("pimple_time_max", "nan"))
    casedir = case_dir(re_value)
    columns, force = read_force_series(casedir)
    probe_t, probe_v = read_probe_v_series(casedir)

    result: dict[str, Any] = {
        "Re": re_value,
        "pimple_time_min": pmin,
        "pimple_time_max": pmax,
        "tail_fraction": tail_fraction,
        "Cd_tail_mean": math.nan,
        "Cd_tail_std": math.nan,
        "Cd_tail_relative_std": math.nan,
        "Cl_tail_mean": math.nan,
        "Cl_tail_std": math.nan,
        "probe_v_tail_std_max": math.nan,
        "Cd_std_ratio_to_Re20": math.nan,
        "Cl_std_ratio_to_Re20": math.nan,
        "probe_v_std_ratio_to_Re20": math.nan,
        "peak_count": 0,
        "estimated_period": math.nan,
        "estimated_frequency": math.nan,
        "period_cv": math.nan,
        "refined_status": "no-diagnostics",
        "periodicity_note": "insufficient_time_for_periodicity",
    }

    if force.size:
        force = force[(force[:, 0] >= pmin - 1e-10) & (force[:, 0] <= pmax + 1e-10)]
        if force.size:
            mask = tail_mask(force[:, 0], tail_fraction)
            tail = force[mask]
            cd_idx = columns.index("Cd") if "Cd" in columns else (2 if force.shape[1] > 2 else None)
            cl_idx = columns.index("Cl") if "Cl" in columns else (3 if force.shape[1] > 3 else None)
            if cd_idx is not None and tail.size:
                cd = tail[:, cd_idx]
                result["Cd_tail_mean"] = float(np.mean(cd))
                result["Cd_tail_std"] = float(np.std(cd))
                if abs(result["Cd_tail_mean"]) > 1e-15:
                    result["Cd_tail_relative_std"] = float(
                        result["Cd_tail_std"] / abs(result["Cd_tail_mean"])
                    )
            if cl_idx is not None and tail.size:
                cl = tail[:, cl_idx]
                result["Cl_tail_mean"] = float(np.mean(cl))
                result["Cl_tail_std"] = float(np.std(cl))
                peak_count, period, freq, cv = peaks_and_period(tail[:, 0], cl)
                result["peak_count"] = peak_count
                result["estimated_period"] = period
                result["estimated_frequency"] = freq
                result["period_cv"] = cv
                if peak_count >= 3:
                    result["periodicity_note"] = "period_check_available"

    if probe_t.size and probe_v.size:
        mask = (probe_t >= pmin - 1e-10) & (probe_t <= pmax + 1e-10)
        probe_t = probe_t[mask]
        probe_v = probe_v[mask]
        if probe_t.size:
            tmask = tail_mask(probe_t, tail_fraction)
            tail_probe = probe_v[tmask]
            if tail_probe.size:
                result["probe_v_tail_std_max"] = float(np.max(np.std(tail_probe, axis=0)))

    return result


def classify(rows: list[dict[str, Any]], baseline_re: float, factor: float) -> None:
    baseline = min(rows, key=lambda item: abs(float(item["Re"]) - baseline_re))
    b_cd = float(baseline["Cd_tail_std"])
    b_cl = float(baseline["Cl_tail_std"])
    b_probe = float(baseline["probe_v_tail_std_max"])

    for row in rows:
        row["Cd_std_ratio_to_Re20"] = finite_ratio(float(row["Cd_tail_std"]), b_cd)
        row["Cl_std_ratio_to_Re20"] = finite_ratio(float(row["Cl_tail_std"]), b_cl)
        row["probe_v_std_ratio_to_Re20"] = finite_ratio(
            float(row["probe_v_tail_std_max"]), b_probe
        )

        cl_std = float(row["Cl_tail_std"])
        probe_std = float(row["probe_v_tail_std_max"])
        peak_count = int(row["peak_count"])
        period_cv = float(row["period_cv"])
        cl_high = math.isfinite(cl_std) and math.isfinite(b_cl) and cl_std > factor * b_cl
        probe_high = (
            math.isfinite(probe_std)
            and math.isfinite(b_probe)
            and probe_std > factor * b_probe
        )

        if peak_count >= 3 and math.isfinite(period_cv) and period_cv < 0.15:
            row["refined_status"] = "periodic-candidate"
            row["periodicity_note"] = "stable_peak_periods_detected"
        elif not cl_high and not probe_high:
            row["refined_status"] = "steady-like"
            if peak_count < 3:
                row["periodicity_note"] = "insufficient_time_for_periodicity"
        else:
            row["refined_status"] = "unsteady-candidate"
            if peak_count < 3:
                row["periodicity_note"] = "insufficient_time_for_periodicity"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    re_values = np.asarray([float(r["Re"]) for r in rows])
    cl_ratio = np.asarray([float(r["Cl_std_ratio_to_Re20"]) for r in rows])
    probe_ratio = np.asarray([float(r["probe_v_std_ratio_to_Re20"]) for r in rows])
    cd_ratio = np.asarray([float(r["Cd_std_ratio_to_Re20"]) for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(re_values, cl_ratio, marker="o", label="Cl tail std / Re20")
    axes[0].plot(re_values, probe_ratio, marker="s", label="probe v tail std / Re20")
    axes[0].axhline(2.5, color="k", linestyle="--", linewidth=1, label="2.5x baseline")
    axes[0].set_ylabel("ratio")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(re_values, cd_ratio, marker="o", label="Cd tail std / Re20")
    axes[1].set_xlabel("Re")
    axes[1].set_ylabel("ratio")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]], tail_fraction: float, factor: float) -> None:
    rows = sorted(rows, key=lambda item: float(item["Re"]))
    unsteady = [int(float(r["Re"])) for r in rows if r["refined_status"] == "unsteady-candidate"]
    periodic = [int(float(r["Re"])) for r in rows if r["refined_status"] == "periodic-candidate"]
    steady = [int(float(r["Re"])) for r in rows if r["refined_status"] == "steady-like"]
    candidates = [re for re in [60, 80, 100, 150] if re in {int(float(r["Re"])) for r in rows}]

    lines = [
        "# Pilot Fast Refined Report",
        "",
        f"Tail fraction: `{tail_fraction}`. Baseline: `Re=20`. Unsteady threshold: `{factor}x` baseline.",
        "",
        "`pilot_fast` is a short forces/probes scan. It can identify preliminary unsteady candidates, but it must not be used as the final regime label for the attractor database. If the time window does not contain one or several shedding periods, it cannot confirm a periodic attractor.",
        "",
        "## Refined Status",
        "",
        "| Re | Cl tail std ratio | probe-v tail std ratio | peaks | period CV | refined status |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {Re:g} | {cl:.3g} | {probe:.3g} | {peaks} | {cv:.3g} | {status} |".format(
                Re=float(row["Re"]),
                cl=float(row["Cl_std_ratio_to_Re20"]),
                probe=float(row["probe_v_std_ratio_to_Re20"]),
                peaks=int(row["peak_count"]),
                cv=float(row["period_cv"]),
                status=row["refined_status"],
            )
        )

    lines += [
        "",
        "## Summary",
        "",
        f"- steady-like: {steady if steady else 'none'}",
        f"- unsteady-candidate: {unsteady if unsteady else 'none'}",
        f"- periodic-candidate: {periodic if periodic else 'none'}",
        "",
        "## Recommended Next Step",
        "",
    ]
    if unsteady and not periodic:
        lines += [
            "The short scan shows elevated unsteady metrics but does not establish stable periodicity. Run a longer forces/probes-only confirmation on a few candidates:",
            "",
            "```bash",
        ]
        for re_value in candidates:
            lines.append(
                f"python3 scripts/01_run_single_re.py --re {re_value} --mode probe_long --nprocs 4 --overwrite"
            )
        lines.append("```")
    else:
        lines.append("Use the refined CSV to choose a smaller set of Re for `probe_long` confirmation before production data generation.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine existing pilot_fast diagnostics without running OpenFOAM.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "manifest" / "pilot_fast_diagnostics.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "manifest" / "pilot_fast_refined_diagnostics.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "manifest" / "pilot_fast_report.md")
    parser.add_argument("--figure", type=Path, default=ROOT / "data" / "logs" / "figures" / "pilot_fast_refined_metrics.png")
    parser.add_argument("--tail-fraction", type=float, default=0.5)
    parser.add_argument("--baseline-re", type=float, default=20.0)
    parser.add_argument("--threshold-factor", type=float, default=2.5)
    args = parser.parse_args()

    source_rows = load_rows(args.input)
    if not source_rows:
        raise SystemExit(f"No rows found in {args.input}")
    rows = [refine_one(row, args.tail_fraction) for row in source_rows]
    rows.sort(key=lambda item: float(item["Re"]))
    classify(rows, args.baseline_re, args.threshold_factor)
    write_csv(args.output, rows)
    write_plot(args.figure, rows)
    write_report(args.report, rows, args.tail_fraction, args.threshold_factor)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
