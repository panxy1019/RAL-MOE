#!/usr/bin/env python3
"""Finalize completed rollout CSVs when optional plotting dependencies are absent."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def match(rows, split, boundary, horizon, method):
    found = [r for r in rows if r["split"] == split and r["boundary"] == boundary
             and int(r["horizon"]) == horizon and r["method"] == method]
    if len(found) != 1:
        raise RuntimeError((split, boundary, horizon, method, len(found)))
    return found[0]


def svg_line_chart(path: Path, title: str, panels: list[tuple[str, list[tuple[str, list[tuple[float, float]]]]]],
                   x_label: str, y_label: str) -> None:
    width, height = 1100, 430
    panel_width, top, bottom = 500, 55, 365
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="550" y="25" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">{title}</text>']
    for panel_index, (panel_title, series) in enumerate(panels):
        left = 65 + panel_index * 535
        all_points = [point for _, values in series for point in values]
        xs, ys = [p[0] for p in all_points], [p[1] for p in all_points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = 0.0, max(ys) * 1.08 if max(ys) > 0 else 1.0
        sx = lambda value: left + (value - xmin) / max(xmax - xmin, 1e-12) * panel_width
        sy = lambda value: bottom - (value - ymin) / max(ymax - ymin, 1e-12) * (bottom - top)
        content.extend([
            f'<line x1="{left}" y1="{bottom}" x2="{left+panel_width}" y2="{bottom}" stroke="#555"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#555"/>',
            f'<text x="{left+panel_width/2}" y="48" text-anchor="middle" font-family="sans-serif" font-size="15">{panel_title}</text>',
            f'<text x="{left+panel_width/2}" y="415" text-anchor="middle" font-family="sans-serif" font-size="12">{x_label}</text>',
            f'<text x="15" y="{(top+bottom)/2}" transform="rotate(-90 15 {(top+bottom)/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{y_label}</text>',
        ])
        for tick in range(5):
            value = ymax * tick / 4
            y = sy(value)
            content.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_width}" y2="{y:.1f}" stroke="#ddd"/>')
            content.append(f'<text x="{left-7}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{value:.3f}</text>')
        for series_index, (label, values) in enumerate(series):
            points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
            color = colors[series_index % len(colors)]
            content.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y in values:
                content.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{color}"/>')
            ly = top + 15 + 16 * series_index
            content.append(f'<line x1="{left+panel_width-120}" y1="{ly}" x2="{left+panel_width-100}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
            content.append(f'<text x="{left+panel_width-95}" y="{ly+4}" font-family="sans-serif" font-size="10">{label}</text>')
    content.append("</svg>")
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    output = args.result_dir.resolve()
    aggregate = read_csv(output / "ROLLOUT_AGGREGATE.csv")
    details = read_csv(output / "ROLLOUT_BY_RE.csv")
    growth = read_csv(output / "ERROR_GROWTH_CURVES.csv")
    weights = read_csv(output / "WEIGHT_DIAGNOSTICS.csv")
    seal_path = output / "FINAL_TEST_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if len(aggregate) != 144 or len(details) != 396 or len(growth) != 3288 or len(weights) != 22:
        raise RuntimeError({"aggregate": len(aggregate), "details": len(details), "growth": len(growth), "weights": len(weights)})
    stability = {}
    for boundary, source_name in (("SH", "Steady"), ("HP", "Periodic")):
        for split in ("validation", "final_test"):
            source = match(aggregate, split, boundary, 56, "source_only")
            target = match(aggregate, split, boundary, 56, "hopf_only")
            stability[f"{boundary}_{split}"] = {
                "source_expert": source_name, "target_expert": "Hopf",
                "windows": int(source["windows"]), "horizon": 56,
                "source_finite_fraction": float(source["finite_fraction"]),
                "target_finite_fraction": float(target["finite_fraction"]),
                "source_divergent_windows": int(source["divergent_windows"]),
                "target_divergent_windows": int(target["divergent_windows"]),
                "source_worst_joint_relative_error": float(source["joint_worst_point"]),
                "target_worst_joint_relative_error": float(target["joint_worst_point"]),
            }
    numeric_aggregate = []
    for row in aggregate:
        converted = dict(row)
        for key in ("horizon", "windows", "divergent_windows"):
            converted[key] = int(converted[key])
        for key in ("velocity_mean", "pressure_mean", "joint_mean", "joint_terminal", "joint_worst_point",
                    "joint_worst_window_mean", "finite_fraction", "alpha_mean", "alpha_min", "alpha_max"):
            converted[key] = float(converted[key])
        numeric_aggregate.append(converted)
    evaluated_re = {}
    for split in ("validation", "final_test"):
        for boundary in ("SH", "HP"):
            evaluated_re[f"{boundary}_{split}"] = sorted({float(r["Re"]) for r in details
                                                           if r["split"] == split and r["boundary"] == boundary})
    elapsed = max(0.0, time.time() - float(seal["started_unix"]))
    summary = {
        "schema_version": 1, "status": "FROZEN_FULL_ROLLOUT_COMPLETE",
        "protocol": {
            "splits": ["validation", "final_test"], "boundaries": ["SH", "HP"],
            "horizons": [1, 8, 16, 24, 32, 56], "windows_per_re": 64,
            "window_sampling": "64 evenly spaced legal K56 starts per Re; shorter horizons are prefixes of the identical trajectories",
            "fusion_weight_scope": "one scalar per trajectory fixed from the initial 3-frame history",
            "fusion_feedback": False, "final_test_used_for_selection": False,
            "training_after_final_test_unseal": False, "divergence_threshold_relative_component_error": 20.0,
            "oracle_grid_spacing": 0.0005,
        },
        "checkpoint_sha256": seal["checkpoint_sha256"],
        "evaluated_reynolds": evaluated_re, "stability": stability,
        "aggregate": numeric_aggregate, "elapsed_seconds": elapsed,
        "postprocessing_note": "Core CSVs completed; matplotlib was absent in the cluster environment. Summary and dependency-free SVG plots were finalized from those immutable CSVs without rerunning or changing weights.",
    }
    summary_path = output / "FULL_ROLLOUT_SUMMARY.json"
    atomic_json(summary_path, summary)
    methods = ("source_only", "hopf_only", "E2_pair_blend", "T2-C", "per_window_oracle")
    for split in ("validation", "final_test"):
        panels = []
        for boundary in ("SH", "HP"):
            series = []
            for method in methods:
                rows = sorted([r for r in aggregate if r["split"] == split and r["boundary"] == boundary and r["method"] == method], key=lambda r: int(r["horizon"]))
                series.append((method, [(int(r["horizon"]), 100 * float(r["joint_mean"])) for r in rows]))
            panels.append((boundary, series))
        svg_line_chart(output / f"{split}_error_by_horizon.svg", f"{split}: joint error vs horizon", panels, "Horizon K", "Joint error (%)")
    panels = []
    for boundary in ("SH", "HP"):
        series = []
        for split in ("validation", "final_test"):
            rows = sorted([r for r in weights if r["boundary"] == boundary and r["split"] == split and r["method"] == "T2-C"], key=lambda r: float(r["Re"]))
            series.append((split, [(float(r["Re"]), float(r["alpha_mean"])) for r in rows]))
        panels.append((boundary, series))
    svg_line_chart(output / "t2c_weight_vs_re.svg", "T2-C source weight vs Reynolds number", panels, "Re", "Source weight alpha")
    for split in ("validation", "final_test"):
        panels = []
        for boundary in ("SH", "HP"):
            series = []
            for method in ("source_only", "E2_pair_blend", "T2-C", "per_window_oracle"):
                rows = sorted([r for r in growth if r["split"] == split and r["boundary"] == boundary
                               and int(r["horizon"]) == 56 and r["method"] == method], key=lambda r: int(r["step"]))
                series.append((method, [(int(r["step"]), 100 * float(r["joint_mean"])) for r in rows]))
            panels.append((boundary, series))
        svg_line_chart(output / f"{split}_K56_error_growth.svg", f"{split}: K56 error growth", panels, "Autonomous step", "Joint error (%)")
    seal.update({"status": "FINAL_TEST_UNSEAL_COMPLETE", "completed_unix": time.time(),
                 "training_after_unseal": False, "result_sha256": sha256(summary_path),
                 "postprocessing_recovered_from_completed_csv": True})
    atomic_json(seal_path, seal)
    print(json.dumps({"summary": str(summary_path), "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
