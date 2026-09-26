#!/usr/bin/env python3
"""Independent, compact recomputation of the frozen Global-MoE result."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--direct-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = read(args.steps)
    target = read(args.target)
    reported = read(args.summary)
    target_windows: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for row in target:
        if int(row["step"]) == 1:
            target_windows[row["regime"]].append(
                (float(row["Re"]), float(row["start_time"]), row["window_id"])
            )
    canonical = []
    for row in raw:
        candidates = [
            item
            for item in target_windows[row["regime"]]
            if abs(item[0] - float(row["Re"])) < 2e-5
            and abs(item[1] - float(row["start_time"])) < 1e-3
        ]
        if len(candidates) != 1:
            raise RuntimeError((row, candidates))
        canonical.append(
            (
                row["regime"],
                candidates[0][2],
                float(row["Re"]),
                int(row["step"]),
                float(row["Eu"]),
                float(row["Ep"]),
                row["finite"].lower() == "true",
            )
        )
    target_keys = {(row["regime"], row["window_id"], int(row["step"])) for row in target}
    actual_keys = {(row[0], row[1], row[3]) for row in canonical}
    if actual_keys != target_keys:
        raise RuntimeError("target/result step keys differ")
    if not all(row[6] and math.isfinite(row[4]) and math.isfinite(row[5]) for row in canonical):
        raise RuntimeError("non-finite row")

    windows: dict[tuple[str, str, float], list[tuple[int, float, float]]] = defaultdict(list)
    for regime, window, re_value, step, eu, ep, _ in canonical:
        windows[(regime, window, re_value)].append((step, eu, ep))
    independent = []
    for regime in ("Hopf", "Periodic"):
        for horizon in (24, 48):
            re_values: dict[float, list[tuple[float, float]]] = defaultdict(list)
            for (row_regime, _, re_value), values in windows.items():
                if row_regime != regime:
                    continue
                ordered = sorted(values)
                if [value[0] for value in ordered] != list(range(1, 49)):
                    raise RuntimeError("non-contiguous window")
                re_values[re_value].append(
                    (
                        float(np.mean([value[1] for value in ordered[:horizon]])),
                        float(np.mean([value[2] for value in ordered[:horizon]])),
                    )
                )
            eu = float(np.mean([np.mean([value[0] for value in values]) for values in re_values.values()]))
            ep = float(np.mean([np.mean([value[1] for value in values]) for values in re_values.values()]))
            independent.append({"regime": regime, "K": horizon, "Eu": eu, "Ep": ep, "joint": eu + ep})

    differences = []
    for row in independent:
        match = [value for value in reported if value["regime"] == row["regime"] and int(value["K"]) == row["K"]]
        if len(match) != 1:
            raise RuntimeError((row, match))
        for key, report_key in (("Eu", "Eu_mean"), ("Ep", "Ep_mean"), ("joint", "joint_mean")):
            differences.append(abs(row[key] - float(match[0][report_key])))
    direct = json.loads(args.direct_audit.read_text(encoding="utf-8"))
    direct_max = max(
        max(float(row["Eu_abs_difference"]), float(row["Ep_abs_difference"])) for row in direct
    )
    result = {
        "passed": max(differences) < 1e-12 and direct_max < 1e-9,
        "target_step_keys": len(target_keys),
        "actual_step_keys": len(actual_keys),
        "unique_windows": len(windows),
        "finite_rows": len(canonical),
        "independent_summary": independent,
        "max_difference_vs_primary_aggregate": max(differences),
        "direct_decode_checks": len(direct),
        "max_cross_basis_vs_direct_difference_percentage_points": direct_max,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(result)


if __name__ == "__main__":
    main()
