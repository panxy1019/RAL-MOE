#!/usr/bin/env python3
"""Validate and aggregate Global-MoE per-step errors using the frozen protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def canonical(row: dict[str, str], default_regime: str, default_seed: str) -> dict[str, object]:
    regime = row.get("regime") or default_regime
    seed = row.get("seed") or default_seed
    window = row.get("window_id") or row.get("start") or row.get("start_global_index")
    if not regime or not window:
        raise ValueError(f"row lacks regime/window identity: {row}")
    eu_key = "Eu" if "Eu" in row else "Eu_percent"
    ep_key = "Ep" if "Ep" in row else "Ep_percent"
    finite = truthy(row.get("finite", "true"))
    eu = float(row[eu_key])
    ep = float(row[ep_key])
    finite = finite and math.isfinite(eu) and math.isfinite(ep)
    return {
        "regime": regime,
        "seed": seed,
        "Re": float(row["Re"]),
        "window_id": str(window),
        "start_time": row.get("start_time", ""),
        "step": int(row["step"]),
        "physical_time": row.get("physical_time", row.get("time", "")),
        "Eu": eu,
        "Ep": ep,
        "finite": finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-windows", type=Path)
    parser.add_argument("--default-regime", default="")
    parser.add_argument("--default-seed", default="single_checkpoint")
    args = parser.parse_args()

    rows = [canonical(row, args.default_regime, args.default_seed) for row in read_csv(args.input)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected: set[tuple[str, str, int]] | None = None
    if args.target_windows:
        target_rows = read_csv(args.target_windows)
        target_windows: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for target in target_rows:
            if int(target["step"]) == 1:
                target_windows[target["regime"]].append(
                    (float(target["Re"]), float(target["start_time"]), target["window_id"])
                )
        # Evaluators may use an internal window identifier.  Reconcile it to
        # the frozen manifest only through the unique physical key
        # (regime, Re, start_time), then require exact step-key equality.
        for row in rows:
            candidates = [
                target
                for target in target_windows[str(row["regime"])]
                if abs(target[0] - float(row["Re"])) < 2e-5
                and abs(target[1] - float(row["start_time"])) < 1e-3
            ]
            if len(candidates) != 1:
                raise RuntimeError(f"cannot uniquely map result window: {row}; candidates={candidates}")
            row["window_id"] = candidates[0][2]
        expected = {(row["regime"], row["window_id"], int(row["step"])) for row in target_rows}
        actual = {(str(r["regime"]), str(r["window_id"]), int(r["step"])) for r in rows}
        if actual != expected:
            missing = sorted(expected - actual)[:20]
            extra = sorted(actual - expected)[:20]
            raise RuntimeError(f"window-key mismatch; missing={missing}, extra={extra}")

    grouped: dict[tuple[str, str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["regime"]), str(row["seed"]), float(row["Re"]), str(row["window_id"]))].append(row)
    window_rows: list[dict[str, object]] = []
    for (regime, seed, re_value, window_id), values in sorted(grouped.items()):
        values.sort(key=lambda value: int(value["step"]))
        steps = [int(value["step"]) for value in values]
        if steps != list(range(1, 49)):
            raise RuntimeError(f"{regime}/{window_id}: expected steps 1..48, got {steps}")
        for horizon in (24, 48):
            subset = values[:horizon]
            complete = all(bool(value["finite"]) for value in subset)
            if not complete:
                eu = ep = joint = float("nan")
            else:
                eu = mean(float(value["Eu"]) for value in subset)
                ep = mean(float(value["Ep"]) for value in subset)
                joint = eu + ep
            window_rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "K": horizon,
                    "Re": re_value,
                    "window_id": window_id,
                    "complete": complete,
                    "Eu": eu,
                    "Ep": ep,
                    "joint": joint,
                }
            )

    by_re: dict[tuple[str, str, int, float], list[dict[str, object]]] = defaultdict(list)
    for row in window_rows:
        by_re[(str(row["regime"]), str(row["seed"]), int(row["K"]), float(row["Re"]))].append(row)
    re_rows: list[dict[str, object]] = []
    for (regime, seed, horizon, re_value), values in sorted(by_re.items()):
        complete = all(bool(value["complete"]) for value in values)
        re_rows.append(
            {
                "regime": regime,
                "seed": seed,
                "K": horizon,
                "Re": re_value,
                "complete_windows": sum(bool(value["complete"]) for value in values),
                "planned_windows": len(values),
                "Eu": mean(float(value["Eu"]) for value in values) if complete else float("nan"),
                "Ep": mean(float(value["Ep"]) for value in values) if complete else float("nan"),
                "joint": mean(float(value["joint"]) for value in values) if complete else float("nan"),
            }
        )

    by_seed: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in re_rows:
        by_seed[(str(row["regime"]), str(row["seed"]), int(row["K"]))].append(row)
    seed_rows: list[dict[str, object]] = []
    for (regime, seed, horizon), values in sorted(by_seed.items()):
        complete = all(int(value["complete_windows"]) == int(value["planned_windows"]) for value in values)
        seed_rows.append(
            {
                "regime": regime,
                "seed": seed,
                "K": horizon,
                "complete_windows": sum(int(value["complete_windows"]) for value in values),
                "planned_windows": sum(int(value["planned_windows"]) for value in values),
                "Eu": mean(float(value["Eu"]) for value in values) if complete else float("nan"),
                "Ep": mean(float(value["Ep"]) for value in values) if complete else float("nan"),
                "joint": mean(float(value["joint"]) for value in values) if complete else float("nan"),
            }
        )

    final_groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        final_groups[(str(row["regime"]), int(row["K"]))].append(row)
    final_rows: list[dict[str, object]] = []
    latex: list[str] = []
    for (regime, horizon), values in sorted(final_groups.items()):
        valid = [row for row in values if math.isfinite(float(row["joint"]))]
        n = len(valid)
        eu_values = [float(row["Eu"]) for row in valid]
        ep_values = [float(row["Ep"]) for row in valid]
        joint_values = [float(row["joint"]) for row in valid]
        final_rows.append(
            {
                "regime": regime,
                "K": horizon,
                "Eu_mean": mean(eu_values) if valid else float("nan"),
                "Eu_sample_sd": stdev(eu_values) if n >= 2 else "",
                "Ep_mean": mean(ep_values) if valid else float("nan"),
                "Ep_sample_sd": stdev(ep_values) if n >= 2 else "",
                "joint_mean": mean(joint_values) if valid else float("nan"),
                "joint_sample_sd": stdev(joint_values) if n >= 2 else "",
                "accepted_checkpoints": n,
                "attempted_checkpoints": len(values),
                "complete_windows": sum(int(row["complete_windows"]) for row in valid),
                "planned_windows": sum(int(row["planned_windows"]) for row in values),
            }
        )
        if n == 1:
            value = f"{joint_values[0]:.4f}"
        elif n >= 2:
            value = f"${mean(joint_values):.4f}\\pm{stdev(joint_values):.4f}$"
        else:
            value = "--"
        latex.append(f"{regime} & Global MoE & {horizon} & {value} & {n}/{len(values)} \\\\")

    write_csv(args.output_dir / "per_window.csv", window_rows)
    write_csv(args.output_dir / "per_re.csv", re_rows)
    write_csv(args.output_dir / "per_seed.csv", seed_rows)
    write_csv(args.output_dir / "final_summary.csv", final_rows)
    (args.output_dir / "latex_rows.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    audit = {
        "input_rows": len(rows),
        "unique_windows": len(grouped),
        "target_key_check": expected is not None,
        "all_rows_finite": all(bool(row["finite"]) for row in rows),
        "aggregation": "step mean -> window mean -> Re mean -> equal-Re mean; joint=Eu+Ep; sample SD across checkpoints only",
        "no_denominator_epsilon_added": True,
        "passed": all(
            int(row["complete_windows"]) == int(row["planned_windows"]) for row in seed_rows
        ),
    }
    (args.output_dir / "recomputation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
