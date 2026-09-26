#!/usr/bin/env python3
"""Summarize the development-only formal H-P preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["rows"]
    by_split = {}
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        by_split[split] = {
            "count": len(selected),
            "Periodic_finite": sum(row["Periodic_finite"] for row in selected),
            "Hopf_finite": sum(row["Hopf_finite"] for row in selected),
            "Periodic_divergent": sum(row["Periodic_divergent"] for row in selected),
            "Hopf_divergent": sum(row["Hopf_divergent"] for row in selected),
        }
    ordered = sorted(rows, key=lambda row: row["Re"])
    both_valid = [
        row
        for row in ordered
        if row["Periodic_finite"]
        and row["Hopf_finite"]
        and not row["Periodic_divergent"]
        and not row["Hopf_divergent"]
    ]
    boundary = [row for row in ordered if row["Re"] <= 79.3811581027]
    boundary_valid = [
        row
        for row in boundary
        if row["Periodic_finite"]
        and row["Hopf_finite"]
        and not row["Periodic_divergent"]
        and not row["Hopf_divergent"]
    ]
    first_hopf_nonfinite = next(
        (row for row in ordered if not row["Hopf_finite"]), None
    )
    first_hopf_divergent = next(
        (row for row in ordered if row["Hopf_divergent"]), None
    )
    result = {
        "decision": data["decision"],
        "rows": len(rows),
        "by_split": by_split,
        "both_valid_Re": [
            {"split": row["split"], "Re_label": row["Re_label"], "Re": row["Re"]}
            for row in both_valid
        ],
        "boundary_upper_Re": 79.3811581027,
        "boundary_rows": len(boundary),
        "boundary_both_valid_rows": len(boundary_valid),
        "boundary_both_valid_Re": [row["Re"] for row in boundary_valid],
        "first_Hopf_nonfinite": (
            {
                "split": first_hopf_nonfinite["split"],
                "Re_label": first_hopf_nonfinite["Re_label"],
                "Re": first_hopf_nonfinite["Re"],
            }
            if first_hopf_nonfinite
            else None
        ),
        "first_Hopf_divergent": (
            {
                "split": first_hopf_divergent["split"],
                "Re_label": first_hopf_divergent["Re_label"],
                "Re": first_hopf_divergent["Re"],
                "finite": first_hopf_divergent["Hopf_finite"],
            }
            if first_hopf_divergent
            else None
        ),
        "per_Re": [
            {
                key: row[key]
                for key in (
                    "split",
                    "Re_label",
                    "Re",
                    "Periodic_finite",
                    "Periodic_divergent",
                    "Hopf_finite",
                    "Hopf_divergent",
                    "Periodic_final_velocity_field_error",
                    "Periodic_final_pressure_field_error",
                    "Hopf_final_velocity_field_error",
                    "Hopf_final_pressure_field_error",
                )
            }
            for row in ordered
        ],
        "authorized_next_action": "do not construct H-P cache or train routes",
        "heldout_access": False,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in result if key != "per_Re"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
