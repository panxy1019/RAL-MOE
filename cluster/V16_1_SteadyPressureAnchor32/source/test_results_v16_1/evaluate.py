#!/usr/bin/env python3
"""Aggregate V15 metrics JSON files into compact console tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n# {data['settings']['experiment_name']}")
        print("Re,best_epoch,rhs,pressure,auto_a1,auto_b1,roll_a,roll_b")
        for item in data["results"]:
            deep = item["deep_moe"]
            one = item["one_step_autonomous_pressure"]
            roll = item["rollout_autonomous_pressure"]
            print(
                f"{item['test_Re']:.6g},{item['best_epoch']},"
                f"{deep['rhs_relative_l2']:.6g},"
                f"{deep['pressure_head_relative_l2']:.6g},"
                f"{one['a_relative_l2']:.6g},"
                f"{one['b_relative_l2']:.6g},"
                f"{roll['a_relative_l2_mean']:.6g},"
                f"{roll['b_relative_l2_mean']:.6g}"
            )


if __name__ == "__main__":
    main()
