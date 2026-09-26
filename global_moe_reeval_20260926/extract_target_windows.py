#!/usr/bin/env python3
"""Extract the frozen H/P test-window keys from the shared split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    regimes = {"H": "Hopf", "P": "Periodic"}
    for short, regime in regimes.items():
        section = manifest[short]
        horizon = int(section["horizon"])
        if horizon != 48:
            raise ValueError(f"{short}: expected horizon 48, got {horizon}")
        by_index: dict[int, tuple[dict[str, object], int]] = {}
        for trajectory in section["trajectories"]:
            indices = [int(value) for value in trajectory["indices"]]
            for position, index in enumerate(indices):
                if index in by_index:
                    raise ValueError(f"{short}: duplicate global index {index}")
                by_index[index] = (trajectory, position)
        starts = [int(value) for value in section["test_starts"]]
        counts[regime] = len(starts)
        for window_number, start in enumerate(starts):
            if start not in by_index:
                raise KeyError(f"{short}: start {start} absent from trajectories")
            trajectory, position = by_index[start]
            indices = [int(value) for value in trajectory["indices"]]
            times = [float(value) for value in trajectory["times"]]
            if position < int(section["history"]) - 1:
                raise ValueError(f"{short}: insufficient history at start {start}")
            if position + horizon >= len(indices):
                raise ValueError(f"{short}: incomplete future at start {start}")
            initial_time = times[position]
            window_id = f"{short}:{trajectory['label']}:{start}"
            for step in range(1, horizon + 1):
                rows.append(
                    {
                        "regime": regime,
                        "window_number": window_number,
                        "window_id": window_id,
                        "trajectory_label": trajectory["label"],
                        "Re": float(trajectory["Re"]),
                        "start_global_index": start,
                        "start_time": initial_time,
                        "step": step,
                        "target_global_index": indices[position + step],
                        "physical_time": times[position + step],
                        "elapsed_time": times[position + step] - initial_time,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "history": {key: int(manifest[key]["history"]) for key in ("H", "P")},
        "horizon": {key: int(manifest[key]["horizon"]) for key in ("H", "P")},
        "windows": counts,
        "step_rows": len(rows),
        "expected_step_rows": 48 * sum(counts.values()),
        "passed": counts == {"Hopf": 42, "Periodic": 32}
        and len(rows) == 48 * (42 + 32),
    }
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError(f"unexpected frozen-window counts: {audit}")


if __name__ == "__main__":
    main()
