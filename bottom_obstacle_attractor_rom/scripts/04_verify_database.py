#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from common import ROOT


def summarize_npz(path: Path, coords_ref: np.ndarray | None) -> tuple[dict, np.ndarray | None]:
    row = {
        "file": str(path),
        "Re": math.nan,
        "nu": math.nan,
        "nProcs": "",
        "T": "",
        "N": "",
        "time_min": "",
        "time_max": "",
        "coords_match_first_Re": "",
        "U_has_nan_or_inf": "",
        "p_has_nan_or_inf": "",
        "U_mean": "",
        "U_std": "",
        "p_mean": "",
        "p_std": "",
        "file_size_mb": path.stat().st_size / 1024**2,
        "status": "ok",
    }
    try:
        with np.load(path, allow_pickle=True) as data:
            coords = data["coords"]
            times = data["times"]
            u = data["U"]
            p = data["p"]
            row["Re"] = float(data["Re"])
            row["nu"] = float(data["nu"])
            metadata = json.loads(str(data["metadata_json"]))
            row["nProcs"] = metadata.get("nProcs", "")
            row["T"] = int(times.shape[0])
            row["N"] = int(coords.shape[0])
            row["time_min"] = float(np.min(times)) if times.size else math.nan
            row["time_max"] = float(np.max(times)) if times.size else math.nan
            row["U_has_nan_or_inf"] = not bool(np.isfinite(u).all())
            row["p_has_nan_or_inf"] = not bool(np.isfinite(p).all())
            row["U_mean"] = float(np.mean(u))
            row["U_std"] = float(np.std(u))
            row["p_mean"] = float(np.mean(p))
            row["p_std"] = float(np.std(p))
            if coords_ref is None:
                row["coords_match_first_Re"] = True
                coords_ref = coords.copy()
            else:
                row["coords_match_first_Re"] = bool(
                    coords.shape == coords_ref.shape and np.allclose(coords, coords_ref, atol=1e-10)
                )
                if not row["coords_match_first_Re"]:
                    row["status"] = "WARNING: coordinates differ from first Re"
    except Exception as exc:  # noqa: BLE001
        row["status"] = f"ERROR: {exc}"
    return row, coords_ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all compressed npz case files.")
    parser.add_argument("--npz-dir", type=Path, default=ROOT / "data" / "npz")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifest" / "database_summary.csv",
    )
    args = parser.parse_args()

    files = sorted(args.npz_dir.glob("Re_*.npz"))
    if not files:
        raise SystemExit(f"No npz files found in {args.npz_dir}")

    rows = []
    coords_ref = None
    for path in files:
        row, coords_ref = summarize_npz(path, coords_ref)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    for row in rows:
        print(
            f"Re={row['Re']} T={row['T']} N={row['N']} "
            f"coords_match={row['coords_match_first_Re']} status={row['status']}"
        )

    bad = [row for row in rows if str(row["status"]).startswith("ERROR")]
    if bad:
        raise SystemExit(1)
    mismatch = [row for row in rows if row["coords_match_first_Re"] is False]
    if mismatch:
        print("WARNING: Some meshes differ from the first Re; POD expects a common grid.")


if __name__ == "__main__":
    main()
