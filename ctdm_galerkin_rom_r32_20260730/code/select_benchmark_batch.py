#!/usr/bin/env python3
"""Select the largest successful B4 batch under the registered memory ceiling."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peak-memory-ceiling-gb", type=float, default=9.0)
    parser.add_argument("--effective-batch", type=int, default=16)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for micro_batch in (4, 8, 16):
        name = f"B4_K56_benchmark_mb{micro_batch}"
        run_dir = args.benchmark_root / name
        rc_path = args.benchmark_root / f"{name}.rc"
        throughput_path = run_dir / "THROUGHPUT.json"
        return_code = int(rc_path.read_text().strip()) if rc_path.is_file() else None
        throughput = (
            json.loads(throughput_path.read_text(encoding="utf-8"))
            if throughput_path.is_file()
            else None
        )
        peak = (
            float(throughput["peak_gpu_memory_gb"])
            if throughput is not None
            else float("nan")
        )
        eligible = (
            return_code == 0
            and throughput is not None
            and math.isfinite(peak)
            and peak <= args.peak_memory_ceiling_gb
            and args.effective_batch % micro_batch == 0
        )
        rows.append(
            {
                "micro_batch": micro_batch,
                "return_code": return_code,
                "throughput": throughput,
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        raise RuntimeError("no benchmark batch satisfies the safety contract")
    selected = max(eligible_rows, key=lambda row: row["micro_batch"])
    micro_batch = int(selected["micro_batch"])
    gradient_accumulation = args.effective_batch // micro_batch
    payload = {
        "schema_version": 1,
        "selection_rule": "largest successful batch with peak allocation <= ceiling",
        "peak_memory_ceiling_gb": args.peak_memory_ceiling_gb,
        "effective_batch": args.effective_batch,
        "selected_micro_batch": micro_batch,
        "selected_gradient_accumulation": gradient_accumulation,
        "benchmarks": rows,
    }
    atomic_json(payload, args.output)
    print(f"{micro_batch} {gradient_accumulation}")


if __name__ == "__main__":
    main()
