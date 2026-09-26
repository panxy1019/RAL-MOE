"""Re-run the archived Circular P benchmark with only its output root changed.

The source, checkpoints and original results are read-only. Each model runs in
its own process. The wrapper records the exact one-line source transformation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path("/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/experiments/round2_20260917")
SOURCE = ROOT / "code/benchmark_periodic_idle.py"
OLD = "out=R/'P_isolated_runtime'/a.kind"
NEW = "out=R/'P_latency_audit_20260923'/a.kind"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["proposed", "dense", "structured"])
    kind = parser.parse_args().kind
    text = SOURCE.read_text()
    if text.count(OLD) != 1:
        raise RuntimeError("Archived benchmark output-root anchor changed")
    modified = text.replace(OLD, NEW)
    output_root = ROOT / "P_latency_audit_20260923"
    output_root.mkdir(exist_ok=True)
    manifest = output_root / "reproduction_manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "source": str(SOURCE),
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "executed_sha256": hashlib.sha256(modified.encode()).hexdigest(),
            "only_change": [OLD, NEW],
            "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        }, indent=2))
    sys.path.insert(0, str(SOURCE.parent))
    sys.argv = [str(SOURCE), "--kind", kind]
    namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
    exec(compile(modified, str(SOURCE), "exec"), namespace)


if __name__ == "__main__":
    main()
