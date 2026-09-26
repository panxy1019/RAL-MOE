#!/usr/bin/env python3
"""Refresh the finalization inventory after adding paper handoff documents."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    artifacts = []
    for path in sorted(item for item in args.root.rglob("*") if item.is_file()):
        if path.name in {"artifact_inventory.json", "h4_expanded_final_small.tar.gz"}:
            continue
        artifacts.append({
            "path": str(path.relative_to(args.root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    output = {"schema_version": 2, "root": str(args.root), "artifacts": artifacts}
    target = args.root / "artifact_inventory.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    print(json.dumps({"status": "PASS", "artifacts": len(artifacts), "inventory": str(target)}))


if __name__ == "__main__":
    main()
