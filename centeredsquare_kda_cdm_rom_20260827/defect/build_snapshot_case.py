#!/usr/bin/env python3
"""Create one temporary OpenFOAM case containing every snapshot of one NPZ trajectory."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np


INTERNAL = re.compile(
    r"internalField\s+(?:uniform\s+[^;]+;|nonuniform\s+List<[^>]+>\s+\d+\s*\(\s*.*?\s*\)\s*;)",
    flags=re.S,
)


def block(values, vector):
    if vector:
        rows = "\n".join(f"({x:.16g} {y:.16g} 0)" for x, y in values)
        kind = "vector"
    else:
        rows = "\n".join(f"{value:.16g}" for value in values)
        kind = "scalar"
    return f"internalField nonuniform List<{kind}>\n{len(values)}\n(\n{rows}\n)\n;"


def write_field(template, path, values, vector):
    text, count = INTERNAL.subn(block(values, vector), template, count=1)
    if count != 1:
        raise RuntimeError(f"could not replace internalField in {path}")
    text = re.sub(r'location\s+"[^"]+";', f'location "{path.parent.name}";', text, count=1)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--snapshot-npz", type=Path, required=True)
    parser.add_argument("--template-time", default="100")
    parser.add_argument("--output-case", type=Path, required=True)
    args = parser.parse_args()
    if args.output_case.exists():
        raise FileExistsError(args.output_case)
    with np.load(args.snapshot_npz, allow_pickle=False) as source:
        velocity = np.asarray(source["U"], dtype=np.float64)
        pressure = np.asarray(source["p"], dtype=np.float64)
        times = np.asarray(source["times"], dtype=np.float64)
        re_value = float(source["Re"])
    shutil.copytree(args.source_case / "constant", args.output_case / "constant")
    shutil.copytree(args.source_case / "system", args.output_case / "system")
    u_template = (args.source_case / args.template_time / "U").read_text()
    p_template = (args.source_case / args.template_time / "p").read_text()
    entries = []
    for index, physical_time in enumerate(times, start=1):
        directory = args.output_case / str(index)
        directory.mkdir()
        write_field(u_template, directory / "U", velocity[index - 1], True)
        write_field(p_template, directory / "p", pressure[index - 1], False)
        entries.append({"foam_time": index, "snapshot_index": index - 1, "physical_time": float(physical_time)})
    manifest = {
        "schema_version": 1, "Re": re_value, "nu": 1.0 / re_value,
        "snapshot_npz": str(args.snapshot_npz), "snapshot_count": len(times),
        "entries": entries, "validation_loaded": False, "heldout_loaded": False,
    }
    (args.output_case / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "entries"}, indent=2))


if __name__ == "__main__":
    main()
