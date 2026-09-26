#!/usr/bin/env python3
"""Generate OpenFOAM probe states for discrete r11 Galerkin tensor assembly."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--pressure-pod", type=Path, required=True)
    parser.add_argument("--output-case", type=Path, required=True)
    parser.add_argument("--velocity-rank", type=int, default=11)
    parser.add_argument("--pressure-rank", type=int, default=11)
    parser.add_argument("--velocity-epsilon", type=float, default=0.05)
    parser.add_argument("--pressure-epsilon", type=float, default=0.05)
    return parser.parse_args()


INTERNAL_PATTERN = re.compile(
    r"internalField\s+(?:uniform\s+[^;]+;|nonuniform\s+List<[^>]+>\s+\d+\s*\(\s*.*?\s*\)\s*;)",
    flags=re.S,
)


def internal_block(values: np.ndarray, vector: bool) -> str:
    if vector:
        rows = "\n".join(f"({x:.16g} {y:.16g} 0)" for x, y in values)
        kind = "vector"
    else:
        rows = "\n".join(f"{value:.16g}" for value in values)
        kind = "scalar"
    return f"internalField   nonuniform List<{kind}>\n{len(values)}\n(\n{rows}\n)\n;"


def write_field(template: str, path: Path, values: np.ndarray, vector: bool) -> None:
    replacement = internal_block(values, vector)
    text, count = INTERNAL_PATTERN.subn(replacement, template, count=1)
    if count != 1:
        raise RuntimeError(f"failed to replace internalField for {path}")
    text = re.sub(r'location\s+"[^"]+";', f'location    "{path.parent.name}";', text, count=1)
    path.write_text(text)


def main():
    args = parse_args()
    if args.output_case.exists():
        raise FileExistsError(args.output_case)
    if args.velocity_epsilon <= 0 or args.pressure_epsilon <= 0:
        raise ValueError("probe epsilons must be positive")
    shutil.copytree(args.source_case / "constant", args.output_case / "constant", copy_function=shutil.copy2)
    shutil.copytree(args.source_case / "system", args.output_case / "system", copy_function=shutil.copy2)
    with np.load(args.velocity_pod, allow_pickle=False) as pod:
        u_mean = np.asarray(pod["mean"], dtype=np.float64)
        u_modes = np.asarray(pod["modes"], dtype=np.float64)[: args.velocity_rank]
    with np.load(args.pressure_pod, allow_pickle=False) as pod:
        p_mean = np.asarray(pod["mean"], dtype=np.float64)
        p_modes = np.asarray(pod["modes"], dtype=np.float64)[: args.pressure_rank]
    u_modes = u_modes.reshape(args.velocity_rank, len(u_mean), 2)
    u_template = (args.source_case / "100/U").read_text()
    p_template = (args.source_case / "100/p").read_text()
    states = []

    def add_state(label, velocity, pressure, metadata):
        index = len(states) + 1
        directory = args.output_case / str(index)
        directory.mkdir()
        write_field(u_template, directory / "U", velocity, True)
        write_field(p_template, directory / "p", pressure, False)
        states.append({"index": index, "label": label, **metadata})

    add_state("base", u_mean, p_mean, {"kind": "base"})
    eps_u = args.velocity_epsilon
    for i in range(args.velocity_rank):
        for sign in (-1, 1):
            add_state(
                f"u_{i}_{sign:+d}", u_mean + sign * eps_u * u_modes[i], p_mean,
                {"kind": "u_axis", "i": i, "sign": sign},
            )
    for i in range(args.velocity_rank):
        for j in range(i + 1, args.velocity_rank):
            for sign_i in (-1, 1):
                for sign_j in (-1, 1):
                    add_state(
                        f"u_{i}_{sign_i:+d}_{j}_{sign_j:+d}",
                        u_mean + eps_u * (sign_i * u_modes[i] + sign_j * u_modes[j]),
                        p_mean,
                        {"kind": "u_mixed", "i": i, "j": j, "sign_i": sign_i, "sign_j": sign_j},
                    )
    eps_p = args.pressure_epsilon
    for i in range(args.pressure_rank):
        for sign in (-1, 1):
            add_state(
                f"p_{i}_{sign:+d}", u_mean, p_mean + sign * eps_p * p_modes[i],
                {"kind": "p_axis", "i": i, "sign": sign},
            )
    manifest = {
        "schema_version": 1,
        "velocity_rank": args.velocity_rank,
        "pressure_rank": args.pressure_rank,
        "velocity_epsilon": eps_u,
        "pressure_epsilon": eps_p,
        "state_count": len(states),
        "states": states,
        "heldout_loaded": False,
    }
    (args.output_case / "probe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in manifest if key != "states"}, indent=2))


if __name__ == "__main__":
    main()
