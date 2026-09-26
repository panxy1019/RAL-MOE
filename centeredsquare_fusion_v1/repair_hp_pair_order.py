#!/usr/bin/env python3
"""Atomically repair the HP cache's candidate/probability order metadata."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from common import atomic_json, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--input-preflight", type=Path, required=True)
    parser.add_argument("--output-preflight", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input_cache, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if str(data["boundary"].item()) != "HP" or str(data["source_name"].item()) != "Periodic":
        raise RuntimeError("repair is only valid for Periodic-native HP cache")
    observed = np.unique(data["pair_indices"], axis=0)
    if observed.shape != (1, 2) or not np.array_equal(observed[0], [1, 2]):
        raise RuntimeError(f"unexpected source pair order: {observed.tolist()}")
    data["pair_indices"] = np.repeat(
        np.asarray([[2, 1]], dtype=np.int64),
        len(data["split"]),
        axis=0,
    )
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_cache.with_suffix(args.output_cache.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    os.replace(temporary, args.output_cache)
    import json
    preflight = json.loads(args.input_preflight.read_text())
    preflight.update(
        {
            "cache_path": str(args.output_cache),
            "cache_sha256": sha256(args.output_cache),
            "pair_probability_order": ["Periodic", "Hopf"],
            "candidate_order": ["Periodic", "Hopf"],
            "metadata_repair": (
                "pair_indices changed from [Hopf,Periodic] to [Periodic,Hopf]; "
                "rollouts, truth, features, and quadratic statistics are unchanged"
            ),
        }
    )
    atomic_json(args.output_preflight, preflight)
    print(preflight["cache_sha256"])


if __name__ == "__main__":
    main()
