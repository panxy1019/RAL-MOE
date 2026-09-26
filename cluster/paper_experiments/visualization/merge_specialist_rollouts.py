#!/usr/bin/env python3
"""Merge three frozen-method rollout caches onto one exact query population."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--vanilla", type=Path, required=True)
    parser.add_argument("--dataonly", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.proposed, allow_pickle=False) as proposed, np.load(
        args.vanilla, allow_pickle=False
    ) as vanilla, np.load(args.dataonly, allow_pickle=False) as dataonly:
        population = ("re", "split")
        for key in population:
            if not np.array_equal(proposed[key], vanilla[key]):
                raise RuntimeError(f"proposed/vanilla population mismatch: {key}")
            if not np.array_equal(proposed[key], dataonly[key]):
                raise RuntimeError(f"proposed/dataonly population mismatch: {key}")
        for candidate in (vanilla, dataonly):
            if not np.allclose(
                proposed["times"][:, : args.horizon],
                candidate["times"][:, : args.horizon],
                atol=1.0e-5,
                rtol=1.0e-7,
            ):
                raise RuntimeError("query-time mismatch")
            for key in ("true_a", "true_b"):
                if not np.allclose(
                    proposed[key][:, : args.horizon],
                    candidate[key][:, : args.horizon],
                    atol=2.0e-5,
                    rtol=1.0e-6,
                ):
                    raise RuntimeError(f"truth mismatch: {key}")
        payload = {
            "re": proposed["re"],
            "split": proposed["split"],
            "start": proposed["start"],
            "times": proposed["times"][:, : args.horizon],
            "truth_a": proposed["true_a"][:, : args.horizon],
            "truth_b": proposed["true_b"][:, : args.horizon],
            "proposed_a": proposed["pred_a"][:, : args.horizon],
            "proposed_b": proposed["pred_b"][:, : args.horizon],
            "vanilla_a": vanilla["pred_a"][:, : args.horizon],
            "vanilla_b": vanilla["pred_b"][:, : args.horizon],
            "dataonly_a": dataonly["pred_a"][:, : args.horizon],
            "dataonly_b": dataonly["pred_b"][:, : args.horizon],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)


if __name__ == "__main__":
    main()
