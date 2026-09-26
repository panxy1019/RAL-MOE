#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(quad_u: np.ndarray, quad_p: np.ndarray, index: int) -> dict:
    eu = np.sqrt(np.maximum(quad_u[:, :, index] / np.maximum(quad_u[:, :, 3], 1e-12), 0.0))
    ep = np.sqrt(np.maximum(quad_p[:, :, index] / np.maximum(quad_p[:, :, 3], 1e-12), 0.0))
    return {
        "velocity_mean": float(eu.mean()),
        "pressure_mean": float(ep.mean()),
        "joint_mean": float((eu + ep).mean()),
        "joint_worst": float((eu + ep).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    report = {
        "candidate_order": [str(data["source_name"].item()), "Hopf"],
        "all": {
            "candidate_1": summarize(data["quad_u"], data["quad_p"], 0),
            "candidate_2": summarize(data["quad_u"], data["quad_p"], 1),
        },
        "by_split": {},
    }
    for split in np.unique(data["split"]):
        mask = data["split"] == split
        report["by_split"][str(split)] = {
            "candidate_1": summarize(data["quad_u"][mask], data["quad_p"][mask], 0),
            "candidate_2": summarize(data["quad_u"][mask], data["quad_p"][mask], 1),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
