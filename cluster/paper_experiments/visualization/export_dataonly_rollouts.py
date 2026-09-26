#!/usr/bin/env python3
"""Export frozen DataOnly modal rollouts on an externally fixed start set."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--regime", choices=("steady", "hopf", "periodic"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()

    code = args.root / "paper_experiments/code"
    evaluator = load_module(
        "visualization_ablation_evaluator",
        code / "evaluate_specialist_ablation_final.py",
    )
    trainer = load_module("visualization_dataonly_trainer", code / "train_dataonly_moe.py")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data = evaluator.load_full_data(args.root, args.regime, trainer)
    with np.load(args.reference_bundle, allow_pickle=False) as reference:
        source_starts = np.asarray(reference["start"], dtype=np.int64)
        re_values = np.asarray(reference["re"], dtype=np.float64)
        split = np.asarray(reference["split"]).astype(str)
        reference_times = np.asarray(reference["times"], dtype=np.float64)
    starts = []
    for re_value, first_query_time in zip(re_values, reference_times[:, 0]):
        next_ids = np.flatnonzero(
            np.isclose(data["re"], re_value, atol=5.0e-4, rtol=0)
            & np.isclose(data["time"], first_query_time, atol=1.0e-4, rtol=1.0e-7)
        )
        if len(next_ids) != 1:
            raise RuntimeError(
                f"cannot uniquely align Re={re_value}, t={first_query_time}: {next_ids}"
            )
        start = int(data["prev"][int(next_ids[0])])
        if start < 0:
            raise RuntimeError("aligned query has no three-step-capable predecessor")
        starts.append(start)
    starts = np.asarray(starts, dtype=np.int64)
    pred_a, pred_b, true_a, true_b = evaluator.dataonly_predictions(
        trainer, checkpoint, data, starts, args.horizon
    )
    times = np.asarray(
        [
            [data["time"][np.asarray(_follow(data, int(start), args.horizon))]]
            for start in starts
        ],
        dtype=np.float64,
    ).reshape(len(starts), args.horizon)
    if not np.allclose(times, reference_times, atol=1.0e-5, rtol=1.0e-7):
        raise RuntimeError("DataOnly/reference query-time mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        re=re_values,
        split=split,
        start=starts,
        source_start=source_starts,
        times=times,
        pred_a=pred_a.astype(np.float32),
        pred_b=pred_b.astype(np.float32),
        true_a=true_a.astype(np.float32),
        true_b=true_b.astype(np.float32),
    )


def _follow(data: dict, start: int, horizon: int) -> list[int]:
    ids = []
    current = start
    for _ in range(horizon):
        current = int(data["next"][current])
        if current < 0:
            raise RuntimeError(f"trajectory ended before K{horizon}, start={start}")
        ids.append(current)
    return ids


if __name__ == "__main__":
    main()
