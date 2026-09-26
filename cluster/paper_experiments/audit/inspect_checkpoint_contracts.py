#!/usr/bin/env python3
"""Read-only checkpoint and implementation contract inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() <= 256:
            return value.detach().cpu().tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return repr(value)


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = None
    state_key = None
    for candidate in (
        "model_state",
        "model_state_dict",
        "best_model_state",
        "state_dict",
        "model",
    ):
        if candidate in checkpoint and isinstance(checkpoint[candidate], dict):
            state = checkpoint[candidate]
            state_key = candidate
            break
    tensor_shapes = {}
    if state is not None:
        tensor_shapes = {
            key: list(value.shape)
            for key, value in state.items()
            if isinstance(value, torch.Tensor)
        }
    selected_metadata = {}
    for key in (
        "args",
        "config",
        "epoch",
        "step",
        "optimizer_step",
        "best_epoch",
        "best_val_score",
        "fluctuation_contract_sha256",
        "split_stats",
        "scalers",
    ):
        if key in checkpoint:
            selected_metadata[key] = jsonable(checkpoint[key])
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "top_level_keys": sorted(checkpoint),
        "state_key": state_key,
        "state_tensor_count": len(tensor_shapes),
        "state_element_count": int(
            sum(np.prod(shape, dtype=np.int64) for shape in tensor_shapes.values())
        ),
        "expert_tensor_shapes": {
            key: shape
            for key, shape in tensor_shapes.items()
            if "expert" in key.lower() or "router" in key.lower()
        },
        "metadata": selected_metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoints = {
        "steady": root
        / "steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt",
        "hopf": root
        / "Hopf/migrated_h4_expanded/runs/HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt",
        "periodic": root
        / "periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt",
        "global_moe": root
        / "V16_1_SteadyPressureAnchor32/source/test_results_v16_1/results/"
        "V16_1_SteadyPressureAnchor32/V16_1_SteadyPressureAnchor32_ru32_rp32/"
        "V16_1_SteadyPressureAnchor32_ru32_rp32_Re_24p630436_checkpoint.pt",
    }
    sources = {
        "steady_trainer": root
        / "steady_specialist_v1/code/train_v16_4_v2_r32_compat.py",
        "steady_s4": root / "steady_specialist_v1/code/train_s4_steady.py",
        "hopf_trainer": root
        / "Hopf/migrated_h4_expanded/code/train_hopf_moe_expanded.py",
        "periodic_trainer": root
        / "periodic_specialist_r32/code/train_periodic_moe.py",
        "global_trainer": root
        / "V16_1_SteadyPressureAnchor32/source/test_results_v16_1/"
        "train_v16_1_train_stable_attractor_moe.py",
        "global_evaluator": root
        / "V16_1_SteadyPressureAnchor32/source/test_results_v16_1/evaluate.py",
    }
    missing = [
        str(path)
        for path in list(checkpoints.values()) + list(sources.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))
    payload = {
        "schema": "paper_experiment_checkpoint_audit/v1",
        "root": str(root),
        "checkpoints": {
            name: inspect_checkpoint(path) for name, path in checkpoints.items()
        },
        "source_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in sources.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
