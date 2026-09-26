#!/usr/bin/env python3
"""Print non-tensor checkpoint metadata and representative parameter shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", {})
    payload = {
        "checkpoint": str(args.checkpoint),
        "variant": checkpoint.get("variant"),
        "optimizer_step": checkpoint.get("optimizer_step"),
        "best_step": checkpoint.get("best_step"),
        "best_score": checkpoint.get("best_score"),
        "args": jsonable(checkpoint.get("args", {})),
        "parameter_count": int(
            sum(tensor.numel() for tensor in state.values() if isinstance(tensor, torch.Tensor))
        ),
        "representative_parameter_shapes": {
            name: list(tensor.shape)
            for name, tensor in list(state.items())[:20]
            if isinstance(tensor, torch.Tensor)
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
