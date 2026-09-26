"""Read only the three user-specified frozen checkpoints; write audit metadata."""
import argparse
import gc
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import torch

CHECKPOINTS = {
    "steady": "steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt",
    "hopf": "Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/HopfExpanded34_H4_NormalFormRadial_r32/final.pt",
    "periodic": "periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe(value, depth=0):
    if depth > 6:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if re.search(r"password|secret|api.?key|access.?token", str(k), re.I)
                else safe(v, depth + 1) for k, v in value.items()}
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        if np.prod(value.shape) <= 20 and not torch.is_tensor(value):
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [safe(v, depth + 1) for v in value] if len(value) < 100 else {"length": len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return safe(vars(value), depth + 1)
    return {"type": type(value).__name__}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    paths = {chart: a.root / rel for chart, rel in CHECKPOINTS.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"STOP: exact checkpoint missing: {path}")
    report = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "cuda_runtime": torch.version.cuda},
              "checkpoints": {}}
    for chart, path in paths.items():
        # Trusted user-owned training artifacts also contain NumPy/RNG metadata.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        item = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
                "top_level_keys": list(ckpt), "metadata": {}, "state_dicts": {}}
        for key, value in ckpt.items():
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                item["state_dicts"][key] = {
                    "tensor_count": len(value), "elements": sum(v.numel() for v in value.values()),
                    "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in value.items()},
                }
            elif not re.search(r"optimizer|scheduler|rng|history|scaler", key, re.I):
                item["metadata"][key] = safe(value)
            elif "scaler" in key.lower():
                item["metadata"][key] = safe(value)
        report["checkpoints"][chart] = item
        print(json.dumps({"chart": chart, "sha256": item["sha256"], "keys": item["top_level_keys"],
                          "states": {k: {x: v[x] for x in ("tensor_count", "elements")} for k, v in item["state_dicts"].items()}}), flush=True)
        del ckpt
        gc.collect()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
