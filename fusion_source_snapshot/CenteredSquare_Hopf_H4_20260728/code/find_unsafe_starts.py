#!/usr/bin/env python3
"""Find train windows that are unsafe under the fresh physical-ROM initialization."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trainer", type=Path, required=True)
    parser.add_argument("--h4-trainer", type=Path, required=True)
    parser.add_argument("--coefficient-view", type=Path, required=True)
    parser.add_argument("--galerkin-path", type=Path, required=True)
    parser.add_argument("--pressure-path", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    B = load_module("unsafe_scan_base", args.baseline_trainer)
    H4 = load_module("unsafe_scan_h4", args.h4_trainer)
    data_args = SimpleNamespace(
        coefficient_view=args.coefficient_view,
        galerkin_path=args.galerkin_path,
        pressure_path=args.pressure_path,
        history_len=3,
    )
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(0.28)
    data = B.load_coefficients(data_args)
    rom_np = B.load_train_rom(data_args)
    rom = {key: torch.as_tensor(value, device=device) for key, value in rom_np.items()}
    pressure_alpha = float(torch.sigmoid(torch.tensor(6.0)))

    def physical_rollout(starts: np.ndarray, horizon: int):
        a = torch.as_tensor(data["a"][starts], device=device)
        b = torch.as_tensor(data["b"][starts], device=device)
        reynolds = torch.as_tensor(data["re"][starts], device=device)
        current = starts.copy()
        output = {key: [] for key in ("pred_a", "pred_b", "true_a", "true_b")}
        for _ in range(horizon):
            nxt = data["next"][current]
            dt = torch.as_tensor(
                (data["time"][nxt] - data["time"][current])[:, None],
                device=device,
            )
            k1 = B.galerkin(a, b, reynolds, rom)
            k2 = B.galerkin(a + 0.5 * dt * k1, b, reynolds, rom)
            k3 = B.galerkin(a + 0.5 * dt * k2, b, reynolds, rom)
            k4 = B.galerkin(a + dt * k3, b, reynolds, rom)
            a = a + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            b = pressure_alpha * B.pressure_base(a, reynolds, rom)
            output["pred_a"].append(a)
            output["pred_b"].append(b)
            output["true_a"].append(torch.as_tensor(data["a"][nxt], device=device))
            output["true_b"].append(torch.as_tensor(data["b"][nxt], device=device))
            current = nxt
        return output
    unsafe: set[int] = set()
    details: list[dict] = []
    with torch.no_grad():
        for horizon in (1, 2, 4, 8):
            for reynolds in H4.TRAIN:
                re_ids = data["train_ids"][
                    np.isclose(data["re"][data["train_ids"]], reynolds, atol=5e-6)
                ]
                starts = B.legal_starts(data, re_ids, horizon)
                for offset in range(0, len(starts), args.batch_size):
                    batch = starts[offset:offset + args.batch_size]
                    output = physical_rollout(batch, horizon)
                    bad, safety = H4.rollout_safety(output)
                    bad_indices = torch.where(bad)[0].cpu().numpy()
                    for index in bad_indices:
                        start = int(batch[index])
                        unsafe.add(start)
                        details.append({
                            "start": start,
                            "Re": float(reynolds),
                            "horizon": horizon,
                            "max_norm_ratio_batch": safety["max_norm_ratio"],
                        })
    payload = {
        "schema_version": 1,
        "unsafe_starts": sorted(unsafe),
        "unsafe_count": len(unsafe),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"unsafe_count": len(unsafe), "output": str(args.output)}))


if __name__ == "__main__":
    main()
