#!/usr/bin/env python3
"""Export frozen Steady Proposed/Vanilla modal rollouts on heldout windows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def relocate(value, root: Path):
    if isinstance(value, dict):
        return {key: relocate(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, root) for item in value]
    if isinstance(value, str):
        return value.replace("/root/panxy/particalMOE", str(root))
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=("proposed", "vanilla"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-stride", type=int, default=8)
    args = parser.parse_args()
    steady = args.root / "steady_specialist_v1"
    s3 = load_module("visualization_steady_s3", steady / "code/train_s3.py")
    if args.method == "vanilla":
        wrapper = load_module(
            "visualization_vanilla_patch",
            args.root / "paper_experiments/code/run_vanilla_fnn_moe.py",
        )
        wrapper.patch_nested_loaders(s3)
        task = next(
            parent for parent in args.checkpoint.parents if parent.name == "vanilla-fnn"
        )
        config = task / "config.json"
        s2_candidates = [
            task / "training/s2/checkpoints/best-qualified.pt",
            task / "training/s2/checkpoints/final.pt",
            task / "training/s2/checkpoints/latest.pt",
        ]
        source = next(path for path in s2_candidates if path.is_file())
    else:
        config = steady / "code/training_s2b_portable.json"
        source = steady / "checkpoint/frozen_s2b_validation.pt"
    relocated_config = args.output.parent / f"{args.method}_relocated_config.json"
    relocated_config.parent.mkdir(parents=True, exist_ok=True)
    relocated_config.write_text(
        json.dumps(relocate(json.loads(config.read_text()), args.root), indent=2) + "\n",
        encoding="utf-8",
    )
    init = SimpleNamespace(
        experiment="S3-B",
        run_dir=str(args.output.parent / f"{args.method}_runtime"),
        config=str(relocated_config),
        trainer=str(steady / "code/train_s2b_3090.py"),
        finalizer=str(steady / "code/finalize_s2b_3090.py"),
        checkpoint=str(source),
        bank=str(steady / "data/perturbation_bank_train_validation.npz"),
        validation_lock=str(args.output.parent / f"{args.method}.lock"),
        learning_rate=1.5516372391099407e-5,
    )
    runtime = s3.S3(init)
    runtime.init_eval_assets()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    runtime.exp.model.load_state_dict(checkpoint["model"], strict=True)
    runtime.exp.model.eval()
    windows = runtime.exp.build_windows(56)["heldout"]
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=runtime.exp.amp_dtype):
        for label, all_ids in sorted(windows.items()):
            starts = np.asarray(all_ids[:: args.window_stride], dtype=np.int64)
            for offset in range(0, len(starts), 16):
                batch = starts[offset : offset + 16]
                ids = torch.as_tensor(batch, dtype=torch.long, device=runtime.exp.device)
                out = runtime.exp.rollout(ids, 56)
                pa, pb, ta, tb = [
                    out[key].float().permute(1, 0, 2).cpu().numpy()
                    for key in ("pa", "pb", "ta", "tb")
                ]
                indices = runtime.exp.indices(ids, 56)[:, 1:].cpu().numpy()
                times = np.asarray(runtime.exp.a["time"])[indices]
                for row, start in enumerate(batch.tolist()):
                    records.append(
                        (
                            float(runtime.exp.a["re"][start]),
                            int(start),
                            times[row],
                            pa[row],
                            pb[row],
                            ta[row],
                            tb[row],
                        )
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        re=np.asarray([row[0] for row in records], np.float64),
        split=np.asarray(["heldout"] * len(records)),
        start=np.asarray([row[1] for row in records], np.int64),
        times=np.asarray([row[2] for row in records], np.float64),
        pred_a=np.asarray([row[3] for row in records], np.float32),
        pred_b=np.asarray([row[4] for row in records], np.float32),
        true_a=np.asarray([row[5] for row in records], np.float32),
        true_b=np.asarray([row[6] for row in records], np.float32),
    )


if __name__ == "__main__":
    main()
