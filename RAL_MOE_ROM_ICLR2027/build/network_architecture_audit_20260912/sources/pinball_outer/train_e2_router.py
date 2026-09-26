#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import CLASS_NAMES, Router, atomic_json, clean_numeric


def collect(root: Path) -> dict[str, dict[str, list[float]]]:
    result = {name: {split: [] for split in ("train", "validation")} for name in CLASS_NAMES}
    membership = {
        "Steady": "belongs_to_steady",
        "Hopf": "belongs_to_hopf",
        "Periodic": "belongs_to_periodic",
    }
    with (root / "config/global_re_split.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split = str(row["split"])
            if split not in ("train", "validation"):
                continue
            for name, column in membership.items():
                if str(row[column]).strip().lower() == "true":
                    result[name][split].append(float(row["Re"]))
    for regimes in result.values():
        for split, values in regimes.items():
            regimes[split] = sorted(set(values))
    return result


def trajectory_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = np.argmax(probabilities, axis=1)
    recalls, f1s = [], []
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, pred in zip(labels, prediction):
        confusion[int(truth), int(pred)] += 1
    for cls in range(3):
        tp = int(np.sum((labels == cls) & (prediction == cls)))
        fp = int(np.sum((labels != cls) & (prediction == cls)))
        fn = int(np.sum((labels == cls) & (prediction != cls)))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1.0e-12))
    top2 = np.argsort(-probabilities, axis=1)[:, :2]
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": confusion.tolist(),
        "illegal_S_P_top2_count": int(np.sum(np.ptp(top2, axis=1) == 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split_re = collect(args.root)
    arrays = {}
    for split in ("train", "validation"):
        rows = [(value, label) for label, name in enumerate(CLASS_NAMES) for value in split_re[name][split]]
        arrays[split] = {
            "x": np.asarray([[row[0]] for row in rows], dtype=np.float32),
            "y": np.asarray([row[1] for row in rows], dtype=np.int64),
        }
    mean = arrays["train"]["x"].mean(0, keepdims=True)
    std = arrays["train"]["x"].std(0, keepdims=True)
    std[std < 1.0e-7] = 1.0
    audit = {
        "split_Re": split_re,
        "trajectory_counts": {split: len(arrays[split]["y"]) for split in arrays},
        "heldout_used_for_selection": False,
        "heldout_rows_used": False,
        "heldout_fields_or_metrics_loaded": False,
        "router_feature": "Re_only",
    }
    atomic_json(args.output_dir / "DATA_AUDIT.json", audit)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Router().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=1.5e-5)
    counts = np.bincount(arrays["train"]["y"], minlength=3).astype(np.float32)
    class_weight = torch.as_tensor(counts.sum() / (3.0 * counts), device=device)
    generator = np.random.default_rng(args.seed + 17)
    run = None
    if args.swanlab_mode != "disabled":
        import swanlab
        run = swanlab.init(
            project="FluidicPinball_Fusion_E2_T2C",
            group="E2_ReOnly",
            name=f"E2_fluidic_pinball_seed{args.seed}",
            mode=args.swanlab_mode,
            logdir=str(args.output_dir / "swanlog"),
            config={"steps": args.steps, "seed": args.seed, **audit},
            reinit=True,
        )
    best = None
    history = []
    started = time.time()
    try:
        for step in range(1, args.steps + 1):
            ids = generator.integers(0, len(arrays["train"]["y"]), size=min(256, len(arrays["train"]["y"])))
            x = torch.as_tensor((arrays["train"]["x"][ids] - mean) / std, device=device)
            y = torch.as_tensor(arrays["train"]["y"][ids], device=device)
            probabilities = torch.softmax(model(x), dim=1)
            cross_entropy = nn.functional.cross_entropy(model(x), y, weight=class_weight)
            adjacency = torch.mean(probabilities[:, 0] * probabilities[:, 2])
            loss = cross_entropy + 0.05 * adjacency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                with torch.inference_mode():
                    vx = torch.as_tensor((arrays["validation"]["x"] - mean) / std, device=device)
                    vp = torch.softmax(model(vx), dim=1).cpu().numpy()
                metrics = trajectory_metrics(arrays["validation"]["y"], vp)
                loss_value = float(loss.detach())
                score = metrics["balanced_accuracy"] + metrics["macro_f1"] - 0.001 * loss_value
                row = {"step": step, "loss": loss_value, "validation_score": score, **metrics}
                history.append(row)
                if best is None or score > best[0]:
                    best = (score, step, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, metrics)
                if run is not None:
                    import swanlab
                    swanlab.log(clean_numeric({f"validation/{key}": value for key, value in row.items()}), step=step)
                atomic_json(args.output_dir / "runtime_status.json", {"status": "running", **row, "updated_unix": time.time()})
        assert best is not None
        model.load_state_dict(best[2])
        results = {}
        for split in ("validation",):
            with torch.inference_mode():
                x = torch.as_tensor((arrays[split]["x"] - mean) / std, device=device)
                probabilities = torch.softmax(model(x), dim=1).cpu().numpy()
            results[split] = trajectory_metrics(arrays[split]["y"], probabilities)
        checkpoint = {
            "schema_version": 1,
            "model_state": best[2],
            "feature_mean": mean,
            "feature_std": std,
            "best_step": best[1],
            "validation_metrics": best[3],
            "seed": args.seed,
            "class_names": CLASS_NAMES,
            "split_Re": split_re,
        }
        torch.save(checkpoint, args.output_dir / "best.pt")
        atomic_json(args.output_dir / "history.json", history)
        atomic_json(args.output_dir / "metrics.json", results)
        atomic_json(args.output_dir / "runtime_status.json", {"status": "complete", "best_step": best[1], "elapsed_seconds": time.time() - started})
    finally:
        if run is not None:
            import swanlab
            swanlab.finish()


if __name__ == "__main__":
    main()
