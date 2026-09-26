#!/usr/bin/env python3
"""E1/E2/E3 trajectory-level Top-1 router experiments.

The three frozen HPRS-MoE specialists are never loaded or modified here.  This
entry point trains only a small outer router from complete-Re trajectory
splits.  Native rollout evidence is inherited from the independently reproduced
E0 artifacts and is bound by SHA256.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path("/root/panxy/particalMOE")
CLASS_NAMES = ("Steady", "Hopf", "Periodic")
EPS = 1.0e-12


ASSETS = {
    0: {
        "name": "Steady",
        "velocity": ROOT / "steady_specialist_v1/source_artifacts/steady/velocity_pod_steady.npz",
        "pressure": ROOT / "steady_specialist_v1/source_artifacts/steady/pressure_pod_steady.npz",
        "index": ROOT / "steady_specialist_v1/source_artifacts/steady/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv",
    },
    1: {
        "name": "Hopf",
        "velocity": ROOT / "Hopf/artifacts/hopf/velocity_pod_hopf.npz",
        "pressure": ROOT / "Hopf/artifacts/hopf/pressure_pod_hopf.npz",
        "index": ROOT / "Hopf/artifacts/hopf/projection_snapshots_velocity_hopf.csv",
    },
    2: {
        "name": "Periodic",
        "velocity": ROOT / "periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz",
        "pressure": ROOT / "periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz",
        "index": ROOT / "periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv",
    },
}

E0_FILES = {
    "Steady": ROOT / "e0_reproduction_20260722/steady/raw_metrics.json",
    "Hopf": ROOT / "e0_reproduction_20260722/hopf/heldout_metrics.json",
    "Periodic": ROOT / "e0_reproduction_20260722/periodic/periodic_r32_multihorizon_evaluation.json",
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_split(value: str) -> str:
    value = value.strip().lower()
    if value in {"heldout", "test"}:
        return "test"
    if value in {"validation", "val"}:
        return "validation"
    if value == "train":
        return "train"
    raise ValueError(f"unsupported split {value!r}")


def load_index(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["snapshot_id"]))
    return {
        "re": np.asarray([float(row["Re"]) for row in rows], dtype=np.float64),
        "time": np.asarray([float(row["time"]) for row in rows], dtype=np.float64),
        "split": np.asarray([normalize_split(row["split"]) for row in rows]),
    }


def weighted_geometry(phi: np.ndarray, mean: np.ndarray, weights: np.ndarray, vector: bool) -> tuple[np.ndarray, np.ndarray, float]:
    w = np.concatenate((weights, weights)) if vector else weights
    p = np.asarray(phi[:32], dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    gram = (p * w[None, :]) @ p.T
    cross = (p * w[None, :]) @ m
    mean_energy = float(np.dot(m * w, m))
    return gram, cross, mean_energy


def state_energy(coeff: np.ndarray, geometry: tuple[np.ndarray, np.ndarray, float]) -> float:
    gram, cross, mean_energy = geometry
    c = np.asarray(coeff, dtype=np.float64)
    return max(mean_energy + 2.0 * float(c @ cross) + float(c @ gram @ c), EPS)


def difference_norm(left: np.ndarray, right: np.ndarray, gram: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return math.sqrt(max(float(delta @ gram @ delta), 0.0))


def physical_descriptors(
    a0: np.ndarray, a1: np.ndarray, a2: np.ndarray,
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
    t0: float, t1: float, t2: float,
    u_geom: tuple[np.ndarray, np.ndarray, float],
    p_geom: tuple[np.ndarray, np.ndarray, float],
    total_area: float,
) -> np.ndarray:
    dt1, dt2 = max(t1 - t0, EPS), max(t2 - t1, EPS)
    ue0, ue1, ue2 = (state_energy(a, u_geom) for a in (a0, a1, a2))
    pe2 = state_energy(b2, p_geom)
    u_scale = math.sqrt(ue2) + EPS
    p_scale = math.sqrt(pe2) + EPS
    du1 = difference_norm(a1, a0, u_geom[0]) / dt1
    du2 = difference_norm(a2, a1, u_geom[0]) / dt2
    dp2 = difference_norm(b2, b1, p_geom[0]) / dt2
    v1 = (a1 - a0) / dt1
    v2 = (a2 - a1) / dt2
    curvature = math.sqrt(max(float((v2 - v1) @ u_geom[0] @ (v2 - v1)), 0.0)) / u_scale
    log_energy_growth = (math.log(ue2) - math.log(ue1)) / dt2
    previous_log_energy_growth = (math.log(ue1) - math.log(ue0)) / dt1
    return np.asarray([
        0.5 * math.log(max(ue2 / max(total_area, EPS), EPS)),
        0.5 * math.log(max(pe2 / max(total_area, EPS), EPS)),
        math.log1p(du2 / u_scale),
        math.log1p(du1 / (math.sqrt(ue1) + EPS)),
        math.tanh(log_energy_growth),
        math.tanh(log_energy_growth - previous_log_energy_growth),
        math.log1p(dp2 / p_scale),
        math.log1p(curvature),
    ], dtype=np.float64)


def evenly_spaced(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    locations = np.linspace(0, len(indices) - 1, num=limit, dtype=np.int64)
    return indices[locations]


def load_dataset(include_descriptors: bool, windows_per_re: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    split_re: dict[str, dict[str, list[float]]] = {name: {s: [] for s in ("train", "validation", "test")} for name in CLASS_NAMES}
    for label, spec in ASSETS.items():
        index = load_index(spec["index"])
        with np.load(spec["velocity"]) as velocity, np.load(spec["pressure"]) as pressure:
            a = np.asarray(velocity["coeff_uv"][:, :32], dtype=np.float64)
            b = np.asarray(pressure["coeff_p"][:, :32], dtype=np.float64)
            if len(a) != len(index["re"]) or len(b) != len(a):
                raise RuntimeError(f"{spec['name']} coefficient/index length mismatch")
            if include_descriptors:
                areas = np.asarray(velocity["point_areas"], dtype=np.float64)
                if not np.array_equal(areas, np.asarray(pressure["point_areas"], dtype=np.float64)):
                    raise RuntimeError(f"{spec['name']} velocity/pressure area mismatch")
                u_geom = weighted_geometry(velocity["phi_uv"], velocity["mean_uv_regime"], areas, True)
                p_geom = weighted_geometry(pressure["phi_p"], pressure["mean_p_regime"], areas, False)
                total_area = float(np.sum(areas))
            for re_value in np.unique(index["re"]):
                ids = np.flatnonzero(np.isclose(index["re"], re_value, rtol=0.0, atol=5e-8))
                ids = ids[np.argsort(index["time"][ids])]
                trajectory_splits = np.unique(index["split"][ids])
                if len(trajectory_splits) != 1:
                    raise RuntimeError(f"trajectory leakage inside {spec['name']} Re={re_value}: {trajectory_splits}")
                split = str(trajectory_splits[0])
                split_re[spec["name"]][split].append(float(re_value))
                valid = ids[2:]
                chosen = evenly_spaced(valid, windows_per_re)
                for current in chosen:
                    local = int(np.where(ids == current)[0][0])
                    history = ids[local - 2: local + 1]
                    descriptor = (
                        physical_descriptors(
                            a[history[0]], a[history[1]], a[history[2]],
                            b[history[0]], b[history[1]], b[history[2]],
                            index["time"][history[0]], index["time"][history[1]], index["time"][history[2]],
                            u_geom, p_geom, total_area,
                        ) if include_descriptors else np.empty(0, dtype=np.float64)
                    )
                    records.append({
                        "split": split, "label": label, "Re": float(re_value),
                        "features": np.concatenate(([float(re_value)], descriptor)),
                    })
    # Prove complete-Re isolation separately for each specialist and globally by (label, Re).
    for name, splits in split_re.items():
        sets = {key: set(np.round(values, 8)) for key, values in splits.items()}
        if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
            raise RuntimeError(f"complete-Re split leakage for {name}")
    output: dict[str, Any] = {"split_re": split_re}
    for split in ("train", "validation", "test"):
        subset = [row for row in records if row["split"] == split]
        output[split] = {
            "x": np.stack([row["features"] for row in subset]).astype(np.float32),
            "y": np.asarray([row["label"] for row in subset], dtype=np.int64),
            "re": np.asarray([row["Re"] for row in subset], dtype=np.float64),
        }
    return output


class Router(nn.Module):
    def __init__(self, input_dim: int, experiment: str) -> None:
        super().__init__()
        if experiment == "E2":
            self.net = nn.Sequential(nn.Linear(input_dim, 24), nn.Tanh(), nn.Linear(24, 3))
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64), nn.SiLU(), nn.LayerNorm(64),
                nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 3),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def confusion(true: np.ndarray, pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(true.tolist(), pred.tolist()):
        matrix[int(t), int(p)] += 1
    return matrix.tolist()


def classification_metrics(true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(probs, axis=1)
    recalls, f1s = [], []
    for c in range(3):
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, EPS))
    return {
        "accuracy": float(np.mean(pred == true)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": confusion(true, pred),
        "illegal_S_P_top2_count": int(np.sum(np.ptp(np.argsort(-probs, axis=1)[:, :2], axis=1) == 2)),
    }


def aggregate_by_trajectory(data: dict[str, np.ndarray], probs: np.ndarray) -> dict[str, Any]:
    rows = []
    true_trajectory, prob_trajectory = [], []
    for label in range(3):
        values = np.unique(data["re"][data["y"] == label])
        for value in values:
            mask = (data["y"] == label) & np.isclose(data["re"], value, atol=5e-8, rtol=0.0)
            mean_prob = np.mean(probs[mask], axis=0)
            selected = int(np.argmax(mean_prob))
            rows.append({
                "Re": float(value), "true": CLASS_NAMES[label], "selected": CLASS_NAMES[selected],
                "probabilities": {name: float(mean_prob[i]) for i, name in enumerate(CLASS_NAMES)},
                "correct": bool(selected == label), "windows": int(np.sum(mask)),
            })
            true_trajectory.append(label)
            prob_trajectory.append(mean_prob)
    metrics = classification_metrics(np.asarray(true_trajectory), np.stack(prob_trajectory))
    return {"metrics": metrics, "rows": rows}


def e0_manifest() -> dict[str, Any]:
    result = {}
    for name, path in E0_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {"path": str(path), "sha256": sha256(path)}
    return result


def init_swanlab(experiment: str, output_dir: Path, config: dict[str, Any], disabled: bool):
    if disabled:
        return None
    import swanlab
    return swanlab.init(
        project="V17_TrajectoryLevel_Hierarchical_MoE",
        group="E1_E2_E3_Top1",
        name=f"{experiment}_trajectory_top1_seed{config['seed']}",
        config=config,
        mode="online",
        logdir=str(output_dir / "swanlog"),
        reinit=True,
    )


def swan_log(run, payload: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    if step is None:
        run.log(payload)
    else:
        run.log(payload, step=step)


def oracle_experiment(data: dict[str, Any], output_dir: Path, run) -> None:
    test = data["test"]
    probs = np.eye(3, dtype=np.float64)[test["y"]]
    trajectory = aggregate_by_trajectory(test, probs)
    payload = {
        "experiment": "E1", "router": "oracle_trajectory_label", "trainable_parameters": 0,
        "test": trajectory, "e0_evidence": e0_manifest(),
    }
    atomic_json(output_dir / "oracle_results.json", payload)
    atomic_json(output_dir / "EVALUATION_STARTED.json", {"started_unix": time.time(), "pid": os.getpid()})
    swan_log(run, {f"test/{k}": v for k, v in trajectory["metrics"].items() if isinstance(v, (int, float))})
    atomic_json(output_dir / "COMPLETE.json", {"status": "COMPLETE", "finished_unix": time.time()})


def predict(model: nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        inputs = torch.as_tensor((x - mean) / std, device=device)
        return torch.softmax(model(inputs), dim=1).cpu().numpy()


def train_experiment(args: argparse.Namespace, data: dict[str, Any], output_dir: Path, run) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train, validation, test = data["train"], data["validation"], data["test"]
    mean = np.mean(train["x"], axis=0, keepdims=True).astype(np.float32)
    std = np.std(train["x"], axis=0, keepdims=True).astype(np.float32)
    std[std < 1.0e-7] = 1.0
    model = Router(train["x"].shape[1], args.experiment).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05)
    rng = np.random.default_rng(args.seed + 17)
    best = {"score": -float("inf"), "step": -1, "state": None, "metrics": None}
    class_counts = np.bincount(train["y"], minlength=3).astype(np.float64)
    class_weights = torch.as_tensor(class_counts.sum() / np.maximum(3.0 * class_counts, 1.0), dtype=torch.float32, device=device)
    atomic_json(output_dir / "TRAINING_STARTED.json", {
        "started_unix": time.time(), "pid": os.getpid(), "device": str(device),
        "specialists_frozen_and_not_loaded": True,
    })
    for step in range(1, args.steps + 1):
        ids = rng.integers(0, len(train["y"]), size=min(args.batch_size, len(train["y"])))
        x = torch.as_tensor((train["x"][ids] - mean) / std, device=device)
        y = torch.as_tensor(train["y"][ids], device=device)
        logits = model(x)
        ce = nn.functional.cross_entropy(logits, y, weight=class_weights)
        probabilities = torch.softmax(logits, dim=1)
        adjacency = torch.mean(probabilities[:, 0] * probabilities[:, 2])
        loss = ce + 0.05 * adjacency
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1:
            atomic_json(output_dir / "FIRST_STEP_COMPLETE.json", {"step": 1, "loss": float(loss.detach().cpu())})
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val_probs = predict(model, validation["x"], mean, std, device)
            val_trajectory = aggregate_by_trajectory(validation, val_probs)
            metrics = val_trajectory["metrics"]
            score = metrics["balanced_accuracy"] + metrics["macro_f1"] - 0.001 * float(loss.detach().cpu())
            if score > best["score"]:
                best = {
                    "score": float(score), "step": step,
                    "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "metrics": metrics,
                }
                torch.save({
                    "experiment": args.experiment, "step": step, "model_state": best["state"],
                    "feature_mean": mean, "feature_std": std, "validation_metrics": metrics,
                    "seed": args.seed, "input_dim": train["x"].shape[1],
                }, output_dir / "best.pt")
            swan_log(run, {
                "train/loss": float(loss.detach().cpu()), "train/cross_entropy": float(ce.detach().cpu()),
                "train/adjacency_penalty": float(adjacency.detach().cpu()),
                "validation/balanced_accuracy": metrics["balanced_accuracy"],
                "validation/macro_f1": metrics["macro_f1"], "lr": scheduler.get_last_lr()[0],
            }, step)
            print(json.dumps({"event": "evaluation", "experiment": args.experiment, "step": step, "loss": float(loss.detach().cpu()), **metrics}), flush=True)
            model.train()
    if best["state"] is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best["state"])
    model.eval()
    val_probs = predict(model, validation["x"], mean, std, device)
    test_probs = predict(model, test["x"], mean, std, device)
    result = {
        "experiment": args.experiment,
        "router": "Re-only" if args.experiment == "E2" else "Re+physical-descriptors",
        "specialists_frozen_and_not_loaded": True,
        "best_step": best["step"], "best_validation_metrics": best["metrics"],
        "validation": aggregate_by_trajectory(validation, val_probs),
        "test": aggregate_by_trajectory(test, test_probs),
        "split_Re": data["split_re"], "e0_evidence": e0_manifest(),
    }
    torch.save({
        "experiment": args.experiment, "step": args.steps, "model_state": model.state_dict(),
        "feature_mean": mean, "feature_std": std, "seed": args.seed,
    }, output_dir / "last.pt")
    atomic_json(output_dir / "metrics.json", result)
    atomic_json(output_dir / "COMPLETE.json", {"status": "COMPLETE", "finished_unix": time.time(), "best_step": best["step"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("E1", "E2", "E3"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--windows-per-re", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-swanlab", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = {
        "experiment": args.experiment, "seed": args.seed, "steps": args.steps,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "windows_per_Re": args.windows_per_re,
        "routing_granularity": "trajectory-level fixed Top-1",
        "specialists": "frozen and not loaded during outer-router training",
        "split": "complete Re and complete trajectories",
        "feature_contract": "Re only" if args.experiment in {"E1", "E2"} else "Re plus current/previous-two physical L2 descriptors",
    }
    atomic_json(args.output_dir / "STARTED.json", {**config, "started_unix": time.time(), "pid": os.getpid()})
    data = load_dataset(include_descriptors=args.experiment == "E3", windows_per_re=args.windows_per_re)
    atomic_json(args.output_dir / "DATA_AUDIT.json", {
        "split_Re": data["split_re"],
        "samples": {split: int(len(data[split]["y"])) for split in ("train", "validation", "test")},
        "input_dim": int(data["train"]["x"].shape[1]),
        "complete_Re_isolation": True,
    })
    run = init_swanlab(args.experiment, args.output_dir, config, args.disable_swanlab)
    try:
        if args.experiment == "E1":
            oracle_experiment(data, args.output_dir, run)
        else:
            train_experiment(args, data, args.output_dir, run)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
