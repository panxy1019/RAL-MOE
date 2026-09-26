#!/usr/bin/env python3
"""Leak-free E3: Re plus chart-independent fixed-sensor descriptors.

The router contract is deliberately pre-routing: it receives Re, three physical
timestamps, and velocity/pressure values at a fixed set of mesh sensors.  The
descriptor function has no chart or class argument.  POD assets are used only
to deserialize the historical training databases into those physical sensor
observations and to audit representation invariance.
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
RANK = 32
SENSOR_COUNT = 64

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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def fixed_sensor_indices(points: np.ndarray, count: int = SENSOR_COUNT) -> np.ndarray:
    """Choose sensors from geometry only, independent of chart and labels."""
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    locations = np.linspace(0, len(order) - 1, num=count, dtype=np.int64)
    return np.asarray(order[locations], dtype=np.int64)


def sensor_state(coeff_u: np.ndarray, coeff_p: np.ndarray, velocity: Any,
                 pressure: Any, sensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(velocity["point_areas"])
    phi_u = np.asarray(velocity["phi_uv"][:RANK], dtype=np.float64)
    mean_u = np.asarray(velocity["mean_uv_regime"], dtype=np.float64)
    indices_u = np.concatenate((sensors, sensors + n))
    u = mean_u[indices_u] + np.asarray(coeff_u[:RANK], dtype=np.float64) @ phi_u[:, indices_u]
    phi_p = np.asarray(pressure["phi_p"][:RANK], dtype=np.float64)
    mean_p = np.asarray(pressure["mean_p_regime"], dtype=np.float64)
    p = mean_p[sensors] + np.asarray(coeff_p[:RANK], dtype=np.float64) @ phi_p[:, sensors]
    return u.reshape(2, -1), p


def fixed_sensor_operator(velocity: Any, pressure: Any, sensors: np.ndarray) -> tuple[np.ndarray, ...]:
    """Materialize the small observation operator once per database chart."""
    n = len(velocity["point_areas"])
    indices_u = np.concatenate((sensors, sensors + n))
    return (
        np.asarray(velocity["phi_uv"][:RANK, indices_u], dtype=np.float64),
        np.asarray(velocity["mean_uv_regime"][indices_u], dtype=np.float64),
        np.asarray(pressure["phi_p"][:RANK, sensors], dtype=np.float64),
        np.asarray(pressure["mean_p_regime"][sensors], dtype=np.float64),
    )


def observed_sensor_state(coeff_u: np.ndarray, coeff_p: np.ndarray,
                          operator: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    phi_u, mean_u, phi_p, mean_p = operator
    u = mean_u + np.asarray(coeff_u[:RANK], dtype=np.float64) @ phi_u
    p = mean_p + np.asarray(coeff_p[:RANK], dtype=np.float64) @ phi_p
    return u.reshape(2, -1), p


def physical_sensor_descriptors(u0: np.ndarray, u1: np.ndarray, u2: np.ndarray,
                                p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                                t0: float, t1: float, t2: float) -> np.ndarray:
    """Eight chart-independent scalars from observed physical sensor values."""
    dt1, dt2 = max(t1 - t0, EPS), max(t2 - t1, EPS)
    urms = [math.sqrt(max(float(np.mean(u * u)), EPS)) for u in (u0, u1, u2)]
    prms2 = math.sqrt(max(float(np.mean(p2 * p2)), EPS))
    du1 = math.sqrt(float(np.mean(((u1 - u0) / dt1) ** 2)))
    du2 = math.sqrt(float(np.mean(((u2 - u1) / dt2) ** 2)))
    dp2 = math.sqrt(float(np.mean(((p2 - p1) / dt2) ** 2)))
    v1, v2 = (u1 - u0) / dt1, (u2 - u1) / dt2
    curvature = math.sqrt(float(np.mean((v2 - v1) ** 2))) / (urms[2] + EPS)
    growth1 = (math.log(urms[1]) - math.log(urms[0])) / dt1
    growth2 = (math.log(urms[2]) - math.log(urms[1])) / dt2
    return np.asarray([
        math.log(urms[2]),
        math.log(prms2),
        math.log1p(du2 / (urms[2] + EPS)),
        math.log1p(du1 / (urms[1] + EPS)),
        math.tanh(growth2),
        math.tanh(growth2 - growth1),
        math.log1p(dp2 / (prms2 + EPS)),
        math.log1p(curvature),
    ], dtype=np.float64)


def evenly_spaced(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices) - 1, num=limit, dtype=np.int64)]


def verify_common_mesh() -> tuple[np.ndarray, dict[str, Any]]:
    reference_points = reference_areas = reference_gauge = None
    audit: dict[str, Any] = {"charts": {}}
    for label, spec in ASSETS.items():
        with np.load(spec["velocity"]) as velocity, np.load(spec["pressure"]) as pressure:
            points = np.asarray(velocity["points"])
            areas = np.asarray(velocity["point_areas"])
            gauge = str(np.asarray(pressure["pressure_gauge"]).item())
            local_ok = np.array_equal(points, np.asarray(pressure["points"])) and np.array_equal(
                areas, np.asarray(pressure["point_areas"])
            )
            if reference_points is None:
                reference_points, reference_areas, reference_gauge = points.copy(), areas.copy(), gauge
            common_ok = np.array_equal(reference_points, points) and np.array_equal(reference_areas, areas)
            gauge_ok = reference_gauge == gauge
            audit["charts"][spec["name"]] = {
                "velocity_pressure_mesh_equal": bool(local_ok),
                "common_point_order_exact": bool(common_ok),
                "common_cell_area_exact": bool(common_ok),
                "pressure_gauge": gauge,
                "common_pressure_gauge": bool(gauge_ok),
            }
            if not (local_ok and common_ok and gauge_ok):
                raise RuntimeError(f"common physical observation contract failed for {spec['name']}")
    sensors = fixed_sensor_indices(reference_points)
    audit.update({
        "passed": True,
        "sensor_count": int(len(sensors)),
        "sensor_indices": sensors.tolist(),
        "selection": "64 indices evenly spaced after lexicographic sorting of common mesh coordinates",
        "input_available_before_routing": True,
    })
    return sensors, audit


def load_dataset(sensors: np.ndarray, windows_per_re: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    split_re = {name: {s: [] for s in ("train", "validation", "test")} for name in CLASS_NAMES}
    for label, spec in ASSETS.items():
        index = load_index(spec["index"])
        with np.load(spec["velocity"]) as velocity, np.load(spec["pressure"]) as pressure:
            coeff_u = np.asarray(velocity["coeff_uv"][:, :RANK], dtype=np.float64)
            coeff_p = np.asarray(pressure["coeff_p"][:, :RANK], dtype=np.float64)
            if len(coeff_u) != len(index["re"]) or len(coeff_p) != len(coeff_u):
                raise RuntimeError(f"{spec['name']} coefficient/index length mismatch")
            observation_operator = fixed_sensor_operator(velocity, pressure, sensors)
            sensor_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for re_value in np.unique(index["re"]):
                ids = np.flatnonzero(np.isclose(index["re"], re_value, rtol=0.0, atol=5e-8))
                ids = ids[np.argsort(index["time"][ids])]
                trajectory_splits = np.unique(index["split"][ids])
                if len(trajectory_splits) != 1:
                    raise RuntimeError(f"trajectory leakage inside {spec['name']} Re={re_value}")
                split = str(trajectory_splits[0])
                split_re[spec["name"]][split].append(float(re_value))
                for current in evenly_spaced(ids[2:], windows_per_re):
                    local = int(np.where(ids == current)[0][0])
                    history = ids[local - 2:local + 1]
                    states = []
                    for snapshot in history:
                        key = int(snapshot)
                        if key not in sensor_cache:
                            sensor_cache[key] = observed_sensor_state(coeff_u[key], coeff_p[key], observation_operator)
                        states.append(sensor_cache[key])
                    descriptor = physical_sensor_descriptors(
                        states[0][0], states[1][0], states[2][0],
                        states[0][1], states[1][1], states[2][1],
                        *(float(index["time"][i]) for i in history),
                    )
                    records.append({
                        "split": split, "label": label, "Re": float(re_value),
                        "features": np.concatenate(([float(re_value)], descriptor)),
                    })
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


def affine_map(source: Any, target: Any, vector: bool) -> tuple[np.ndarray, np.ndarray]:
    areas = np.asarray(source["point_areas"], dtype=np.float64)
    weights = np.concatenate((areas, areas)) if vector else areas
    phi_s = np.asarray(source["phi_uv" if vector else "phi_p"][:RANK], dtype=np.float64)
    phi_t = np.asarray(target["phi_uv" if vector else "phi_p"][:RANK], dtype=np.float64)
    mean_s = np.asarray(source["mean_uv_regime" if vector else "mean_p_regime"], dtype=np.float64)
    mean_t = np.asarray(target["mean_uv_regime" if vector else "mean_p_regime"], dtype=np.float64)
    linear = (phi_t * weights[None, :]) @ phi_s.T
    offset = (phi_t * weights[None, :]) @ (mean_s - mean_t)
    return linear, offset


def representation_audit(sensors: np.ndarray, histories_per_direction: int = 8) -> dict[str, Any]:
    directions = ((0, 1), (1, 0), (1, 2), (2, 1))
    result: dict[str, Any] = {
        "definition": "same source database physical state projected to target chart; never pairs different-Re states",
        "S_H_sampling_note": "Re ranges overlap, but there are no exactly shared sampled Re values",
        "H_P_sampling_note": "Re ranges do not overlap; H-P results are extrapolative representation diagnostics, not overlap certification",
        "directions": {},
    }
    for source_id, target_id in directions:
        source_spec, target_spec = ASSETS[source_id], ASSETS[target_id]
        source_index = load_index(source_spec["index"])
        with np.load(source_spec["velocity"]) as su, np.load(source_spec["pressure"]) as sp, \
             np.load(target_spec["velocity"]) as tu, np.load(target_spec["pressure"]) as tp:
            map_u, offset_u = affine_map(su, tu, True)
            map_p, offset_p = affine_map(sp, tp, False)
            coeff_u = np.asarray(su["coeff_uv"][:, :RANK], dtype=np.float64)
            coeff_p = np.asarray(sp["coeff_p"][:, :RANK], dtype=np.float64)
            unique_re = np.unique(source_index["re"])
            boundary_re = unique_re[-min(4, len(unique_re)):] if source_id < target_id else unique_re[:min(4, len(unique_re))]
            candidates: list[tuple[np.ndarray, float]] = []
            for re_value in boundary_re:
                ids = np.flatnonzero(np.isclose(source_index["re"], re_value, atol=5e-8, rtol=0.0))
                ids = ids[np.argsort(source_index["time"][ids])]
                for current in evenly_spaced(ids[2:], max(1, histories_per_direction // len(boundary_re))):
                    local = int(np.where(ids == current)[0][0])
                    candidates.append((ids[local - 2:local + 1], float(re_value)))
            absolute, relative, sensor_errors, rows = [], [], [], []
            for history, re_value in candidates[:histories_per_direction]:
                source_states, target_states = [], []
                for snapshot in history:
                    source_states.append(sensor_state(coeff_u[snapshot], coeff_p[snapshot], su, sp, sensors))
                    mapped_u = map_u @ coeff_u[snapshot] + offset_u
                    mapped_p = map_p @ coeff_p[snapshot] + offset_p
                    target_states.append(sensor_state(mapped_u, mapped_p, tu, tp, sensors))
                times = [float(source_index["time"][i]) for i in history]
                d_source = physical_sensor_descriptors(
                    source_states[0][0], source_states[1][0], source_states[2][0],
                    source_states[0][1], source_states[1][1], source_states[2][1], *times)
                d_target = physical_sensor_descriptors(
                    target_states[0][0], target_states[1][0], target_states[2][0],
                    target_states[0][1], target_states[1][1], target_states[2][1], *times)
                abs_error = np.abs(d_target - d_source)
                rel_error = abs_error / np.maximum(np.abs(d_source), 1.0e-6)
                u_err = np.linalg.norm(target_states[-1][0] - source_states[-1][0]) / max(np.linalg.norm(source_states[-1][0]), EPS)
                p_err = np.linalg.norm(target_states[-1][1] - source_states[-1][1]) / max(np.linalg.norm(source_states[-1][1]), EPS)
                absolute.append(abs_error); relative.append(rel_error); sensor_errors.append([u_err, p_err])
                rows.append({"Re": re_value, "time": times[-1], "descriptor_abs_max": float(abs_error.max()),
                             "descriptor_rel_l2": float(np.linalg.norm(d_target-d_source)/max(np.linalg.norm(d_source), EPS)),
                             "velocity_sensor_relative_error": float(u_err), "pressure_sensor_relative_error": float(p_err)})
            absolute_a, relative_a, sensor_a = np.stack(absolute), np.stack(relative), np.asarray(sensor_errors)
            key = f"{source_spec['name']}->{target_spec['name']}"
            result["directions"][key] = {
                "histories": len(rows), "source_Re": sorted({row["Re"] for row in rows}),
                "descriptor_abs_mean_by_feature": absolute_a.mean(axis=0).tolist(),
                "descriptor_abs_max": float(absolute_a.max()),
                "descriptor_relative_median": float(np.median(relative_a)),
                "descriptor_relative_p95": float(np.percentile(relative_a, 95)),
                "descriptor_relative_l2_mean": float(np.mean([row["descriptor_rel_l2"] for row in rows])),
                "velocity_sensor_relative_error_mean": float(sensor_a[:, 0].mean()),
                "pressure_sensor_relative_error_mean": float(sensor_a[:, 1].mean()),
                "rows": rows,
            }
    return result


class Router(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.SiLU(), nn.LayerNorm(64),
                                 nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def classification_metrics(true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(probs, axis=1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    recalls, f1s = [], []
    for t, p in zip(true.tolist(), pred.tolist()):
        matrix[int(t), int(p)] += 1
    for c in range(3):
        tp, fp, fn = matrix[c, c], matrix[:, c].sum()-matrix[c, c], matrix[c, :].sum()-matrix[c, c]
        recall, precision = tp/max(tp+fn, 1), tp/max(tp+fp, 1)
        recalls.append(recall); f1s.append(2*precision*recall/max(precision+recall, EPS))
    nll = -float(np.mean(np.log(np.maximum(probs[np.arange(len(true)), true], EPS))))
    onehot = np.eye(3)[true]
    return {"accuracy": float(np.mean(pred == true)), "balanced_accuracy": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s)), "confusion_matrix": matrix.tolist(), "nll": nll,
            "brier": float(np.mean(np.sum((probs-onehot)**2, axis=1))),
            "illegal_S_P_top2_count": int(np.sum(np.ptp(np.argsort(-probs, axis=1)[:, :2], axis=1) == 2))}


def aggregate(data: dict[str, np.ndarray], probs: np.ndarray) -> dict[str, Any]:
    rows, truths, trajectory_probs = [], [], []
    for label in range(3):
        for re_value in np.unique(data["re"][data["y"] == label]):
            mask = (data["y"] == label) & np.isclose(data["re"], re_value, atol=5e-8, rtol=0.0)
            mean_prob = probs[mask].mean(axis=0)
            selected = int(np.argmax(mean_prob))
            rows.append({"Re": float(re_value), "true": CLASS_NAMES[label], "selected": CLASS_NAMES[selected],
                         "probabilities": {name: float(mean_prob[i]) for i, name in enumerate(CLASS_NAMES)},
                         "correct": bool(selected == label), "windows": int(mask.sum())})
            truths.append(label); trajectory_probs.append(mean_prob)
    return {"metrics": classification_metrics(np.asarray(truths), np.stack(trajectory_probs)), "rows": rows}


def predict(model: nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(model(torch.as_tensor((x-mean)/std, device=device)), dim=1).cpu().numpy()


def e0_manifest() -> dict[str, Any]:
    return {name: {"path": str(path), "sha256": sha256(path)} for name, path in E0_FILES.items()}


def train(args: argparse.Namespace, data: dict[str, Any], output: Path, sensors: np.ndarray, run: Any) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_data, validation, test = data["train"], data["validation"], data["test"]
    mean = train_data["x"].mean(axis=0, keepdims=True).astype(np.float32)
    std = train_data["x"].std(axis=0, keepdims=True).astype(np.float32); std[std < 1e-7] = 1.0
    model = Router(train_data["x"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate*0.05)
    counts = np.bincount(train_data["y"], minlength=3).astype(np.float64)
    weights = torch.as_tensor(counts.sum()/np.maximum(3*counts, 1), dtype=torch.float32, device=device)
    rng = np.random.default_rng(args.seed + 17)
    best: dict[str, Any] = {"score": -float("inf")}
    atomic_json(output/"TRAINING_STARTED.json", {"pid": os.getpid(), "started_unix": time.time(), "device": str(device)})
    for step in range(1, args.steps+1):
        ids = rng.integers(0, len(train_data["y"]), size=min(args.batch_size, len(train_data["y"])))
        x = torch.as_tensor((train_data["x"][ids]-mean)/std, device=device)
        y = torch.as_tensor(train_data["y"][ids], device=device)
        logits = model(x); probabilities = torch.softmax(logits, dim=1)
        ce = nn.functional.cross_entropy(logits, y, weight=weights)
        adjacency = torch.mean(probabilities[:, 0]*probabilities[:, 2]); loss = ce + 0.05*adjacency
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        if step == 1:
            atomic_json(output/"FIRST_STEP_COMPLETE.json", {"step": 1, "loss": float(loss.detach().cpu())})
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval(); val = aggregate(validation, predict(model, validation["x"], mean, std, device)); metrics = val["metrics"]
            score = metrics["balanced_accuracy"] + metrics["macro_f1"] - 0.001*float(loss.detach().cpu())
            if score > best["score"]:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best = {"score": float(score), "step": step, "state": state, "metrics": metrics}
                torch.save({"experiment": "E3_chart_independent", "step": step, "model_state": state,
                            "feature_mean": mean, "feature_std": std, "validation_metrics": metrics,
                            "seed": args.seed, "input_dim": 9, "sensor_indices": sensors}, output/"best.pt")
            if run is not None:
                run.log({"train/loss": float(loss.detach().cpu()), "train/cross_entropy": float(ce.detach().cpu()),
                         "train/adjacency_penalty": float(adjacency.detach().cpu()),
                         "validation/balanced_accuracy": metrics["balanced_accuracy"],
                         "validation/macro_f1": metrics["macro_f1"], "validation/nll": metrics["nll"],
                         "lr": scheduler.get_last_lr()[0]}, step=step)
            print(json.dumps({"event": "evaluation", "step": step, "loss": float(loss.detach().cpu()), **metrics}), flush=True)
            model.train()
    model.load_state_dict(best["state"]); model.eval()
    val = aggregate(validation, predict(model, validation["x"], mean, std, device))
    test_result = aggregate(test, predict(model, test["x"], mean, std, device))
    torch.save({"experiment": "E3_chart_independent", "step": args.steps, "model_state": model.state_dict(),
                "feature_mean": mean, "feature_std": std, "seed": args.seed, "sensor_indices": sensors}, output/"last.pt")
    atomic_json(output/"metrics.json", {"experiment": "E3_chart_independent", "router": "Re+fixed-physical-sensor-descriptors",
                "descriptor_contract": "pre-routing Re + 3 timestamps + fixed physical sensor observations; no chart/label input",
                "best_step": best["step"], "best_validation_metrics": best["metrics"], "validation": val, "test": test_result,
                "split_Re": data["split_re"], "e0_evidence": e0_manifest()})
    atomic_json(output/"COMPLETE.json", {"status": "COMPLETE", "finished_unix": time.time(), "best_step": best["step"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31003)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--windows-per-re", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-swanlab", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    config = {"experiment": "E3_chart_independent", "seed": args.seed, "steps": args.steps,
              "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "windows_per_Re": args.windows_per_re, "split": "original complete Re and complete trajectories",
              "feature_contract": "Re + 8 chart-independent fixed-physical-sensor descriptors",
              "sensor_count": SENSOR_COUNT, "specialists": "frozen and not loaded"}
    atomic_json(args.output_dir/"STARTED.json", {**config, "pid": os.getpid(), "started_unix": time.time()})
    try:
        sensors, mesh_audit = verify_common_mesh(); atomic_json(args.output_dir/"MESH_CONTRACT_AUDIT.json", mesh_audit)
        descriptor_audit = representation_audit(sensors); atomic_json(args.output_dir/"DESCRIPTOR_AUDIT.json", descriptor_audit)
        if args.audit_only:
            atomic_json(args.output_dir/"COMPLETE.json", {"status": "AUDIT_COMPLETE", "finished_unix": time.time()}); return
        data = load_dataset(sensors, args.windows_per_re)
        atomic_json(args.output_dir/"DATA_AUDIT.json", {"split_Re": data["split_re"],
                    "samples": {s: int(len(data[s]["y"])) for s in ("train", "validation", "test")},
                    "input_dim": 9, "complete_Re_isolation": True, "trajectory_isolation": True,
                    "descriptor_function_uses_chart_or_label": False})
        run = None
        if not args.disable_swanlab:
            import swanlab
            run = swanlab.init(project="V17_TrajectoryLevel_Hierarchical_MoE", group="E3_chart_independent_repair",
                               name=f"E3_chart_independent_seed{args.seed}", config=config, mode="online",
                               logdir=str(args.output_dir/"swanlog"), reinit=True)
        try: train(args, data, args.output_dir, sensors, run)
        finally:
            if run is not None: run.finish()
    except Exception as exc:
        atomic_json(args.output_dir/"BLOCKED.json", {"status": "BLOCKED", "error": repr(exc), "finished_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
