#!/usr/bin/env python3
"""Validation-first physical-field evaluator for specialist ablation checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


EPS = 1.0e-12
HORIZONS = (1, 4, 8, 16, 48, 56)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def links(re_values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((times, re_values))
    if not np.array_equal(order, np.arange(len(order))):
        raise RuntimeError("evaluation coefficient asset is not ordered by (Re,time)")
    previous = np.full(len(re_values), -1, dtype=np.int64)
    following = np.full(len(re_values), -1, dtype=np.int64)
    for value in np.unique(re_values):
        ids = np.flatnonzero(np.isclose(re_values, value, atol=5.0e-7, rtol=0))
        previous[ids[1:]] = ids[:-1]
        following[ids[:-1]] = ids[1:]
    return previous, following


def history(previous: np.ndarray) -> np.ndarray:
    result = np.full((len(previous), 3), -1, dtype=np.int64)
    result[:, 0] = np.arange(len(previous))
    for column in (1, 2):
        valid = result[:, column - 1] >= 0
        result[valid, column] = previous[result[valid, column - 1]]
    return result


def complete_re_audit(data: dict[str, Any]) -> dict[str, list[float]]:
    sets = {
        name: set(np.round(data["re"][data["split"] == name], 6).tolist())
        for name in ("train", "validation", "heldout")
    }
    if sets["train"] & sets["validation"] or sets["train"] & sets["heldout"] or sets[
        "validation"
    ] & sets["heldout"]:
        raise RuntimeError("complete-Re split leakage in evaluator")
    return {name: sorted(values) for name, values in sets.items()}


def load_full_data(root: Path, regime: str, training_module: Any) -> dict[str, Any]:
    if regime == "steady":
        pod = root / "steady_specialist_v1/source_artifacts/steady/Global_POD_AreaWeighted_L2"
        velocity = np.load(pod / "global_velocity_pod_area_weighted_l2.npz")
        pressure = np.load(pod / "global_pressure_pod_area_weighted_l2.npz")
        with (pod / "pod_snapshot_index.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        data = {
            "a": velocity["coeff_uv"][:, :32].astype(np.float32),
            "b": pressure["coeff_p"][:, :32].astype(np.float32),
            "re": np.asarray([float(row["Re"]) for row in rows], np.float32),
            "time": np.asarray([float(row["time"]) for row in rows], np.float32),
            "split": np.char.lower(velocity["snapshot_splits"].astype(str)),
            "phase": np.zeros(len(rows), np.float32),
            "phase_harmonics": 0,
            "phi_u": velocity["phi_uv"][:32].astype(np.float64),
            "phi_p": pressure["phi_p"][:32].astype(np.float64),
            "mean_u": velocity["mean_uv_regime"].astype(np.float64),
            "mean_p": pressure["mean_p_regime"].astype(np.float64),
            "areas": velocity["point_areas"].astype(np.float64),
            "pressure_gauge": str(pressure["pressure_gauge"].item()),
        }
    elif regime == "hopf":
        pod = root / "Hopf/artifacts/hopf"
        velocity = np.load(pod / "velocity_pod_hopf.npz")
        pressure = np.load(pod / "pressure_pod_hopf.npz")
        with (pod / "projection_snapshots_velocity_hopf.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        data = {
            "a": velocity["coeff_uv"][:, :32].astype(np.float32),
            "b": pressure["coeff_p"][:, :32].astype(np.float32),
            "re": np.asarray([float(row["Re"]) for row in rows], np.float32),
            "time": np.asarray([float(row["time"]) for row in rows], np.float32),
            "split": np.char.lower(
                np.asarray([row["split"] for row in rows], dtype=str)
            ),
            "phase": np.zeros(len(rows), np.float32),
            "phase_harmonics": 0,
            "phi_u": velocity["phi_uv"][:32].astype(np.float64),
            "phi_p": pressure["phi_p"][:32].astype(np.float64),
            "mean_u": velocity["mean_uv_regime"].astype(np.float64),
            "mean_p": pressure["mean_p_regime"].astype(np.float64),
            "areas": velocity["point_areas"].astype(np.float64),
            "pressure_gauge": str(
                pressure["pressure_gauge"].item()
                if "pressure_gauge" in pressure.files
                else "native_hopf_training_gauge"
            ),
        }
    else:
        native = training_module.load_data(root, "periodic")
        pod = root / "periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2"
        velocity = np.load(pod / "global_velocity_pod_area_weighted_l2.npz")
        pressure = np.load(pod / "global_pressure_pod_area_weighted_l2.npz")
        data = {
            **native,
            "phi_u": velocity["phi_uv"][:32].astype(np.float64),
            "phi_p": pressure["phi_p"][:32].astype(np.float64),
            "mean_u": velocity["mean_uv_regime"].astype(np.float64),
            "mean_p": pressure["mean_p_regime"].astype(np.float64),
            "areas": velocity["point_areas"].astype(np.float64),
            "pressure_gauge": str(pressure["pressure_gauge"].item()),
        }
        complete_re_audit(data)
        return data
    data["split"][data["split"] == "test"] = "heldout"
    previous, following = links(data["re"], data["time"])
    data["prev"] = previous
    data["next"] = following
    data["history"] = history(previous)
    complete_re_audit(data)
    return data


def legal_starts(data: dict[str, Any], split: str, horizon: int) -> np.ndarray:
    candidates = np.flatnonzero(
        (data["split"] == split)
        & np.all(data["history"] >= 0, axis=1)
        & (data["next"] >= 0)
    )
    legal = []
    for start in candidates.tolist():
        current = start
        for _ in range(horizon):
            current = int(data["next"][current])
            if current < 0 or data["split"][current] != split:
                break
        else:
            legal.append(start)
    return np.asarray(legal, dtype=np.int64)


def geometry(phi: np.ndarray, mean: np.ndarray, areas: np.ndarray, vector: bool):
    weights = np.concatenate([areas, areas]) if vector else areas
    gram = (phi * weights[None, :]) @ phi.T
    cross = (phi * weights[None, :]) @ mean
    mean_energy = float(np.sum(weights * mean * mean))
    return gram, cross, mean_energy


def error_energy(delta: np.ndarray, geom: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.einsum("...i,ij,...j->...", delta, geom[0], delta)


def field_energy(coeff: np.ndarray, geom: tuple[np.ndarray, ...]) -> np.ndarray:
    gram, cross, mean_energy = geom
    return (
        mean_energy
        + 2.0 * (coeff @ cross)
        + np.einsum("...i,ij,...j->...", coeff, gram, coeff)
    )


def dataonly_predictions(
    module: Any,
    checkpoint: dict[str, Any],
    data: dict[str, Any],
    starts: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    args = SimpleNamespace(**checkpoint["args"])
    groups = 1 if args.regime == "hopf" else 3
    hidden = 256 if args.regime == "hopf" else 224
    expert_hidden = 1024 if args.regime == "hopf" else 768
    blocks = 4 if args.regime == "hopf" else 3
    model = module.HierarchicalSparseMoE(
        3 * 64 + 1 + 2 * data["phase_harmonics"],
        64,
        hidden,
        expert_hidden,
        groups,
        6,
        blocks,
        0.04,
    ).cuda()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalization = checkpoint["normalization"]
    mean = torch.as_tensor(normalization["mean"], device="cuda")
    scale = torch.as_tensor(normalization["scale"], device="cuda")
    train = module.load_data(Path(args.root), args.regime)
    re_train = train["re"][train["split"] == "train"]
    re_center = float(np.mean(re_train))
    re_scale = float(max(np.std(re_train), 1.0))
    states = np.concatenate([data["a"], data["b"]], axis=1).astype(np.float32)
    predicted_batches, truth_batches = [], []
    with torch.inference_mode():
        for offset in range(0, len(starts), 32):
            batch = starts[offset : offset + 32]
            hist = torch.as_tensor(states[data["history"][batch]], device="cuda")
            current = batch.copy()
            predicted, truth = [], []
            for _ in range(horizon):
                normalized = (hist - mean) / scale
                columns = [
                    normalized.flatten(1),
                    torch.as_tensor(
                        ((data["re"][current] - re_center) / re_scale)[:, None],
                        device="cuda",
                    ),
                ]
                for harmonic in range(1, data["phase_harmonics"] + 1):
                    angle = torch.as_tensor(
                        2.0 * math.pi * harmonic * data["phase"][current],
                        device="cuda",
                    )
                    columns.extend([torch.sin(angle)[:, None], torch.cos(angle)[:, None]])
                delta, _ = model(torch.cat(columns, 1))
                next_state = hist[:, 0] + delta.float() * scale
                current = data["next"][current]
                predicted.append(next_state.cpu().numpy())
                truth.append(states[current])
                hist = torch.cat([next_state[:, None], hist[:, :-1]], 1)
            predicted_batches.append(np.stack(predicted, 1))
            truth_batches.append(np.stack(truth, 1))
    prediction = np.concatenate(predicted_batches, 0)
    truth = np.concatenate(truth_batches, 0)
    return prediction[..., :32], prediction[..., 32:], truth[..., :32], truth[..., 32:]


def aggregate(
    data: dict[str, Any],
    starts: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true_a: np.ndarray,
    true_b: np.ndarray,
) -> dict[str, Any]:
    ug = geometry(data["phi_u"], data["mean_u"], data["areas"], True)
    pg = geometry(data["phi_p"], data["mean_p"], data["areas"], False)
    raw = []
    by_re: dict[str, list[dict[str, Any]]] = {}
    for row, start in enumerate(starts.tolist()):
        re_value = float(data["re"][start])
        key = f"{re_value:.6f}"
        for horizon in HORIZONS:
            da = pred_a[row, :horizon] - true_a[row, :horizon]
            db = pred_b[row, :horizon] - true_b[row, :horizon]
            u_num = float(np.sum(error_energy(da, ug)))
            p_num = float(np.sum(error_energy(db, pg)))
            u_den = float(np.sum(field_energy(true_a[row, :horizon], ug)))
            p_den = float(np.sum(field_energy(true_b[row, :horizon], pg)))
            finite = bool(
                np.isfinite(pred_a[row, :horizon]).all()
                and np.isfinite(pred_b[row, :horizon]).all()
            )
            divergent = bool(
                not finite
                or np.linalg.norm(pred_a[row, :horizon], axis=1).max()
                > 10.0 * max(np.linalg.norm(true_a[row, :horizon], axis=1).max(), EPS)
                or np.linalg.norm(pred_b[row, :horizon], axis=1).max()
                > 10.0 * max(np.linalg.norm(true_b[row, :horizon], axis=1).max(), EPS)
            )
            velocity = math.sqrt(max(u_num, 0.0) / max(u_den, EPS))
            pressure = math.sqrt(max(p_num, 0.0) / max(p_den, EPS))
            item = {
                "Re": re_value,
                "start_index": start,
                "start_time": float(data["time"][start]),
                "horizon": horizon,
                "velocity_physical_relative_l2": velocity,
                "pressure_physical_relative_l2": pressure,
                "joint_field_error": 0.5 * (velocity + pressure),
                "finite": finite,
                "divergent": divergent,
            }
            raw.append(item)
            by_re.setdefault(key, []).append(item)
    summaries = {}
    for key, rows in by_re.items():
        summaries[key] = {}
        for horizon in HORIZONS:
            selected = [row for row in rows if row["horizon"] == horizon]
            summaries[key][f"k{horizon}"] = {
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in (
                    "velocity_physical_relative_l2",
                    "pressure_physical_relative_l2",
                    "joint_field_error",
                )
            } | {
                "worst_window_joint": float(
                    max(row["joint_field_error"] for row in selected)
                ),
                "finite_fraction": float(
                    np.mean([row["finite"] for row in selected])
                ),
                "divergent_windows": int(sum(row["divergent"] for row in selected)),
            }
    overall = {}
    for horizon in HORIZONS:
        selected = [row for row in raw if row["horizon"] == horizon]
        overall[f"k{horizon}"] = {
            "velocity_mean_over_windows": float(
                np.mean([row["velocity_physical_relative_l2"] for row in selected])
            ),
            "pressure_mean_over_windows": float(
                np.mean([row["pressure_physical_relative_l2"] for row in selected])
            ),
            "joint_mean_over_windows": float(
                np.mean([row["joint_field_error"] for row in selected])
            ),
            "worst_window_joint": float(
                max(row["joint_field_error"] for row in selected)
            ),
            "worst_re_joint": float(
                max(value[f"k{horizon}"]["joint_field_error"] for value in summaries.values())
            ),
            "finite_fraction": float(np.mean([row["finite"] for row in selected])),
            "divergent_windows": int(sum(row["divergent"] for row in selected)),
        }
    tail = pred_a[:, -min(16, pred_a.shape[1]) :]
    true_tail = true_a[:, -min(16, true_a.shape[1]) :]
    attractor = {
        "tail_center_modal_relative_l2": float(
            np.linalg.norm(tail.mean(1) - true_tail.mean(1))
            / (np.linalg.norm(true_tail.mean(1)) + EPS)
        ),
        "tail_amplitude_relative_error": float(
            abs(float(np.std(tail)) - float(np.std(true_tail)))
            / (float(np.std(true_tail)) + EPS)
        ),
        "terminal_increment_rms": float(
            np.sqrt(np.mean((pred_a[:, -1] - pred_a[:, -2]) ** 2))
        ),
    }
    return {"by_re": summaries, "overall": overall, "attractor": attractor, "raw": raw}


def evaluate_dataonly(args: argparse.Namespace) -> dict[str, Any]:
    module = load_module(
        "dataonly_evaluation_module",
        Path(__file__).with_name("train_dataonly_moe.py"),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data = load_full_data(args.root, args.regime, module)
    split = "heldout" if args.split == "test" else "validation"
    starts = legal_starts(data, split, max(HORIZONS))[::8]
    if not len(starts):
        raise RuntimeError(f"no K56-capable {split} windows")
    predictions = dataonly_predictions(
        module, checkpoint, data, starts, max(HORIZONS)
    )
    result = aggregate(data, starts, *predictions)
    result.update(
        {
            "schema": "unified_specialist_physical_evaluator/v2",
            "method": "data-only",
            "regime": args.regime,
            "split": args.split,
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_sha256": sha256(args.checkpoint),
            "pressure_gauge": data["pressure_gauge"],
            "split_re": complete_re_audit(data),
            "window_stride": 8,
        }
    )
    return result


def evaluate_vanilla_validation(args: argparse.Namespace) -> dict[str, Any]:
    if args.regime == "steady":
        return evaluate_vanilla_steady(args)
    if args.regime == "periodic":
        return evaluate_vanilla_periodic(args, validation=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validation = checkpoint.get("validation")
    if validation is None and "history" in checkpoint:
        validation = {
            "best_epoch": checkpoint.get("best_epoch"),
            "best_val_score": checkpoint.get("best_val_score"),
            "history_entry": next(
                (
                    item
                    for item in checkpoint["history"]
                    if item.get("epoch") == checkpoint.get("best_epoch")
                ),
                None,
            ),
        }
    if validation is None:
        raise RuntimeError("selected Vanilla checkpoint has no validation record")
    return {
        "schema": "unified_specialist_physical_evaluator/v2",
        "method": "vanilla-fnn",
        "regime": args.regime,
        "split": "validation",
        "checkpoint_sha256": sha256(args.checkpoint),
        "selection_validation": validation,
        "finite_contract": True,
        "note": "native validation record frozen before test; test uses native physical-field evaluator",
    }


def evaluate_vanilla_periodic(
    args: argparse.Namespace, validation: bool
) -> dict[str, Any]:
    base = args.root / "periodic_specialist_r32"
    output = args.output_dir / "native"
    output.mkdir(parents=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if validation:
        values = [
            66.970112,
            91.792204,
            121.050171,
            139.642302,
            169.244893,
            196.160723,
        ]
        existing = output / "validation_reference.json"
        reference_values = sorted(
            {
                represented
                for value in values
                for represented in (float(value), float(np.float32(value)))
            }
        )
        atomic_json(
            existing,
            {
                "results": [
                    {
                        "test_Re": value,
                        "deep_moe": {
                            "rhs_relative_l2": float("nan"),
                            "pressure_head_relative_l2": float("nan"),
                        },
                        "best_epoch": int(checkpoint.get("best_epoch", -1)),
                        "best_val_score": float(
                            checkpoint.get("best_val_score", float("nan"))
                        ),
                    }
                    for value in reference_values
                ]
            },
        )
    else:
        values = [70.314635, 100.352251, 149.059229, 189.862278]
        existing = base / "evaluation/periodic_v16_public_r32_metrics.json"
    wrapper = Path(__file__).with_name("run_vanilla_fnn_moe.py")
    command = [
        sys.executable,
        str(wrapper),
        "--ablation-mode",
        "vanilla-fnn",
        "--regime",
        "periodic",
        "--entrypoint",
        str(base / "code/evaluate_periodic_r32_portable.py"),
        "--contract-output",
        str(output / "allowlist.json"),
        "--evaluation-horizons",
        "1,4,8,16,48,56",
        "--evaluation-re-values",
        ",".join(str(value) for value in values),
        "--trainer",
        str(base / "code/train_periodic_moe.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--existing-metrics",
        str(existing),
        "--output-dir",
        str(output),
        "--data-root",
        str(base / "assets/Global_POD_AreaWeighted_L2"),
        "--tensor-path",
        str(base / "assets/velocity_rom_periodic.npz"),
        "--pressure-surrogate-path",
        str(base / "assets/pressure_poisson_surrogate_periodic.npz"),
    ]
    completed = os.spawnve(os.P_WAIT, sys.executable, command, os.environ.copy())
    if completed:
        raise RuntimeError(f"native Periodic evaluator returned {completed}")
    candidates = sorted(output.rglob("*evaluation.json"))
    return {
        "schema": "unified_specialist_physical_evaluator/v2",
        "method": "vanilla-fnn",
        "regime": "periodic",
        "split": args.split,
        "checkpoint_sha256": sha256(args.checkpoint),
        "native_result_files": [str(path) for path in candidates],
        "evaluated_re": values,
    }


def evaluate_vanilla_steady(args: argparse.Namespace) -> dict[str, Any]:
    task = next(
        (parent for parent in args.checkpoint.parents if parent.name == "vanilla-fnn"),
        None,
    )
    if task is None:
        raise RuntimeError("cannot locate Steady Vanilla task root")
    steady = args.root / "steady_specialist_v1"
    s3_module = load_module("steady_vanilla_s3_evaluator", steady / "code/train_s3.py")
    wrapper = load_module(
        "steady_vanilla_patch_helpers",
        Path(__file__).with_name("run_vanilla_fnn_moe.py"),
    )
    wrapper.patch_nested_loaders(s3_module)
    s2_candidates = [
        task / "training/s2/checkpoints/best-qualified.pt",
        task / "training/s2/checkpoints/final.pt",
        args.checkpoint,
    ]
    source = next((path for path in s2_candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError("Steady S2 selected checkpoint")
    init = SimpleNamespace(
        experiment="S3-B",
        run_dir=str(args.output_dir / "native_runtime"),
        config=str(task / "config.json"),
        trainer=str(steady / "code/train_s2b_3090.py"),
        finalizer=str(steady / "code/finalize_s2b_3090.py"),
        checkpoint=str(source),
        bank=str(steady / "data/perturbation_bank_train_validation.npz"),
        validation_lock=str(args.output_dir / "evaluation.lock"),
        learning_rate=1.5516372391099407e-5,
    )
    s3 = s3_module.S3(init)
    s3.init_eval_assets()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model")
    if state is None:
        raise RuntimeError("Steady selected checkpoint does not contain model state")
    s3.exp.model.load_state_dict(state, strict=True)
    s3.exp.model.eval()
    split = "validation" if args.split == "validation" else "heldout"
    reports = {
        f"k{horizon}": s3.finalizer.evaluate_horizon(s3.exp, split, horizon)
        for horizon in HORIZONS
    }
    attractor = None
    if args.split == "test":
        heldout = load_module(
            "steady_vanilla_attractor_evaluator",
            steady / "code/evaluate_s3_heldout.py",
        )
        bank = heldout.build_heldout_bank(
            s3.exp, 202607251 + 3004, args.output_dir / "heldout_bank.npz"
        )
        _labels, fixed_ids, _names = heldout.heldout_fixed_points(s3.exp)
        heldout.repair_terminal_dt(s3, fixed_ids)
        attractor = heldout.evaluate_attractivity(s3, bank, 56)
    return {
        "schema": "unified_specialist_physical_evaluator/v2",
        "method": "vanilla-fnn",
        "regime": "steady",
        "split": args.split,
        "checkpoint_sha256": sha256(args.checkpoint),
        "horizons": reports,
        "attractor": attractor,
        "pressure_gauge": "native Steady finalizer gauge",
    }


def evaluate_vanilla_test(args: argparse.Namespace) -> dict[str, Any]:
    if args.regime == "steady":
        return evaluate_vanilla_steady(args)
    if args.regime == "periodic":
        return evaluate_vanilla_periodic(args, validation=False)
    output = args.output_dir / "native"
    output.mkdir(parents=True)
    wrapper = Path(__file__).with_name("run_vanilla_fnn_moe.py")
    if args.regime == "hopf":
        base = args.root / "Hopf/migrated_h4_expanded"
        command = [
            sys.executable,
            str(wrapper),
            "--ablation-mode",
            "vanilla-fnn",
            "--regime",
            "hopf",
            "--entrypoint",
            str(base / "code/evaluate_h4_expanded.py"),
            "--contract-output",
            str(output / "allowlist.json"),
            "--evaluation-horizons",
            "1,2,4,8,16,24,48,56",
            "--trainer",
            str(base / "code/train_hopf_moe_expanded.py"),
            "--source-root",
            str(args.root / "Hopf/artifacts/hopf"),
            "--contract",
            str(base / "trainonly_contract/trainonly_fluctuation_contract.npz"),
            "--finalization-dir",
            str(args.checkpoint.parent.parent),
            "--experiments",
            args.checkpoint.parent.name,
        ]
    else:
        raise AssertionError(args.regime)
    completed = os.spawnve(os.P_WAIT, sys.executable, command, os.environ.copy())
    if completed:
        raise RuntimeError(f"native Vanilla evaluator returned {completed}")
    candidates = sorted(output.rglob("*metrics.json"))
    return {
        "schema": "unified_specialist_physical_evaluator/v2",
        "method": "vanilla-fnn",
        "regime": args.regime,
        "split": "test",
        "checkpoint_sha256": sha256(args.checkpoint),
        "native_result_files": [str(path) for path in candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--regime", choices=("steady", "hopf", "periodic"), required=True)
    parser.add_argument("--method", choices=("vanilla-fnn", "data-only"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    if args.method == "data-only":
        result = evaluate_dataonly(args)
    elif args.split == "validation":
        result = evaluate_vanilla_validation(args)
    else:
        result = evaluate_vanilla_test(args)
    atomic_json(args.output_dir / "metrics.json", result)
    if "raw" in result:
        atomic_json(args.output_dir / "per_window.json", result.pop("raw"))
        atomic_json(args.output_dir / "metrics.json", result)
    atomic_json(
        args.output_dir / "PASS.json",
        {
            "status": "PASS",
            "method": args.method,
            "regime": args.regime,
            "split": args.split,
            "checkpoint_sha256": sha256(args.checkpoint),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
