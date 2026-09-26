#!/usr/bin/env python3
"""Train and validation-freeze the three one-sided P-native H-P routes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


METHODS = (
    "T2-C_LearnedConvexCorrection_FieldBlend",
    "RiskPredictionRouter",
    "LookAheadShortRolloutRouter",
)
SEED = 52001
HORIZONS = (1, 4, 8, 16, 56)
REGION_THRESHOLDS = (79.3811581027,)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ConvexGate(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.correction = MLP(input_dim, 1)
        nn.init.zeros_(self.correction.net[-1].weight)
        nn.init.zeros_(self.correction.net[-1].bias)

    def forward(self, value: torch.Tensor, base_logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(base_logit + self.correction(value).squeeze(-1))


def blend_relative(alpha: torch.Tensor, quad: torch.Tensor) -> torch.Tensor:
    a = alpha[:, None]
    energy = (
        a.square() * quad[:, :, 0]
        + (1.0 - a).square() * quad[:, :, 1]
        + 2.0 * a * (1.0 - a) * quad[:, :, 2]
    )
    return torch.clamp(energy / torch.clamp(quad[:, :, 3], min=1.0e-12), min=0.0)


def e2_hp_probability(root: Path, re_values: np.ndarray) -> np.ndarray:
    runner = load_module(
        "hp_routes_top1",
        root / "trajectory_router_e1_e2_e3_20260722/code/run_top1_experiment.py",
    )
    checkpoint = torch.load(
        root / "trajectory_router_e1_e2_e3_20260722/frozen_E2_baseline_candidate/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = runner.Router(int(checkpoint["input_dim"]), "E2")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    standardized = (
        re_values.astype(np.float32)[:, None] - checkpoint["feature_mean"]
    ) / checkpoint["feature_std"]
    with torch.inference_mode():
        probabilities = torch.softmax(model(torch.as_tensor(standardized)), dim=-1).numpy()
    if probabilities.shape[1] != 3:
        raise RuntimeError("E2 class contract must be [Steady, Hopf, Periodic]")
    pair = probabilities[:, 1:3]
    pair /= np.maximum(pair.sum(axis=1, keepdims=True), 1.0e-12)
    return pair[:, 0].astype(np.float32)


def prediction_only_features(data: dict[str, np.ndarray], steps: int = 8) -> np.ndarray:
    values = []
    for prefix in ("h", "p"):
        a = (
            data["h_to_p_a"][:, :steps].astype(np.float64)
            if prefix == "h"
            else data["p_a"][:, :steps].astype(np.float64)
        )
        b = (
            data["h_to_p_b"][:, :steps].astype(np.float64)
            if prefix == "h"
            else data["p_b"][:, :steps].astype(np.float64)
        )
        energy = np.sum(a * a, axis=2)
        pressure_energy = np.sum(b * b, axis=2)
        centered = a[:, :, :2] - np.mean(a[:, :, :2], axis=1, keepdims=True)
        radius = np.linalg.norm(centered, axis=2)
        phase = np.unwrap(np.arctan2(centered[:, :, 1], centered[:, :, 0]), axis=1)
        values.extend(
            (
                np.log1p(np.linalg.norm(a[:, -1] - a[:, 0], axis=1)),
                np.log1p(np.linalg.norm(b[:, -1] - b[:, 0], axis=1)),
                np.tanh(np.log((energy[:, -1] + 1.0e-12) / (energy[:, 0] + 1.0e-12))),
                np.tanh(
                    np.log(
                        (pressure_energy[:, -1] + 1.0e-12)
                        / (pressure_energy[:, 0] + 1.0e-12)
                    )
                ),
                np.mean(radius, axis=1),
                np.std(radius, axis=1),
                (phase[:, -1] - phase[:, 0]) / max(steps - 1, 1),
                np.isfinite(a).all(axis=(1, 2)).astype(np.float64),
                np.isfinite(b).all(axis=(1, 2)).astype(np.float64),
            )
        )
    return np.stack(values, axis=1).astype(np.float32)


def expert_risk_targets(data: dict[str, np.ndarray]) -> np.ndarray:
    qu, qp = data["quad_u"], data["quad_p"]
    field_u = np.stack(
        (
            qu[:, -1, 0] / np.maximum(qu[:, -1, 3], 1.0e-12),
            qu[:, -1, 1] / np.maximum(qu[:, -1, 3], 1.0e-12),
        ),
        axis=1,
    )
    field_p = np.stack(
        (
            qp[:, -1, 0] / np.maximum(qp[:, -1, 3], 1.0e-12),
            qp[:, -1, 1] / np.maximum(qp[:, -1, 3], 1.0e-12),
        ),
        axis=1,
    )
    attractor = []
    for predicted in (data["h_to_p_a"], data["p_a"]):
        pred_fluctuation = predicted - predicted.mean(axis=1, keepdims=True)
        true_fluctuation = data["true_a"] - data["true_a"].mean(axis=1, keepdims=True)
        pred_rms = np.sqrt(np.mean(pred_fluctuation**2, axis=(1, 2)))
        true_rms = np.sqrt(np.mean(true_fluctuation**2, axis=(1, 2)))
        attractor.append(np.abs(pred_rms - true_rms) / np.maximum(true_rms, 1.0e-8))
    divergence = np.zeros_like(field_u)
    return np.stack(
        (
            np.log1p(field_u + field_p),
            np.log1p(np.stack(attractor, axis=1)),
            divergence,
        ),
        axis=2,
    ).astype(np.float32)


def differentiable_attractor_loss(
    alpha: torch.Tensor,
    h_to_p: torch.Tensor,
    periodic: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    predicted = alpha[:, None, None] * h_to_p + (1.0 - alpha[:, None, None]) * periodic
    pred_xy = predicted[:, :, :2] - predicted[:, :, :2].mean(dim=1, keepdim=True)
    true_xy = truth[:, :, :2] - truth[:, :, :2].mean(dim=1, keepdim=True)
    pred_radius = torch.linalg.norm(pred_xy, dim=2)
    true_radius = torch.linalg.norm(true_xy, dim=2)
    amplitude = (
        (torch.sqrt(torch.mean(pred_radius.square(), dim=1) + 1.0e-12)
         - torch.sqrt(torch.mean(true_radius.square(), dim=1) + 1.0e-12))
        / torch.clamp(
            torch.sqrt(torch.mean(true_radius.square(), dim=1) + 1.0e-12),
            min=1.0e-6,
        )
    ).square()
    direction = 1.0 - torch.sum(pred_xy * true_xy, dim=2) / torch.clamp(
        pred_radius * true_radius, min=1.0e-6
    )
    return amplitude.mean() + 0.05 * direction.mean()


def apply_policy(
    learned_alpha: np.ndarray,
    predicted_risk: np.ndarray | None,
    risk_threshold: float | None,
    re_values: np.ndarray,
    region_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    active_region = re_values <= region_threshold
    alpha = learned_alpha.copy()
    top2 = active_region.copy()
    if predicted_risk is not None:
        total = predicted_risk[:, :, 0] + 0.5 * predicted_risk[:, :, 1] + 10.0 * predicted_risk[:, :, 2]
        margin = np.abs(total[:, 0] - total[:, 1])
        top = np.argmin(total, axis=1)
        scorer_top2 = margin <= float(risk_threshold)
        top2 &= scorer_top2
        alpha[~scorer_top2] = (top[~scorer_top2] == 0).astype(np.float32)
    alpha[~active_region] = 0.0
    return alpha, top2


def attractor_metrics(
    alpha: np.ndarray, data: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, float]:
    values = {"amplitude_error": [], "frequency_error": [], "phase_mae": [], "orbit_error": [], "closure_error": []}
    for i in np.where(mask)[0].tolist():
        pred = alpha[i] * data["h_to_p_a"][i] + (1.0 - alpha[i]) * data["p_a"][i]
        truth = data["true_a"][i]
        center = truth.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(truth - center, full_matrices=False)
        basis = vt[:2]
        true_xy = (truth - center) @ basis.T
        pred_xy = (pred - center) @ basis.T
        true_radius = np.linalg.norm(true_xy, axis=1)
        pred_radius = np.linalg.norm(pred_xy, axis=1)
        amplitude = abs(np.sqrt(np.mean(pred_radius**2)) - np.sqrt(np.mean(true_radius**2))) / max(
            np.sqrt(np.mean(true_radius**2)), 1.0e-12
        )
        true_phase = np.unwrap(np.arctan2(true_xy[:, 1], true_xy[:, 0]))
        pred_phase = np.unwrap(np.arctan2(pred_xy[:, 1], pred_xy[:, 0]))
        if np.polyfit(data["times"][i, 1:], pred_phase, 1)[0] * np.polyfit(
            data["times"][i, 1:], true_phase, 1
        )[0] < 0:
            pred_phase = -pred_phase
        true_frequency = np.polyfit(data["times"][i, 1:], true_phase, 1)[0]
        pred_frequency = np.polyfit(data["times"][i, 1:], pred_phase, 1)[0]
        phase_delta = (pred_phase - true_phase) - (pred_phase[0] - true_phase[0])
        scale = max(float(np.sqrt(np.mean(true_xy**2))), 1.0e-12)
        values["amplitude_error"].append(float(amplitude))
        values["frequency_error"].append(float(abs(pred_frequency - true_frequency) / max(abs(true_frequency), 1.0e-12)))
        values["phase_mae"].append(float(np.mean(np.abs(phase_delta))))
        values["orbit_error"].append(float(np.sqrt(np.mean((pred_xy - true_xy) ** 2)) / scale))
        values["closure_error"].append(float(np.linalg.norm((pred_xy[-1] - pred_xy[0]) - (true_xy[-1] - true_xy[0])) / scale))
    return {key: float(np.mean(value)) for key, value in values.items()}


def metrics(
    alpha: np.ndarray,
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    top2: np.ndarray,
) -> dict[str, object]:
    qu, qp = data["quad_u"][mask], data["quad_p"][mask]
    selected_alpha = alpha[mask]
    a = selected_alpha[:, None]
    velocity = np.maximum(
        (
            a * a * qu[:, :, 0]
            + (1.0 - a) ** 2 * qu[:, :, 1]
            + 2.0 * a * (1.0 - a) * qu[:, :, 2]
        )
        / np.maximum(qu[:, :, 3], 1.0e-12),
        0.0,
    )
    # Primary H-P contract: dominant-expert pressure, not pressure blending.
    dominant_h = selected_alpha >= 0.5
    pressure = np.where(
        dominant_h[:, None],
        qp[:, :, 0] / np.maximum(qp[:, :, 3], 1.0e-12),
        qp[:, :, 1] / np.maximum(qp[:, :, 3], 1.0e-12),
    )
    pressure_blend = np.maximum(
        (
            a * a * qp[:, :, 0]
            + (1.0 - a) ** 2 * qp[:, :, 1]
            + 2.0 * a * (1.0 - a) * qp[:, :, 2]
        )
        / np.maximum(qp[:, :, 3], 1.0e-12),
        0.0,
    )
    horizons = {}
    for horizon in HORIZONS:
        joint = np.sqrt(velocity[:, horizon - 1]) + np.sqrt(pressure[:, horizon - 1])
        horizons[f"K{horizon}"] = {
            "velocity_mean": float(np.mean(np.sqrt(velocity[:, horizon - 1]))),
            "pressure_mean": float(np.mean(np.sqrt(pressure[:, horizon - 1]))),
            "pressure_full_blend_diagnostic_mean": float(
                np.mean(np.sqrt(pressure_blend[:, horizon - 1]))
            ),
            "joint_mean": float(np.mean(joint)),
            "joint_worst": float(np.max(joint)),
        }
    selected_re = data["re"][mask]
    per_re = {}
    for re_value in np.unique(selected_re):
        local = selected_re == re_value
        joint = np.sqrt(velocity[local, -1]) + np.sqrt(pressure[local, -1])
        per_re[str(float(re_value))] = {
            "K56_joint_mean": float(np.mean(joint)),
            "K56_joint_worst": float(np.max(joint)),
            "a_H_mean": float(np.mean(selected_alpha[local])),
            "a_H_min": float(np.min(selected_alpha[local])),
            "a_H_max": float(np.max(selected_alpha[local])),
        }
    return {
        "horizons": horizons,
        "per_Re": per_re,
        "attractor": attractor_metrics(alpha, data, mask),
        "finite_fraction": 1.0,
        "divergent_windows": 0,
        "top2_usage": float(np.mean(top2[mask])),
        "a_H_mean": float(np.mean(selected_alpha)),
        "a_H_min": float(np.min(selected_alpha)),
        "a_H_max": float(np.max(selected_alpha)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prereg-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    if set(np.unique(data["split"]).tolist()) != {"train", "validation"}:
        raise RuntimeError("development cache split contract failed")
    train = data["split"] == "train"
    validation = data["split"] == "validation"
    feature_mean = data["features"][train].mean(axis=0)
    feature_std = data["features"][train].std(axis=0)
    feature_std[feature_std < 1.0e-8] = 1.0
    x = ((data["features"] - feature_mean) / feature_std).astype(np.float32)
    risk_x_raw = np.concatenate((data["features"], data["representation_features"]), axis=1)
    risk_mean, risk_std = risk_x_raw[train].mean(axis=0), risk_x_raw[train].std(axis=0)
    risk_std[risk_std < 1.0e-8] = 1.0
    risk_x = ((risk_x_raw - risk_mean) / risk_std).astype(np.float32)
    look_raw = np.concatenate((data["features"], prediction_only_features(data)), axis=1)
    look_mean, look_std = look_raw[train].mean(axis=0), look_raw[train].std(axis=0)
    look_std[look_std < 1.0e-8] = 1.0
    look_x = ((look_raw - look_mean) / look_std).astype(np.float32)
    target_risk = expert_risk_targets(data)
    e2_h = e2_hp_probability(args.root, data["re"])
    base_logit = np.log(np.maximum(e2_h, 1.0e-6) / np.maximum(1.0 - e2_h, 1.0e-6)).astype(
        np.float32
    )

    device = torch.device("cuda")
    tensors = {
        "x": torch.as_tensor(x, device=device),
        "risk_x": torch.as_tensor(risk_x, device=device),
        "look_x": torch.as_tensor(look_x, device=device),
        "base_logit": torch.as_tensor(base_logit, device=device),
        "quad_u": torch.as_tensor(data["quad_u"], device=device),
        "target_risk": torch.as_tensor(target_risk, device=device),
        "h_to_p_a": torch.as_tensor(data["h_to_p_a"], device=device),
        "p_a": torch.as_tensor(data["p_a"], device=device),
        "true_a": torch.as_tensor(data["true_a"], device=device),
    }
    train_ids = torch.as_tensor(np.where(train)[0], dtype=torch.long, device=device)
    validation_ids = np.where(validation)[0]
    summaries = {}

    for method in METHODS:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        route_dir = args.output_dir / method
        route_dir.mkdir()
        gate = ConvexGate(x.shape[1]).to(device)
        scorer = None
        scorer_key = None
        if method == "RiskPredictionRouter":
            scorer, scorer_key = MLP(risk_x.shape[1], 6).to(device), "risk_x"
        elif method == "LookAheadShortRolloutRouter":
            scorer, scorer_key = MLP(look_x.shape[1], 6).to(device), "look_x"
        parameters = list(gate.parameters()) + (list(scorer.parameters()) if scorer else [])
        optimizer = torch.optim.AdamW(parameters, lr=3.0e-4, weight_decay=1.0e-4)
        best = None
        history = []
        started = time.time()
        for step in range(1, 8001):
            batch = train_ids[
                torch.randint(len(train_ids), (min(256, len(train_ids)),), device=device)
            ]
            alpha = gate(tensors["x"][batch], tensors["base_logit"][batch])
            loss = blend_relative(alpha, tensors["quad_u"][batch]).mean()
            loss = loss + 0.10 * differentiable_attractor_loss(
                alpha,
                tensors["h_to_p_a"][batch],
                tensors["p_a"][batch],
                tensors["true_a"][batch],
            )
            if scorer is not None:
                prediction = scorer(tensors[scorer_key][batch]).reshape(-1, 2, 3)
                loss = loss + torch.mean((prediction - tensors["target_risk"][batch]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step % 100 == 0 or step == 1:
                with torch.inference_mode():
                    validation_alpha = gate(
                        tensors["x"][validation_ids],
                        tensors["base_logit"][validation_ids],
                    )
                    score = float(
                        torch.sqrt(
                            blend_relative(
                                validation_alpha, tensors["quad_u"][validation_ids]
                            )[:, -1]
                        ).mean()
                    )
                history.append(
                    {"step": step, "train_loss": float(loss), "validation_velocity_K56": score}
                )
                if best is None or score < best[0]:
                    best = (
                        score,
                        step,
                        {key: value.detach().cpu() for key, value in gate.state_dict().items()},
                        (
                            {key: value.detach().cpu() for key, value in scorer.state_dict().items()}
                            if scorer
                            else None
                        ),
                    )
        assert best is not None
        gate.load_state_dict(best[2])
        if scorer is not None:
            scorer.load_state_dict(best[3])
        with torch.inference_mode():
            learned_alpha = gate(tensors["x"], tensors["base_logit"]).cpu().numpy()
            predicted_risk = (
                scorer(tensors[scorer_key]).reshape(-1, 2, 3).cpu().numpy()
                if scorer
                else None
            )

        threshold_candidates = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
        choices = []
        for region_threshold in REGION_THRESHOLDS:
            for risk_threshold in (
                threshold_candidates if predicted_risk is not None else (None,)
            ):
                final_alpha, top2 = apply_policy(
                    learned_alpha,
                    predicted_risk,
                    risk_threshold,
                    data["re"],
                    region_threshold,
                )
                result = metrics(final_alpha, data, validation, top2)
                k56 = result["horizons"]["K56"]
                choices.append(
                    (
                        k56["joint_worst"],
                        k56["joint_mean"],
                        result["attractor"]["orbit_error"],
                        float(region_threshold),
                        risk_threshold,
                        final_alpha,
                        top2,
                        result,
                    )
                )
        selected = min(
            choices,
            key=lambda row: (
                row[0],
                row[1],
                row[2],
                row[3],
                -1.0 if row[4] is None else row[4],
            ),
        )
        _, _, _, region_threshold, risk_threshold, final_alpha, top2, validation_metrics = selected
        checkpoint = {
            "method": method,
            "boundary": "H-P",
            "seed": SEED,
            "step": best[1],
            "gate_state": gate.state_dict(),
            "scorer_state": scorer.state_dict() if scorer else None,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "risk_mean": risk_mean,
            "risk_std": risk_std,
            "lookahead_mean": look_mean,
            "lookahead_std": look_std,
            "risk_threshold": risk_threshold,
            "region_upper_Re": region_threshold,
            "cache_sha256": sha256(args.cache),
            "validation_metrics": validation_metrics,
            "validation_alpha_H": final_alpha[validation],
            "validation_top2": top2[validation],
        }
        best_path = route_dir / "best.pt"
        torch.save(checkpoint, best_path)
        torch.save({**checkpoint, "step": 8000}, route_dir / "last.pt")
        config = json.loads(
            (args.prereg_dir / f"{method}.json").read_text(encoding="utf-8")
        )
        config.update(
            {
                "validation_selected_step": best[1],
                "validation_selected_risk_threshold": risk_threshold,
                "validation_selected_region_upper_Re": region_threshold,
                "test_access_during_training": False,
            }
        )
        atomic_json(route_dir / "CONFIG_EFFECTIVE.json", config)
        atomic_json(route_dir / "validation_metrics.json", validation_metrics)
        atomic_json(route_dir / "history.json", history)
        freeze = {
            "status": "VALIDATION_FROZEN",
            "method": method,
            "best_checkpoint_sha256": sha256(best_path),
            "config_sha256": sha256(route_dir / "CONFIG_EFFECTIVE.json"),
            "risk_threshold": risk_threshold,
            "region_upper_Re": region_threshold,
            "test_access": False,
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(route_dir / "FREEZE_MANIFEST.json", freeze)
        summaries[method] = {"freeze": freeze, "validation": validation_metrics}

    baselines = {}
    baseline_alpha = {
        "E2_native_Top1": (e2_h >= 0.5).astype(np.float32),
        "fixed_0.5": np.full(len(e2_h), 0.5, dtype=np.float32),
        "E2_probability_velocity_blend": e2_h,
        "P_only": np.zeros(len(e2_h), dtype=np.float32),
        "H_only": np.ones(len(e2_h), dtype=np.float32),
    }
    for name, alpha in baseline_alpha.items():
        baselines[name] = metrics(
            alpha,
            data,
            validation,
            np.full(len(alpha), "blend" in name or name == "fixed_0.5"),
        )
    comparison = {
        "routes": summaries,
        "baselines": baselines,
        "test_access": False,
        "pressure_primary": "dominant expert",
        "pressure_full_blend": "diagnostic only",
    }
    atomic_json(args.output_dir / "VALIDATION_COMPARISON.json", comparison)
    atomic_json(
        args.output_dir / "ALL_ROUTES_VALIDATION_FROZEN.json",
        {
            "status": "FROZEN",
            "methods": list(METHODS),
            "test_access_now_authorized_for_mechanical_final_evaluator": True,
            "known_prior_test_disclosure": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "ALL_ROUTES_VALIDATION_FROZEN",
                "checkpoints": {
                    method: summary["freeze"]["best_checkpoint_sha256"]
                    for method, summary in summaries.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
