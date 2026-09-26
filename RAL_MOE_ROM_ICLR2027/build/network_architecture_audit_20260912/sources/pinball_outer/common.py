from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

EPS = 1.0e-12
CLASS_NAMES = ("Steady", "Hopf", "Periodic")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced(values: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if len(values) <= limit:
        return values
    return values[np.linspace(0, len(values) - 1, limit, dtype=np.int64)]


def chain(next_idx: np.ndarray, start: int, horizon: int) -> np.ndarray | None:
    result = [int(start)]
    for _ in range(horizon):
        nxt = int(next_idx[result[-1]])
        if nxt < 0:
            return None
        result.append(nxt)
    return np.asarray(result, dtype=np.int64)


def history_matrix(prev_idx: np.ndarray, history_len: int = 3) -> np.ndarray:
    output = np.full((len(prev_idx), history_len), -1, dtype=np.int64)
    for current in range(len(prev_idx)):
        value = current
        for slot in range(history_len):
            output[current, slot] = value
            value = int(prev_idx[value])
            if value < 0 and slot + 1 < history_len:
                output[current, slot + 1 :] = -1
                break
    return output


def pressure_gauge(fields: np.ndarray, areas: np.ndarray) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    mean = np.sum(values * areas, axis=-1, keepdims=True) / np.sum(areas)
    return values - mean


def affine_project(
    coefficients: np.ndarray,
    source_phi: np.ndarray,
    source_mean: np.ndarray,
    target_phi: np.ndarray,
    target_mean: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    source_phi = np.asarray(source_phi, dtype=np.float64)
    target_phi = np.asarray(target_phi, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    gram = (target_phi * weights[None, :]) @ target_phi.T
    if not np.allclose(gram, np.eye(len(target_phi)), atol=3.0e-4, rtol=3.0e-4):
        inverse = np.linalg.inv(gram)
    else:
        inverse = np.eye(len(target_phi))
    linear = (source_phi * weights[None, :]) @ target_phi.T @ inverse
    offset = ((source_mean - target_mean) * weights) @ target_phi.T @ inverse
    return (np.asarray(coefficients, dtype=np.float64) @ linear + offset).astype(np.float32)


def reconstruct(coefficients: np.ndarray, phi: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return np.asarray(mean, dtype=np.float64) + np.asarray(coefficients, dtype=np.float64) @ np.asarray(phi, dtype=np.float64)


def physical_descriptors(
    velocity_history: np.ndarray,
    pressure_history: np.ndarray,
    times: np.ndarray,
    areas: np.ndarray,
) -> np.ndarray:
    velocity_history = np.asarray(velocity_history, dtype=np.float64).reshape(3, -1, 2)
    pressure_history = pressure_gauge(np.asarray(pressure_history, dtype=np.float64), areas)
    dt1 = max(float(times[1] - times[0]), EPS)
    dt2 = max(float(times[2] - times[1]), EPS)
    velocity_weight = areas[:, None]
    ue = np.sum(np.square(velocity_history) * velocity_weight[None, :, :], axis=(1, 2))
    pe = np.sum(np.square(pressure_history) * areas[None, :], axis=1)
    du1 = math.sqrt(max(float(np.sum((velocity_history[1] - velocity_history[0]) ** 2 * velocity_weight)), 0.0)) / dt1
    du2 = math.sqrt(max(float(np.sum((velocity_history[2] - velocity_history[1]) ** 2 * velocity_weight)), 0.0)) / dt2
    dp2 = math.sqrt(max(float(np.sum((pressure_history[2] - pressure_history[1]) ** 2 * areas)), 0.0)) / dt2
    scale_u = math.sqrt(max(float(ue[2]), EPS))
    scale_p = math.sqrt(max(float(pe[2]), EPS))
    growth1 = (math.log(max(float(ue[1]), EPS)) - math.log(max(float(ue[0]), EPS))) / dt1
    growth2 = (math.log(max(float(ue[2]), EPS)) - math.log(max(float(ue[1]), EPS))) / dt2
    v1 = (velocity_history[1] - velocity_history[0]) / dt1
    v2 = (velocity_history[2] - velocity_history[1]) / dt2
    curvature = math.sqrt(max(float(np.sum((v2 - v1) ** 2 * velocity_weight)), 0.0)) / (scale_u + EPS)
    total_area = float(np.sum(areas))
    return np.asarray(
        [
            0.5 * math.log(max(float(ue[2]) / total_area, EPS)),
            0.5 * math.log(max(float(pe[2]) / total_area, EPS)),
            math.log1p(du2 / (scale_u + EPS)),
            math.log1p(du1 / (math.sqrt(max(float(ue[1]), EPS)) + EPS)),
            math.tanh(growth2),
            math.tanh(growth2 - growth1),
            math.log1p(dp2 / (scale_p + EPS)),
            math.log1p(curvature),
        ],
        dtype=np.float32,
    )


def quadratic_statistics(
    first: np.ndarray,
    second: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    first_delta = np.asarray(first, dtype=np.float64) - truth
    second_delta = np.asarray(second, dtype=np.float64) - truth
    weights = np.asarray(weights, dtype=np.float64)
    axes = tuple(range(2, first_delta.ndim))
    reshape = (1, 1) + weights.shape
    weighted = weights.reshape(reshape)
    return np.stack(
        [
            np.sum(first_delta * first_delta * weighted, axis=axes),
            np.sum(second_delta * second_delta * weighted, axis=axes),
            np.sum(first_delta * second_delta * weighted, axis=axes),
            np.sum(np.asarray(truth, dtype=np.float64) ** 2 * weighted, axis=axes),
        ],
        axis=-1,
    ).astype(np.float64)


class Router(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 24), nn.Tanh(), nn.Linear(24, 3))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class MLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ConvexGate(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.correction = MLP(input_dim)
        nn.init.zeros_(self.correction.net[-1].weight)
        nn.init.zeros_(self.correction.net[-1].bias)

    def forward(self, features: torch.Tensor, base_logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(base_logit + self.correction(features).squeeze(-1))


def router_probabilities(checkpoint_path: Path, reynolds: np.ndarray) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Router()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    values = np.asarray(reynolds, dtype=np.float32)[:, None]
    normalized = (values - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    with torch.inference_mode():
        return torch.softmax(model(torch.as_tensor(normalized)), dim=1).numpy()


def blend_relative(alpha: torch.Tensor, quadratic: torch.Tensor) -> torch.Tensor:
    weight = alpha[:, None]
    numerator = (
        weight.square() * quadratic[:, :, 0]
        + (1.0 - weight).square() * quadratic[:, :, 1]
        + 2.0 * weight * (1.0 - weight) * quadratic[:, :, 2]
    )
    return torch.clamp(numerator / quadratic[:, :, 3].clamp_min(EPS), min=0.0)


def clean_numeric(payload: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))
    }
