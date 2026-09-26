#!/usr/bin/env python3
"""Matched B1--B4 Hopf r32 comparison with discrete or continuous memory.

The program can load only the sanitized train+validation coefficient view and
train-only POD/ROM assets.  Held-out evaluation is intentionally implemented in
a separate gated program.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ctdm_memory import DeltaMemory, MemoryConfig


EPS = 1.0e-12
RANK = 32
CURRENT_BASE_DIM = 109
CURRENT_DIM = 109
FULL_HISTORY_DIM = 493
VARIANTS = ("b1", "b2", "b3", "b4")
MODEL_NAMES = {
    "b1": "B1_Deep_FNN_H3",
    "b2": "B2_Deep_FNN_current",
    "b3": "B3_Discrete_KDA_FNN",
    "b4": "B4_CTDM_Galerkin_ROM",
}
STAGES = (
    (0, 1200, 4),
    (1200, 2800, 8),
    (2800, 4800, 16),
    (4800, 6400, 32),
    (6400, 8000, 56),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--baseline-trainer", type=Path, required=True)
    parser.add_argument("--h4-trainer", type=Path, required=True)
    parser.add_argument("--coefficient-view", type=Path, required=True)
    parser.add_argument("--galerkin-path", type=Path, required=True)
    parser.add_argument("--pressure-path", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--fluctuation-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--seed", type=int, default=1248)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--tbptt-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-memory", type=float, default=1.0e-6)
    parser.add_argument("--gamma-min", type=float, default=1.0e-3)
    parser.add_argument("--eta-max", type=float, default=0.25)
    parser.add_argument("--v-max", type=float, default=1.0)
    parser.add_argument("--warmup-length", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--validation-windows-per-re", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.42)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fused-adamw", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--benchmark-horizon", type=int, default=56)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--swanlab-mode",
        choices=("disabled", "online", "offline", "local"),
        default="online",
    )
    parser.add_argument("--swanlab-project", default="CTDM_Galerkin_ROM_R32")
    parser.add_argument("--swanlab-group", default="B1_B4_single_seed")
    parser.add_argument("--swanlab-log-every", type=int, default=20)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=True, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_math(allow_tf32: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def stage_for(step: int) -> tuple[int, int]:
    for start, end, horizon in STAGES:
        if start <= step < end:
            return horizon, start
    return 56, 6400


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, gate_dim: int = 0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        previous = in_dim
        for _ in range(4):
            layers.extend(
                (
                    nn.Linear(previous, 512),
                    nn.LayerNorm(512),
                    nn.SiLU(),
                )
            )
            previous = 512
        layers.extend((nn.Linear(512, 256), nn.SiLU()))
        self.body = nn.Sequential(*layers)
        self.output = nn.Linear(256, out_dim + gate_dim)
        self.out_dim = out_dim
        self.gate_dim = gate_dim
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if gate_dim:
            with torch.no_grad():
                self.output.bias[out_dim:].fill_(math.log(0.99 / 0.01))

    def forward(self, features: torch.Tensor):
        output = self.output(self.body(features))
        if not self.gate_dim:
            return output
        residual, logits = torch.split(output, (self.out_dim, self.gate_dim), dim=1)
        return residual, torch.sigmoid(logits)


class ClosureHeads(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.velocity = ResidualMLP(in_dim, RANK)
        self.pressure = ResidualMLP(in_dim, RANK, RANK)

    def velocity_output(self, features: torch.Tensor) -> torch.Tensor:
        return self.velocity(features)

    def pressure_output(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pressure(features)


class DirectFNN(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.heads = ClosureHeads(in_dim)


class MemoryFNN(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        config = MemoryConfig(
            current_dim=CURRENT_DIM,
            token_dim=128,
            num_heads=4,
            d_k=16,
            d_v=16,
            gamma_min=args.gamma_min,
            eta_max=args.eta_max,
            v_max=args.v_max,
        )
        self.memory = DeltaMemory(config)
        self.heads = ClosureHeads(config.token_dim + config.num_heads * config.d_v)

    def contexts(
        self, current: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = self.memory.encode(current)
        memory_u, memory_p = self.memory.parameters_net.read(state, token)
        return token, torch.cat((token, memory_u), 1), torch.cat((token, memory_p), 1)


def build_model(variant: str, args: argparse.Namespace) -> nn.Module:
    if variant == "b1":
        return DirectFNN(FULL_HISTORY_DIM)
    if variant == "b2":
        return DirectFNN(CURRENT_DIM)
    return MemoryFNN(args)


def parameter_manifest(model: nn.Module, variant: str) -> dict[str, Any]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        prefix = name.split(".", 1)[0]
        groups[prefix] = groups.get(prefix, 0) + parameter.numel()
    runtime = 0
    if variant in ("b3", "b4"):
        config = model.memory.config
        runtime = config.num_heads * config.d_k * config.d_v
    return {
        "variant": variant,
        "model_name": MODEL_NAMES[variant],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "parameter_groups": groups,
        "runtime_memory_state_per_trajectory": runtime,
    }


def stats_tensors(stats: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in asdict(stats).items()
    }


def current_features(
    B: Any,
    fc: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    re: torch.Tensor,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    galerkin = B.galerkin(a, b, re, rom)
    base = B.make_base_features(a, b, galerkin, re)
    features = (base - stats["x_mean"][:CURRENT_BASE_DIM]) / stats["x_scale"][
        :CURRENT_BASE_DIM
    ]
    if features.shape[1] != CURRENT_DIM:
        raise RuntimeError(f"current feature dimension {features.shape[1]} != {CURRENT_DIM}")
    return features, galerkin


def full_history_features(
    B: Any,
    fc: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    re: torch.Tensor,
    a_history: torch.Tensor,
    b_history: torch.Tensor,
    rhs_history: torch.Tensor,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    raw, galerkin = B.state_features(
        a, b, re, a_history, b_history, rhs_history, rom, stats
    )
    if raw.shape[1] != FULL_HISTORY_DIM:
        raise RuntimeError(f"history feature dimension {raw.shape[1]} != {FULL_HISTORY_DIM}")
    return raw, galerkin


def velocity_rhs_direct(
    model: DirectFNN,
    features: torch.Tensor,
    galerkin: torch.Tensor,
    stats: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    standardized = model.heads.velocity_output(features)
    return galerkin + standardized * stats["rhs_scale"] + stats["rhs_mean"]


def pressure_close(
    B: Any,
    model: nn.Module,
    pressure_features: torch.Tensor,
    a_next: torch.Tensor,
    re: torch.Tensor,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_std, gate = model.heads.pressure_output(pressure_features)
    residual = residual_std * stats["pressure_scale"] + stats["pressure_mean"]
    pressure_base = B.pressure_base(a_next, re, rom)
    pressure = gate * pressure_base + residual
    return pressure, gate


def direct_step(
    variant: str,
    model: DirectFNN,
    B: Any,
    fc: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    re: torch.Tensor,
    dt: torch.Tensor,
    a_history: torch.Tensor,
    b_history: torch.Tensor,
    rhs_history: torch.Tensor,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    def evaluate(trial_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if variant == "b1":
            features, galerkin = full_history_features(
                B,
                fc,
                trial_a,
                b,
                re,
                a_history,
                b_history,
                rhs_history,
                rom,
                stats,
            )
        else:
            features, galerkin = current_features(B, fc, trial_a, b, re, rom, stats)
        return velocity_rhs_direct(model, features, galerkin, stats), galerkin

    k1, g1 = evaluate(a)
    k2, _ = evaluate(a + 0.5 * dt * k1)
    k3, _ = evaluate(a + 0.5 * dt * k2)
    k4, _ = evaluate(a + dt * k3)
    a_next = a + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if variant == "b1":
        pressure_features, _ = full_history_features(
            B,
            fc,
            a_next,
            b,
            re,
            a_history,
            b_history,
            rhs_history,
            rom,
            stats,
        )
    else:
        pressure_features, _ = current_features(
            B, fc, a_next, b, re, rom, stats
        )
    b_next, gate = pressure_close(
        B, model, pressure_features, a_next, re, rom, stats
    )
    return a_next, b_next, g1, k1, {"pressure_gate": gate}


def memory_velocity_rhs(
    model: MemoryFNN,
    B: Any,
    fc: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    re: torch.Tensor,
    memory: torch.Tensor,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    current, galerkin = current_features(B, fc, a, b, re, rom, stats)
    token, velocity_context, _ = model.contexts(current, memory)
    residual_std = model.heads.velocity_output(velocity_context)
    derivative_a = galerkin + residual_std * stats["rhs_scale"] + stats["rhs_mean"]
    derivative_memory, diagnostics = model.memory.parameters_net.derivative(
        memory, token
    )
    return derivative_a, derivative_memory, current, token, diagnostics


def discrete_memory_step(
    model: MemoryFNN,
    B: Any,
    fc: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    re: torch.Tensor,
    dt: torch.Tensor,
    memory: torch.Tensor,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    current_start, g1 = current_features(B, fc, a, b, re, rom, stats)
    token_start = model.memory.encode(current_start)
    memory_next, diagnostics = model.memory.parameters_net.discrete_update(
        memory, token_start, dt
    )

    def evaluate(trial_a: torch.Tensor) -> torch.Tensor:
        current, galerkin = current_features(B, fc, trial_a, b, re, rom, stats)
        _, velocity_context, _ = model.contexts(current, memory)
        standardized = model.heads.velocity_output(velocity_context)
        return galerkin + standardized * stats["rhs_scale"] + stats["rhs_mean"]

    k1 = evaluate(a)
    k2 = evaluate(a + 0.5 * dt * k1)
    k3 = evaluate(a + 0.5 * dt * k2)
    k4 = evaluate(a + dt * k3)
    a_next = a + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    current_next, _ = current_features(B, fc, a_next, b, re, rom, stats)
    _, _, pressure_context = model.contexts(current_next, memory_next)
    b_next, gate = pressure_close(
        B, model, pressure_context, a_next, re, rom, stats
    )
    diagnostics = {
        **diagnostics,
        "pressure_gate": gate,
        "rk4_stage_memory_delta": torch.zeros(
            (), device=memory.device, dtype=torch.float32
        ),
    }
    return a_next, b_next, memory_next, g1, diagnostics


def continuous_memory_step(
    model: MemoryFNN,
    B: Any,
    fc: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    re: torch.Tensor,
    dt: torch.Tensor,
    memory: torch.Tensor,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    def evaluate(
        trial_a: torch.Tensor, trial_memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        derivative_a, derivative_memory, _, _, diagnostics = memory_velocity_rhs(
            model,
            B,
            fc,
            trial_a,
            b,
            re,
            trial_memory,
            rom,
            stats,
        )
        return derivative_a, derivative_memory, diagnostics

    k1_a, k1_s, diagnostics = evaluate(a, memory)
    memory_stage_2 = memory + 0.5 * dt[..., None, None] * k1_s
    k2_a, k2_s, _ = evaluate(a + 0.5 * dt * k1_a, memory_stage_2)
    memory_stage_3 = memory + 0.5 * dt[..., None, None] * k2_s
    k3_a, k3_s, _ = evaluate(a + 0.5 * dt * k2_a, memory_stage_3)
    memory_stage_4 = memory + dt[..., None, None] * k3_s
    k4_a, k4_s, _ = evaluate(a + dt * k3_a, memory_stage_4)
    a_next = a + dt * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a) / 6.0
    memory_next = memory + dt[..., None, None] * (
        k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s
    ) / 6.0
    current_next, _ = current_features(B, fc, a_next, b, re, rom, stats)
    _, _, pressure_context = model.contexts(current_next, memory_next)
    b_next, gate = pressure_close(
        B, model, pressure_context, a_next, re, rom, stats
    )
    diagnostics = {
        **diagnostics,
        "memory_norm": torch.linalg.matrix_norm(
            memory_next.float(), ord="fro", dim=(-2, -1)
        ),
        "pressure_gate": gate,
        "rk4_stage_memory_delta": torch.stack(
            (
                torch.linalg.vector_norm((memory_stage_2 - memory).reshape(len(memory), -1), dim=1),
                torch.linalg.vector_norm((memory_stage_3 - memory).reshape(len(memory), -1), dim=1),
                torch.linalg.vector_norm((memory_stage_4 - memory).reshape(len(memory), -1), dim=1),
            ),
            1,
        ).max(),
    }
    return a_next, b_next, memory_next, k1_a, diagnostics


def previous_chain(data: Mapping[str, np.ndarray], starts: np.ndarray, length: int) -> np.ndarray:
    columns = [starts.astype(np.int64)]
    current = starts.astype(np.int64)
    for _ in range(length - 1):
        current = data["prev"][current]
        if np.any(current < 0):
            raise RuntimeError("warm-up prefix crosses a trajectory boundary")
        columns.append(current.copy())
    return np.stack(columns[::-1], axis=1)


def warm_memory(
    variant: str,
    model: MemoryFNN,
    B: Any,
    fc: Any,
    data: Mapping[str, np.ndarray],
    starts: np.ndarray,
    re: torch.Tensor,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
    device: torch.device,
    length: int,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    chain = previous_chain(data, starts, length)
    memory = model.memory.parameters_net.zero_state(len(starts), device)
    diagnostics: list[dict[str, torch.Tensor]] = []
    for interval in range(length - 1):
        left, right = chain[:, interval], chain[:, interval + 1]
        a0 = torch.as_tensor(data["a"][left], device=device)
        a1 = torch.as_tensor(data["a"][right], device=device)
        b0 = torch.as_tensor(data["b"][left], device=device)
        b1 = torch.as_tensor(data["b"][right], device=device)
        dt = torch.as_tensor(
            (data["time"][right] - data["time"][left])[:, None], device=device
        )
        if variant == "b3":
            current, _ = current_features(B, fc, a0, b0, re, rom, stats)
            token = model.memory.encode(current)
            memory, diag = model.memory.parameters_net.discrete_update(
                memory, token, dt
            )
        else:
            def derivative(
                trial_memory: torch.Tensor, fraction: float
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                a_trial = torch.lerp(a0, a1, fraction)
                b_trial = torch.lerp(b0, b1, fraction)
                current, _ = current_features(
                    B, fc, a_trial, b_trial, re, rom, stats
                )
                token = model.memory.encode(current)
                return model.memory.parameters_net.derivative(trial_memory, token)

            k1, diag = derivative(memory, 0.0)
            k2, _ = derivative(memory + 0.5 * dt[..., None, None] * k1, 0.5)
            k3, _ = derivative(memory + 0.5 * dt[..., None, None] * k2, 0.5)
            k4, _ = derivative(memory + dt[..., None, None] * k3, 1.0)
            memory = memory + dt[..., None, None] * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            ) / 6.0
            diag = {
                **diag,
                "memory_norm": torch.linalg.matrix_norm(
                    memory.float(), ord="fro", dim=(-2, -1)
                ),
            }
        diagnostics.append(diag)
    return memory, diagnostics


def legal_starts(
    data: Mapping[str, np.ndarray],
    ids: np.ndarray,
    horizon: int,
    warmup_length: int,
) -> np.ndarray:
    allowed = set(int(index) for index in ids.tolist())
    starts: list[int] = []
    for start in ids:
        backward = int(start)
        valid = True
        for _ in range(warmup_length - 1):
            backward = int(data["prev"][backward])
            if backward < 0 or backward not in allowed:
                valid = False
                break
        current = int(start)
        if valid:
            for _ in range(horizon):
                current = int(data["next"][current])
                if current < 0 or current not in allowed:
                    valid = False
                    break
        if valid:
            starts.append(int(start))
    return np.asarray(starts, dtype=np.int64)


def equal_re_sample(
    pools: Mapping[float, np.ndarray], batch: int, rng: np.random.Generator
) -> np.ndarray:
    reynolds = np.asarray(sorted(pools), dtype=np.float64)
    choices = rng.choice(reynolds, size=batch, replace=True)
    return np.asarray(
        [rng.choice(pools[float(value)]) for value in choices], dtype=np.int64
    )


def rollout(
    model: nn.Module,
    variant: str,
    B: Any,
    fc: Any,
    data: Mapping[str, np.ndarray],
    starts: np.ndarray,
    horizon: int,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
    device: torch.device,
    warmup_length: int = 3,
    tbptt_steps: int = 0,
    substeps: int = 1,
    collect_diagnostics: bool = True,
) -> dict[str, Any]:
    query = B.batch_from_ids(data, starts, device)
    a, b, re = query["a"], query["b"], query["re"]
    a_history, b_history = query["ah"], query["bh"]
    rhs_history = B.galerkin(
        a_history.reshape(-1, RANK),
        b_history.reshape(-1, RANK),
        re[:, None].expand(-1, a_history.shape[1]).reshape(-1),
        rom,
    ).reshape_as(a_history)
    memory = None
    memory_diagnostics: list[dict[str, torch.Tensor]] = []
    if variant in ("b3", "b4"):
        memory, warm_diagnostics = warm_memory(
            variant,
            model,
            B,
            fc,
            data,
            starts,
            re,
            rom,
            stats,
            device,
            warmup_length,
        )
        if collect_diagnostics:
            memory_diagnostics.extend(warm_diagnostics)
    current_indices = starts.copy()
    output: dict[str, Any] = {
        key: []
        for key in (
            "pred_a",
            "pred_b",
            "true_a",
            "true_b",
            "dt",
            "pressure_gate",
        )
    }
    output["memory_diagnostics"] = memory_diagnostics
    output["memory"] = memory
    output["re"] = re
    output["discrete_memory_updates"] = 0
    output["continuous_stage_memory_changed"] = False
    for step_index in range(horizon):
        next_indices = data["next"][current_indices]
        macro_dt = torch.as_tensor(
            (data["time"][next_indices] - data["time"][current_indices])[:, None],
            device=device,
        )
        for _ in range(substeps):
            dt = macro_dt / float(substeps)
            if variant in ("b1", "b2"):
                a, b, galerkin, _, diagnostics = direct_step(
                    variant,
                    model,
                    B,
                    fc,
                    a,
                    b,
                    re,
                    dt,
                    a_history,
                    b_history,
                    rhs_history,
                    rom,
                    stats,
                )
            elif variant == "b3":
                a, b, memory, galerkin, diagnostics = discrete_memory_step(
                    model, B, fc, a, b, re, dt, memory, rom, stats
                )
                output["discrete_memory_updates"] += 1
            else:
                a, b, memory, galerkin, diagnostics = continuous_memory_step(
                    model, B, fc, a, b, re, dt, memory, rom, stats
                )
                output["continuous_stage_memory_changed"] = (
                    output["continuous_stage_memory_changed"]
                    or float(diagnostics["rk4_stage_memory_delta"].detach()) > 0.0
                )
            if variant == "b1":
                a_history = torch.cat((a[:, None], a_history[:, :-1]), 1)
                b_history = torch.cat((b[:, None], b_history[:, :-1]), 1)
                rhs_history = torch.cat(
                    (galerkin[:, None], rhs_history[:, :-1]), 1
                )
            if variant in ("b3", "b4"):
                if collect_diagnostics:
                    output["memory_diagnostics"].append(diagnostics)
            output["pressure_gate"].append(diagnostics["pressure_gate"])
        true_a = torch.as_tensor(data["a"][next_indices], device=device)
        true_b = torch.as_tensor(data["b"][next_indices], device=device)
        output["pred_a"].append(a)
        output["pred_b"].append(b)
        output["true_a"].append(true_a)
        output["true_b"].append(true_b)
        output["dt"].append(macro_dt)
        current_indices = next_indices
        if tbptt_steps and (step_index + 1) % tbptt_steps == 0:
            a, b = a.detach(), b.detach()
            a_history, b_history, rhs_history = (
                a_history.detach(),
                b_history.detach(),
                rhs_history.detach(),
            )
            if memory is not None:
                memory = memory.detach()
    output["memory"] = memory
    return output


def relative_loss(
    prediction: torch.Tensor, target: torch.Tensor, floor: torch.Tensor
) -> torch.Tensor:
    numerator = torch.sum((prediction - target).square(), 1)
    denominator = torch.clamp(torch.sum(target.square(), 1), min=floor)
    return torch.mean(numerator / denominator)


def objective(
    model: nn.Module,
    variant: str,
    B: Any,
    fc: Any,
    data: Mapping[str, np.ndarray],
    starts: np.ndarray,
    horizon: int,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    output = rollout(
        model,
        variant,
        B,
        fc,
        data,
        starts,
        horizon,
        rom,
        stats,
        device,
        warmup_length=args.warmup_length,
        tbptt_steps=args.tbptt_steps,
        collect_diagnostics=False,
    )
    coefficient = torch.stack(
        [
            F.mse_loss(
                (prediction - target) / stats["a_scale"],
                torch.zeros_like(prediction),
            )
            for prediction, target in zip(output["pred_a"], output["true_a"])
        ]
    ).mean()
    pressure = torch.stack(
        [
            F.mse_loss(
                (prediction - target) / stats["b_state_scale"],
                torch.zeros_like(prediction),
            )
            for prediction, target in zip(output["pred_b"], output["true_b"])
        ]
    ).mean()
    velocity_relative = torch.stack(
        [
            relative_loss(prediction, target, stats["a_rel_floor"])
            for prediction, target in zip(output["pred_a"], output["true_a"])
        ]
    ).mean()
    pressure_relative = torch.stack(
        [
            relative_loss(prediction, target, stats["b_rel_floor"])
            for prediction, target in zip(output["pred_b"], output["true_b"])
        ]
    ).mean()
    memory_penalty = coefficient * 0.0
    if output["memory"] is not None:
        memory_penalty = output["memory"].float().square().mean()
    total = (
        coefficient
        + 0.55 * pressure
        + 0.38 * velocity_relative
        + 0.095 * pressure_relative
        + args.lambda_memory * memory_penalty
    )
    return (
        total,
        {
            "Ea": coefficient,
            "Eb": pressure,
            "Eu_modal": velocity_relative,
            "Ep_modal": pressure_relative,
            "memory": memory_penalty,
            "total": total,
        },
        output,
    )


def physical_relative(
    prediction: torch.Tensor,
    target: torch.Tensor,
    modes: torch.Tensor,
    mean: torch.Tensor,
    sqrt_weight: torch.Tensor,
) -> torch.Tensor:
    prediction_field = mean + prediction @ modes
    target_field = mean + target @ modes
    numerator = torch.sum(((prediction_field - target_field) * sqrt_weight).square(), 1)
    denominator = torch.clamp(
        torch.sum((target_field * sqrt_weight).square(), 1), min=EPS
    )
    return torch.sqrt(numerator / denominator)


def memory_summary(
    output: Mapping[str, Any], re_value: float
) -> dict[str, float]:
    if output["memory"] is None:
        return {}
    diagnostics = output["memory_diagnostics"]
    gamma = torch.cat([item["gamma"].reshape(-1) for item in diagnostics])
    eta = torch.cat([item["eta"].reshape(-1) for item in diagnostics])
    delta = torch.cat([item["delta_error"].reshape(-1) for item in diagnostics])
    state = output["memory"].float()
    singular = torch.linalg.svdvals(state)
    probabilities = singular / singular.sum(-1, keepdim=True).clamp_min(EPS)
    effective_rank = torch.exp(
        -torch.sum(probabilities * torch.log(probabilities.clamp_min(EPS)), -1)
    )
    return {
        "Re": re_value,
        "gamma_mean": float(gamma.mean()),
        "gamma_min": float(gamma.min()),
        "gamma_max": float(gamma.max()),
        "timescale_mean": float((1.0 / gamma).mean()),
        "half_life_mean": float((math.log(2.0) / gamma).mean()),
        "eta_mean": float(eta.mean()),
        "eta_min": float(eta.min()),
        "eta_max": float(eta.max()),
        "memory_frobenius_mean": float(
            torch.linalg.matrix_norm(state, ord="fro", dim=(-2, -1)).mean()
        ),
        "memory_max_singular_mean": float(singular[..., 0].mean()),
        "memory_effective_rank_mean": float(effective_rank.mean()),
        "delta_error_mean": float(delta.mean()),
    }


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    variant: str,
    B: Any,
    H: Any,
    fc: Any,
    data: Mapping[str, np.ndarray],
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    warmup_length: int = 3,
) -> dict[str, Any]:
    model.eval()
    modes_u = torch.as_tensor(data["phi_u"], device=device)
    modes_p = torch.as_tensor(data["phi_p"], device=device)
    mean_u = torch.as_tensor(data["mean_u"], device=device)
    mean_p = torch.as_tensor(data["mean_p"], device=device)
    area = torch.sqrt(torch.as_tensor(data["areas"], device=device))
    weight_u = torch.cat((area, area))
    by_re: dict[str, Any] = {}
    memory_rows: list[dict[str, float]] = []
    all_eu: list[float] = []
    all_ep: list[float] = []
    all_worst: list[float] = []
    total_divergent = 0
    finite_min = 1.0
    started = time.perf_counter()
    for re_value in H.VAL:
        ids = data["val_ids"][
            np.isclose(data["re"][data["val_ids"]], re_value, atol=5e-6)
        ]
        starts = legal_starts(data, ids, 56, warmup_length)
        if len(starts) > args.validation_windows_per_re:
            starts = starts[
                np.linspace(
                    0,
                    len(starts) - 1,
                    args.validation_windows_per_re,
                    dtype=int,
                )
            ]
        output = rollout(
            model,
            variant,
            B,
            fc,
            data,
            starts,
            56,
            rom,
            stats,
            device,
            warmup_length=warmup_length,
        )
        eu_steps: list[torch.Tensor] = []
        ep_steps: list[torch.Tensor] = []
        ea_steps: list[torch.Tensor] = []
        eb_steps: list[torch.Tensor] = []
        ratios: list[torch.Tensor] = []
        pressure_drifts: list[torch.Tensor] = []
        curve: dict[str, float] = {}
        for index, (pa, pb, ta, tb) in enumerate(
            zip(
                output["pred_a"],
                output["pred_b"],
                output["true_a"],
                output["true_b"],
            ),
            start=1,
        ):
            eu = physical_relative(pa, ta, modes_u, mean_u, weight_u)
            ep = physical_relative(pb, tb, modes_p, mean_p, area)
            eu_steps.append(eu)
            ep_steps.append(ep)
            ea_steps.append(torch.sqrt(torch.mean(((pa - ta) / stats["a_scale"]).square(), 1)))
            eb_steps.append(torch.sqrt(torch.mean(((pb - tb) / stats["b_state_scale"]).square(), 1)))
            ratio_u = torch.linalg.vector_norm(pa, dim=1) / torch.linalg.vector_norm(
                ta, dim=1
            ).clamp_min(EPS)
            ratio_p = torch.linalg.vector_norm(pb, dim=1) / torch.linalg.vector_norm(
                tb, dim=1
            ).clamp_min(EPS)
            ratios.append(torch.maximum(ratio_u, ratio_p))
            true_energy = torch.sum(tb.square(), 1)
            pressure_drifts.append(
                torch.abs(torch.sum(pb.square(), 1) - true_energy)
                / true_energy.clamp_min(stats["b_rel_floor"])
            )
            if index in (1, 4, 8, 16, 32, 56):
                curve[str(index)] = float(torch.mean(eu + ep))
        eu_tensor = torch.stack(eu_steps, 1)
        ep_tensor = torch.stack(ep_steps, 1)
        ea_tensor = torch.stack(ea_steps, 1)
        eb_tensor = torch.stack(eb_steps, 1)
        ratio_tensor = torch.stack(ratios, 1)
        finite = (
            torch.isfinite(eu_tensor).all(1)
            & torch.isfinite(ep_tensor).all(1)
            & torch.isfinite(ratio_tensor).all(1)
        )
        divergent = (~finite) | (ratio_tensor.max(1).values > 10.0)
        joint_by_window = (eu_tensor + ep_tensor).mean(1)
        row = {
            "Re": float(re_value),
            "windows": int(len(starts)),
            "Eu": float(torch.nanmean(eu_tensor)),
            "Ep": float(torch.nanmean(ep_tensor)),
            "Ea": float(torch.nanmean(ea_tensor)),
            "Eb": float(torch.nanmean(eb_tensor)),
            "terminal_Eu": float(torch.nanmean(eu_tensor[:, -1])),
            "terminal_Ep": float(torch.nanmean(ep_tensor[:, -1])),
            "worst_window_joint": float(torch.nan_to_num(joint_by_window, nan=1e9).max()),
            "pressure_drift": float(torch.nanmean(torch.stack(pressure_drifts, 1))),
            "finite_fraction": float(finite.float().mean()),
            "divergent_windows": int(divergent.sum()),
            "max_norm_ratio": float(
                torch.nan_to_num(ratio_tensor, nan=1e9, posinf=1e9).max()
            ),
            "error_growth_curve": curve,
        }
        by_re[f"{float(re_value):.6f}"] = row
        memory_row = memory_summary(output, float(re_value))
        if memory_row:
            memory_rows.append(memory_row)
        all_eu.append(row["Eu"])
        all_ep.append(row["Ep"])
        all_worst.append(row["worst_window_joint"])
        total_divergent += row["divergent_windows"]
        finite_min = min(finite_min, row["finite_fraction"])
    aggregate = {
        "Eu_mean": float(np.mean(all_eu)),
        "Ep_mean": float(np.mean(all_ep)),
        "worst_window_joint_max": float(np.max(all_worst)),
        "finite_fraction_min": finite_min,
        "divergent_windows_total": total_divergent,
    }
    hard_gate = finite_min == 1.0 and total_divergent == 0
    score = (
        aggregate["Eu_mean"]
        + aggregate["Ep_mean"]
        + 0.25 * aggregate["worst_window_joint_max"]
    )
    model.train()
    return {
        "schema_version": 1,
        "selection_split": "validation_only",
        "step": step,
        "variant": variant,
        "warmup_length": warmup_length,
        "by_re": by_re,
        "aggregate": aggregate,
        "memory_diagnostics": memory_rows,
        "hard_gate": hard_gate,
        "score": score,
        "elapsed_seconds": time.perf_counter() - started,
    }


def checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: argparse.Namespace,
    step: int,
    best_score: float,
    best_step: int,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract": "CTDM_R32_TRAIN_VALIDATION_ONLY",
        "variant": args.variant,
        "model_name": MODEL_NAMES[args.variant],
        "optimizer_step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "best_step": best_step,
        "validation_history": list(history),
        "args": vars(args),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "heldout_evaluation_performed": False,
    }


def run_smoke(
    model: nn.Module,
    variant: str,
    B: Any,
    H: Any,
    fc: Any,
    data: Mapping[str, np.ndarray],
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    model.train()
    rows: list[dict[str, Any]] = []
    smoke_optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-5)
    for horizon in (4, 8, 16, 32, 56):
        ids = data["train_ids"][
            np.isclose(data["re"][data["train_ids"]], H.TRAIN[0], atol=5e-6)
        ]
        starts = legal_starts(data, ids, horizon, args.warmup_length)[:1]
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=args.amp
        ):
            loss, parts, output = objective(
                model,
                variant,
                B,
                fc,
                data,
                starts,
                horizon,
                rom,
                stats,
                device,
                args,
            )
        loss.backward()
        memory_gradient_norm = 0.0
        if variant in ("b3", "b4"):
            memory_gradient_norm = math.sqrt(
                sum(
                    float(parameter.grad.detach().float().square().sum())
                    for parameter in model.memory.parameters()
                    if parameter.grad is not None
                )
            )
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(loss) or not math.isfinite(float(gradient)):
            raise RuntimeError(f"nonfinite smoke at K{horizon}")
        memory_dtype = None
        memory_changed_in_rk4 = None
        if output["memory"] is not None:
            memory_dtype = str(output["memory"].dtype)
            if output["memory"].dtype != torch.float32:
                raise RuntimeError("memory state is not float32")
            memory_changed_in_rk4 = output["continuous_stage_memory_changed"]
        if variant == "b3" and output["discrete_memory_updates"] != horizon:
            raise RuntimeError(
                f"discrete update count {output['discrete_memory_updates']} != {horizon}"
            )
        if variant == "b4" and not output["continuous_stage_memory_changed"]:
            raise RuntimeError("continuous memory did not change within RK4 stages")
        if variant == "b4" and output["discrete_memory_updates"] != 0:
            raise RuntimeError("CTDM executed a forbidden discrete memory update")
        if (
            variant in ("b3", "b4")
            and horizon > 4
            and memory_gradient_norm <= 0.0
        ):
            raise RuntimeError("memory path has zero gradient after closure warm-start")
        rows.append(
            {
                "horizon": horizon,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient),
                "parts": {key: float(value.detach()) for key, value in parts.items()},
                "memory_dtype": memory_dtype,
                "memory_changed_in_rk4_expected": memory_changed_in_rk4,
                "discrete_memory_updates": output["discrete_memory_updates"],
                "memory_gradient_norm": memory_gradient_norm,
            }
        )
        smoke_optimizer.step()
    state_path = output_dir / "smoke_checkpoint.pt"
    atomic_torch({"model_state": model.state_dict(), "variant": variant}, state_path)
    clone = build_model(variant, args).to(device)
    loaded = torch.load(state_path, map_location=device, weights_only=False)
    clone.load_state_dict(loaded["model_state"], strict=True)
    for left, right in zip(model.parameters(), clone.parameters()):
        if not torch.equal(left, right):
            raise RuntimeError("checkpoint roundtrip mismatch")
    report = {
        "schema_version": 1,
        "variant": variant,
        "shape_forward_backward": True,
        "checkpoint_roundtrip": True,
        "memory_fp32": variant not in ("b3", "b4") or True,
        "continuous_memory_joint_rk4": variant != "b4" or True,
        "discrete_memory_one_macro_update": variant != "b3" or True,
        "rows": rows,
        "checkpoint_sha256": sha256(state_path),
    }
    atomic_json(report, output_dir / "SMOKE_TEST.json")
    print(json.dumps(report, indent=2))


def init_swanlab(args: argparse.Namespace, config: Mapping[str, Any]):
    if args.swanlab_mode == "disabled":
        return None
    import swanlab

    return swanlab.init(
        project=args.swanlab_project,
        group=args.swanlab_group,
        name=args.experiment_name,
        mode=args.swanlab_mode,
        config=dict(config),
        reinit=True,
        parallel="shared",
    )


def main() -> None:
    args = parse_args()
    if args.max_steps != 8000 and not (args.smoke_only or args.benchmark_steps):
        raise ValueError("formal matched budget is exactly 8000 optimizer steps")
    args.experiment_name = args.experiment_name or (
        f"{MODEL_NAMES[args.variant]}_seed{args.seed}"
    )
    seed_all(args.seed)
    configure_math(args.allow_tf32)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    B = load_module("ctdm_frozen_hopf_base", args.baseline_trainer)
    H = load_module("ctdm_frozen_h4", args.h4_trainer)
    if not np.array_equal(np.asarray(H.TRAIN), np.asarray(B.TRAIN_RE)):
        raise RuntimeError("baseline and H4 train split mismatch")
    if not np.array_equal(np.asarray(H.VAL), np.asarray(B.VAL_RE)):
        raise RuntimeError("baseline and H4 validation split mismatch")
    if not np.array_equal(np.asarray(H.HELD), np.asarray(B.HELDOUT_RE)):
        raise RuntimeError("baseline and H4 heldout split mismatch")
    asset_audit = B.audit_assets(
        SimpleNamespace(
            coefficient_view=args.coefficient_view,
            galerkin_path=args.galerkin_path,
            pressure_path=args.pressure_path,
            asset_manifest=args.asset_manifest,
        )
    )
    data = B.load_coefficients(
        SimpleNamespace(coefficient_view=args.coefficient_view, history_len=3)
    )
    rom_np = B.load_train_rom(
        SimpleNamespace(
            galerkin_path=args.galerkin_path, pressure_path=args.pressure_path
        )
    )
    base_args = SimpleNamespace(history_len=3, scale_floor_quantile=0.1)
    fitted_stats, _ = B.fit_stats(data, rom_np, base_args)
    stats = stats_tensors(fitted_stats, device)
    rom = {key: torch.as_tensor(value, device=device) for key, value in rom_np.items()}
    fc = H.FluctuationContract(args.fluctuation_contract, device)
    model = build_model(args.variant, args).to(device)
    parameters = parameter_manifest(model, args.variant)
    if args.variant in ("b3", "b4"):
        paired_variant = "b4" if args.variant == "b3" else "b3"
        paired_model = build_model(paired_variant, args)
        paired_parameters = parameter_manifest(paired_model, paired_variant)
        left_shapes = {
            name: tuple(parameter.shape)
            for name, parameter in model.named_parameters()
        }
        right_shapes = {
            name: tuple(parameter.shape)
            for name, parameter in paired_model.named_parameters()
        }
        if (
            parameters["trainable_parameters"]
            != paired_parameters["trainable_parameters"]
            or left_shapes != right_shapes
        ):
            raise RuntimeError("B3/B4 parameter contract is not exactly matched")
        parameters["paired_variant"] = paired_variant
        parameters["paired_trainable_parameters"] = paired_parameters[
            "trainable_parameters"
        ]
        parameters["paired_parameter_shapes_identical"] = True
        del paired_model
    output_dir = args.output_root / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "schema_version": 1,
        "model_name": MODEL_NAMES[args.variant],
        "train_re": H.TRAIN.tolist(),
        "validation_re": H.VAL.tolist(),
        "heldout_re_hard_disabled": H.HELD.tolist(),
        "curriculum": STAGES,
        "current_feature_dimension": CURRENT_DIM,
        "full_history_dimension": FULL_HISTORY_DIM,
        "asset_audit": asset_audit,
        "source_hashes": {
            "trainer": sha256(Path(__file__)),
            "memory": sha256(Path(__file__).with_name("ctdm_memory.py")),
            "baseline": sha256(args.baseline_trainer),
            "h4": sha256(args.h4_trainer),
            "coefficient_view": sha256(args.coefficient_view),
            "galerkin": sha256(args.galerkin_path),
            "pressure": sha256(args.pressure_path),
            "asset_manifest": sha256(args.asset_manifest),
            "fluctuation_contract": sha256(args.fluctuation_contract),
        },
        "test_access": "not_loaded_by_this_program",
    }
    atomic_json(config, output_dir / "CONFIG_MANIFEST.json")
    atomic_json(parameters, output_dir / "PARAMETER_COUNT.json")
    if args.smoke_only:
        run_smoke(
            model,
            args.variant,
            B,
            H,
            fc,
            data,
            rom,
            stats,
            device,
            args,
            output_dir,
        )
        return
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), fused=args.fused_adamw, **optimizer_kwargs
        )
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps
    )
    step = 0
    best_score = float("inf")
    best_step = -1
    history: list[dict[str, Any]] = []
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        step = int(saved["optimizer_step"])
        best_score = float(saved["best_score"])
        best_step = int(saved["best_step"])
        history = list(saved.get("validation_history", []))
    pools: dict[int, dict[float, np.ndarray]] = {}
    for horizon in (4, 8, 16, 32, 56):
        pools[horizon] = {}
        for re_value in H.TRAIN:
            ids = data["train_ids"][
                np.isclose(data["re"][data["train_ids"]], re_value, atol=5e-6)
            ]
            legal = legal_starts(data, ids, horizon, args.warmup_length)
            if not len(legal):
                raise RuntimeError(f"no train windows Re={re_value} K={horizon}")
            pools[horizon][float(re_value)] = legal
    run = init_swanlab(args, config)
    rng = np.random.default_rng(args.seed + step)
    total_steps = args.benchmark_steps if args.benchmark_steps else args.max_steps
    started = time.perf_counter()
    running: dict[str, float] = {}
    running_count = 0
    try:
        while step < total_steps:
            horizon, _ = stage_for(step)
            if args.benchmark_steps:
                horizon = args.benchmark_horizon
            optimizer.zero_grad(set_to_none=True)
            last_parts: dict[str, torch.Tensor] = {}
            for _ in range(args.grad_accum):
                starts = equal_re_sample(pools[horizon], args.micro_batch, rng)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.amp,
                ):
                    loss, parts, _ = objective(
                        model,
                        args.variant,
                        B,
                        fc,
                        data,
                        starts,
                        horizon,
                        rom,
                        stats,
                        device,
                        args,
                    )
                    scaled_loss = loss / args.grad_accum
                scaled_loss.backward()
                last_parts = parts
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            if not math.isfinite(float(gradient_norm)):
                raise FloatingPointError(f"nonfinite gradient at step {step}")
            optimizer.step()
            scheduler.step()
            step += 1
            for key, value in last_parts.items():
                running[key] = running.get(key, 0.0) + float(value.detach())
            running_count += 1
            if step == 1 or step % args.swanlab_log_every == 0:
                status = {
                    "status": "running",
                    "variant": args.variant,
                    "optimizer_step": step,
                    "horizon": horizon,
                    "pid": os.getpid(),
                    "last_update_unix": time.time(),
                }
                atomic_json(status, output_dir / "runtime_status.json")
            if step % args.swanlab_log_every == 0:
                payload = {
                    f"train/{key}": value / max(running_count, 1)
                    for key, value in running.items()
                }
                payload.update(
                    {
                        "train/horizon": horizon,
                        "train/gradient_norm": float(gradient_norm),
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                )
                if run:
                    import swanlab

                    swanlab.log(payload, step=step)
                running = {}
                running_count = 0
            if not args.benchmark_steps and (
                step % args.eval_every == 0 or step == total_steps
            ):
                validation = evaluate_validation(
                    model,
                    args.variant,
                    B,
                    H,
                    fc,
                    data,
                    rom,
                    stats,
                    device,
                    args,
                    step,
                )
                history.append(validation)
                atomic_json(
                    {"schema_version": 1, "history": history},
                    output_dir / "VALIDATION_HISTORY.json",
                )
                if run:
                    import swanlab

                    swanlab.log(
                        {
                            "validation/score": validation["score"],
                            "validation/Eu": validation["aggregate"]["Eu_mean"],
                            "validation/Ep": validation["aggregate"]["Ep_mean"],
                            "validation/worst": validation["aggregate"][
                                "worst_window_joint_max"
                            ],
                            "validation/hard_gate": float(validation["hard_gate"]),
                        },
                        step=step,
                    )
                if (
                    step >= 4000
                    and validation["hard_gate"]
                    and validation["score"] < best_score
                ):
                    best_score = float(validation["score"])
                    best_step = step
                    atomic_torch(
                        checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            args,
                            step,
                            best_score,
                            best_step,
                            history,
                        ),
                        output_dir / "best_validation.pt",
                    )
                atomic_torch(
                    checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        args,
                        step,
                        best_score,
                        best_step,
                        history,
                    ),
                    output_dir / "latest.pt",
                )
        elapsed = time.perf_counter() - started
        performance = {
            "elapsed_seconds": elapsed,
            "optimizer_steps": step,
            "steps_per_minute": 60.0 * step / max(elapsed, EPS),
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda"
                else 0.0
            ),
        }
        atomic_json(performance, output_dir / "THROUGHPUT.json")
        final = checkpoint(
            model,
            optimizer,
            scheduler,
            args,
            step,
            best_score,
            best_step,
            history,
        )
        final["status"] = "training_complete_heldout_not_run"
        final["performance"] = performance
        atomic_torch(final, output_dir / "final_training.pt")
        atomic_torch(
            {
                "status": "PENDING_VALIDATION_FREEZE_AND_GATED_TEST",
                "best_validation": (
                    "best_validation.pt" if best_step >= 0 else None
                ),
                "heldout_evaluation_performed": False,
            },
            output_dir / "final.pt",
        )
        atomic_json(
            {
                "status": "training_complete_heldout_not_run",
                "variant": args.variant,
                "optimizer_step": step,
                "horizon": stage_for(step)[0],
                "pid": os.getpid(),
                "last_update_unix": time.time(),
            },
            output_dir / "runtime_status.json",
        )
    finally:
        if run:
            import swanlab

            swanlab.finish()


if __name__ == "__main__":
    main()
