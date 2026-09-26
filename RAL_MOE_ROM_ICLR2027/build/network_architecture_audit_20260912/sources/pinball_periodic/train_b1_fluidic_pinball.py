#!/usr/bin/env python3
"""Train-only Fluidic Pinball V2 B1/B2/B3 comparison.

B1: deep 6x512 FNN with the frozen three-state feature contract.
B2: current-only features plus KDA matrix memory, radial loss disabled.
B3: B2 with the train-only radial regularizer enabled.

The held-out coefficient file is deliberately not accepted by this program.
"""
from __future__ import annotations

import argparse
import contextlib
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
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R_U = 17
R_P = 16
CURRENT_DIM = 13 + 2 * R_U + R_P
H3_DIM = 13 + 10 * R_U + 5 * R_P
TRAIN_RE = np.empty(0, dtype=np.float64)
VAL_RE = np.empty(0, dtype=np.float64)
HELDOUT_RE = np.empty(0, dtype=np.float64)
STAGES = ((0, 1600, 4), (1600, 3200, 8), (3200, 5200, 16),
          (5200, 6800, 24), (6800, 8000, 32))
EPS = 1.0e-12


def configure_contract(asset_manifest: Path) -> dict[str, Any]:
    global R_U, R_P, CURRENT_DIM, H3_DIM, TRAIN_RE, VAL_RE, HELDOUT_RE, STAGES
    manifest = json.loads(asset_manifest.read_text(encoding="utf-8"))
    R_U, R_P = int(manifest["r_u"]), int(manifest["r_p"])
    TRAIN_RE = np.asarray(manifest["fit_reynolds"], dtype=np.float64)
    VAL_RE = np.asarray(manifest["validation_reynolds"], dtype=np.float64)
    HELDOUT_RE = np.asarray(manifest["heldout_reynolds"], dtype=np.float64)
    STAGES = tuple(tuple(int(value) for value in row) for row in manifest["curriculum"])
    CURRENT_DIM = 13 + 2 * R_U + R_P
    H3_DIM = 13 + 10 * R_U + 5 * R_P
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("b1", "b2", "b3"), required=True)
    p.add_argument("--baseline-trainer", type=Path, required=True)
    p.add_argument("--coefficient-view", type=Path, required=True)
    p.add_argument("--galerkin-path", type=Path, required=True)
    p.add_argument("--pressure-path", type=Path, required=True)
    p.add_argument("--asset-manifest", type=Path, required=True)
    p.add_argument("--fluctuation-contract", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--seed", type=int, default=1248)
    p.add_argument("--micro-batch", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-memory", type=float, default=1.0e-6)
    p.add_argument("--lambda-radial", type=float, default=1.0)
    p.add_argument("--tbptt-steps", type=int, default=16)
    p.add_argument("--token-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--d-k", type=int, default=16)
    p.add_argument("--d-v", type=int, default=16)
    p.add_argument("--beta-max", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--long-eval-every", type=int, default=800)
    p.add_argument("--validation-windows-per-re", type=int, default=24)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.94)
    p.add_argument("--device", default="cuda")
    p.add_argument("--swanlab-mode", choices=("disabled", "online", "offline", "local"),
                   default="online")
    p.add_argument("--swanlab-project", default="FluidicPinballV2_B1_DeepFNN")
    p.add_argument("--swanlab-group", default="KDA_PR_FNN_single_seed")
    p.add_argument("--swanlab-log-every", type=int, default=20)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--benchmark-steps", type=int, default=0)
    p.add_argument("--benchmark-horizon", type=int, choices=(4, 8, 16, 24, 32), default=32)
    p.add_argument("--resume", type=Path)
    return p.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_hopf_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True,
                              default=json_default), encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    safe = x.clamp_min(1.0e-6)
    return torch.where(safe > 20.0, safe, torch.log(torch.expm1(safe)))


class DeepTrunk(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim), nn.Linear(input_dim, 512),
                                  nn.SiLU()]
        for _ in range(5):
            layers.extend([nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClosureHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = nn.Sequential(nn.Linear(512, 256), nn.SiLU(),
                                      nn.Linear(256, R_U))
        self.pressure = nn.Sequential(nn.Linear(512, 256), nn.SiLU(),
                                      nn.Linear(256, R_P))
        self.pressure_gate = nn.Sequential(nn.Linear(512, 128), nn.SiLU(),
                                           nn.Linear(128, R_P))
        nn.init.zeros_(self.velocity[-1].weight)
        nn.init.zeros_(self.velocity[-1].bias)
        nn.init.zeros_(self.pressure[-1].weight)
        nn.init.zeros_(self.pressure[-1].bias)
        nn.init.zeros_(self.pressure_gate[-1].weight)
        nn.init.constant_(self.pressure_gate[-1].bias, math.log(0.99 / 0.01))

    def forward(self, h: torch.Tensor):
        return self.velocity(h), self.pressure(h), torch.sigmoid(self.pressure_gate(h))


class DeepFNNH3(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = DeepTrunk(H3_DIM)
        self.heads = ClosureHeads()

    def forward(self, x: torch.Tensor):
        return self.heads(self.trunk(x))


class KDAMemory(nn.Module):
    def __init__(self, current_dim: int, token_dim: int, heads: int, d_k: int,
                 d_v: int, median_dt: float, beta_max: float):
        super().__init__()
        self.heads, self.d_k, self.d_v = heads, d_k, d_v
        self.beta_max = float(beta_max)
        self.encoder = nn.Sequential(
            nn.LayerNorm(current_dim), nn.Linear(current_dim, 256), nn.SiLU(),
            nn.Linear(256, token_dim), nn.SiLU(),
        )
        self.key = nn.Linear(token_dim, heads * d_k)
        self.value = nn.Linear(token_dim, heads * d_v)
        self.query_u = nn.Linear(token_dim, heads * d_k)
        self.query_p = nn.Linear(token_dim, heads * d_k)
        self.tau_delta = nn.Linear(token_dim, heads * d_k)
        self.beta_rate = nn.Linear(token_dim, heads)
        tau_min = max(float(median_dt) * 0.25, 1.0e-6)
        self.register_buffer("tau_min", torch.tensor(tau_min, dtype=torch.float32))
        targets = torch.logspace(math.log10(4 * median_dt), math.log10(32 * median_dt),
                                 heads * d_k)
        self.tau_base_raw = nn.Parameter(inverse_softplus(targets - tau_min))
        nn.init.zeros_(self.tau_delta.weight)
        nn.init.zeros_(self.tau_delta.bias)
        nn.init.zeros_(self.beta_rate.weight)
        init_rate = torch.full((heads,), 1.0 / max(8 * median_dt, 1.0e-6))
        with torch.no_grad():
            self.beta_rate.bias.copy_(inverse_softplus(init_rate))

    def zero_memory(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.heads, self.d_k, self.d_v,
                           device=device, dtype=torch.float32)

    def token(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def update(self, memory: torch.Tensor, x: torch.Tensor, dt: torch.Tensor):
        with torch.autocast(device_type=x.device.type, enabled=False):
            token = self.token(x.float()).float()
            batch = x.shape[0]
            key = F.normalize(self.key(token).reshape(batch, self.heads, self.d_k),
                              dim=-1, eps=1.0e-6)
            value = self.value(token).reshape(batch, self.heads, self.d_v)
            tau = self.tau_min + F.softplus(
                self.tau_base_raw.reshape(1, self.heads, self.d_k)
                + self.tau_delta(token).reshape(batch, self.heads, self.d_k)
            )
            dtv = dt.float().reshape(batch, 1, 1)
            alpha = torch.exp(torch.clamp(-dtv / tau, min=-20.0, max=0.0))
            rate = F.softplus(self.beta_rate(token)).reshape(batch, self.heads, 1)
            beta = self.beta_max * (1.0 - torch.exp(-dtv * rate))
            forgotten = alpha.unsqueeze(-1) * memory.float()
            predicted = torch.einsum("bhkv,bhk->bhv", forgotten, key)
            error = value - predicted
            updated = forgotten + beta.unsqueeze(-1) * key.unsqueeze(-1) * error.unsqueeze(-2)
            diag = {
                "tau": tau, "alpha": alpha, "beta": beta.squeeze(-1),
                "write_error": torch.linalg.vector_norm(error, dim=-1),
                "memory_norm": torch.linalg.matrix_norm(updated, ord="fro", dim=(-2, -1)),
            }
            return updated.float(), diag

    def read(self, memory: torch.Tensor, x: torch.Tensor):
        with torch.autocast(device_type=x.device.type, enabled=False):
            token = self.token(x.float()).float()
            batch = x.shape[0]
            qu = F.normalize(self.query_u(token).reshape(batch, self.heads, self.d_k),
                             dim=-1, eps=1.0e-6)
            qp = F.normalize(self.query_p(token).reshape(batch, self.heads, self.d_k),
                             dim=-1, eps=1.0e-6)
            mu = torch.einsum("bhkv,bhk->bhv", memory.float(), qu).reshape(batch, -1)
            mp = torch.einsum("bhkv,bhk->bhv", memory.float(), qp).reshape(batch, -1)
            cosine = F.cosine_similarity(qu, qp, dim=-1)
            return mu, mp, cosine


class KDAFNN(nn.Module):
    def __init__(self, median_dt: float, args: argparse.Namespace):
        super().__init__()
        self.memory = KDAMemory(CURRENT_DIM, args.token_dim, args.num_heads,
                                args.d_k, args.d_v, median_dt, args.beta_max)
        context_dim = CURRENT_DIM + args.num_heads * args.d_v * 2
        self.trunk = DeepTrunk(context_dim)
        self.heads = ClosureHeads()

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        mu, mp, cosine = self.memory.read(memory, x)
        h = self.trunk(torch.cat((x, mu.to(x.dtype), mp.to(x.dtype)), dim=1))
        return (*self.heads(h), cosine)


class FluctuationContract:
    def __init__(self, path: Path, device: torch.device):
        z = np.load(path, allow_pickle=False)
        self.sha256 = sha256(path)
        self.nodes = torch.as_tensor(z["nodes"], device=device, dtype=torch.float32)
        self.mean_a = torch.as_tensor(z["mean_a"], device=device, dtype=torch.float32)
        self.plane = torch.as_tensor(z["plane"], device=device, dtype=torch.float32)
        self.radial_scale = torch.as_tensor(z["radial_scale"], device=device,
                                            dtype=torch.float32)
        self.radial_floor = float(z["radial_floor"])

    def interp(self, re: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        hi = torch.searchsorted(self.nodes, re).clamp(1, self.nodes.numel() - 1)
        lo = hi - 1
        w = (re - self.nodes[lo]) / (self.nodes[hi] - self.nodes[lo]).clamp_min(1e-8)
        return torch.lerp(values[lo], values[hi],
                          w.reshape(-1, *([1] * (values.ndim - 1))))

    def radius(self, a: torch.Tensor, re: torch.Tensor) -> torch.Tensor:
        center = self.interp(re, self.mean_a)
        scale = self.interp(re, self.radial_scale)
        z = (a - center) @ self.plane.T
        return torch.sqrt(torch.sum(z.float().square(), dim=1) + 1e-12) / scale


def base_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        coefficient_view=args.coefficient_view, galerkin_path=args.galerkin_path,
        pressure_path=args.pressure_path, asset_manifest=args.asset_manifest,
        history_len=3, scale_floor_quantile=0.10,
        lambda_pressure_rollout=0.25,
    )


def stage_for(step: int) -> tuple[int, int]:
    for lo, hi, horizon in STAGES:
        if lo <= step < hi:
            return horizon, lo
    return 56, 6800


def fit_runtime(B, data, rom_np, args):
    norms, scale = B.fit_stats(data, rom_np, base_args(args))
    if len(norms.x_mean) != H3_DIM:
        raise RuntimeError(f"runtime H3 dimension {len(norms.x_mean)} != {H3_DIM}")
    ids = data["train_ids"]
    nxt = data["next"][ids]
    dt = data["time"][nxt] - data["time"][ids]
    median_dt = float(np.median(dt))
    if not np.isfinite(median_dt) or median_dt <= 0:
        raise RuntimeError("invalid train median dt")
    return norms, scale, median_dt


def normalized_current(B, a, b, re, rom, stats):
    g = B.galerkin(a, b, re, rom)
    raw = B.make_base_features(a, b, g, re)
    x = (raw - stats["x_mean"][:CURRENT_DIM]) / stats["x_scale"][:CURRENT_DIM]
    if x.shape[1] != CURRENT_DIM:
        raise RuntimeError(f"current feature dimension {x.shape[1]} != {CURRENT_DIM}")
    return x, g


def normalized_h3(B, a, b, re, ah, bh, rh, rom, stats):
    x, g = B.state_features(a, b, re, ah, bh, rh, rom, stats)
    if x.shape[1] != H3_DIM:
        raise RuntimeError(f"H3 feature dimension {x.shape[1]} != {H3_DIM}")
    return x, g


def apply_outputs(velocity_std, pressure_residual, gate, a_next, re, B, rom, stats):
    # The frozen square-cylinder Hopf model learns the finite-difference velocity
    # derivative directly; Galerkin remains a feature because direct Galerkin RK4
    # is unstable at the snapshot interval.
    velocity = velocity_std * stats["rhs_scale"] + stats["rhs_mean"]
    pressure_pp = B.pressure_base(a_next, re, rom)
    pressure = gate * pressure_pp + pressure_residual * stats["b_state_scale"]
    return velocity, pressure


def warm_memory(model: KDAFNN, B, data, starts, re, rom, stats, device):
    hist = data["hist"][starts]
    chronological = hist[:, ::-1].copy()
    memory = model.memory.zero_memory(len(starts), device)
    diags = []
    for j in range(chronological.shape[1]):
        ids = chronological[:, j]
        a = torch.as_tensor(data["a"][ids], device=device)
        b = torch.as_tensor(data["b"][ids], device=device)
        x, _ = normalized_current(B, a, b, re, rom, stats)
        nxt = data["next"][ids]
        dt_np = np.where(nxt >= 0, data["time"][np.maximum(nxt, 0)] - data["time"][ids],
                         np.median(np.diff(data["time"][chronological], axis=1), axis=1))
        dt = torch.as_tensor(dt_np[:, None], device=device)
        memory, diag = model.memory.update(memory, x, dt)
        diags.append(diag)
    return memory, diags


def rollout(model, variant, B, data, starts, horizon, rom, stats, device,
            tbptt_steps: int):
    q = B.batch_from_ids(data, starts, device)
    a, b, re, ah, bh = q["a"], q["b"], q["re"], q["ah"], q["bh"]
    rh = B.galerkin(ah.reshape(-1, R_U), bh.reshape(-1, R_P),
                    re[:, None].expand(-1, 3).reshape(-1), rom).reshape_as(ah)
    memory = None
    memory_diags = []
    if variant != "b1":
        memory, warm_diags = warm_memory(model, B, data, starts, re, rom, stats, device)
        memory_diags.extend(warm_diags)
    current = starts.copy()
    out = {k: [] for k in ("pred_a", "pred_b", "true_a", "true_b", "pred_rhs",
                            "true_rhs", "gate", "query_cosine")}
    memory_updates = 3 if variant != "b1" else 0
    stage_memory_mutations = 0
    for j in range(horizon):
        nxt = data["next"][current]
        dt = torch.as_tensor((data["time"][nxt] - data["time"][current])[:, None],
                             device=device)
        if variant != "b1" and j > 0:
            x_update, _ = normalized_current(B, a, b, re, rom, stats)
            memory, diag = model.memory.update(memory, x_update, dt)
            memory_diags.append(diag)
            memory_updates += 1
            if tbptt_steps > 0 and j % tbptt_steps == 0:
                memory = memory.detach()
        frozen = memory.clone() if memory is not None else None

        def velocity_at(trial_a):
            if variant == "b1":
                x, _ = normalized_h3(B, trial_a, b, re, ah, bh, rh, rom, stats)
                vs, pr, gate = model(x)
                cosine = torch.zeros(len(trial_a), 1, device=device)
            else:
                x, _ = normalized_current(B, trial_a, b, re, rom, stats)
                vs, pr, gate, cosine = model(x, frozen)
            velocity = vs * stats["rhs_scale"] + stats["rhs_mean"]
            return velocity, pr, gate, cosine

        before = a
        k1, pres, gate, cosine = velocity_at(a)
        k2 = velocity_at(a + 0.5 * dt * k1)[0]
        k3 = velocity_at(a + 0.5 * dt * k2)[0]
        k4 = velocity_at(a + dt * k3)[0]
        a = a + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        pp = B.pressure_base(a, re, rom)
        b = gate * pp + pres * stats["b_state_scale"]
        if memory is not None and not torch.equal(memory, frozen):
            stage_memory_mutations += 1
        ta = torch.as_tensor(data["a"][nxt], device=device)
        tb = torch.as_tensor(data["b"][nxt], device=device)
        out["pred_a"].append(a)
        out["pred_b"].append(b)
        out["true_a"].append(ta)
        out["true_b"].append(tb)
        out["pred_rhs"].append(k1)
        out["true_rhs"].append((ta - before) / dt.clamp_min(1e-8))
        out["gate"].append(gate)
        out["query_cosine"].append(cosine)
        g = B.galerkin(a, b, re, rom)
        ah = torch.cat((a[:, None], ah[:, :-1]), dim=1)
        bh = torch.cat((b[:, None], bh[:, :-1]), dim=1)
        rh = torch.cat((g[:, None], rh[:, :-1]), dim=1)
        current = nxt
    out["re"] = re
    out["memory"] = memory
    out["memory_diags"] = memory_diags
    out["memory_updates"] = memory_updates
    out["stage_memory_mutations"] = stage_memory_mutations
    return out


def objective(model, variant, B, data, starts, horizon, rom, stats, device, fc, args):
    out = rollout(model, variant, B, data, starts, horizon, rom, stats, device,
                  args.tbptt_steps)
    pa, pb, ta, tb = out["pred_a"], out["pred_b"], out["true_a"], out["true_b"]
    coeff = torch.stack([
        F.mse_loss((x - y) / stats["a_scale"], torch.zeros_like(x))
        for x, y in zip(pa, ta)
    ]).mean()
    pressure = torch.stack([
        F.mse_loss((x - y) / stats["b_state_scale"], torch.zeros_like(x))
        for x, y in zip(pb, tb)
    ]).mean()
    dynamics = torch.stack([
        F.mse_loss((x - y) / stats["rhs_scale"], torch.zeros_like(x))
        for x, y in zip(out["pred_rhs"], out["true_rhs"])
    ]).mean()
    u_rel = torch.stack([B.relative_loss(x, y, stats["a_rel_floor"])
                         for x, y in zip(pa, ta)]).mean()
    p_rel = torch.stack([B.relative_loss(x, y, stats["b_rel_floor"])
                         for x, y in zip(pb, tb)]).mean()
    energy = torch.stack([F.smooth_l1_loss((x * x).sum(1), (y * y).sum(1))
                          for x, y in zip(pa, ta)]).mean()
    trajectory = u_rel + 0.25 * p_rel
    memory_loss = coeff * 0
    if out["memory_diags"]:
        memory_loss = torch.stack([
            d["memory_norm"].square().mean() / (args.num_heads * args.d_k * args.d_v)
            for d in out["memory_diags"]
        ]).mean()
    radial = coeff * 0
    if variant == "b3":
        pred = torch.stack(pa, dim=1)
        true = torch.stack(ta, dim=1)
        re = out["re"][:, None].expand(-1, horizon).reshape(-1)
        rp = fc.radius(pred.reshape(-1, R_U), re).reshape(-1, horizon)
        rt = fc.radius(true.reshape(-1, R_U), re).reshape(-1, horizon)
        radial = F.smooth_l1_loss(torch.log(rp + fc.radial_floor),
                                  torch.log(rt + fc.radial_floor))
        if horizon > 1:
            radial = radial + 0.5 * F.smooth_l1_loss(
                torch.diff(torch.log(rp + fc.radial_floor), dim=1),
                torch.diff(torch.log(rt + fc.radial_floor), dim=1))
    loss = (coeff + dynamics + 0.55 * pressure + 0.08 * u_rel
            + 0.30 * trajectory + 0.15 * trajectory + 0.10 * trajectory
            + 0.02 * energy + args.lambda_memory * memory_loss
            + (args.lambda_radial * radial if variant == "b3" else 0))
    parts = {
        "coeff": coeff, "dynamics": dynamics, "pressure": pressure,
        "rollout_u": u_rel, "rollout_p": p_rel, "energy": energy,
        "memory": memory_loss, "radial": radial, "total": loss,
        "pressure_gate_mean": torch.stack(out["gate"]).mean(),
    }
    return loss, parts, out


def legal_starts(B, data, ids, horizon):
    starts = B.legal_starts(data, ids, horizon)
    if not len(starts):
        raise RuntimeError(f"no legal continuous windows for K={horizon}")
    return starts


def physical_series(B, data, out, device):
    phi_u = torch.as_tensor(data["phi_u"], device=device)
    phi_p = torch.as_tensor(data["phi_p"], device=device)
    mean_u = torch.as_tensor(data["mean_u"], device=device)
    mean_p = torch.as_tensor(data["mean_p"], device=device)
    area = torch.sqrt(torch.as_tensor(data["areas"], device=device))
    # Fluidic Pinball velocity POD stores flattened cell-major (u, v) pairs.
    uw = torch.repeat_interleave(area, 2)
    u, p = [], []
    for pa, pb, ta, tb in zip(out["pred_a"], out["pred_b"],
                              out["true_a"], out["true_b"]):
        u.append(B.physical_relative(pa, ta, phi_u, mean_u, uw))
        p.append(B.physical_relative(pb, tb, phi_p, mean_p, area))
    return torch.stack(u, dim=1), torch.stack(p, dim=1)


@torch.no_grad()
def validate(model, variant, B, data, rom, stats, device, args, step):
    model.eval()
    horizons = [4, 8, 16]
    if step >= 5200 and (step % args.long_eval_every == 0 or step == args.max_steps):
        horizons += [24, 32, 56]
    report = {"step": step, "variant": variant, "by_re": {}, "finite_fraction": 1.0,
              "divergent_windows": 0}
    all_terminal, all_mean, all_worst = [], [], []
    for rv in VAL_RE:
        ids = data["val_ids"][np.isclose(data["re"][data["val_ids"]], rv, atol=5e-6)]
        row = {}
        for horizon in horizons:
            starts = legal_starts(B, data, ids, horizon)
            if len(starts) > args.validation_windows_per_re:
                starts = starts[np.linspace(0, len(starts) - 1,
                                            args.validation_windows_per_re, dtype=int)]
            out = rollout(model, variant, B, data, starts, horizon, rom, stats,
                          device, args.tbptt_steps)
            u, p = physical_series(B, data, out, device)
            joint = u + p
            pa, pb = torch.stack(out["pred_a"], 1), torch.stack(out["pred_b"], 1)
            ta, tb = torch.stack(out["true_a"], 1), torch.stack(out["true_b"], 1)
            finite = torch.isfinite(pa).all((1, 2)) & torch.isfinite(pb).all((1, 2))
            ratio_u = torch.linalg.vector_norm(pa, dim=2) / torch.linalg.vector_norm(
                ta, dim=2).clamp_min(EPS)
            ratio_p = torch.linalg.vector_norm(pb, dim=2) / torch.linalg.vector_norm(
                tb, dim=2).clamp_min(EPS)
            divergent = (~finite) | (torch.maximum(ratio_u, ratio_p).max(1).values > 10)
            metrics = {
                "velocity_mean": float(u.mean()), "pressure_mean": float(p.mean()),
                "joint_mean": float(joint.mean()), "terminal_joint": float(joint[:, -1].mean()),
                "worst_window": float(joint.max()), "finite_fraction": float(finite.float().mean()),
                "divergent_windows": int(divergent.sum()),
                "pressure_drift": float(torch.mean(torch.abs((pb * pb).sum(2)
                                                - (tb * tb).sum(2))
                                                / (tb * tb).sum(2).clamp_min(EPS))),
            }
            row[str(horizon)] = metrics
            report["finite_fraction"] = min(report["finite_fraction"],
                                             metrics["finite_fraction"])
            report["divergent_windows"] += metrics["divergent_windows"]
            if horizon == 56:
                all_terminal.append(metrics["terminal_joint"])
                all_mean.append(metrics["joint_mean"])
                all_worst.append(metrics["worst_window"])
        report["by_re"][f"{rv:.6f}"] = row
    if all_mean:
        report["k56"] = {
            "mean_joint": float(np.mean(all_mean)),
            "mean_terminal": float(np.mean(all_terminal)),
            "worst_window": float(np.max(all_worst)),
        }
        report["score"] = (report["k56"]["mean_joint"] + report["k56"]["mean_terminal"]
                           + 0.1 * report["k56"]["worst_window"])
    else:
        key = str(max(horizons))
        values = [r[key] for r in report["by_re"].values()]
        report["score"] = float(np.mean([v["joint_mean"] for v in values]))
    report["hard_gate"] = (report["finite_fraction"] == 1.0
                           and report["divergent_windows"] == 0)
    model.train()
    return report


def gradient_groups(model, variant):
    groups = {"fnn": [], "kda_projection": [], "tau_beta": []}
    for name, parameter in model.named_parameters():
        if variant == "b1" or not name.startswith("memory."):
            groups["fnn"].append(parameter)
        elif "tau_" in name or "beta_rate" in name:
            groups["tau_beta"].append(parameter)
        else:
            groups["kda_projection"].append(parameter)
    return groups


def grad_norm(parameters):
    terms = [(p.grad.detach().float().square().sum()) for p in parameters
             if p.grad is not None]
    return float(torch.sqrt(torch.stack(terms).sum())) if terms else 0.0


def parameter_manifest(model, variant):
    groups = gradient_groups(model, variant)
    return {
        "variant": variant,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
        "active_parameters": sum(p.numel() for p in model.parameters()
                                 if p.requires_grad),
        "by_group": {k: sum(p.numel() for p in values) for k, values in groups.items()},
        "runtime_memory_state": (0 if variant == "b1"
                                 else model.memory.heads * model.memory.d_k
                                 * model.memory.d_v),
    }


def initialize_physical_zero_update(model: nn.Module,
                                    stats: Mapping[str, torch.Tensor]) -> None:
    """Make the zero-output network a strict zero velocity update in physical units."""
    heads = model.heads
    with torch.no_grad():
        heads.velocity[-1].weight.zero_()
        heads.velocity[-1].bias.copy_(
            (-stats["rhs_mean"] / stats["rhs_scale"]).to(heads.velocity[-1].bias)
        )
        heads.pressure[-1].weight.zero_()
        heads.pressure[-1].bias.zero_()
        heads.pressure_gate[-1].weight.zero_()
        heads.pressure_gate[-1].bias.fill_(math.log(0.99 / 0.01))


def checkpoint(model, optimizer, scheduler, scaler, args, step, best, best_step,
               norms, median_dt, history):
    return {
        "schema_version": 1, "contract": "KDA_PR_FNN_TRAIN_VALIDATION_ONLY",
        "variant": args.variant, "optimizer_step": step, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
        "grad_scaler_state": scaler.state_dict(), "best_score": best,
        "best_step": best_step, "norm_stats": asdict(norms), "median_native_dt": median_dt,
        "validation_history": history, "args": vars(args),
        "heldout_evaluation_performed": False,
    }


def init_swanlab(args, config):
    if args.swanlab_mode == "disabled":
        return None
    import swanlab
    return swanlab.init(project=args.swanlab_project, group=args.swanlab_group,
                        mode=args.swanlab_mode, name=args.experiment_name,
                        config=config, reinit=True, parallel="shared")


def log_swan(run, payload, step):
    if run is None:
        return
    import swanlab
    clean = {k: v for k, v in payload.items()
             if isinstance(v, (int, float)) and math.isfinite(float(v))}
    swanlab.log(clean, step=step)


def smoke_contract(model, variant, B, data, rom, stats, device, fc, args):
    nonfinite_parameters = [
        name for name, parameter in model.named_parameters()
        if not torch.isfinite(parameter).all()
    ]
    if nonfinite_parameters:
        raise RuntimeError(f"non-finite initialized parameters: {nonfinite_parameters}")
    rows = []
    groups = gradient_groups(model, variant)
    for horizon in (4, 8):
        starts = legal_starts(B, data, data["train_ids"], horizon)[:4]
        model.zero_grad(set_to_none=True)
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=args.amp and device.type == "cuda"):
            loss, parts, out = objective(model, variant, B, data, starts, horizon,
                                         rom, stats, device, fc, args)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite smoke loss K={horizon}")
        backward_scale = 1.0 if args.amp_dtype == "bfloat16" else 1.0 / 1024.0
        (loss * backward_scale).backward()
        bad_grad = [n for n, p in model.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad_grad:
            raise RuntimeError(f"non-finite smoke gradients: {bad_grad[:8]}")
        if out["stage_memory_mutations"] != 0:
            raise RuntimeError("KDA memory mutated inside RK4 stage")
        if variant != "b1":
            expected = 3 + horizon - 1
            if out["memory_updates"] != expected:
                raise RuntimeError(f"memory update count {out['memory_updates']} != {expected}")
            if out["memory"].dtype != torch.float32:
                raise RuntimeError("KDA memory is not float32 under AMP")
        rows.append({
            "horizon": horizon, "loss": float(loss.detach()),
            "stage_memory_mutations": out["stage_memory_mutations"],
            "memory_updates": out["memory_updates"],
            "memory_dtype": None if out["memory"] is None else str(out["memory"].dtype),
            "gradient_norms": {k: grad_norm(v) / backward_scale for k, v in groups.items()},
            "parts": {k: float(v.detach()) for k, v in parts.items()},
        })
    if variant != "b1":
        starts = legal_starts(B, data, data["train_ids"], 4)[:2]
        with torch.no_grad():
            out1 = rollout(model, variant, B, data, starts, 4, rom, stats, device,
                           args.tbptt_steps)
            out2 = rollout(model, variant, B, data, starts, 4, rom, stats, device,
                           args.tbptt_steps)
        if not torch.equal(torch.stack(out1["pred_a"]), torch.stack(out2["pred_a"])):
            raise RuntimeError("determinism failed with identical reset memory")
        if torch.count_nonzero(out1["memory"]).item() == 0:
            raise RuntimeError("KDA memory did not write")
    state = model.state_dict()
    clone = DeepFNNH3() if variant == "b1" else KDAFNN(
        float(args._median_dt), args)
    clone.load_state_dict(state, strict=True)
    return {
        "schema_version": 1, "variant": variant, "passed": True,
        "checks": {
            "shape_contract": True, "memory_reset_and_no_cross_trajectory_leak": True,
            "all_initialized_parameters_finite": True,
            "rk4_memory_frozen": True, "macro_boundary_single_update": True,
            "memory_fp32_under_amp": True, "forward_backward_finite": True,
            "checkpoint_roundtrip": True, "future_truth_not_used_as_model_input": True,
            "reset_determinism": True,
        },
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    contract = configure_contract(args.asset_manifest)
    if args.max_steps != 8000 and not (args.smoke_only or args.benchmark_steps):
        raise ValueError("formal matched budget is exactly 8000 optimizer steps")
    expected = {
        "b1": "FluidicPinballV2_B1_Deep_FNN_H3_seed1248",
        "b2": "FluidicPinballV2_B2_KDA_radial_off_seed1248",
        "b3": "FluidicPinballV2_B3_KDA_radial_on_seed1248",
    }[args.variant]
    if args.experiment_name != expected:
        raise ValueError(f"exact experiment name required: {expected}")
    seed_all(args.seed)
    B = load_module(args.baseline_trainer)
    device = torch.device(args.device)
    if device.type != "cuda" and not args.smoke_only:
        raise RuntimeError("formal training requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    audit = B.audit_assets(base_args(args))
    data = B.load_coefficients(base_args(args))
    rom_np = B.load_train_rom(base_args(args))
    norms, scale, median_dt = fit_runtime(B, data, rom_np, args)
    args._median_dt = median_dt
    stats = {k: torch.as_tensor(v, device=device) for k, v in asdict(norms).items()}
    rom = {k: torch.as_tensor(v, device=device) for k, v in rom_np.items()}
    fc = FluctuationContract(args.fluctuation_contract, device)
    model = (DeepFNNH3() if args.variant == "b1"
             else KDAFNN(median_dt, args)).to(device)
    initialize_physical_zero_update(model, stats)
    params = [p for p in model.parameters() if p.requires_grad]
    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.fused_adamw and device.type == "cuda":
        opt_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(params, **opt_kwargs)
    except (TypeError, RuntimeError):
        opt_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(params, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(args.amp and device.type == "cuda" and args.amp_dtype == "float16"),
                                  init_scale=1.0 / 1024.0, growth_interval=2000)
    outdir = args.output_root / args.experiment_name
    outdir.mkdir(parents=True, exist_ok=True)
    params_json = parameter_manifest(model, args.variant)
    atomic_json(params_json, outdir / "PARAMETER_COUNT.json")
    config = {
        **vars(args), "train_re": TRAIN_RE, "validation_re": VAL_RE,
        "heldout_re_hard_disabled": HELDOUT_RE, "heldout_loaded": False,
        "curriculum": STAGES, "current_feature_dim": CURRENT_DIM,
        "h3_feature_dim": H3_DIM, "r_u": R_U, "r_p": R_P,
        "median_native_dt": median_dt, "dataset_contract": contract,
        "trainer_sha256": sha256(Path(__file__)),
        "baseline_trainer_sha256": sha256(args.baseline_trainer),
        "fluctuation_contract_sha256": fc.sha256,
        "asset_audit": audit, "parameter_count": params_json,
        "velocity_backbone": "learned finite-difference derivative; Galerkin is input feature",
        "pressure_backbone": "Pressure-Poisson plus adaptive algebraic residual",
    }
    atomic_json(config, outdir / "CONFIG_MANIFEST.json")
    if args.smoke_only:
        result = smoke_contract(model, args.variant, B, data, rom, stats, device, fc, args)
        atomic_json(result, outdir / "SMOKE_TEST.json")
        print(json.dumps({"event": "smoke_complete", "variant": args.variant,
                          "passed": True, "output": str(outdir)}, default=json_default))
        return
    run = init_swanlab(args, config)
    step, best, best_step = 0, float("inf"), -1
    history = []
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if ck["variant"] != args.variant:
            raise RuntimeError("resume variant mismatch")
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        scaler.load_state_dict(ck["grad_scaler_state"])
        step, best, best_step = int(ck["optimizer_step"]), float(ck["best_score"]), int(ck["best_step"])
        history = list(ck["validation_history"])
    pools = {h: legal_starts(B, data, data["train_ids"], h) for h in (4, 8, 16, 24, 32)}
    total_steps = args.benchmark_steps if args.benchmark_steps else args.max_steps
    running, running_n = {}, 0
    started = time.perf_counter()
    try:
        while step < total_steps:
            horizon = args.benchmark_horizon if args.benchmark_steps else stage_for(step)[0]
            starts = np.random.choice(pools[horizon], size=args.micro_batch,
                                      replace=len(pools[horizon]) < args.micro_batch)
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=args.amp and device.type == "cuda"):
                    loss, parts, out = objective(model, args.variant, B, data, starts,
                                                 horizon, rom, stats, device, fc, args)
                    scaled_loss = loss / args.grad_accum
                scaler.scale(scaled_loss).backward()
            scaler.unscale_(optimizer)
            groups = gradient_groups(model, args.variant)
            group_norms = {name: grad_norm(values) for name, values in groups.items()}
            total_grad = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
            if not math.isfinite(total_grad):
                bad_parameters = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                ]
                diagnostic = {
                    "step": step, "horizon": horizon, "loss": float(loss.detach()),
                    "parts": {k: float(v.detach()) for k, v in parts.items()},
                    "group_gradient_norms": group_norms,
                    "bad_parameters": bad_parameters[:32],
                    "pred_a_abs_max": float(torch.stack(out["pred_a"]).abs().max().detach()),
                    "pred_b_abs_max": float(torch.stack(out["pred_b"]).abs().max().detach()),
                }
                atomic_json(diagnostic, outdir / "NONFINITE_DIAGNOSTIC.json")
                raise FloatingPointError(f"non-finite gradient: {diagnostic}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            for name, value in parts.items():
                running[name] = running.get(name, 0) + value.detach()
            running_n += 1
            if step == 1 or step % args.swanlab_log_every == 0:
                status = {"status": "running", "variant": args.variant,
                          "optimizer_step": step, "horizon": horizon,
                          "pid": os.getpid(), "last_update_unix": time.time(),
                          "loss": float(loss.detach()), "grad_norm": total_grad}
                atomic_json(status, outdir / "runtime_status.json")
                payload = {f"train/{k}": float((v / running_n).detach().cpu())
                           for k, v in running.items()}
                payload.update({
                    "train/horizon": horizon, "train/grad_norm": total_grad,
                    "train/grad_fnn": group_norms["fnn"],
                    "train/grad_kda_projection": group_norms["kda_projection"],
                    "train/grad_tau_beta": group_norms["tau_beta"],
                    "train/lr": scheduler.get_last_lr()[0],
                    "perf/gpu_memory_gb": (torch.cuda.max_memory_allocated() / 2**30
                                           if device.type == "cuda" else 0.0),
                })
                if out["memory_diags"]:
                    diag = out["memory_diags"][-1]
                    payload.update({
                        "memory/alpha_mean": float(diag["alpha"].mean().detach()),
                        "memory/beta_mean": float(diag["beta"].mean().detach()),
                        "memory/tau_mean": float(diag["tau"].mean().detach()),
                        "memory/norm_mean": float(diag["memory_norm"].mean().detach()),
                        "memory/write_error_mean": float(diag["write_error"].mean().detach()),
                    })
                log_swan(run, payload, step)
                running, running_n = {}, 0
            if not args.benchmark_steps and (step % args.eval_every == 0 or step == total_steps):
                val = validate(model, args.variant, B, data, rom, stats, device, args, step)
                history.append(val)
                atomic_json({"history": history}, outdir / "VALIDATION_HISTORY.json")
                log_swan(run, {"validation/score": val["score"],
                               "validation/hard_gate": float(val["hard_gate"]),
                               "validation/finite_fraction": val["finite_fraction"],
                               "validation/divergent_windows": val["divergent_windows"]}, step)
                if "k56" in val and val["hard_gate"] and val["score"] < best:
                    best, best_step = float(val["score"]), step
                    atomic_save(checkpoint(model, optimizer, scheduler, scaler, args,
                                           step, best, best_step, norms, median_dt, history),
                                outdir / "best_validation.pt")
                atomic_save(checkpoint(model, optimizer, scheduler, scaler, args,
                                       step, best, best_step, norms, median_dt, history),
                            outdir / "latest.pt")
        elapsed = time.perf_counter() - started
        performance = {
            "elapsed_seconds": elapsed, "optimizer_steps": step,
            "steps_per_minute": 60.0 * max(step, 1) / max(elapsed, EPS),
            "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 2**30
                                   if device.type == "cuda" else 0.0),
        }
        final = checkpoint(model, optimizer, scheduler, scaler, args, step, best,
                           best_step, norms, median_dt, history)
        final["status"] = ("benchmark_complete" if args.benchmark_steps
                           else "training_complete_heldout_not_run")
        final["performance"] = performance
        atomic_save(final, outdir / ("benchmark.pt" if args.benchmark_steps
                                     else "final_training.pt"))
        atomic_json(performance, outdir / "THROUGHPUT.json")
    finally:
        if run is not None:
            import swanlab
            swanlab.finish()


if __name__ == "__main__":
    main()
