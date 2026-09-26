#!/usr/bin/env python3
"""Strict CenteredSquare Hopf-local r11 HPRS-MoE trainer.

This entry point deliberately consumes a *sanitized* train+validation coefficient
view and train-only ROM interfaces.  Heldout rows are not accepted by the process.
It reuses the mature V16_1 encoder/expert/AdaptiveGate implementation, while the
training, rollout and split contracts are Hopf-only and optimizer-step based.

Scientific invariants enforced here:
  * ru=rp=11; exactly 23 train and 6 validation Reynolds numbers;
  * no heldout Reynolds number may occur in any loaded POD/coefficient asset;
  * Galerkin/pressure files contain exactly the 12 train interfaces; validation
    bases are linearly interpolated from those interfaces (never looked up);
  * phase is absent from the feature contract and autonomous rollout never reads
    future metadata other than dt and the fixed parameter Re;
  * B differs from A only by the train-only PCA-plane scale-aware objective.

The final heldout evaluator is intentionally a separate, later workflow.  This
script never loads or evaluates heldout data.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from periodic_moe_3090 import train_periodic_moe as v16  # noqa: E402


EPS = 1.0e-12
RANK = 11
TRAIN_RE = np.asarray(
    [
        94.0, 95.0, 95.05, 95.15, 95.2, 95.35, 95.4, 95.45,
        95.55, 95.6, 95.75, 96.0, 96.25, 96.75, 97.0, 97.25,
        97.75, 98.0, 98.5, 99.5, 100.0, 101.0, 101.724137931034,
    ],
    dtype=np.float64,
)
VAL_RE = np.asarray([94.5, 95.25, 95.5, 97.5, 99.0, 101.5], dtype=np.float64)
HELDOUT_RE = np.asarray([95.1, 95.3, 96.5, 100.5, 102.0], dtype=np.float64)
ALLOWED_VIEW_RE = np.sort(np.concatenate([TRAIN_RE, VAL_RE]))
STAGES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1200, 1), (1200, 2800, 2), (2800, 4800, 4), (4800, 7500, 8)
)


@dataclass(frozen=True)
class ScaleStats:
    center: np.ndarray
    plane: np.ndarray
    r_floor: float
    train_log_growth_median: float
    train_log_growth_mad: float


@dataclass
class NormStats:
    x_mean: np.ndarray
    x_scale: np.ndarray
    rhs_mean: np.ndarray
    rhs_scale: np.ndarray
    pressure_mean: np.ndarray
    pressure_scale: np.ndarray
    a_scale: np.ndarray
    b_state_scale: np.ndarray
    a_rel_floor: float
    b_rel_floor: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("common", "scale_aware"), required=True)
    p.add_argument("--coefficient-view", type=Path, required=True,
                   help="Sanitized Hopf-local train+validation NPZ; heldout rows forbidden.")
    p.add_argument("--galerkin-path", type=Path, required=True,
                   help="Sanitized train-only r11 Galerkin NPZ.")
    p.add_argument("--pressure-path", type=Path, required=True,
                   help="Sanitized train-only r11 pressure-Poisson NPZ.")
    p.add_argument("--asset-manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--seed", type=int, default=1248)
    p.add_argument("--r-u", type=int, default=RANK)
    p.add_argument("--r-p", type=int, default=RANK)
    p.add_argument("--history-len", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--experts", type=int, default=6)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--expert-hidden", type=int, default=1024)
    p.add_argument("--expert-blocks", type=int, default=4)
    p.add_argument("--quadratic-rank", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--rollout-batch", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=7500)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adaptive-gate-initial-logit", type=float, default=6.0)
    p.add_argument("--lambda-coeff", type=float, default=1.0)
    p.add_argument("--lambda-dynamics", type=float, default=1.0)
    p.add_argument("--lambda-pressure", type=float, default=0.55)
    p.add_argument("--lambda-reconstruction", type=float, default=0.08)
    p.add_argument("--lambda-rollout", type=float, default=0.30)
    p.add_argument("--lambda-pressure-rollout", type=float, default=0.25)
    p.add_argument("--lambda-consistency", type=float, default=0.15)
    p.add_argument("--lambda-trajectory", type=float, default=0.10)
    p.add_argument("--lambda-energy", type=float, default=0.02)
    p.add_argument("--lambda-router-balance", type=float, default=0.02)
    p.add_argument("--lambda-router-entropy", type=float, default=0.002)
    p.add_argument("--lambda-expert-diversity", type=float, default=0.01)
    p.add_argument("--lambda-scale-amplitude", type=float, default=1.0)
    p.add_argument("--lambda-scale-growth", type=float, default=0.5)
    p.add_argument("--lambda-scale-sign", type=float, default=0.1)
    p.add_argument("--scale-floor-quantile", type=float, default=0.10)
    p.add_argument("--scale-ramp-steps", type=int, default=200)
    p.add_argument("--scale-grad-min", type=float, default=0.05)
    p.add_argument("--scale-grad-max", type=float, default=0.15)
    p.add_argument("--scale-grad-target", type=float, default=0.10)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--long-eval-every", type=int, default=400)
    p.add_argument("--validation-windows-per-re", type=int, default=32)
    p.add_argument("--early-stop-patience-evals", type=int, default=8)
    p.add_argument("--early-stop-min-delta", type=float, default=1.0e-3)
    p.add_argument("--validation-lock", type=Path, default=Path("/tmp/hopf_validation_gpu.lock"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.42)
    p.add_argument("--benchmark-steps", type=int, default=0,
                   help="Run an isolated throughput benchmark instead of the 7500-step formal budget.")
    p.add_argument("--benchmark-horizon", type=int, choices=(1, 2, 4, 8), default=8)
    p.add_argument("--swanlab-mode", choices=("disabled", "online", "local", "offline"),
                   default="online")
    p.add_argument("--swanlab-project", default="CenteredSquare_Hopf_H4")
    p.add_argument("--swanlab-workspace", default=None)
    p.add_argument("--swanlab-group", default="CenteredSquare_Hopf_r11")
    p.add_argument("--swanlab-log-every", type=int, default=20)
    p.add_argument("--smoke-only", action="store_true")
    return p.parse_args()


def exact_name(variant: str) -> str:
    return "CenteredSquareHopf_r11_Common" if variant == "common" else "CenteredSquareHopf_r11_ScaleAware"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def values_match(actual: Iterable[float], expected: np.ndarray, tol: float = 5e-6) -> bool:
    a = np.sort(np.asarray(list(actual), dtype=np.float64))
    e = np.sort(np.asarray(expected, dtype=np.float64))
    return a.shape == e.shape and bool(np.all(np.abs(a - e) <= tol))


def assert_no_heldout(values: Iterable[float], source: str) -> None:
    a = np.asarray(list(values), dtype=np.float64)
    for value in HELDOUT_RE:
        if np.any(np.abs(a - value) <= 5e-6):
            raise RuntimeError(f"HELDOUT HARD-GATE: {source} contains Re={value:.6f}")


def manifest_list(m: Mapping[str, Any], *keys: str) -> List[float]:
    for key in keys:
        if key in m:
            return [float(v) for v in m[key]]
    return []


def audit_assets(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    required_true = (
        "pressure_gauge_verified", "projection_error_verified",
        "split_contract_verified", "rom_tensor_verified",
    )
    failures: List[str] = []
    if manifest.get("scope") not in ("hopf_local_train_only", "hopf_local_train_val_view"):
        failures.append("manifest.scope is not Hopf-local/train-only")
    if int(manifest.get("r_u", -1)) != RANK or int(manifest.get("r_p", -1)) != RANK:
        failures.append(f"manifest ranks are not ru=rp={RANK}")
    fit_re = manifest_list(manifest, "fit_reynolds", "train_reynolds", "pod_fit_reynolds")
    if not values_match(fit_re, TRAIN_RE):
        failures.append("POD/statistics fit Reynolds numbers are not exactly the 23 train Re")
    for key in required_true:
        if manifest.get(key) is not True:
            failures.append(f"{key} is not true")
    if "projection_error" not in manifest and "projection_errors" not in manifest:
        failures.append("projection error values are missing")
    files = {
        "coefficient_view": args.coefficient_view,
        "galerkin": args.galerkin_path,
        "pressure": args.pressure_path,
    }
    declared = manifest.get("sha256", {})
    observed: Dict[str, str] = {}
    for name, path in files.items():
        if not path.is_file():
            failures.append(f"missing {name}: {path}")
            continue
        observed[name] = sha256(path)
        expected = declared.get(name) or declared.get(path.name)
        if not expected or str(expected).lower() != observed[name]:
            failures.append(f"SHA mismatch/missing for {name}")
    if failures:
        raise RuntimeError("Asset audit failed closed:\n- " + "\n- ".join(failures))
    return {"manifest": manifest, "observed_sha256": observed, "passed": True}


def key(npz: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in npz.files:
            return npz[name]
    raise KeyError(f"None of {names} found; available={npz.files}")


def load_coefficients(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    z = np.load(args.coefficient_view, allow_pickle=False)
    a = key(z, "coeff_uv", "a").astype(np.float32)[:, :RANK]
    b = key(z, "coeff_p", "b").astype(np.float32)[:, :RANK]
    re = key(z, "Re", "re", "re_values_per_snapshot").astype(np.float64).reshape(-1)
    t = key(z, "time", "t").astype(np.float64).reshape(-1)
    if not (len(a) == len(b) == len(re) == len(t)):
        raise ValueError("coefficient view arrays have inconsistent lengths")
    unique_re = np.unique(np.round(re, 6))
    assert_no_heldout(unique_re, "coefficient view")
    if not values_match(unique_re, ALLOWED_VIEW_RE, tol=1.1e-6):
        raise RuntimeError(f"coefficient view must contain exactly 23 train + 6 val Re; got {unique_re}")
    if a.shape[1] != RANK or b.shape[1] != RANK:
        raise RuntimeError(f"coefficient view is not r{RANK}/rp{RANK}")
    order = np.lexsort((t, re))
    a, b, re, t = a[order], b[order], re[order], t[order]
    next_idx = np.full(len(re), -1, dtype=np.int64)
    prev_idx = np.full(len(re), -1, dtype=np.int64)
    for rv in np.unique(re):
        ids = np.flatnonzero(np.abs(re - rv) <= 5e-7)
        next_idx[ids[:-1]] = ids[1:]
        prev_idx[ids[1:]] = ids[:-1]
    hist = v16.history_index_matrix(np.arange(len(re), dtype=np.int64), prev_idx, args.history_len)
    valid = np.flatnonzero((next_idx >= 0) & np.all(hist >= 0, axis=1))
    train_ids = valid[np.any(np.abs(re[valid, None] - TRAIN_RE[None, :]) <= 5e-6, axis=1)]
    val_ids = valid[np.any(np.abs(re[valid, None] - VAL_RE[None, :]) <= 5e-6, axis=1)]
    if not len(train_ids) or not len(val_ids):
        raise RuntimeError("empty strict train or validation split")
    # Physical reconstruction assets. Only the single 23-Re train-fitted mean is accepted.
    phi_u = key(z, "phi_uv", "velocity_modes")[:RANK].astype(np.float32)
    phi_p = key(z, "phi_p", "pressure_modes")[:RANK].astype(np.float32)
    areas = key(z, "point_areas", "areas", "cell_areas").astype(np.float32).reshape(-1)
    if np.any(areas <= 0) or not np.all(np.isfinite(areas)):
        raise RuntimeError("invalid area weights")
    mean_u = key(z, "mean_uv_train", "mean_uv").astype(np.float32)
    mean_p = key(z, "mean_p_train", "mean_p").astype(np.float32)
    if mean_u.ndim != 1 or mean_p.ndim != 1:
        raise RuntimeError("per-Re means are forbidden in sanitized trainer; provide single train-only mean")
    return {
        "a": a, "b": b, "re": re.astype(np.float32), "time": t.astype(np.float32),
        "next": next_idx, "prev": prev_idx, "hist": hist, "valid": valid,
        "train_ids": train_ids, "val_ids": val_ids, "phi_u": phi_u, "phi_p": phi_p,
        "areas": areas, "mean_u": mean_u, "mean_p": mean_p,
    }


def load_train_rom(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    g = np.load(args.galerkin_path, allow_pickle=False)
    p = np.load(args.pressure_path, allow_pickle=False)
    gre = key(g, "Re_values_computed", "Re_values", "re_nodes").astype(np.float64)
    pre = key(p, "Re_values_computed", "Re_values", "re_nodes").astype(np.float64)
    assert_no_heldout(gre, "Galerkin interfaces")
    assert_no_heldout(pre, "pressure interfaces")
    if not values_match(gre, TRAIN_RE) or not values_match(pre, TRAIN_RE):
        raise RuntimeError("ROM files must expose exactly the 23 train Re interfaces")
    gi = np.argsort(gre)
    pi = np.argsort(pre)
    out = {
        "nodes": gre[gi].astype(np.float32),
        "c": key(g, "c_all", "c")[gi, :RANK].astype(np.float32),
        "A": key(g, "A_all", "A")[gi, :RANK, :RANK].astype(np.float32),
        "H": key(g, "H")[:RANK, :RANK, :RANK].astype(np.float32),
        "P": key(g, "P")[:RANK, :RANK].astype(np.float32),
        "pc": key(p, "c_tilde_all", "c_tilde")[pi, :RANK].astype(np.float32),
        "pA": key(p, "A_tilde_all", "A_tilde")[pi, :RANK, :RANK].astype(np.float32),
        "pH": key(p, "H_tilde")[:RANK, :RANK, :RANK].astype(np.float32),
    }
    for name, value in out.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite ROM tensor: {name}")
    return out


def interp_indices(re: torch.Tensor, nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hi = torch.searchsorted(nodes, re).clamp(1, nodes.numel() - 1)
    lo = hi - 1
    w = ((re - nodes[lo]) / torch.clamp(nodes[hi] - nodes[lo], min=1e-8)).clamp(0.0, 1.0)
    return lo, hi, w


def galerkin(a: torch.Tensor, b: torch.Tensor, re: torch.Tensor, rom: Mapping[str, torch.Tensor]) -> torch.Tensor:
    # Tensor contractions and the RK4 physical backbone remain FP32 even when
    # the learned encoder/experts run under FP16 autocast.
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, b, re = a.float(), b.float(), re.float()
        lo, hi, w = interp_indices(re, rom["nodes"].float())
        c = torch.lerp(rom["c"][lo].float(), rom["c"][hi].float(), w[:, None])
        A = torch.lerp(rom["A"][lo].float(), rom["A"][hi].float(), w[:, None, None])
        return c + torch.einsum("bij,bj->bi", A, a) + torch.einsum("ijk,bj,bk->bi", rom["H"].float(), a, a) + b @ rom["P"].float().T


def pressure_base(a: torch.Tensor, re: torch.Tensor, rom: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, re = a.float(), re.float()
        lo, hi, w = interp_indices(re, rom["nodes"].float())
        c = torch.lerp(rom["pc"][lo].float(), rom["pc"][hi].float(), w[:, None])
        A = torch.lerp(rom["pA"][lo].float(), rom["pA"][hi].float(), w[:, None, None])
        return c + torch.einsum("bij,bj->bi", A, a) + torch.einsum("ijk,bj,bk->bi", rom["pH"].float(), a, a)


def make_base_features(a: torch.Tensor, b: torch.Tensor, rhs: torch.Tensor, re: torch.Tensor) -> torch.Tensor:
    # phase-free by construction: no phase argument exists in this API.
    zeros = torch.zeros_like(re)
    return v16.make_features_torch(a, b, rhs, re, zeros, 0)


def make_history(base_x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, rhs: torch.Tensor,
                 ah: torch.Tensor, bh: torch.Tensor, rh: torch.Tensor) -> torch.Tensor:
    return v16.make_history_features_from_states_torch(base_x, a, b, rhs, ah, bh, rh)


def fit_stats(data: Mapping[str, np.ndarray], rom_np: Mapping[str, np.ndarray], args: argparse.Namespace) -> Tuple[NormStats, ScaleStats]:
    ids = data["train_ids"]
    a, b, re = data["a"], data["b"], data["re"]
    # CPU fitting only; validation is never referenced.
    rom = {k: torch.from_numpy(v) for k, v in rom_np.items()}
    at = torch.from_numpy(a[ids]); bt = torch.from_numpy(b[ids]); rt = torch.from_numpy(re[ids])
    rhs = galerkin(at, bt, rt, rom).numpy()
    hist = data["hist"][ids]
    base = make_base_features(at, bt, torch.from_numpy(rhs), rt)
    ah = torch.from_numpy(a[hist]); bh = torch.from_numpy(b[hist])
    # Galerkin history is computed from historical autonomous states/parameters, never phase.
    rh = galerkin(ah.reshape(-1, RANK), bh.reshape(-1, RANK),
                  torch.from_numpy(re[hist]).reshape(-1), rom).reshape(len(ids), args.history_len, RANK)
    x = make_history(base, at, bt, torch.from_numpy(rhs), ah, bh, rh).numpy()
    nxt = data["next"][ids]
    dt = (data["time"][nxt] - data["time"][ids])[:, None]
    fd = (a[nxt] - a[ids]) / np.maximum(dt, 1e-8)
    # CenteredSquare snapshots are separated by a large physical interval. The
    # continuous Galerkin operator remains an input feature, but directly using
    # it as an RK4 backbone is unstable at this sampling interval. Fit the
    # learned update to the finite-difference derivative itself.
    residual = fd
    pres_res = b[nxt]
    def ms(xv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, s = xv.mean(0).astype(np.float32), xv.std(0).astype(np.float32)
        s[s < 1e-8] = 1.0
        return m, s
    xm, xs = ms(x); rm, rs = ms(residual); pm, ps = ms(pres_res)
    _, asc = ms(a[nxt]); _, bsc = ms(b[nxt])
    per_re_centered = []
    for value in TRAIN_RE:
        re_ids = ids[np.abs(re[ids] - value) <= 5e-6]
        per_re_centered.append(a[re_ids] - a[re_ids].mean(0, keepdims=True))
    centered = np.concatenate(per_re_centered, axis=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    plane = eigenvectors[:, np.argsort(eigenvalues)[-2:][::-1]].T.astype(np.float32)
    center = a[ids].mean(0).astype(np.float32)
    r = np.linalg.norm(centered @ plane.T, axis=1)
    positive = r[r > 1e-10]
    if not len(positive):
        raise RuntimeError("cannot identify train-only Hopf plane/floor")
    r_floor = float(np.quantile(positive, args.scale_floor_quantile))
    growth: List[np.ndarray] = []
    for rv in TRAIN_RE:
        rid = ids[np.abs(re[ids] - rv) <= 5e-6]
        rr = np.linalg.norm((a[rid] - center) @ plane.T, axis=1)
        growth.append(np.diff(np.log(rr + r_floor)))
    gg = np.concatenate(growth)
    scale = ScaleStats(center, plane, r_floor, float(np.median(gg)), float(np.median(np.abs(gg - np.median(gg))) + EPS))
    stats = NormStats(xm, xs, rm, rs, pm, ps, asc, bsc,
                      float(np.quantile(np.linalg.norm(a[nxt], axis=1) ** 2, .02) + EPS),
                      float(np.quantile(np.linalg.norm(b[nxt], axis=1) ** 2, .02) + EPS))
    return stats, scale


def build_model(in_dim: int, args: argparse.Namespace, stats_t: Mapping[str, torch.Tensor], device: torch.device) -> nn.Module:
    model = v16.OperatorSpaceMoEROM(
        in_dim=in_dim, out_dim=RANK, pressure_dim=RANK, hidden_dim=args.hidden_dim,
        expert_hidden=args.expert_hidden, num_blocks=args.num_blocks, num_experts=args.experts,
        num_operator_spaces=1, num_regime_groups=1, experts_per_group=args.experts,
        top_k=args.top_k, group_top_k=1, dropout=args.dropout, temperature=args.temperature,
        gate_floor=0.0, group_temperature=1.0, group_gate_floor=0.0,
        shared_scale=1.0, routed_scale=.75, expert_blocks=args.expert_blocks,
        quadratic_rank=args.quadratic_rank, quadratic_scale=.05, phase_harmonics=0,
        closure_mode="adaptive_gate", pressure_base_mode="static", film_base_hidden=64,
        film_base_scale=.20, attractor_conditioned=False,
    ).to(device)
    # The scalar outer group router is mathematically inert and excluded from optimization.
    model.group_router.requires_grad_(False)
    v16.initialize_physical_zero_residual(model, {
        "rhs_op_mean": stats_t["rhs_mean"], "rhs_op_scale": stats_t["rhs_scale"],
        "pressure_mean": stats_t["pressure_mean"], "pressure_scale": stats_t["pressure_scale"],
    }, args.adaptive_gate_initial_logit)
    return model


def state_features(a: torch.Tensor, b: torch.Tensor, re: torch.Tensor,
                   ah: torch.Tensor, bh: torch.Tensor, rh: torch.Tensor,
                   rom: Mapping[str, torch.Tensor], stats: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    g = galerkin(a, b, re, rom)
    x = make_history(make_base_features(a, b, g, re), a, b, g, ah, bh, rh)
    return (x - stats["x_mean"]) / stats["x_scale"], g


def outputs(model: nn.Module, a: torch.Tensor, b: torch.Tensor, re: torch.Tensor,
            ah: torch.Tensor, bh: torch.Tensor, rh: torch.Tensor,
            rom: Mapping[str, torch.Tensor], stats: Mapping[str, torch.Tensor],
            return_stack: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    x, g = state_features(a, b, re, ah, bh, rh, rom, stats)
    rstd, pstd, gates, stack, closure = model(x, return_expert_stack=return_stack, return_closure_params=True)
    rhs = g + rstd * stats["rhs_scale"] + stats["rhs_mean"]
    pres_res = pstd * stats["pressure_scale"] + stats["pressure_mean"]
    return rhs, pres_res, gates, stack, closure, g


def autonomous_step(model: nn.Module, a: torch.Tensor, b: torch.Tensor, re: torch.Tensor, dt: torch.Tensor,
                    ah: torch.Tensor, bh: torch.Tensor, rh: torch.Tensor,
                    rom: Mapping[str, torch.Tensor], stats: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    k1, pres, gates, stack, closure, g = outputs(model, a, b, re, ah, bh, rh, rom, stats, True)
    k2 = outputs(model, a + .5 * dt * k1, b, re, ah, bh, rh, rom, stats)[0]
    k3 = outputs(model, a + .5 * dt * k2, b, re, ah, bh, rh, rom, stats)[0]
    k4 = outputs(model, a + dt * k3, b, re, ah, bh, rh, rom, stats)[0]
    an = a + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
    pb = pressure_base(an, re, rom)
    bn = v16.closure_components_torch("adaptive_gate", pb, pres, closure)[0]
    return an, bn, g, k1, gates, stack


def relative_loss(pred: torch.Tensor, true: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum((pred - true) ** 2, 1) / torch.clamp(torch.sum(true ** 2, 1), min=floor))


def scale_loss(pred: Sequence[torch.Tensor], true: Sequence[torch.Tensor], scale: Mapping[str, torch.Tensor],
               args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pp = torch.stack(pred, 1); tt = torch.stack(true, 1)
    pxy = (pp - scale["center"]) @ scale["plane"].T
    txy = (tt - scale["center"]) @ scale["plane"].T
    # sqrt(sum(x^2)+eps) has a finite derivative at the newly added near-steady rows;
    # torch.linalg.norm has an undefined zero-vector gradient on some CUDA paths.
    pr = torch.sqrt(torch.sum(pxy.float().square(), dim=-1) + 1e-12)
    tr = torch.sqrt(torch.sum(txy.float().square(), dim=-1) + 1e-12)
    plog = torch.log(pr + scale["r_floor"]); tlog = torch.log(tr + scale["r_floor"])
    amp = F.smooth_l1_loss(plog, tlog)
    if plog.shape[1] > 1:
        pg, tg = torch.diff(plog, dim=1), torch.diff(tlog, dim=1)
        growth = F.smooth_l1_loss(pg, tg)
        sign = torch.mean(F.relu(-pg * tg) / (torch.abs(tg) + 1e-4))
    else:
        growth = amp * 0; sign = amp * 0
    total = (args.lambda_scale_amplitude * amp + args.lambda_scale_growth * growth
             + args.lambda_scale_sign * sign)
    return total, {"amplitude": amp, "growth": growth, "sign": sign}


def stage_for(step: int) -> Tuple[int, int]:
    for start, end, k in STAGES:
        if start <= step < end:
            return k, start
    return 8, 4800


def legal_starts(data: Mapping[str, np.ndarray], ids: np.ndarray, horizon: int) -> np.ndarray:
    allowed = set(int(i) for i in ids.tolist())
    starts: List[int] = []
    for sid in ids:
        cur = int(sid); ok = True
        for _ in range(horizon):
            cur = int(data["next"][cur])
            if cur not in allowed:
                ok = False; break
        if ok:
            starts.append(int(sid))
    return np.asarray(starts, dtype=np.int64)


def batch_from_ids(data: Mapping[str, np.ndarray], ids: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    hist = data["hist"][ids]
    def t(name: str, idx: np.ndarray = ids) -> torch.Tensor:
        return torch.as_tensor(data[name][idx], device=device)
    return {"ids": torch.as_tensor(ids, device=device), "a": t("a"), "b": t("b"),
            "re": t("re"), "time": t("time"), "ah": t("a", hist), "bh": t("b", hist)}


def rollout_batch(model: nn.Module, data: Mapping[str, np.ndarray], starts: np.ndarray, horizon: int,
                  rom: Mapping[str, torch.Tensor], stats: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, Any]:
    q = batch_from_ids(data, starts, device)
    a, b, re, ah, bh = q["a"], q["b"], q["re"], q["ah"], q["bh"]
    rhs_hist = galerkin(ah.reshape(-1, RANK), bh.reshape(-1, RANK),
                         re[:, None].expand(-1, ah.shape[1]).reshape(-1), rom).reshape_as(ah)
    current = starts.copy(); pred_a: List[torch.Tensor] = []; pred_b: List[torch.Tensor] = []
    true_a: List[torch.Tensor] = []; true_b: List[torch.Tensor] = []
    pred_rhs: List[torch.Tensor] = []; true_rhs: List[torch.Tensor] = []
    gates_all: List[List[torch.Tensor]] = []; stacks: List[torch.Tensor] = []
    for _ in range(horizon):
        nxt = data["next"][current]
        dt = torch.as_tensor((data["time"][nxt] - data["time"][current])[:, None], device=device)
        a_before = a
        a, b, g, rhs, gates, stack = autonomous_step(model, a, b, re, dt, ah, bh, rhs_hist, rom, stats)
        ta = torch.as_tensor(data["a"][nxt], device=device); tb = torch.as_tensor(data["b"][nxt], device=device)
        pred_a.append(a); pred_b.append(b); true_a.append(ta); true_b.append(tb)
        pred_rhs.append(rhs); true_rhs.append((ta-a_before)/torch.clamp(dt,min=1.0e-8))
        gates_all.append(gates); stacks.append(stack)
        ah = torch.cat((a[:, None], ah[:, :-1]), 1); bh = torch.cat((b[:, None], bh[:, :-1]), 1)
        rhs_hist = torch.cat((g[:, None], rhs_hist[:, :-1]), 1); current = nxt
    return {"pred_a": pred_a, "pred_b": pred_b, "true_a": true_a, "true_b": true_b,
            "pred_rhs": pred_rhs, "true_rhs": true_rhs,
            "gates": gates_all, "stacks": stacks}


def common_objective(model: nn.Module, data: Mapping[str, np.ndarray], starts: np.ndarray, horizon: int,
                     rom: Mapping[str, torch.Tensor], stats: Mapping[str, torch.Tensor],
                     device: torch.device, args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    out = rollout_batch(model, data, starts, horizon, rom, stats, device)
    pa, pb = out["pred_a"], out["pred_b"]; ta, tb = out["true_a"], out["true_b"]
    a_rel = torch.stack([relative_loss(x, y, stats["a_rel_floor"]) for x, y in zip(pa, ta)]).mean()
    b_rel = torch.stack([relative_loss(x, y, stats["b_rel_floor"]) for x, y in zip(pb, tb)]).mean()
    coeff = torch.stack([F.mse_loss((x-y)/stats["a_scale"], torch.zeros_like(x)) for x,y in zip(pa,ta)]).mean()
    pressure = torch.stack([F.mse_loss((x-y)/stats["b_state_scale"], torch.zeros_like(x)) for x,y in zip(pb,tb)]).mean()
    energy = torch.stack([F.smooth_l1_loss(torch.sum(x*x,1), torch.sum(y*y,1)) for x,y in zip(pa,ta)]).mean()
    trajectory = a_rel + args.lambda_pressure_rollout*b_rel
    recon = a_rel  # area-weighted orthonormal POD coefficient error; full field is used in validation.
    gates = [g for step_gates in out["gates"] for g in step_gates]
    balance, entropy, _ = v16.router_regularization(gates)
    diversity = torch.stack([v16.expert_diversity_loss(s) for s in out["stacks"]]).mean()
    dynamics = torch.stack([
        F.mse_loss((x-y)/stats["rhs_scale"], torch.zeros_like(x))
        for x,y in zip(out["pred_rhs"],out["true_rhs"])
    ]).mean()
    consistency = trajectory
    loss = (args.lambda_coeff*coeff + args.lambda_dynamics*dynamics + args.lambda_pressure*pressure
            + args.lambda_reconstruction*recon + args.lambda_rollout*(a_rel + args.lambda_pressure_rollout*b_rel)
            + args.lambda_consistency*consistency + args.lambda_trajectory*trajectory
            + args.lambda_energy*energy + args.lambda_router_balance*balance
            - args.lambda_router_entropy*entropy + args.lambda_expert_diversity*diversity)
    return loss, {"coeff": coeff, "dynamics": dynamics, "pressure": pressure, "reconstruction": recon,
                  "rollout_u": a_rel, "rollout_p": b_rel, "trajectory": trajectory, "energy": energy,
                  "router_balance": balance, "router_entropy": entropy, "expert_diversity": diversity}, out


def grad_norm(loss: torch.Tensor, params: Sequence[nn.Parameter], retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    terms = [torch.sum(g.float() ** 2) for g in grads if g is not None]
    return torch.sqrt(torch.stack(terms).sum()) if terms else loss.new_zeros(())


def audit_scale_gradient(model: nn.Module, common: torch.Tensor, raw_scale: torch.Tensor,
                         multiplier: float | None, args: argparse.Namespace, step: int) -> Tuple[float, Dict[str, float]]:
    params = [p for p in model.parameters() if p.requires_grad]
    common_g = grad_norm(common, params, True)
    scale_g = grad_norm(raw_scale, params, True)
    base_ratio = float((scale_g / torch.clamp(common_g, min=EPS)).detach())
    if not math.isfinite(base_ratio) or base_ratio <= EPS:
        raise RuntimeError(f"scale-aware raw gradient ratio is invalid: {base_ratio}")
    previous_multiplier = multiplier
    # Explicitly recalibrate at every registered curriculum boundary. This is
    # logged and then hard-checked; no out-of-range stage is trained silently.
    multiplier = args.scale_grad_target / base_ratio
    ratio = base_ratio * multiplier
    record = {"step": step, "common_grad_norm": float(common_g.detach()),
              "raw_scale_grad_norm": float(scale_g.detach()), "raw_ratio": base_ratio,
              "weight_multiplier": float(multiplier), "weighted_ratio": float(ratio),
              "previous_weight_multiplier": None if previous_multiplier is None else float(previous_multiplier),
              "raw_scale_loss": float(raw_scale.detach()),
              "weighted_scale_loss": float((raw_scale*multiplier).detach())}
    if not (args.scale_grad_min <= ratio <= args.scale_grad_max):
        raise RuntimeError(f"scale-aware gradient ratio {ratio:.4f} outside fail-closed [{args.scale_grad_min}, {args.scale_grad_max}]")
    return float(multiplier), record


@contextlib.contextmanager
def validation_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+")
    try:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def physical_relative(pred: torch.Tensor, true: torch.Tensor, phi: torch.Tensor,
                      mean: torch.Tensor, sqrt_w: torch.Tensor) -> torch.Tensor:
    pf = mean + pred @ phi; tf = mean + true @ phi
    return torch.sqrt(torch.sum(((pf-tf)*sqrt_w)**2, 1) / torch.clamp(torch.sum((tf*sqrt_w)**2, 1), min=EPS))


@torch.no_grad()
def validate(model: nn.Module, data: Mapping[str, np.ndarray], rom: Mapping[str, torch.Tensor],
             stats: Mapping[str, torch.Tensor], scale: Mapping[str, torch.Tensor], device: torch.device,
             args: argparse.Namespace, step: int) -> Dict[str, Any]:
    model.eval(); horizons = [1,2,4,8,16]
    if step >= 4800 and step % args.long_eval_every == 0:
        horizons += [24,48]
    phi_u = torch.as_tensor(data["phi_u"], device=device); phi_p = torch.as_tensor(data["phi_p"], device=device)
    mean_u = torch.as_tensor(data["mean_u"], device=device); mean_p = torch.as_tensor(data["mean_p"], device=device)
    area = torch.as_tensor(np.sqrt(data["areas"]), device=device)
    uw = torch.cat((area, area)); pw = area
    report: Dict[str, Any] = {"step": step, "by_re": {}, "hard_gate": True}
    for rv in VAL_RE:
        ids = data["val_ids"][np.abs(data["re"][data["val_ids"]] - rv) <= 5e-6]
        rr: Dict[str, Any] = {}
        for h in horizons:
            starts = legal_starts(data, ids, h)
            if not len(starts):
                raise RuntimeError(f"no validation windows Re={rv} K={h}")
            if len(starts) > args.validation_windows_per_re:
                starts = starts[np.linspace(0, len(starts)-1, args.validation_windows_per_re, dtype=int)]
            out = rollout_batch(model, data, starts, h, rom, stats, device)
            pa, pb, ta, tb = out["pred_a"][-1], out["pred_b"][-1], out["true_a"][-1], out["true_b"][-1]
            finite = torch.isfinite(pa).all(1) & torch.isfinite(pb).all(1)
            uerr = physical_relative(pa, ta, phi_u, mean_u, uw)
            perr = physical_relative(pb, tb, phi_p, mean_p, pw)
            ratio_u = torch.linalg.norm(pa,dim=1)/torch.clamp(torch.linalg.norm(ta,dim=1),min=EPS)
            ratio_p = torch.linalg.norm(pb,dim=1)/torch.clamp(torch.linalg.norm(tb,dim=1),min=EPS)
            divergent = (~finite) | (ratio_u>10) | (ratio_p>10)
            penergy = torch.sum(pb*pb,1); tpenergy=torch.sum(tb*tb,1)
            pdrift=torch.abs(penergy-tpenergy)/torch.clamp(tpenergy,min=stats["b_rel_floor"])
            rr[str(h)] = {"u_physical_rel_l2": float(torch.nanmean(uerr)),
                          "p_physical_rel_l2": float(torch.nanmean(perr)),
                          "u_modal_rel_l2": float(torch.sqrt(relative_loss(pa,ta,stats["a_rel_floor"]))),
                          "p_modal_rel_l2": float(torch.sqrt(relative_loss(pb,tb,stats["b_rel_floor"]))),
                          "finite_fraction": float(finite.float().mean()),
                          "divergent_windows": int(divergent.sum()),
                          "max_u_modal_norm_ratio": float(torch.nan_to_num(ratio_u,nan=1e9).max()),
                          "max_p_modal_norm_ratio": float(torch.nan_to_num(ratio_p,nan=1e9).max()),
                          "pressure_energy_drift": float(torch.nanmean(pdrift))}
            if h in (8,16) and (int(divergent.sum()) or not bool(finite.all()) or float(pdrift.mean()) > 10):
                report["hard_gate"] = False
        report["by_re"][f"{rv:.6f}"] = rr
    report["score"] = max(max(v["8"]["u_physical_rel_l2"] + v["8"]["p_physical_rel_l2"],
                              v["16"]["u_physical_rel_l2"] + v["16"]["p_physical_rel_l2"])
                          for v in report["by_re"].values())
    model.train()
    return report


def init_swanlab(args: argparse.Namespace, config: Mapping[str, Any]) -> Any:
    if args.swanlab_mode == "disabled": return None
    import swanlab
    kw = dict(project=args.swanlab_project, group=args.swanlab_group, mode=args.swanlab_mode,
              name=args.experiment_name, config=dict(config), reinit=True, parallel="shared")
    if args.swanlab_workspace: kw["workspace"] = args.swanlab_workspace
    return swanlab.init(**kw)


def log_swan(run: Any, payload: Mapping[str, Any], step: int) -> None:
    if run is None: return
    import swanlab
    swanlab.log({k: v for k,v in payload.items() if isinstance(v,(int,float)) and math.isfinite(float(v))}, step=step)


def checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any,
               args: argparse.Namespace, step: int, best: float, best_step: int,
               scale_multiplier: float | None, audits: List[Dict[str, float]], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"schema_version": 1, "contract": "CenteredSquareHopf_r11_strict_train_only", "experiment_name": args.experiment_name,
            "variant": args.variant, "optimizer_step": step, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "grad_scaler_state": scaler.state_dict(), "rng": v16.rng_state_dict(), "best_score": best,
            "best_step": best_step, "scale_multiplier": scale_multiplier, "gradient_audits": audits,
            "history": history, "args": vars(args)}


def main() -> None:
    args = parse_args()
    if args.r_u != RANK or args.r_p != RANK:
        raise ValueError(f"strict contract fixes ru=rp={RANK}")
    if args.max_steps != 7500 and not (args.smoke_only or args.benchmark_steps > 0):
        raise ValueError("formal budget is exactly 7500 optimizer steps")
    wanted = exact_name(args.variant)
    if args.experiment_name is None: args.experiment_name = wanted
    if args.experiment_name != wanted: raise ValueError(f"experiment name must be {wanted}")
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" and not args.smoke_only: raise RuntimeError("formal training requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    if device.type == "cuda" and 0.0 < args.gpu_memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_memory_fraction,
            device=device.index if device.index is not None else torch.cuda.current_device(),
        )
    audit = audit_assets(args)
    data = load_coefficients(args); rom_np = load_train_rom(args)
    norms, scale_np = fit_stats(data, rom_np, args)
    outdir = args.output_root / args.experiment_name; outdir.mkdir(parents=True, exist_ok=True)
    code_sha = sha256(Path(__file__))
    config = {**vars(args), "train_re": TRAIN_RE, "validation_re": VAL_RE,
              "heldout_re_hard_disabled": HELDOUT_RE, "curriculum": STAGES,
              "effective_batch": args.micro_batch*args.grad_accum, "code_sha256": code_sha,
              "v16_common_base_sha256": sha256(Path(v16.__file__)),
              "phase_features": 0, "outer_group_router": "frozen_inert_singleton",
              "heldout_evaluation": "not_performed", "asset_audit": audit,
              "scale_stats_train_only": asdict(scale_np)}
    atomic_json(config, outdir/"config.json"); atomic_json(audit, outdir/"data_pod_manifest.json")
    stats_t = {k: torch.as_tensor(v, device=device) for k,v in asdict(norms).items()}
    scale_t = {k: torch.as_tensor(v, device=device) for k,v in asdict(scale_np).items()}
    rom = {k: torch.as_tensor(v, device=device) for k,v in rom_np.items()}
    # Dimension probe is train-only.
    probe = batch_from_ids(data, data["train_ids"][:2], device)
    prh = galerkin(probe["ah"].reshape(-1,RANK), probe["bh"].reshape(-1,RANK),
                    probe["re"][:,None].expand(-1,args.history_len).reshape(-1),rom).reshape_as(probe["ah"])
    in_dim = state_features(probe["a"],probe["b"],probe["re"],probe["ah"],probe["bh"],prh,rom,stats_t)[0].shape[1]
    model = build_model(in_dim,args,stats_t,device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt_kw = dict(lr=args.lr, weight_decay=args.weight_decay)
    if args.fused_adamw and device.type == "cuda": opt_kw["fused"] = True
    try: optimizer = torch.optim.AdamW(params, **opt_kw)
    except (TypeError, RuntimeError):
        opt_kw.pop("fused",None); optimizer=torch.optim.AdamW(params,**opt_kw)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.max_steps)
    scaler=torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type=="cuda",
        # The train-only Hopf coefficient scales make the unscaled K8 objective
        # gradient O(1e7); sub-unit dynamic scaling is required for FP16 safety.
        init_scale=1.0/1024.0, growth_interval=2000,
    )
    run=init_swanlab(args,config)
    step=0; best=float("inf"); best_step=-1; bad_evals=0; k8_evals=0
    scale_multiplier: float|None=None; audits:List[Dict[str,float]]=[]; history:List[Dict[str,Any]]=[]
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"]); scheduler.load_state_dict(ck["scheduler_state"])
        scaler.load_state_dict(ck["grad_scaler_state"]); v16.restore_rng_state(ck["rng"])
        step=int(ck["optimizer_step"]); best=float(ck["best_score"]); best_step=int(ck["best_step"])
        scale_multiplier=ck.get("scale_multiplier"); audits=list(ck.get("gradient_audits",[])); history=list(ck.get("history",[]))
    pools={k:legal_starts(data,data["train_ids"],k) for k in (1,2,4,8)}
    if any(not len(v) for v in pools.values()): raise RuntimeError("insufficient train rollout windows")
    max_steps=(8 if args.smoke_only else args.benchmark_steps if args.benchmark_steps > 0 else args.max_steps)
    benchmark_started=time.perf_counter()
    last_stage=None; running:Dict[str,torch.Tensor]={}; running_n=0
    try:
        while step < max_steps:
            if args.smoke_only:
                k=(1,1,2,2,4,4,8,8)[step % 8]; stage_start=step-(step%2)
            elif args.benchmark_steps > 0:
                k=args.benchmark_horizon; stage_start=0
            else:
                k,stage_start=stage_for(step)
            stage_changed=k!=last_stage
            optimizer.zero_grad(set_to_none=True)
            for _acc in range(args.grad_accum):
                starts=np.random.choice(pools[k],size=args.micro_batch,replace=len(pools[k])<args.micro_batch)
                if args.variant=="scale_aware" and stage_changed and _acc==0:
                    was=model.training; model.eval()
                    with torch.autocast(device_type=device.type,enabled=False):
                        audit_common,_,audit_rollout=common_objective(model,data,starts,k,rom,stats_t,device,args)
                        audit_raw,_=scale_loss(audit_rollout["pred_a"],audit_rollout["true_a"],scale_t,args)
                    scale_multiplier,rec=audit_scale_gradient(
                        model,audit_common,audit_raw,scale_multiplier,args,step
                    )
                    audits.append(rec); atomic_json({"audits":audits},outdir/"scale_gradient_audits.json")
                    model.train(was)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=args.amp and device.type=="cuda"):
                    common, parts, rollout=common_objective(model,data,starts,k,rom,stats_t,device,args)
                    raw_scale=common*0
                    if args.variant=="scale_aware": raw_scale,scale_parts=scale_loss(rollout["pred_a"],rollout["true_a"],scale_t,args)
                    ramp=min(1.0,max(0.0,(step-stage_start)/max(1,args.scale_ramp_steps)))
                    added=(raw_scale*float(scale_multiplier or 0.0)*ramp) if args.variant=="scale_aware" else raw_scale
                    loss=(common+added)/args.grad_accum
                scaler.scale(loss).backward()
                for name,val in {**parts,"scale_raw":raw_scale,"scale_weighted":added,"total":common+added}.items():
                    detached=val.detach()
                    running[name]=running[name]+detached if name in running else detached.clone()
                    running_n+=1 if name=="total" else 0
            scaler.unscale_(optimizer); grad=float(torch.nn.utils.clip_grad_norm_(params,args.grad_clip))
            if args.smoke_only:
                print(json.dumps({"event":"smoke_step","step":step,"horizon":k,
                                  "loss":float((common+added).detach()),"grad_norm":grad,
                                  "grad_scale":float(scaler.get_scale())}),flush=True)
            if not math.isfinite(grad): raise FloatingPointError("non-finite gradient")
            scaler.step(optimizer); scaler.update(); scheduler.step(); step+=1; last_stage=k
            if step%args.swanlab_log_every==0:
                denom=max(1,running_n); payload={f"train/{n}":float((v/denom).cpu()) for n,v in running.items()}
                payload.update({"train/grad_norm":grad,"train/lr":scheduler.get_last_lr()[0],"train/horizon":k,
                                "train/scale_ramp":ramp,"perf/gpu_memory_gb":torch.cuda.max_memory_allocated()/2**30 if device.type=="cuda" else 0})
                log_swan(run,payload,step); running={}; running_n=0
            if step%args.eval_every==0 or step==max_steps:
                with validation_lock(args.validation_lock): val=validate(model,data,rom,stats_t,scale_t,device,args,step)
                history.append(val); atomic_json({"history":history},outdir/"validation_history.json")
                log_swan(run,{"validation/score":val["score"],"validation/hard_gate":float(val["hard_gate"])},step)
                eligible=step>=6000
                if step>=4800: k8_evals+=1
                if eligible and k8_evals>=5 and val["hard_gate"]:
                    if val["score"] < best*(1-args.early_stop_min_delta):
                        best=float(val["score"]);best_step=step;bad_evals=0
                        atomic_save(checkpoint(model,optimizer,scheduler,scaler,args,step,best,best_step,scale_multiplier,audits,history),outdir/"best_validation.pt")
                    else: bad_evals+=1
                payload=checkpoint(model,optimizer,scheduler,scaler,args,step,best,best_step,scale_multiplier,audits,history)
                atomic_save(payload,outdir/"latest.pt")
                if eligible and k8_evals>=5 and bad_evals>=args.early_stop_patience_evals: break
        final=checkpoint(model,optimizer,scheduler,scaler,args,step,best,best_step,scale_multiplier,audits,history)
        elapsed=max(time.perf_counter()-benchmark_started,EPS)
        final["performance"]={"elapsed_seconds":elapsed,"optimizer_steps_per_min":60.0*step/elapsed,
                              "rollout_seed_samples_per_second":step*args.micro_batch*args.grad_accum/elapsed,
                              "peak_gpu_memory_gb":torch.cuda.max_memory_allocated()/2**30 if device.type=="cuda" else 0.0}
        final["status"]="training_complete_heldout_not_run";atomic_save(final,outdir/"final_training.pt")
        atomic_json(final["performance"],outdir/"throughput.json")
        # Contract placeholder: not a frozen selected checkpoint and explicitly unusable as heldout result.
        atomic_save({"status":"PENDING_USER_FREEZE_AND_HELDOUT","final_training":"final_training.pt",
                     "best_validation":"best_validation.pt" if best_step>=0 else None,
                     "heldout_evaluation_performed":False},outdir/"final.pt")
    finally:
        if run is not None:
            import swanlab; swanlab.finish()


if __name__ == "__main__":
    main()
