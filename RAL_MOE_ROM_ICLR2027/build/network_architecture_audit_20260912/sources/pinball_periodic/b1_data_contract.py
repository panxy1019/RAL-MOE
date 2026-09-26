#!/usr/bin/env python3
"""Runtime data/physics contract for the Fluidic Pinball V2 B1 baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


EPS = 1.0e-12
RE_FEATURE_CENTER, RE_FEATURE_SCALE, RE_FEATURE_REF = 110.0, 90.0, 200.0


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _manifest(args) -> dict[str, Any]:
    return json.loads(args.asset_manifest.read_text(encoding="utf-8"))


def audit_assets(args) -> dict[str, Any]:
    manifest = _manifest(args)
    failures = []
    if manifest.get("scope") != "periodic_train_only_with_train_validation_view":
        failures.append("unexpected asset scope")
    for key in ("pressure_gauge_verified", "projection_error_verified",
                "split_contract_verified", "rom_tensor_verified"):
        if manifest.get(key) is not True:
            failures.append(f"{key} is not true")
    observed = {}
    for name, path in {
        "coefficient_view": args.coefficient_view,
        "galerkin": args.galerkin_path,
        "pressure": args.pressure_path,
    }.items():
        observed[name] = sha256(path)
        if observed[name] != manifest.get("sha256", {}).get(name):
            failures.append(f"SHA mismatch: {name}")
    if failures:
        raise RuntimeError("Asset audit failed: " + "; ".join(failures))
    return {"passed": True, "manifest": manifest, "observed_sha256": observed}


def _history_index(prev: np.ndarray, history_len: int) -> np.ndarray:
    hist = np.full((len(prev), history_len), -1, dtype=np.int64)
    for row in range(len(prev)):
        current = row
        for offset in range(history_len):
            if current < 0:
                break
            hist[row, offset] = current
            current = int(prev[current])
    return hist


def load_coefficients(args) -> dict[str, np.ndarray]:
    manifest = _manifest(args)
    ru, rp = int(manifest["r_u"]), int(manifest["r_p"])
    train_re = np.asarray(manifest["fit_reynolds"], dtype=np.float64)
    val_re = np.asarray(manifest["validation_reynolds"], dtype=np.float64)
    heldout = np.asarray(manifest["heldout_reynolds"], dtype=np.float64)
    z = np.load(args.coefficient_view, allow_pickle=False)
    a = np.asarray(z["coeff_uv"][:, :ru], dtype=np.float32)
    b = np.asarray(z["coeff_p"][:, :rp], dtype=np.float32)
    re = np.asarray(z["Re"], dtype=np.float64)
    time = np.asarray(z["time"], dtype=np.float64)
    if np.any(np.isclose(re[:, None], heldout[None, :], atol=5e-7)):
        raise RuntimeError("Heldout leakage in B1 coefficient view.")
    order = np.lexsort((time, re))
    a, b, re, time = a[order], b[order], re[order], time[order]
    nxt = np.full(len(re), -1, dtype=np.int64)
    prev = np.full(len(re), -1, dtype=np.int64)
    for value in np.unique(re):
        ids = np.flatnonzero(np.isclose(re, value, atol=5e-7))
        nxt[ids[:-1]] = ids[1:]
        prev[ids[1:]] = ids[:-1]
    hist = _history_index(prev, args.history_len)
    valid = np.flatnonzero((nxt >= 0) & np.all(hist >= 0, axis=1))
    train_ids = valid[np.any(np.isclose(re[valid, None], train_re[None, :], atol=5e-7), axis=1)]
    val_ids = valid[np.any(np.isclose(re[valid, None], val_re[None, :], atol=5e-7), axis=1)]
    if not len(train_ids) or not len(val_ids):
        raise RuntimeError("Empty B1 train or validation split.")
    areas = np.asarray(z["point_areas"], dtype=np.float32)
    if np.any(areas <= 0) or not np.all(np.isfinite(areas)):
        raise RuntimeError("Invalid cell-volume weights.")
    return {
        "a": a, "b": b, "re": re.astype(np.float32), "time": time.astype(np.float32),
        "next": nxt, "prev": prev, "hist": hist, "valid": valid,
        "train_ids": train_ids, "val_ids": val_ids,
        "phi_u": np.asarray(z["phi_uv"][:ru], dtype=np.float32),
        "phi_p": np.asarray(z["phi_p"][:rp], dtype=np.float32),
        "areas": areas,
        "mean_u": np.asarray(z["mean_uv_train"], dtype=np.float32),
        "mean_p": np.asarray(z["mean_p_train"], dtype=np.float32),
    }


def load_train_rom(args) -> dict[str, np.ndarray]:
    manifest = _manifest(args)
    ru, rp = int(manifest["r_u"]), int(manifest["r_p"])
    train_re = np.asarray(manifest["fit_reynolds"], dtype=np.float64)
    g = np.load(args.galerkin_path, allow_pickle=False)
    p = np.load(args.pressure_path, allow_pickle=False)
    gre = np.asarray(g["Re_values_computed"], dtype=np.float64)
    pre = np.asarray(p["Re_values_computed"], dtype=np.float64)
    if not (np.allclose(np.sort(gre), np.sort(train_re), atol=1e-12)
            and np.allclose(np.sort(pre), np.sort(train_re), atol=1e-12)):
        raise RuntimeError("B1 ROM interfaces are not exactly train-only.")
    gi, pi = np.argsort(gre), np.argsort(pre)
    out = {
        "nodes": gre[gi].astype(np.float32),
        "c": np.asarray(g["c_all"][gi, :ru], dtype=np.float32),
        "A": np.asarray(g["A_all"][gi, :ru, :ru], dtype=np.float32),
        "H": np.asarray(g["H"][:ru, :ru, :ru], dtype=np.float32),
        "P": np.asarray(g["P"][:ru, :rp], dtype=np.float32),
        "pc": np.asarray(p["c_tilde_all"][pi, :rp], dtype=np.float32),
        "pA": np.asarray(p["A_tilde_all"][pi, :rp, :ru], dtype=np.float32),
        "pH": np.asarray(p["H_tilde"][:rp, :ru, :ru], dtype=np.float32),
    }
    if not all(np.all(np.isfinite(value)) for value in out.values()):
        raise RuntimeError("Non-finite B1 ROM tensor.")
    return out


def _interp(re: torch.Tensor, nodes: torch.Tensor):
    hi = torch.searchsorted(nodes, re).clamp(1, nodes.numel() - 1)
    lo = hi - 1
    weight = ((re - nodes[lo]) / (nodes[hi] - nodes[lo]).clamp_min(1e-8)).clamp(0, 1)
    return lo, hi, weight


def galerkin(a: torch.Tensor, b: torch.Tensor, re: torch.Tensor,
             rom: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, b, re = a.float(), b.float(), re.float()
        lo, hi, weight = _interp(re, rom["nodes"].float())
        c = torch.lerp(rom["c"][lo].float(), rom["c"][hi].float(), weight[:, None])
        A = torch.lerp(rom["A"][lo].float(), rom["A"][hi].float(), weight[:, None, None])
        return (c + torch.einsum("bij,bj->bi", A, a)
                + torch.einsum("ijk,bj,bk->bi", rom["H"].float(), a, a)
                + b @ rom["P"].float().T)


def pressure_base(a: torch.Tensor, re: torch.Tensor,
                  rom: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, re = a.float(), re.float()
        lo, hi, weight = _interp(re, rom["nodes"].float())
        c = torch.lerp(rom["pc"][lo].float(), rom["pc"][hi].float(), weight[:, None])
        A = torch.lerp(rom["pA"][lo].float(), rom["pA"][hi].float(), weight[:, None, None])
        return (c + torch.einsum("bij,bj->bi", A, a)
                + torch.einsum("ijk,bj,bk->bi", rom["pH"].float(), a, a))


def make_base_features(a: torch.Tensor, b: torch.Tensor, rhs: torch.Tensor,
                       re: torch.Tensor) -> torch.Tensor:
    re = re.float()
    e_low = torch.linalg.norm(a[:, :min(4, a.shape[1])], dim=1, keepdim=True)
    split = min(12, a.shape[1])
    e_mid = torch.linalg.norm(a[:, min(4, a.shape[1]):split], dim=1, keepdim=True)
    e_high = torch.linalg.norm(a[:, split:], dim=1, keepdim=True)
    a_norm = torch.linalg.norm(a, dim=1, keepdim=True)
    b_norm = torch.linalg.norm(b, dim=1, keepdim=True)
    rhs_norm = torch.linalg.norm(rhs, dim=1, keepdim=True)
    a_energy, b_energy = torch.sum(a * a, 1, keepdim=True), torch.sum(b * b, 1, keepdim=True)
    return torch.cat([
        ((re - RE_FEATURE_CENTER) / RE_FEATURE_SCALE)[:, None],
        (RE_FEATURE_REF / re.clamp_min(EPS))[:, None], a, b, rhs,
        e_low, e_mid, e_high, b_norm, rhs_norm, a_energy, b_energy,
        a_energy + b_energy, e_low.square() / (a_energy + EPS),
        e_high.square() / (a_energy + EPS), b_norm / (a_norm + EPS),
    ], dim=1)


def _history_features(base: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                      rhs: torch.Tensor, ah: torch.Tensor, bh: torch.Tensor,
                      rh: torch.Tensor) -> torch.Tensor:
    columns = [base]
    for offset in range(1, ah.shape[1]):
        columns.extend([ah[:, offset], bh[:, offset], rh[:, offset],
                        a - ah[:, offset], b - bh[:, offset], rhs - rh[:, offset]])
    return torch.cat(columns, dim=1)


def state_features(a: torch.Tensor, b: torch.Tensor, re: torch.Tensor,
                   ah: torch.Tensor, bh: torch.Tensor, rh: torch.Tensor,
                   rom: Mapping[str, torch.Tensor], stats: Mapping[str, torch.Tensor]):
    g = galerkin(a, b, re, rom)
    raw = _history_features(make_base_features(a, b, g, re), a, b, g, ah, bh, rh)
    return (raw - stats["x_mean"]) / stats["x_scale"], g


def fit_stats(data: Mapping[str, np.ndarray], rom_np: Mapping[str, np.ndarray], args):
    ids = data["train_ids"]
    rom = {key: torch.from_numpy(value) for key, value in rom_np.items()}
    a, b, re = data["a"], data["b"], data["re"]
    at, bt, rt = torch.from_numpy(a[ids]), torch.from_numpy(b[ids]), torch.from_numpy(re[ids])
    rhs = galerkin(at, bt, rt, rom)
    hist = data["hist"][ids]
    ah, bh = torch.from_numpy(a[hist]), torch.from_numpy(b[hist])
    rh = galerkin(ah.reshape(-1, a.shape[1]), bh.reshape(-1, b.shape[1]),
                  torch.from_numpy(re[hist]).reshape(-1), rom).reshape(len(ids), args.history_len, a.shape[1])
    x = _history_features(make_base_features(at, bt, rhs, rt), at, bt, rhs, ah, bh, rh).numpy()
    nxt = data["next"][ids]
    dt = (data["time"][nxt] - data["time"][ids])[:, None]
    finite_difference = (a[nxt] - a[ids]) / np.maximum(dt, 1e-8)

    def mean_scale(values: np.ndarray):
        mean = values.mean(0).astype(np.float32)
        scale = values.std(0).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        return mean, scale

    xm, xs = mean_scale(x)
    rm, rs = mean_scale(finite_difference)
    pm, ps = mean_scale(b[nxt])
    _, asc = mean_scale(a[nxt])
    _, bsc = mean_scale(b[nxt])
    stats = NormStats(
        xm, xs, rm, rs, pm, ps, asc, bsc,
        float(np.quantile(np.linalg.norm(a[nxt], axis=1) ** 2, 0.02) + EPS),
        float(np.quantile(np.linalg.norm(b[nxt], axis=1) ** 2, 0.02) + EPS),
    )
    return stats, {"unused_by_b1": True}


def batch_from_ids(data, ids: np.ndarray, device: torch.device):
    hist = data["hist"][ids]
    return {
        "a": torch.as_tensor(data["a"][ids], device=device),
        "b": torch.as_tensor(data["b"][ids], device=device),
        "re": torch.as_tensor(data["re"][ids], device=device),
        "ah": torch.as_tensor(data["a"][hist], device=device),
        "bh": torch.as_tensor(data["b"][hist], device=device),
    }


def legal_starts(data, ids: np.ndarray, horizon: int) -> np.ndarray:
    allowed = set(int(value) for value in ids.tolist())
    starts = []
    for start in ids.tolist():
        current, valid = int(start), True
        for _ in range(horizon):
            current = int(data["next"][current])
            if current not in allowed:
                valid = False
                break
        if valid:
            starts.append(int(start))
    return np.asarray(starts, dtype=np.int64)


def relative_loss(pred: torch.Tensor, true: torch.Tensor, floor: torch.Tensor):
    return torch.mean(torch.sum((pred - true).square(), 1)
                      / torch.sum(true.square(), 1).clamp_min(floor))


def physical_relative(pred: torch.Tensor, true: torch.Tensor, modes: torch.Tensor,
                      mean: torch.Tensor, weights: torch.Tensor):
    pred_field = mean + pred @ modes
    true_field = mean + true @ modes
    numerator = torch.linalg.vector_norm((pred_field - true_field) * weights, dim=1)
    denominator = torch.linalg.vector_norm(true_field * weights, dim=1).clamp_min(EPS)
    return numerator / denominator
