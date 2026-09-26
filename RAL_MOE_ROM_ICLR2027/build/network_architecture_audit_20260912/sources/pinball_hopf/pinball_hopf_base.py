"""Runtime feature/data contract for Fluidic Pinball V2 Hopf rank999 training."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

EPS = 1.0e-12


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


@dataclass
class ScaleStats:
    center: np.ndarray
    plane: np.ndarray
    r_floor: float
    train_log_growth_median: float
    train_log_growth_mad: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _split_groups(path: Path) -> dict[str, np.ndarray]:
    groups: dict[str, list[float]] = {"train": [], "validation": [], "final_test": []}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups[row["split"]].append(float(row["Re"]))
    return {key: np.asarray(value, dtype=np.float64) for key, value in groups.items()}


def _matches(actual: Iterable[float], expected: np.ndarray) -> bool:
    a = np.sort(np.asarray(list(actual), dtype=np.float64))
    e = np.sort(np.asarray(expected, dtype=np.float64))
    return a.shape == e.shape and bool(np.all(np.abs(a - e) <= 5e-6))


def audit_assets(args) -> dict[str, Any]:
    manifest = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    groups = _split_groups(args.split_manifest)
    failures: list[str] = []
    if manifest.get("scope") != "fluidic_pinball_v2_hopf_rank999_train_validation_only":
        failures.append("unexpected asset scope")
    if not _matches(manifest.get("train_reynolds", []), groups["train"]):
        failures.append("manifest train Re mismatch")
    if not _matches(manifest.get("validation_reynolds", []), groups["validation"]):
        failures.append("manifest validation Re mismatch")
    if not _matches(manifest.get("heldout_reynolds_hard_disabled", []), groups["final_test"]):
        failures.append("manifest heldout Re mismatch")
    files = {"coefficient_view": args.coefficient_view,
             "galerkin": args.galerkin_path, "pressure": args.pressure_path}
    observed: dict[str, str] = {}
    for name, path in files.items():
        if not path.is_file():
            failures.append(f"missing {name}: {path}")
            continue
        observed[name] = _sha256(path)
        if observed[name] != manifest.get("sha256", {}).get(name):
            failures.append(f"SHA mismatch for {name}")
    if failures:
        raise RuntimeError("Asset audit failed closed:\n- " + "\n- ".join(failures))
    return {"manifest": manifest, "observed_sha256": observed, "passed": True}


def _history_index_matrix(prev: np.ndarray, history_len: int) -> np.ndarray:
    hist = np.full((len(prev), history_len), -1, dtype=np.int64)
    for row in range(len(prev)):
        cur = row
        for h in range(history_len):
            if cur < 0:
                break
            hist[row, h] = cur
            cur = int(prev[cur])
    return hist


def load_coefficients(args) -> dict[str, np.ndarray]:
    groups = _split_groups(args.split_manifest)
    with np.load(args.coefficient_view, allow_pickle=False) as z:
        a, b = z["a"].astype(np.float32), z["b"].astype(np.float32)
        re, time = z["re"].astype(np.float64), z["time"].astype(np.float64)
        split = z["split"].astype("U10")
        ru, rp = int(z["ru"]), int(z["rp"])
        phi_u, phi_p = z["phi_u"].astype(np.float32), z["phi_p"].astype(np.float32)
        areas = z["cell_volumes"].astype(np.float32)
        mean_u, mean_p = z["mean_u"].astype(np.float32), z["mean_p"].astype(np.float32)
    if a.shape[1] != ru or b.shape[1] != rp or phi_u.shape[0] != ru or phi_p.shape[0] != rp:
        raise RuntimeError("coefficient/POD rank contract mismatch")
    if np.any(split == "final_test"):
        raise RuntimeError("HELDOUT HARD-GATE: final_test rows are present")
    if not _matches(np.unique(re[split == "train"]), groups["train"]):
        raise RuntimeError("coefficient view train Re mismatch")
    if not _matches(np.unique(re[split == "validation"]), groups["validation"]):
        raise RuntimeError("coefficient view validation Re mismatch")
    order = np.lexsort((time, re))
    a, b, re, time, split = a[order], b[order], re[order], time[order], split[order]
    nxt = np.full(len(re), -1, dtype=np.int64)
    prev = np.full(len(re), -1, dtype=np.int64)
    for rv in np.unique(re):
        ids = np.flatnonzero(np.isclose(re, rv, atol=5e-7))
        if np.any(np.diff(time[ids]) <= 0):
            raise RuntimeError(f"non-increasing time for Re={rv}")
        nxt[ids[:-1]], prev[ids[1:]] = ids[1:], ids[:-1]
    hist = _history_index_matrix(prev, int(args.history_len))
    valid = np.flatnonzero((nxt >= 0) & np.all(hist >= 0, axis=1))
    train_ids = valid[split[valid] == "train"]
    val_ids = valid[split[valid] == "validation"]
    return {"a": a, "b": b, "re": re.astype(np.float32), "time": time.astype(np.float32),
            "split": split, "next": nxt, "prev": prev, "hist": hist, "valid": valid,
            "train_ids": train_ids, "val_ids": val_ids, "train_re": groups["train"],
            "validation_re": groups["validation"], "heldout_re": groups["final_test"],
            "ru": np.asarray(ru), "rp": np.asarray(rp), "phi_u": phi_u,
            "phi_p": phi_p, "areas": areas, "mean_u": mean_u, "mean_p": mean_p}


def load_train_rom(args) -> dict[str, np.ndarray]:
    groups = _split_groups(args.split_manifest)
    with np.load(args.galerkin_path, allow_pickle=False) as g, \
         np.load(args.pressure_path, allow_pickle=False) as p:
        gre, pre = g["Re_list"].astype(np.float64), p["Re_list"].astype(np.float64)
        gmask = g["split_list"].astype("U10") == "train"
        pmask = p["split_list"].astype("U10") == "train"
        if not _matches(gre[gmask], groups["train"]) or not _matches(pre[pmask], groups["train"]):
            raise RuntimeError("ROM train nodes do not match frozen split")
        gi, pi = np.argsort(gre[gmask]), np.argsort(pre[pmask])
        ru, rp = int(g["ru"]), int(g["rp"])
        out = {
            "nodes": gre[gmask][gi].astype(np.float32),
            "c": g["c_all"][gmask][gi, :ru].astype(np.float32),
            "A": g["A_all"][gmask][gi, :ru, :ru].astype(np.float32),
            "H": g["H"][:ru, :ru, :ru].astype(np.float32),
            "P": g["P"][:ru, :rp].astype(np.float32),
            "pc": p["c_tilde_all"][pmask][pi, :rp].astype(np.float32),
            "pA": p["A_tilde_all"][pmask][pi, :rp, :ru].astype(np.float32),
            "pH": p["H_tilde"][:rp, :ru, :ru].astype(np.float32),
        }
    if any(not np.all(np.isfinite(value)) for value in out.values()):
        raise RuntimeError("non-finite ROM tensor")
    return out


def _interp(re: torch.Tensor, nodes: torch.Tensor):
    hi = torch.searchsorted(nodes, re).clamp(1, nodes.numel() - 1)
    lo = hi - 1
    w = ((re - nodes[lo]) / (nodes[hi] - nodes[lo]).clamp_min(1e-8)).clamp(0, 1)
    return lo, hi, w


def galerkin(a: torch.Tensor, b: torch.Tensor, re: torch.Tensor,
             rom: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, b, re = a.float(), b.float(), re.float()
        lo, hi, w = _interp(re, rom["nodes"].float())
        c = torch.lerp(rom["c"][lo].float(), rom["c"][hi].float(), w[:, None])
        A = torch.lerp(rom["A"][lo].float(), rom["A"][hi].float(), w[:, None, None])
        return c + torch.einsum("bij,bj->bi", A, a) + torch.einsum(
            "ijk,bj,bk->bi", rom["H"].float(), a, a) + b @ rom["P"].float().T


def pressure_base(a: torch.Tensor, re: torch.Tensor,
                  rom: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, re = a.float(), re.float()
        lo, hi, w = _interp(re, rom["nodes"].float())
        c = torch.lerp(rom["pc"][lo].float(), rom["pc"][hi].float(), w[:, None])
        A = torch.lerp(rom["pA"][lo].float(), rom["pA"][hi].float(), w[:, None, None])
        return c + torch.einsum("bij,bj->bi", A, a) + torch.einsum(
            "ijk,bj,bk->bi", rom["pH"].float(), a, a)


def make_base_features(a: torch.Tensor, b: torch.Tensor, rhs: torch.Tensor,
                       re: torch.Tensor) -> torch.Tensor:
    low, split = min(4, a.shape[1]), min(12, a.shape[1])
    e_low = torch.linalg.vector_norm(a[:, :low], dim=1, keepdim=True)
    e_mid = torch.linalg.vector_norm(a[:, low:split], dim=1, keepdim=True)
    e_high = torch.linalg.vector_norm(a[:, split:], dim=1, keepdim=True)
    an, bn = torch.linalg.vector_norm(a, dim=1, keepdim=True), torch.linalg.vector_norm(b, dim=1, keepdim=True)
    rn = torch.linalg.vector_norm(rhs, dim=1, keepdim=True)
    ae, be = (a * a).sum(1, keepdim=True), (b * b).sum(1, keepdim=True)
    return torch.cat((re[:, None], (1 / re.clamp_min(EPS))[:, None], a, b, rhs,
                      e_low, e_mid, e_high, bn, rn, ae, be, ae + be,
                      e_low.square() / (ae + EPS), e_high.square() / (ae + EPS),
                      bn / (an + EPS)), dim=1)


def _history_features(base, a, b, rhs, ah, bh, rh):
    cols = [base]
    for h in range(1, ah.shape[1]):
        cols.extend((ah[:, h], bh[:, h], rh[:, h], a - ah[:, h], b - bh[:, h], rhs - rh[:, h]))
    return torch.cat(cols, dim=1)


def state_features(a, b, re, ah, bh, rh, rom, stats):
    g = galerkin(a, b, re, rom)
    x = _history_features(make_base_features(a, b, g, re), a, b, g, ah, bh, rh)
    return (x - stats["x_mean"]) / stats["x_scale"], g


def fit_stats(data, rom_np, args):
    ids, a, b, re = data["train_ids"], data["a"], data["b"], data["re"]
    rom = {key: torch.from_numpy(value) for key, value in rom_np.items()}
    at, bt, rt = torch.from_numpy(a[ids]), torch.from_numpy(b[ids]), torch.from_numpy(re[ids])
    rhs = galerkin(at, bt, rt, rom).numpy()
    hist = data["hist"][ids]
    ah, bh = torch.from_numpy(a[hist]), torch.from_numpy(b[hist])
    rh = galerkin(ah.reshape(-1, a.shape[1]), bh.reshape(-1, b.shape[1]),
                  torch.from_numpy(re[hist]).reshape(-1), rom).reshape(len(ids), args.history_len, a.shape[1])
    x = _history_features(make_base_features(at, bt, torch.from_numpy(rhs), rt), at, bt,
                          torch.from_numpy(rhs), ah, bh, rh).numpy()
    nxt = data["next"][ids]
    dt = (data["time"][nxt] - data["time"][ids])[:, None]
    fd = (a[nxt] - a[ids]) / np.maximum(dt, 1e-8)
    def ms(value):
        mean, scale = value.mean(0).astype(np.float32), value.std(0).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        return mean, scale
    xm, xs = ms(x); rm, rs = ms(fd); pm, ps = ms(b[nxt]); _, asc = ms(a[nxt]); _, bsc = ms(b[nxt])
    centered = np.concatenate([a[ids[np.isclose(re[ids], rv, atol=5e-6)]] -
                               a[ids[np.isclose(re[ids], rv, atol=5e-6)]].mean(0, keepdims=True)
                               for rv in data["train_re"]])
    eigval, eigvec = np.linalg.eigh(centered.T @ centered / len(centered))
    plane = eigvec[:, np.argsort(eigval)[-2:][::-1]].T.astype(np.float32)
    center = a[ids].mean(0).astype(np.float32)
    radii = np.linalg.norm(centered @ plane.T, axis=1)
    positive = radii[radii > 1e-10]
    floor = float(np.quantile(positive, args.scale_floor_quantile))
    growth = []
    for rv in data["train_re"]:
        rid = ids[np.isclose(re[ids], rv, atol=5e-6)]
        growth.append(np.diff(np.log(np.linalg.norm((a[rid] - center) @ plane.T, axis=1) + floor)))
    gg = np.concatenate(growth)
    stats = NormStats(xm, xs, rm, rs, pm, ps, asc, bsc,
                      float(np.quantile(np.linalg.norm(a[nxt], axis=1) ** 2, .02) + EPS),
                      float(np.quantile(np.linalg.norm(b[nxt], axis=1) ** 2, .02) + EPS))
    scale = ScaleStats(center, plane, floor, float(np.median(gg)),
                       float(np.median(np.abs(gg - np.median(gg))) + EPS))
    return stats, scale


def legal_starts(data, ids, horizon):
    allowed = set(int(i) for i in ids)
    starts = []
    for sid in ids:
        cur, ok = int(sid), True
        for _ in range(horizon):
            cur = int(data["next"][cur])
            if cur not in allowed:
                ok = False
                break
        if ok:
            starts.append(int(sid))
    return np.asarray(starts, dtype=np.int64)


def batch_from_ids(data, ids, device):
    hist = data["hist"][ids]
    return {"a": torch.as_tensor(data["a"][ids], device=device),
            "b": torch.as_tensor(data["b"][ids], device=device),
            "re": torch.as_tensor(data["re"][ids], device=device),
            "time": torch.as_tensor(data["time"][ids], device=device),
            "ah": torch.as_tensor(data["a"][hist], device=device),
            "bh": torch.as_tensor(data["b"][hist], device=device)}


def relative_loss(pred, true, floor):
    return torch.mean(torch.sum((pred - true).square(), 1) /
                      torch.clamp(torch.sum(true.square(), 1), min=floor))


def physical_relative(pred, true, phi, mean, sqrt_w):
    pf, tf = mean + pred @ phi, mean + true @ phi
    return torch.sqrt(torch.sum(((pf - tf) * sqrt_w).square(), 1) /
                      torch.clamp(torch.sum((tf * sqrt_w).square(), 1), min=EPS))
