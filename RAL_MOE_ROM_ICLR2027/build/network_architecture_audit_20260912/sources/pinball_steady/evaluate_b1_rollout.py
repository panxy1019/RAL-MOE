#!/usr/bin/env python3
"""Frozen-checkpoint validation/final-test rollout for steady B1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

EPS = 1.0e-12
DEFAULT_HORIZONS = (1, 2, 4, 8, 16, 32, 56)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--baseline-trainer", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--asset-manifest", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--galerkin-path", type=Path, required=True)
    p.add_argument("--pressure-path", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True,
                   help="LABEL=PATH; first entry is the preselected official checkpoint")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--windows-per-re", type=int, default=64)
    p.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(value, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_groups(path: Path) -> dict[str, list[float]]:
    groups = {"train": [], "validation": [], "final_test": []}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups[row["split"]].append(float(row["Re"]))
    return groups


def matches(actual, expected) -> bool:
    a = np.sort(np.asarray(actual, dtype=np.float64))
    e = np.sort(np.asarray(expected, dtype=np.float64))
    return a.shape == e.shape and bool(np.all(np.abs(a - e) <= 5e-6))


def history_indices(prev: np.ndarray, history_len: int) -> np.ndarray:
    hist = np.full((len(prev), history_len), -1, dtype=np.int64)
    for row in range(len(prev)):
        cur = row
        for j in range(history_len):
            if cur < 0:
                break
            hist[row, j] = cur
            cur = int(prev[cur])
    return hist


def load_eval_split(root: Path, split: str, expected_re: list[float],
                    history_len: int, ru: int, rp: int,
                    expected_mesh_hash: str) -> tuple[dict, dict[str, str]]:
    directory = root / "eval_pod_coefficients_v2" / "steady" / split
    parts = {key: [] for key in ("a", "b", "re", "time")}
    observed, file_hashes, mesh_hash = [], {}, None
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            if str(z["split"].item()) != split or str(z["expert"].item()) != "steady":
                raise RuntimeError(f"bad evaluation contract: {path}")
            rv = float(z["Re"])
            observed.append(rv)
            current_mesh = str(z["mesh_hash"].item())
            mesh_hash = current_mesh if mesh_hash is None else mesh_hash
            if current_mesh != mesh_hash or current_mesh != expected_mesh_hash:
                raise RuntimeError(f"mesh mismatch: {path}")
            if int(z["rank999_ru"]) != ru or int(z["rank999_rp"]) != rp:
                raise RuntimeError(f"rank mismatch: {path}")
            count = len(z["times"])
            parts["a"].append(z["a_velocity_rank999"].astype(np.float32))
            parts["b"].append(z["b_pressure_rank999"].astype(np.float32))
            parts["re"].append(np.full(count, rv, dtype=np.float64))
            parts["time"].append(z["times"].astype(np.float64))
        file_hashes[str(path.relative_to(root))] = sha256(path)
    if not matches(observed, expected_re):
        raise RuntimeError(f"{split} files do not match frozen split manifest")
    a, b = np.concatenate(parts["a"]), np.concatenate(parts["b"])
    re, times = np.concatenate(parts["re"]), np.concatenate(parts["time"])
    order = np.lexsort((times, re))
    a, b, re, times = a[order], b[order], re[order], times[order]
    nxt = np.full(len(re), -1, dtype=np.int64)
    prev = np.full(len(re), -1, dtype=np.int64)
    for rv in np.unique(re):
        ids = np.flatnonzero(np.isclose(re, rv, atol=5e-7))
        if np.any(np.diff(times[ids]) <= 0):
            raise RuntimeError(f"non-increasing time: split={split}, Re={rv}")
        nxt[ids[:-1]], prev[ids[1:]] = ids[1:], ids[:-1]
    hist = history_indices(prev, history_len)
    valid = np.flatnonzero((nxt >= 0) & np.all(hist >= 0, axis=1))
    pack_path = root / "rom_assets_v2" / "steady" / "rom" / "rank999_ru4_rp3" / "pod_rank_pack.npz"
    with np.load(pack_path, allow_pickle=False) as pack:
        phi_u = pack["Phi_u"].transpose(0, 2, 1).reshape(ru, -1).astype(np.float32)
        phi_p = pack["Phi_p"].reshape(rp, -1).astype(np.float32)
        mean_u = pack["U_mean"].T.reshape(-1).astype(np.float32)
        mean_p = pack["p_mean"].reshape(-1).astype(np.float32)
        areas = pack["cellVolumes"].reshape(-1).astype(np.float32)
    data = {"a": a, "b": b, "re": re.astype(np.float32), "time": times.astype(np.float32),
            "next": nxt, "prev": prev, "hist": hist, "valid": valid, "val_ids": valid,
            "phi_u": phi_u, "phi_p": phi_p, "mean_u": mean_u, "mean_p": mean_p,
            "areas": areas}
    return data, file_hashes


def reconstruct(K, checkpoint: dict, device: torch.device):
    if checkpoint.get("variant") != "b1":
        raise RuntimeError("only B1 checkpoints are accepted")
    model = K.DeepFNNH3()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def evaluate_one(K, B, model, data, rom, stats, device, split, label, step,
                 rv, horizon, windows_per_re):
    ids = data["val_ids"][np.isclose(data["re"][data["val_ids"]], rv, atol=5e-6)]
    starts = B.legal_starts(data, ids, horizon)
    if not len(starts):
        raise RuntimeError(f"no starts: {split}, Re={rv}, K={horizon}")
    if len(starts) > windows_per_re:
        starts = starts[np.linspace(0, len(starts) - 1, windows_per_re, dtype=int)]
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        out = K.rollout(model, "b1", B, data, starts, horizon, rom, stats, device, 16)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        u, p = K.physical_series(B, data, out, device)
        joint = u + p
        pa, pb = torch.stack(out["pred_a"], 1), torch.stack(out["pred_b"], 1)
        ta, tb = torch.stack(out["true_a"], 1), torch.stack(out["true_b"], 1)
        finite = torch.isfinite(pa).all((1, 2)) & torch.isfinite(pb).all((1, 2))
        ratio_u = torch.linalg.vector_norm(pa, dim=2) / torch.linalg.vector_norm(ta, dim=2).clamp_min(EPS)
        ratio_p = torch.linalg.vector_norm(pb, dim=2) / torch.linalg.vector_norm(tb, dim=2).clamp_min(EPS)
        max_ratio = torch.maximum(ratio_u, ratio_p).max(1).values
        divergent = (~finite) | (max_ratio > 10)
        pdrift = torch.abs(pb.square().sum(2) - tb.square().sum(2)) / tb.square().sum(2).clamp_min(EPS)
    row = {"checkpoint": label, "checkpoint_step": step, "split": split,
           "Re": float(rv), "horizon": horizon, "windows": len(starts),
           "velocity_field_time_mean": float(u.mean()),
           "pressure_field_time_mean": float(p.mean()),
           "joint_field_time_mean": float(joint.mean()),
           "velocity_field_terminal": float(u[:, -1].mean()),
           "pressure_field_terminal": float(p[:, -1].mean()),
           "joint_field_terminal": float(joint[:, -1].mean()),
           "joint_field_worst_window_time": float(joint.max()),
           "pressure_drift_time_mean": float(pdrift.mean()),
           "finite_fraction": float(finite.float().mean()),
           "divergent_windows": int(divergent.sum()),
           "max_norm_ratio": float(max_ratio.max()), "inference_seconds": elapsed,
           "window_steps_per_second": len(starts) * horizon / max(elapsed, EPS)}
    growth = [{"checkpoint": label, "checkpoint_step": step, "split": split,
               "Re": float(rv), "horizon": horizon, "step": j + 1,
               "velocity_field_mean": float(u[:, j].mean()),
               "pressure_field_mean": float(p[:, j].mean()),
               "joint_field_mean": float(joint[:, j].mean())}
              for j in range(horizon)]
    return row, growth


def aggregate(rows: list[dict], labels: list[str], horizons: tuple[int, ...]):
    result = {}
    for label in labels:
        result[label] = {}
        for split in ("validation", "final_test"):
            result[label][split] = {}
            for horizon in horizons:
                selected = [r for r in rows if r["checkpoint"] == label and
                            r["split"] == split and r["horizon"] == horizon]
                result[label][split][str(horizon)] = {
                    "velocity_field_time_mean": float(np.mean([r["velocity_field_time_mean"] for r in selected])),
                    "pressure_field_time_mean": float(np.mean([r["pressure_field_time_mean"] for r in selected])),
                    "joint_field_time_mean": float(np.mean([r["joint_field_time_mean"] for r in selected])),
                    "joint_field_terminal": float(np.mean([r["joint_field_terminal"] for r in selected])),
                    "joint_field_worst_window_time": float(np.max([r["joint_field_worst_window_time"] for r in selected])),
                    "pressure_drift_time_mean": float(np.mean([r["pressure_drift_time_mean"] for r in selected])),
                    "finite_fraction_min": float(np.min([r["finite_fraction"] for r in selected])),
                    "divergent_windows_total": int(np.sum([r["divergent_windows"] for r in selected])),
                    "windows_total": int(np.sum([r["windows"] for r in selected])),
                    "inference_seconds_total": float(np.sum([r["inference_seconds"] for r in selected])),
                }
    return result


def plot_results(summary, rows, growth, official, horizons, output_dir):
    try:
        import matplotlib
    except ModuleNotFoundError:
        atomic_json({"status": "skipped", "reason": "matplotlib is not installed"},
                    output_dir / "PLOT_STATUS.json")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(summary)
    for split in ("validation", "final_test"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for label in labels:
            values = summary[label][split]
            axes[0].plot(horizons, [100 * values[str(h)]["velocity_field_time_mean"] for h in horizons], marker="o", label=label)
            axes[1].plot(horizons, [100 * values[str(h)]["pressure_field_time_mean"] for h in horizons], marker="o", label=label)
            axes[2].plot(horizons, [100 * values[str(h)]["joint_field_time_mean"] for h in horizons], marker="o", label=label)
        for ax, title in zip(axes, ("Velocity", "Pressure", "Joint")):
            ax.set(xlabel="Rollout horizon", ylabel="Mean relative error (%)", title=title)
            ax.grid(alpha=.25); ax.legend()
        fig.suptitle(f"Steady B1 {split} rollout")
        fig.tight_layout(); fig.savefig(output_dir / f"{split}_error_by_horizon.png", dpi=180); plt.close(fig)
    selected = [r for r in rows if r["checkpoint"] == official and r["horizon"] == max(horizons)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, split in zip(axes, ("validation", "final_test")):
        ss = sorted((r for r in selected if r["split"] == split), key=lambda r: r["Re"])
        x = [r["Re"] for r in ss]
        ax.plot(x, [100*r["velocity_field_time_mean"] for r in ss], marker="o", label="velocity")
        ax.plot(x, [100*r["pressure_field_time_mean"] for r in ss], marker="o", label="pressure")
        ax.set(xlabel="Re", ylabel="Mean relative error (%)", title=f"{split} K{max(horizons)}")
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "official_kmax_by_re.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, split in zip(axes, ("validation", "final_test")):
        gg = [r for r in growth if r["checkpoint"] == official and r["split"] == split and r["horizon"] == max(horizons)]
        for rv in sorted(set(r["Re"] for r in gg)):
            rr = [r for r in gg if r["Re"] == rv]
            ax.plot([r["step"] for r in rr], [100*r["joint_field_mean"] for r in rr], label=f"Re={rv:g}")
        ax.set(xlabel="Autonomous step", ylabel="Joint relative error (%)", title=f"{split} K{max(horizons)} growth")
        ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(output_dir / "official_kmax_error_growth.png", dpi=180); plt.close(fig)


def main() -> None:
    args = parse_args()
    horizons = tuple(int(v) for v in args.horizons.split(","))
    if not horizons or any(v <= 0 for v in horizons):
        raise ValueError("invalid horizons")
    checkpoints = []
    for item in args.checkpoint:
        label, sep, value = item.partition("=")
        if not sep or not label:
            raise ValueError("--checkpoint must be LABEL=PATH")
        checkpoints.append((label, Path(value)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    K = load_module("steady_b1_trainer", args.trainer)
    B = load_module("steady_b1_base", args.baseline_trainer)
    shape = K.configure_shape_contract(SimpleNamespace(asset_manifest=args.asset_manifest, history_len=3))
    groups = split_groups(args.split_manifest)
    manifest = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    data_by_split, eval_hashes = {}, {}
    for split in ("validation", "final_test"):
        data_by_split[split], hashes = load_eval_split(
            args.dataset_root, split, groups[split], 3, shape["ru"], shape["rp"],
            str(manifest["mesh_hash"]))
        eval_hashes.update(hashes)
    rom_np = B.load_train_rom(SimpleNamespace(galerkin_path=args.galerkin_path,
                                               pressure_path=args.pressure_path,
                                               split_manifest=args.split_manifest))
    device = torch.device(args.device)
    rom = {k: torch.as_tensor(v, device=device) for k, v in rom_np.items()}
    rows, growth_rows, checkpoint_manifest = [], [], {}
    start = time.perf_counter()
    for label, path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract") != "PINBALL_V2_STEADY_TRAIN_VALIDATION_ONLY":
            raise RuntimeError(f"checkpoint contract mismatch: {path}")
        if checkpoint.get("heldout_evaluation_performed") is not False:
            raise RuntimeError(f"checkpoint was not frozen before heldout evaluation: {path}")
        if checkpoint.get("asset_hashes") != {k: manifest["sha256"][k] for k in ("coefficient_view", "galerkin", "pressure")}:
            raise RuntimeError(f"training asset hash mismatch: {path}")
        model = reconstruct(K, checkpoint, device)
        stats = {k: torch.as_tensor(v, device=device) for k, v in checkpoint["norm_stats"].items()}
        step = int(checkpoint["optimizer_step"])
        checkpoint_manifest[label] = {"path": str(path), "sha256": sha256(path),
                                      "optimizer_step": step,
                                      "preselected_official": label == checkpoints[0][0]}
        for split, data in data_by_split.items():
            for rv in groups[split]:
                for horizon in horizons:
                    row, growth = evaluate_one(K, B, model, data, rom, stats, device,
                                               split, label, step, rv, horizon,
                                               args.windows_per_re)
                    rows.append(row); growth_rows.extend(growth)
    elapsed = time.perf_counter() - start
    labels = [label for label, _ in checkpoints]
    summary = aggregate(rows, labels, horizons)
    total_window_steps = int(sum(r["windows"] * r["horizon"] for r in rows))
    payload = {"schema_version": 1,
               "scope": "frozen_validation_and_final_test_rollout",
               "official_checkpoint": labels[0],
               "checkpoint_selection_source": "training-time validation only",
               "final_test_used_for_selection": False,
               "horizons": list(horizons), "windows_per_re": args.windows_per_re,
               "split_reynolds": {k: groups[k] for k in ("validation", "final_test")},
               "shape_contract": shape, "checkpoints": checkpoint_manifest,
               "training_asset_manifest_sha256": sha256(args.asset_manifest),
               "evaluation_file_sha256": eval_hashes, "summary": summary,
               "finite_all": all(r["finite_fraction"] == 1.0 for r in rows),
               "divergent_windows_total": int(sum(r["divergent_windows"] for r in rows)),
               "elapsed_seconds": elapsed, "total_window_steps": total_window_steps,
               "window_steps_per_second": total_window_steps / max(elapsed, EPS)}
    write_csv(rows, args.output_dir / "ROLLOUT_RESULTS.csv")
    write_csv(growth_rows, args.output_dir / "ERROR_GROWTH_CURVES.csv")
    atomic_json(payload, args.output_dir / "ROLLOUT_SUMMARY.json")
    plot_results(summary, rows, growth_rows, labels[0], horizons, args.output_dir)
    print(json.dumps({"event": "evaluation_complete", "output": str(args.output_dir),
                      "finite_all": payload["finite_all"],
                      "divergent_windows_total": payload["divergent_windows_total"],
                      "elapsed_seconds": elapsed}))


if __name__ == "__main__":
    main()
