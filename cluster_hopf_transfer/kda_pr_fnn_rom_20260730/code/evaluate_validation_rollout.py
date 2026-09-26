#!/usr/bin/env python3
"""Validation-only K1--K56 rollout evaluation for B1/B2/B3."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HORIZONS = (1, 2, 4, 8, 16, 32, 56)
VAL_RE = np.asarray([94.5, 95.25, 95.5, 97.5, 99.0, 101.5])
HELDOUT_RE = (95.1, 95.3, 96.5, 100.5, 102.0)
EPS = 1.0e-12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--baseline-trainer", type=Path, required=True)
    p.add_argument("--coefficient-view", type=Path, required=True)
    p.add_argument("--galerkin-path", type=Path, required=True)
    p.add_argument("--pressure-path", type=Path, required=True)
    p.add_argument("--asset-manifest", type=Path, required=True)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--windows-per-re", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(payload, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(x):
    x = x.detach().float().reshape(-1)
    q = torch.quantile(x, torch.tensor([0.05, 0.5, 0.95], device=x.device))
    return float(x.mean()), float(q[0]), float(q[1]), float(q[2])


def reconstruct(K, variant, checkpoint, device):
    saved = checkpoint["args"]
    if variant == "b1":
        model = K.DeepFNNH3()
    else:
        cfg = SimpleNamespace(
            token_dim=int(saved["token_dim"]),
            num_heads=int(saved["num_heads"]),
            d_k=int(saved["d_k"]),
            d_v=int(saved["d_v"]),
            beta_max=float(saved["beta_max"]),
        )
        model = K.KDAFNN(float(checkpoint["median_native_dt"]), cfg)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def modal_relative(pred, true):
    return torch.linalg.vector_norm(pred - true, dim=-1) / torch.linalg.vector_norm(
        true, dim=-1).clamp_min(EPS)


def evaluate_one(K, B, model, variant, data, rom, stats, device, rv, horizon,
                 windows_per_re):
    ids = data["val_ids"][np.isclose(data["re"][data["val_ids"]], rv, atol=5e-6)]
    starts = B.legal_starts(data, ids, horizon)
    if not len(starts):
        raise RuntimeError(f"no validation starts: Re={rv}, K={horizon}")
    if len(starts) > windows_per_re:
        starts = starts[np.linspace(0, len(starts) - 1, windows_per_re, dtype=int)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = K.rollout(model, variant, B, data, starts, horizon, rom, stats,
                        device, 16)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        u, p = K.physical_series(B, data, out, device)
        joint = u + p
        pa = torch.stack(out["pred_a"], 1)
        pb = torch.stack(out["pred_b"], 1)
        ta = torch.stack(out["true_a"], 1)
        tb = torch.stack(out["true_b"], 1)
        finite = torch.isfinite(pa).all((1, 2)) & torch.isfinite(pb).all((1, 2))
        ratio_u = torch.linalg.vector_norm(pa, dim=2) / torch.linalg.vector_norm(
            ta, dim=2).clamp_min(EPS)
        ratio_p = torch.linalg.vector_norm(pb, dim=2) / torch.linalg.vector_norm(
            tb, dim=2).clamp_min(EPS)
        ratio = torch.maximum(ratio_u, ratio_p)
        divergent = (~finite) | (ratio.max(1).values > 10)
        pdrift = torch.abs((pb * pb).sum(2) - (tb * tb).sum(2)) / (
            (tb * tb).sum(2).clamp_min(EPS))
        um = modal_relative(pa, ta)
        pm = modal_relative(pb, tb)
    row = {
        "model": variant, "Re": float(rv), "horizon": horizon,
        "windows": len(starts),
        "velocity_field_time_mean": float(u.mean()),
        "pressure_field_time_mean": float(p.mean()),
        "joint_field_time_mean": float(joint.mean()),
        "velocity_field_terminal": float(u[:, -1].mean()),
        "pressure_field_terminal": float(p[:, -1].mean()),
        "joint_field_terminal": float(joint[:, -1].mean()),
        "joint_field_worst_window_time": float(joint.max()),
        "velocity_modal_time_mean": float(um.mean()),
        "pressure_modal_time_mean": float(pm.mean()),
        "pressure_drift_time_mean": float(pdrift.mean()),
        "finite_fraction": float(finite.float().mean()),
        "divergent_windows": int(divergent.sum()),
        "max_norm_ratio": float(ratio.max()),
        "inference_seconds": wall,
        "window_steps_per_second": len(starts) * horizon / max(wall, EPS),
    }
    growth = []
    for step in range(horizon):
        growth.append({
            "model": variant, "Re": float(rv), "horizon": horizon, "step": step + 1,
            "velocity_field_mean": float(u[:, step].mean()),
            "pressure_field_mean": float(p[:, step].mean()),
            "joint_field_mean": float(joint[:, step].mean()),
        })
    memory_rows = []
    if variant != "b1" and horizon == 56:
        final_memory = out["memory"].float()
        singular = torch.linalg.svdvals(final_memory)
        probability = singular / singular.sum(-1, keepdim=True).clamp_min(EPS)
        effective_rank = torch.exp(
            -(probability * torch.log(probability.clamp_min(EPS))).sum(-1))
        cosine = torch.stack(out["query_cosine"], dim=1)
        diagnostics = out["memory_diags"]
        tau = torch.stack([d["tau"] for d in diagnostics])
        alpha = torch.stack([d["alpha"] for d in diagnostics])
        beta = torch.stack([d["beta"] for d in diagnostics])
        memory_norm = torch.stack([d["memory_norm"] for d in diagnostics])
        write_error = torch.stack([d["write_error"] for d in diagnostics])
        for head in range(final_memory.shape[1]):
            tm, tq05, tq50, tq95 = quantiles(tau[:, :, head])
            am, aq05, aq50, aq95 = quantiles(alpha[:, :, head])
            bm, bq05, bq50, bq95 = quantiles(beta[:, :, head])
            memory_rows.append({
                "model": variant, "Re": float(rv), "horizon": horizon, "head": head,
                "tau_mean": tm, "tau_q05": tq05, "tau_q50": tq50, "tau_q95": tq95,
                "half_life_mean": tm * math.log(2.0),
                "alpha_mean": am, "alpha_q05": aq05, "alpha_q50": aq50,
                "alpha_q95": aq95,
                "beta_mean": bm, "beta_q05": bq05, "beta_q50": bq50,
                "beta_q95": bq95,
                "memory_frobenius_mean": float(memory_norm[:, :, head].mean()),
                "memory_max_singular_mean": float(singular[:, head, 0].mean()),
                "memory_effective_rank_mean": float(effective_rank[:, head].mean()),
                "delta_write_error_mean": float(write_error[:, :, head].mean()),
                "query_u_p_cosine_mean": float(cosine[:, :, head].mean()),
            })
    return row, growth, memory_rows


def aggregate(rows):
    result = {}
    for model in ("b1", "b2", "b3"):
        result[model] = {}
        for horizon in HORIZONS:
            selected = [r for r in rows if r["model"] == model and r["horizon"] == horizon]
            result[model][str(horizon)] = {
                "velocity_field_time_mean": float(np.mean(
                    [r["velocity_field_time_mean"] for r in selected])),
                "pressure_field_time_mean": float(np.mean(
                    [r["pressure_field_time_mean"] for r in selected])),
                "joint_field_time_mean": float(np.mean(
                    [r["joint_field_time_mean"] for r in selected])),
                "joint_field_terminal": float(np.mean(
                    [r["joint_field_terminal"] for r in selected])),
                "joint_field_worst_window_time": float(np.max(
                    [r["joint_field_worst_window_time"] for r in selected])),
                "pressure_drift_time_mean": float(np.mean(
                    [r["pressure_drift_time_mean"] for r in selected])),
                "finite_fraction_min": float(np.min(
                    [r["finite_fraction"] for r in selected])),
                "divergent_windows_total": int(np.sum(
                    [r["divergent_windows"] for r in selected])),
                "inference_seconds_total": float(np.sum(
                    [r["inference_seconds"] for r in selected])),
            }
    return result


def plot_results(summary, growth_rows, memory_rows, output_dir):
    try:
        import matplotlib
    except ModuleNotFoundError:
        print(json.dumps({
            "event": "plot_skipped",
            "reason": "matplotlib is not installed in pt_env",
        }), flush=True)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"b1": "B1 Deep-FNN-H3", "b2": "B2 KDA off", "b3": "B3 KDA radial"}
    colors = {"b1": "#38598b", "b2": "#d1495b", "b3": "#edae49"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in labels:
        y = [summary[model][str(k)]["joint_field_time_mean"] for k in HORIZONS]
        ax.plot(HORIZONS, np.asarray(y) * 100, marker="o", label=labels[model],
                color=colors[model])
    ax.set(xlabel="Rollout horizon K", ylabel="Mean joint field error (%)",
           title="Validation rollout error growth")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "error_growth_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in labels:
        selected = [r for r in growth_rows
                    if r["model"] == model and r["horizon"] == 56]
        by_step = {}
        for row in selected:
            by_step.setdefault(row["step"], []).append(row["joint_field_mean"])
        x = sorted(by_step)
        y = [np.mean(by_step[k]) * 100 for k in x]
        ax.plot(x, y, label=labels[model], color=colors[model])
    ax.set(xlabel="Autonomous rollout step", ylabel="Mean joint field error (%)",
           title="K56 validation error trajectory")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "k56_error_trajectory.png", dpi=180)
    plt.close(fig)

    if memory_rows:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for model in ("b2", "b3"):
            selected = [r for r in memory_rows if r["model"] == model]
            x = np.arange(4)
            for ax, key, title in zip(
                axes, ("tau_mean", "beta_mean", "memory_effective_rank_mean"),
                ("Memory time scale", "Write rate", "Effective rank"),
            ):
                y = [np.mean([r[key] for r in selected if r["head"] == h])
                     for h in range(4)]
                ax.plot(x, y, marker="o", label=labels[model], color=colors[model])
                ax.set_title(title)
                ax.set_xlabel("Head")
                ax.grid(alpha=.25)
        axes[0].set_ylabel("Mean value")
        axes[-1].legend()
        fig.tight_layout()
        fig.savefig(output_dir / "memory_diagnostics.png", dpi=180)
        plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    K = load_module("kda_trainer_eval", args.trainer)
    B = load_module("frozen_hopf_eval", args.baseline_trainer)
    device = torch.device(args.device)
    base_args = SimpleNamespace(
        coefficient_view=args.coefficient_view, galerkin_path=args.galerkin_path,
        pressure_path=args.pressure_path, asset_manifest=args.asset_manifest,
        history_len=3, scale_floor_quantile=0.10, lambda_pressure_rollout=0.25,
    )
    audit = B.audit_assets(base_args)
    data = B.load_coefficients(base_args)
    if any(np.isclose(data["re"], value, atol=5e-6).any() for value in HELDOUT_RE):
        raise RuntimeError("held-out leakage in validation coefficient view")
    rom_np = B.load_train_rom(base_args)
    rom = {k: torch.as_tensor(v, device=device) for k, v in rom_np.items()}
    run_names = {
        "b1": "B1_Deep_FNN_H3_seed1248",
        "b2": "B2_KDA_PR_FNN_ROM_radial_off_seed1248",
        "b3": "B3_KDA_PR_FNN_ROM_radial_on_seed1248",
    }
    rows, growth_rows, memory_rows = [], [], []
    checkpoints = {}
    for variant, name in run_names.items():
        checkpoint_path = args.runs_root / name / "best_validation.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoints[variant] = {
            "path": str(checkpoint_path), "sha256": sha256(checkpoint_path),
            "optimizer_step": int(checkpoint["optimizer_step"]),
            "best_step": int(checkpoint["best_step"]),
        }
        model = reconstruct(K, variant, checkpoint, device)
        stats = {k: torch.as_tensor(v, device=device)
                 for k, v in checkpoint["norm_stats"].items()}
        for rv in VAL_RE:
            for horizon in HORIZONS:
                row, growth, memory = evaluate_one(
                    K, B, model, variant, data, rom, stats, device, rv, horizon,
                    args.windows_per_re)
                rows.append(row)
                growth_rows.extend(growth)
                memory_rows.extend(memory)
                print(json.dumps({
                    "model": variant, "Re": float(rv), "K": horizon,
                    "joint": row["joint_field_time_mean"],
                    "finite": row["finite_fraction"],
                }), flush=True)
        del model
        torch.cuda.empty_cache()
    summary = aggregate(rows)
    payload = {
        "schema_version": 1,
        "scope": "validation_only_no_heldout_loaded",
        "validation_re": VAL_RE.tolist(),
        "heldout_hard_disabled": list(HELDOUT_RE),
        "horizons": list(HORIZONS),
        "windows_per_re_cap": args.windows_per_re,
        "asset_audit": audit,
        "checkpoints": checkpoints,
        "aggregate": summary,
    }
    write_csv(rows, args.output_dir / "ROLLOUT_RESULTS.csv")
    write_csv(growth_rows, args.output_dir / "ERROR_GROWTH_CURVES.csv")
    write_csv(memory_rows, args.output_dir / "MEMORY_DIAGNOSTICS.csv")
    atomic_json(payload, args.output_dir / "ROLLOUT_SUMMARY.json")
    plot_results(summary, growth_rows, memory_rows, args.output_dir)
    print(json.dumps({"event": "evaluation_complete", "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
