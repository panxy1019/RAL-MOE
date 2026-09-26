"""Read-only S-native capability scan for frozen Steady and Hopf specialists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path("/root/panxy/particalMOE")
RANK = 32
K56 = 56
K128 = 128
TAIL = 16
TARGET_RE = np.asarray([
    38.357249, 39.685479, 40.711525, 41.576575, 42.359071,
    43.093925, 43.797402, 44.478353, 45.142703, 45.795194, 46.440072,
], dtype=np.float64)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def geometry(pod: object, vector: bool):
    areas = np.asarray(pod["point_areas"], dtype=np.float64)
    weights = np.concatenate((areas, areas)) if vector else areas
    phi_key, mean_key = ("phi_uv", "mean_uv_regime") if vector else ("phi_p", "mean_p_regime")
    phi = np.asarray(pod[phi_key][:RANK], dtype=np.float64)
    mean = np.asarray(pod[mean_key], dtype=np.float64)
    gram = (phi * weights) @ phi.T
    cross = (phi * weights) @ mean
    mean_energy = float(np.dot(mean * weights, mean))
    return gram, cross, mean_energy


def state_norm(coeff: np.ndarray, geom) -> np.ndarray:
    gram, cross, mean_energy = geom
    x = np.asarray(coeff, dtype=np.float64)
    return np.maximum(mean_energy + 2 * np.einsum("...i,i->...", x, cross) + np.einsum("...i,ij,...j->...", x, gram, x), 1e-30)


def quad_stats(s: np.ndarray, h: np.ndarray, y: np.ndarray, source: object, target: object, vector: bool) -> np.ndarray:
    areas = np.asarray(source["point_areas"], dtype=np.float64)
    w = np.concatenate((areas, areas)) if vector else areas
    pk, mk = ("phi_uv", "mean_uv_regime") if vector else ("phi_p", "mean_p_regime")
    ps = np.asarray(source[pk][:RANK], dtype=np.float64)
    ph = np.asarray(target[pk][:RANK], dtype=np.float64)
    ms = np.asarray(source[mk], dtype=np.float64)
    mh = np.asarray(target[mk], dtype=np.float64)
    gss = (ps * w) @ ps.T
    ghh = (ph * w) @ ph.T
    csh = (ps * w) @ ph.T
    md = mh - ms
    psmd = (ps * w) @ md
    phmd = (ph * w) @ md
    md2 = float(np.dot(md * w, md))
    psms = (ps * w) @ ms
    ms2 = float(np.dot(ms * w, ms))
    ds = s.astype(np.float64) - y.astype(np.float64)
    h = h.astype(np.float64); y = y.astype(np.float64)
    qss = np.einsum("ti,ij,tj->t", ds, gss, ds)
    qhh = md2 + np.einsum("ti,ij,tj->t", h, ghh, h) + np.einsum("ti,ij,tj->t", y, gss, y) + 2 * h @ phmd - 2 * y @ psmd - 2 * np.einsum("ti,ij,tj->t", y, csh, h)
    qsh = np.einsum("ti,ti->t", ds, psmd[None] + h @ csh.T - y @ gss.T)
    yn = ms2 + 2 * y @ psms + np.einsum("ti,ij,tj->t", y, gss, y)
    return np.stack((qss, qhh, qsh, yn), axis=1)


def relative_error(q: np.ndarray, expert: int) -> np.ndarray:
    return np.sqrt(np.maximum(q[:, expert] / np.maximum(q[:, 3], 1e-30), 0.0))


def variation_metrics(pred: np.ndarray, truth: np.ndarray, pred_geom, truth_geom) -> dict[str, float]:
    tail = pred[-min(TAIL, len(pred)):].astype(np.float64)
    tail_mean = tail.mean(axis=0, keepdims=True)
    gram = pred_geom[0]
    variance = float(np.mean(np.einsum("ti,ij,tj->t", tail - tail_mean, gram, tail - tail_mean)))
    terminal = pred[-1].astype(np.float64) - pred[-2].astype(np.float64)
    terminal_norm = float(terminal @ gram @ terminal)
    cumulative = pred[-1].astype(np.float64) - pred[0].astype(np.float64)
    cumulative_norm = float(cumulative @ gram @ cumulative)
    denominator = float(np.mean(state_norm(truth[-min(TAIL, len(truth)):], truth_geom)))
    end_denominator = float(state_norm(truth[-1], truth_geom))
    return {
        "tail_oscillation_rms": math.sqrt(max(variance, 0.0) / max(denominator, 1e-30)),
        "fixed_point_drift": math.sqrt(max(terminal_norm, 0.0) / max(end_denominator, 1e-30)),
        "cumulative_drift": math.sqrt(max(cumulative_norm, 0.0) / max(end_denominator, 1e-30)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "STARTED.json", {
        "mode": "read_only_frozen_specialist_capability_scan",
        "targets": TARGET_RE.tolist(), "K56": K56, "K128": K128,
        "specialists_retrained": False, "new_CFD_data_created": False,
    })
    preflight = load_module("capability_preflight", Path(__file__).with_name("one_sided_preflight_sh.py"))
    device = torch.device("cuda")
    exp, steady_ckpt = preflight.build_steady_runtime(args.output_dir)
    hopf = preflight.HopfRuntime(device)
    sroot = ROOT / "steady_specialist_v1/source_artifacts/steady"
    hroot = ROOT / "Hopf/artifacts/hopf"
    per_window = []
    with np.load(sroot / "velocity_pod_steady.npz") as su, np.load(sroot / "pressure_pod_steady.npz") as sp, \
         np.load(hroot / "velocity_pod_hopf.npz") as hu, np.load(hroot / "pressure_pod_hopf.npz") as hp:
        if not (np.array_equal(su["points"], hu["points"]) and np.array_equal(su["point_areas"], hu["point_areas"])):
            raise RuntimeError("S/H mesh or area mismatch")
        if str(np.asarray(sp["pressure_gauge"]).item()) != str(np.asarray(hp["pressure_gauge"]).item()):
            raise RuntimeError("S/H pressure gauge mismatch")
        map_u, off_u = preflight.affine_map(su, hu, True)
        map_p, off_p = preflight.affine_map(sp, hp, False)
        su_g, sp_g, hu_g, hp_g = geometry(su, True), geometry(sp, False), geometry(hu, True), geometry(hp, False)
        windows56 = exp.build_windows(K56)
        all_labels = []
        for split, label_map in windows56.items():
            for label, starts in label_map.items():
                re_value = float(np.mean(exp.a["re"][exp.a["label_id"] == label]))
                nearest = TARGET_RE[np.argmin(np.abs(TARGET_RE - re_value))]
                if abs(nearest - re_value) <= 5e-4:
                    all_labels.append((split, int(label), re_value, np.asarray(starts, dtype=np.int64)))
        if len(all_labels) != len(TARGET_RE):
            raise RuntimeError(f"expected {len(TARGET_RE)} target Re, found {len(all_labels)}")
        for split, label, re_value, starts in sorted(all_labels, key=lambda item: item[2]):
            for start in starts.tolist():
                start_t = torch.tensor([start], dtype=torch.long, device=device)
                indices = exp.indices(start_t, K56)[0]
                times = np.asarray(exp.a["time"])[indices.cpu().numpy()]
                dts = np.diff(times).astype(np.float32)
                if not (len(dts) == K56 and np.all(dts > 0)):
                    raise RuntimeError("invalid native timestamp sequence")
                current = indices[0:1]
                hist_ids = exp.tensor("hist_idx", current, torch.long)
                a0 = exp.tensor("a", current)[0].float().cpu().numpy(); b0 = exp.tensor("b", current)[0].float().cpu().numpy()
                ah = exp.tensor("a", hist_ids)[0].float().cpu().numpy(); bh = exp.tensor("b", hist_ids)[0].float().cpu().numpy()
                ha0, hb0 = map_u @ a0 + off_u, map_p @ b0 + off_p
                hah, hbh = ah @ map_u.T + off_u, bh @ map_p.T + off_p
                with torch.inference_mode():
                    steady = exp.rollout(start_t, K56)
                hpa, hpb = hopf.rollout(ha0, hb0, hah, hbh, re_value, dts)
                sa = steady["pa"][:, 0].float().cpu().numpy(); sb = steady["pb"][:, 0].float().cpu().numpy()
                ya = steady["ta"][:, 0].float().cpu().numpy(); yb = steady["tb"][:, 0].float().cpu().numpy()
                ha, hb = np.asarray(hpa, dtype=np.float32), np.asarray(hpb, dtype=np.float32)
                if not (np.isfinite(sa).all() and np.isfinite(sb).all() and np.isfinite(ha).all() and np.isfinite(hb).all()):
                    raise RuntimeError(f"non-finite rollout at Re={re_value}, start={start}")
                qu, qp = quad_stats(sa, ha, ya, su, hu, True), quad_stats(sb, hb, yb, sp, hp, False)
                esu, ehu, esp, ehp = relative_error(qu, 0), relative_error(qu, 1), relative_error(qp, 0), relative_error(qp, 1)
                es_joint, eh_joint = esu + esp, ehu + ehp
                s_vel, h_vel = variation_metrics(sa, ya, su_g, su_g), variation_metrics(ha, ya, hu_g, su_g)
                s_pre, h_pre = variation_metrics(sb, yb, sp_g, sp_g), variation_metrics(hb, yb, hp_g, sp_g)
                per_window.append({
                    "split": split, "Re": re_value, "label_id": label, "start_snapshot": start,
                    "start_time": float(times[0]), "end_time_K56": float(times[-1]), "K56_steps": K56,
                    "S_velocity_error_K56": float(esu[-1]), "S_pressure_error_K56": float(esp[-1]), "S_joint_error_K56": float(es_joint[-1]),
                    "H_velocity_error_K56": float(ehu[-1]), "H_pressure_error_K56": float(ehp[-1]), "H_joint_error_K56": float(eh_joint[-1]),
                    "S_minus_H_joint_K56": float(es_joint[-1] - eh_joint[-1]),
                    "S_velocity_fixed_point_drift": s_vel["fixed_point_drift"], "H_velocity_fixed_point_drift": h_vel["fixed_point_drift"],
                    "S_pressure_fixed_point_drift": s_pre["fixed_point_drift"], "H_pressure_fixed_point_drift": h_pre["fixed_point_drift"],
                    "S_velocity_tail_oscillation_rms": s_vel["tail_oscillation_rms"], "H_velocity_tail_oscillation_rms": h_vel["tail_oscillation_rms"],
                    "S_pressure_tail_oscillation_rms": s_pre["tail_oscillation_rms"], "H_pressure_tail_oscillation_rms": h_pre["tail_oscillation_rms"],
                    "S_pressure_cumulative_drift": s_pre["cumulative_drift"], "H_pressure_cumulative_drift": h_pre["cumulative_drift"],
                    "K128_error_status": "UNAVAILABLE_native_truth_trajectory_shorter_than_128",
                })
    fieldnames = list(per_window[0])
    with (args.output_dir / "PER_WINDOW_S_H_CAPABILITY_SCAN.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(per_window)
    curves = []
    for re_value in sorted(set(row["Re"] for row in per_window)):
        items = [row for row in per_window if row["Re"] == re_value]
        curve = {"Re": re_value, "split": items[0]["split"], "K56_window_count": len(items), "K128_error_status": items[0]["K128_error_status"]}
        for key in fieldnames:
            if key.startswith(("S_", "H_")) and key.endswith(("K56", "drift", "rms")):
                values = [float(row[key]) for row in items]
                curve[key + "_mean"] = float(np.mean(values)); curve[key + "_worst"] = float(np.max(values))
        curve["S_minus_H_joint_K56_mean"] = float(np.mean([row["S_minus_H_joint_K56"] for row in items]))
        curve["S_minus_H_joint_K56_worst"] = float(np.max([row["S_minus_H_joint_K56"] for row in items]))
        curves.append(curve)
    with (args.output_dir / "S_H_CAPABILITY_CURVES.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    crossings = []
    for left, right in zip(curves[:-1], curves[1:]):
        dl, dr = left["S_minus_H_joint_K56_mean"], right["S_minus_H_joint_K56_mean"]
        if dl == 0 or dr == 0 or dl * dr < 0:
            if abs(dr - dl) < 1e-30:
                estimate = None
            else:
                estimate = left["Re"] + (right["Re"] - left["Re"]) * (-dl) / (dr - dl)
            crossings.append({"left_Re": left["Re"], "right_Re": right["Re"], "left_difference": dl, "right_difference": dr, "linear_interpolated_Re_cap": estimate})
    summary = {
        "status": "S_NATIVE_FROZEN_CAPABILITY_SCAN_COMPLETE",
        "protocol": "Each S-native state/history is run by frozen S under its original feature builder and frozen phase-free H after analytic physical-history projection; no cross-feedback or fusion.",
        "targets": TARGET_RE.tolist(), "K56": K56,
        "K128": {"error_status": "UNAVAILABLE", "reason": "All requested native S trajectories have 63-64 snapshots; K128 cannot be paired with a true future field after three-step history."},
        "phase_audit": {"S": "The existing S snapshot index has no explicit phase or period column. Its original vendor feature builder derives phase as (t-t0)/(t_end-t0) from the complete native indexed trajectory; this scanner preserves that frozen evaluation contract and does not synthesize a new phase.", "H": "phase-free"},
        "checkpoint_sha256": {"Steady": sha256(steady_ckpt), "Hopf": hopf.checkpoint_sha256},
        "specialists_retrained": False, "specialist_checkpoints_modified": False, "new_CFD_data_created": False,
        "curves": curves, "capacity_crossings": crossings,
        "metric_definitions": {
            "joint_error": "area-weighted velocity relative L2 plus pressure relative L2 at K56",
            "fixed_point_drift": "terminal one-step physical-field change divided by terminal true-field norm",
            "tail_oscillation_rms": "RMS physical deviation over the final 16 forecast steps from that prediction tail mean, normalized by true tail field energy",
            "pressure_cumulative_drift": "physical pressure change from first to K56 predicted step, normalized by terminal true pressure-field norm",
        },
    }
    atomic_json(args.output_dir / "S_H_CAPABILITY_SCAN.json", summary)
    lines = ["# Frozen S-native S/H capability scan", "", "K56 uses all valid native windows for each requested Re. K128 field errors are unavailable because no requested S-native trajectory supplies 128 future true snapshots after the required history.", "", "| Re | split | K56 windows | S joint mean | H joint mean | S-H |", "|---:|---|---:|---:|---:|---:|"]
    for row in curves:
        lines.append(f"| {row['Re']:.6f} | {row['split']} | {row['K56_window_count']} | {row['S_joint_error_K56_mean']:.6g} | {row['H_joint_error_K56_mean']:.6g} | {row['S_minus_H_joint_K56_mean']:.6g} |")
    if crossings:
        lines += ["", "Estimated capability crossing(s):", ""]
        for crossing in crossings:
            lines.append(f"- Re_cap ≈ {crossing['linear_interpolated_Re_cap']:.6f} between {crossing['left_Re']:.6f} and {crossing['right_Re']:.6f}.")
    else:
        lines += ["", "No sign change of S-H K56 mean error was observed across the requested Re grid."]
    (args.output_dir / "S_H_CAPABILITY_SCAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "Re_count": len(curves), "window_count": len(per_window), "crossings": crossings}, indent=2))


if __name__ == "__main__":
    main()
