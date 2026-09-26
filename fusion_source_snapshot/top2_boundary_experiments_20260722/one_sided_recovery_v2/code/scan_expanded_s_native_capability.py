"""Frozen S/H capability scan for externally projected S-native trajectories.

The Steady specialist is evaluated with its original feature/scaler/model
contract.  Because its frozen ROM tables are defined only on discrete Reynolds
nodes, each requested Re uses the nearest native ROM node, exactly as the
vendor compatibility wrapper does.  Hopf remains phase-free and receives the
same three-state physical history after analytic POD projection.
"""

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
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


RANK = 32
K56 = 56
K128 = 128
TAIL = 16
EXPECTED_SPLITS = {
    43.20: "train",
    43.40: "train",
    43.60: "train",
    43.30: "validation",
    43.70: "validation",
    43.50: "heldout",
    43.90: "heldout",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def rewritten_config(source: Path, destination: Path, old_root: str, new_root: Path) -> Path:
    value = json.loads(source.read_text(encoding="utf-8"))

    def rewrite(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: rewrite(val) for key, val in item.items()}
        if isinstance(item, list):
            return [rewrite(val) for val in item]
        if isinstance(item, str):
            return item.replace(old_root, str(new_root))
        return item

    destination.write_text(
        json.dumps(rewrite(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def build_steady_runtime(root: Path, output_dir: Path):
    specialist = root / "steady_specialist_v1"
    trainer = load_module("expanded_scan_steady_s4", specialist / "code/train_s4_steady.py")
    config = rewritten_config(
        specialist / "code/training_s2b_portable.json",
        output_dir / "runtime_config_migrated_paths.json",
        "/root/panxy/particalMOE",
        root,
    )
    cli = SimpleNamespace(
        trainer=str(specialist / "code/train_s2b_3090.py"),
        finalizer=str(specialist / "code/finalize_s2b_3090.py"),
        s3_trainer=str(specialist / "code/train_s3.py"),
        config=str(config),
        s2b_checkpoint=str(specialist / "checkpoint/frozen_s2b_validation.pt"),
        s3b_checkpoint=str(specialist / "checkpoint/frozen_s3b_contraction.pt"),
        bank=str(specialist / "data/perturbation_bank_train_validation.npz"),
        run_dir=str(output_dir / "steady_runtime"),
        learning_rate=3.0e-5,
        preflight_only=False,
    )
    instance = trainer.S4(cli)
    checkpoint_path = specialist / "checkpoint/frozen_s4_validation_step_1200.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    instance.exp.model.load_state_dict(checkpoint["model"], strict=True)
    instance.exp.model.eval()
    return instance.exp, checkpoint_path


def native_rom_for_re(exp: Any, re_value: float) -> tuple[dict, dict, float, str]:
    with np.load(exp.paths["velocity_rom"]) as velocity, np.load(exp.paths["pressure_rom"]) as pressure:
        nodes = np.asarray(velocity["Re_values_computed"], dtype=np.float64)
        labels = np.asarray(velocity["Re_labels_computed"]).astype(str)
        row = int(np.argmin(np.abs(nodes - re_value)))
        label = str(labels[row])
        device = exp.device
        gal = {
            0: {
                "c": torch.as_tensor(velocity["c_all"][row, :RANK], dtype=torch.float32, device=device),
                "A": torch.as_tensor(velocity["A_all"][row, :RANK, :RANK], dtype=torch.float32, device=device),
                "H": torch.as_tensor(velocity["H"][:RANK, :RANK, :RANK], dtype=torch.float32, device=device),
                "P": torch.as_tensor(velocity["P"][:RANK, :RANK], dtype=torch.float32, device=device),
            }
        }
        sur = {
            0: {
                "c_tilde": torch.as_tensor(pressure[f"{label}_c_tilde"][:RANK], dtype=torch.float32, device=device),
                "A_tilde": torch.as_tensor(pressure[f"{label}_A_tilde"][:RANK, :RANK], dtype=torch.float32, device=device),
                "H_tilde": torch.as_tensor(pressure["H_tilde"][:RANK, :RANK, :RANK], dtype=torch.float32, device=device),
            }
        }
    return gal, sur, float(nodes[row]), label


def steady_rollout(
    exp: Any,
    a_all: np.ndarray,
    b_all: np.ndarray,
    time_all: np.ndarray,
    start: int,
    re_value: float,
    gal: dict,
    sur: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = exp.device
    label = torch.zeros(1, dtype=torch.long, device=device)
    re_t = torch.tensor([re_value], dtype=torch.float32, device=device)
    history_ids = np.asarray([start, start - 1, start - 2], dtype=np.int64)
    a = torch.as_tensor(a_all[start : start + 1], dtype=torch.float32, device=device)
    b = torch.as_tensor(b_all[start : start + 1], dtype=torch.float32, device=device)
    ah = torch.as_tensor(a_all[history_ids][None], dtype=torch.float32, device=device)
    bh = torch.as_tensor(b_all[history_ids][None], dtype=torch.float32, device=device)
    rh = exp.vendor.galerkin_rhs_torch(
        ah.reshape(-1, RANK),
        bh.reshape(-1, RANK),
        torch.zeros(3, dtype=torch.long, device=device),
        gal,
    ).reshape_as(ah)
    phase = ((time_all - time_all[0]) / max(float(time_all[-1] - time_all[0]), 1e-12)).astype(np.float32)
    pred_a, pred_b, true_a, true_b = [], [], [], []
    with torch.inference_mode():
        for step in range(K56):
            current = start + step
            base_rhs = exp.vendor.galerkin_rhs_torch(a, b, label, gal)
            base_x = exp.vendor.make_features_torch(
                a,
                b,
                base_rhs,
                re_t,
                torch.tensor([phase[current]], dtype=torch.float32, device=device),
                exp.cfg["model"]["phase_harmonics"],
            )
            features = exp.vendor.make_history_features_from_states_torch(
                base_x, a, b, base_rhs, ah, bh, rh
            )
            raw_u, raw_p = exp._model_call((features - exp.xmt) / exp.xst, K56)
            dt = float(time_all[current + 1] - time_all[current])
            rhs = base_rhs + raw_u.float() * exp.rst + exp.rmt
            an = a + dt * rhs
            pressure_delta = raw_p.float() * exp.pst + exp.pmt
            bn = exp.vendor.pressure_surrogate_torch(an, label, sur) + pressure_delta
            pred_a.append(an[0].float().cpu().numpy())
            pred_b.append(bn[0].float().cpu().numpy())
            true_a.append(a_all[current + 1])
            true_b.append(b_all[current + 1])
            if ah.shape[1] > 1:
                ah = torch.cat([an[:, None], a[:, None], ah[:, 1:-1]], dim=1)
                bh = torch.cat([bn[:, None], b[:, None], bh[:, 1:-1]], dim=1)
                rh = torch.cat([rhs[:, None], base_rhs[:, None], rh[:, 1:-1]], dim=1)
            a, b = an, bn
    return tuple(np.asarray(x, dtype=np.float32) for x in (pred_a, pred_b, true_a, true_b))


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
    return np.maximum(
        mean_energy
        + 2 * np.einsum("...i,i->...", x, cross)
        + np.einsum("...i,ij,...j->...", x, gram, x),
        1e-30,
    )


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
    h = h.astype(np.float64)
    y = y.astype(np.float64)
    qss = np.einsum("ti,ij,tj->t", ds, gss, ds)
    qhh = (
        md2
        + np.einsum("ti,ij,tj->t", h, ghh, h)
        + np.einsum("ti,ij,tj->t", y, gss, y)
        + 2 * h @ phmd
        - 2 * y @ psmd
        - 2 * np.einsum("ti,ij,tj->t", y, csh, h)
    )
    qsh = np.einsum("ti,ti->t", ds, psmd[None] + h @ csh.T - y @ gss.T)
    yn = ms2 + 2 * y @ psms + np.einsum("ti,ij,tj->t", y, gss, y)
    return np.stack((qss, qhh, qsh, yn), axis=1)


def relative_error(q: np.ndarray, expert: int) -> np.ndarray:
    return np.sqrt(np.maximum(q[:, expert] / np.maximum(q[:, 3], 1e-30), 0.0))


def variation_metrics(pred: np.ndarray, truth: np.ndarray, pred_geom, truth_geom) -> dict[str, float]:
    tail = pred[-min(TAIL, len(pred)) :].astype(np.float64)
    tail_mean = tail.mean(axis=0, keepdims=True)
    gram = pred_geom[0]
    variance = float(np.mean(np.einsum("ti,ij,tj->t", tail - tail_mean, gram, tail - tail_mean)))
    terminal = pred[-1].astype(np.float64) - pred[-2].astype(np.float64)
    terminal_norm = float(terminal @ gram @ terminal)
    cumulative = pred[-1].astype(np.float64) - pred[0].astype(np.float64)
    cumulative_norm = float(cumulative @ gram @ cumulative)
    denominator = float(np.mean(state_norm(truth[-min(TAIL, len(truth)) :], truth_geom)))
    end_denominator = float(state_norm(truth[-1], truth_geom))
    return {
        "tail_oscillation_rms": math.sqrt(max(variance, 0.0) / max(denominator, 1e-30)),
        "fixed_point_drift": math.sqrt(max(terminal_norm, 0.0) / max(end_denominator, 1e-30)),
        "cumulative_drift": math.sqrt(max(cumulative_norm, 0.0) / max(end_denominator, 1e-30)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modal_path = args.dataset / "processed/steady_expanded_validation_modal_r32.npz"
    split_path = args.dataset / "split_manifest.json"
    acceptance_path = args.dataset / "FINAL_ACCEPTANCE.json"
    checkpoint_paths = {
        "Steady": args.root / "steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt",
        "Hopf": args.root / "Hopf/migrated_h4_expanded/runs/HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt",
    }
    hashes_before = {key: sha256(path) for key, path in checkpoint_paths.items()}
    atomic_json(
        args.output_dir / "STARTED.json",
        {
            "mode": "expanded_S_native_frozen_dual_capability_scan",
            "root": str(args.root),
            "dataset": str(args.dataset),
            "K56": K56,
            "K128": K128,
            "checkpoint_sha256_before": hashes_before,
            "test_used_for_tuning": False,
        },
    )
    preflight = load_module("expanded_scan_preflight", Path(__file__).with_name("one_sided_preflight_sh.py"))
    preflight.ROOT = args.root
    exp, steady_checkpoint = build_steady_runtime(args.root, args.output_dir)
    hopf = preflight.HopfRuntime(torch.device("cuda"))
    with np.load(modal_path, allow_pickle=True) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    actual = {round(float(re_value), 2): str(split) for re_value, split in zip(data["Re"], data["split"])}
    if any(actual.get(key) != value for key, value in EXPECTED_SPLITS.items()):
        raise RuntimeError(f"split contract mismatch: {actual}")
    sroot = args.root / "steady_specialist_v1/source_artifacts/steady"
    hroot = args.root / "Hopf/artifacts/hopf"
    per_window: list[dict[str, object]] = []
    rom_assignments: list[dict[str, object]] = []
    with (
        np.load(sroot / "velocity_pod_steady.npz") as su,
        np.load(sroot / "pressure_pod_steady.npz") as sp,
        np.load(hroot / "velocity_pod_hopf.npz") as hu,
        np.load(hroot / "pressure_pod_hopf.npz") as hp,
    ):
        if not (
            np.array_equal(su["points"], hu["points"])
            and np.array_equal(su["point_areas"], hu["point_areas"])
        ):
            raise RuntimeError("S/H mesh or area mismatch")
        if str(np.asarray(sp["pressure_gauge"]).item()) != str(np.asarray(hp["pressure_gauge"]).item()):
            raise RuntimeError("S/H pressure gauge mismatch")
        map_u, off_u = preflight.affine_map(su, hu, True)
        map_p, off_p = preflight.affine_map(sp, hp, False)
        su_g, sp_g = geometry(su, True), geometry(sp, False)
        hu_g, hp_g = geometry(hu, True), geometry(hp, False)
        for re_value in sorted(np.unique(data["Re"]).tolist()):
            ids = np.flatnonzero(np.isclose(data["Re"], re_value, rtol=0.0, atol=1e-8))
            if len(ids) != 64 or not np.array_equal(data["local_snapshot_index"][ids], np.arange(64)):
                raise RuntimeError(f"unexpected trajectory layout for Re={re_value}")
            split = str(data["split"][ids[0]])
            a_all = data["a_raw"][ids].astype(np.float32)
            b_all = data["b_raw"][ids].astype(np.float32)
            times = data["time"][ids].astype(np.float64)
            if not np.all(np.diff(times) > 0):
                raise RuntimeError(f"nonmonotonic time for Re={re_value}")
            gal, sur, rom_re, rom_label = native_rom_for_re(exp, float(re_value))
            rom_assignments.append(
                {
                    "Re": float(re_value),
                    "split": split,
                    "native_ROM_Re": rom_re,
                    "native_ROM_label": rom_label,
                    "absolute_Re_gap": abs(float(re_value) - rom_re),
                    "selection": "nearest frozen computed ROM node",
                }
            )
            for start in range(2, len(ids) - K56):
                sa, sb, ya, yb = steady_rollout(
                    exp, a_all, b_all, times, start, float(re_value), gal, sur
                )
                history_ids = np.asarray([start, start - 1, start - 2], dtype=np.int64)
                ha0 = map_u @ a_all[start] + off_u
                hb0 = map_p @ b_all[start] + off_p
                hah = a_all[history_ids] @ map_u.T + off_u
                hbh = b_all[history_ids] @ map_p.T + off_p
                dts = np.diff(times[start : start + K56 + 1]).astype(np.float32)
                hpa, hpb = hopf.rollout(ha0, hb0, hah, hbh, float(re_value), dts)
                ha = np.asarray(hpa, dtype=np.float32)
                hb = np.asarray(hpb, dtype=np.float32)
                if not all(np.isfinite(item).all() for item in (sa, sb, ha, hb)):
                    raise RuntimeError(f"non-finite rollout at Re={re_value}, start={start}")
                qu = quad_stats(sa, ha, ya, su, hu, True)
                qp = quad_stats(sb, hb, yb, sp, hp, False)
                esu, ehu = relative_error(qu, 0), relative_error(qu, 1)
                esp, ehp = relative_error(qp, 0), relative_error(qp, 1)
                es_joint, eh_joint = esu + esp, ehu + ehp
                s_vel = variation_metrics(sa, ya, su_g, su_g)
                h_vel = variation_metrics(ha, ya, hu_g, su_g)
                s_pre = variation_metrics(sb, yb, sp_g, sp_g)
                h_pre = variation_metrics(hb, yb, hp_g, sp_g)
                per_window.append(
                    {
                        "split": split,
                        "Re": float(re_value),
                        "native_ROM_Re": rom_re,
                        "native_ROM_label": rom_label,
                        "start_local_snapshot": start,
                        "start_time": float(times[start]),
                        "end_time_K56": float(times[start + K56]),
                        "K56_steps": K56,
                        "S_velocity_error_K56": float(esu[-1]),
                        "S_pressure_error_K56": float(esp[-1]),
                        "S_joint_error_K56": float(es_joint[-1]),
                        "H_velocity_error_K56": float(ehu[-1]),
                        "H_pressure_error_K56": float(ehp[-1]),
                        "H_joint_error_K56": float(eh_joint[-1]),
                        "S_minus_H_joint_K56": float(es_joint[-1] - eh_joint[-1]),
                        "S_velocity_fixed_point_drift": s_vel["fixed_point_drift"],
                        "H_velocity_fixed_point_drift": h_vel["fixed_point_drift"],
                        "S_pressure_fixed_point_drift": s_pre["fixed_point_drift"],
                        "H_pressure_fixed_point_drift": h_pre["fixed_point_drift"],
                        "S_velocity_tail_oscillation_rms": s_vel["tail_oscillation_rms"],
                        "H_velocity_tail_oscillation_rms": h_vel["tail_oscillation_rms"],
                        "S_pressure_tail_oscillation_rms": s_pre["tail_oscillation_rms"],
                        "H_pressure_tail_oscillation_rms": h_pre["tail_oscillation_rms"],
                        "S_pressure_cumulative_drift": s_pre["cumulative_drift"],
                        "H_pressure_cumulative_drift": h_pre["cumulative_drift"],
                        "K128_error_status": "UNAVAILABLE_64_snapshot_native_trajectory",
                    }
                )
    fieldnames = list(per_window[0])
    with (args.output_dir / "PER_WINDOW_S_H_CAPABILITY_SCAN.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_window)
    curves: list[dict[str, object]] = []
    for re_value in sorted(set(float(row["Re"]) for row in per_window)):
        items = [row for row in per_window if float(row["Re"]) == re_value]
        curve: dict[str, object] = {
            "Re": re_value,
            "split": items[0]["split"],
            "native_ROM_Re": items[0]["native_ROM_Re"],
            "K56_window_count": len(items),
            "K128_error_status": items[0]["K128_error_status"],
        }
        for key in fieldnames:
            if key.startswith(("S_", "H_")) and key.endswith(("K56", "drift", "rms")):
                values = [float(row[key]) for row in items]
                curve[key + "_mean"] = float(np.mean(values))
                curve[key + "_worst"] = float(np.max(values))
        differences = [float(row["S_minus_H_joint_K56"]) for row in items]
        curve["S_minus_H_joint_K56_mean"] = float(np.mean(differences))
        curve["S_minus_H_joint_K56_worst"] = float(np.max(differences))
        curves.append(curve)
    with (args.output_dir / "S_H_CAPABILITY_CURVES.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    crossings = []
    for left, right in zip(curves[:-1], curves[1:]):
        dl = float(left["S_minus_H_joint_K56_mean"])
        dr = float(right["S_minus_H_joint_K56_mean"])
        if dl == 0 or dr == 0 or dl * dr < 0:
            estimate = None if abs(dr - dl) < 1e-30 else float(left["Re"]) + (
                float(right["Re"]) - float(left["Re"])
            ) * (-dl) / (dr - dl)
            crossings.append(
                {
                    "left_Re": left["Re"],
                    "right_Re": right["Re"],
                    "left_difference": dl,
                    "right_difference": dr,
                    "linear_interpolated_Re_cap": estimate,
                    "interpretation_warning": "This is a descriptive crossing only; S uses nearest frozen ROM nodes.",
                }
            )
    hashes_after = {key: sha256(path) for key, path in checkpoint_paths.items()}
    if hashes_before != hashes_after or checkpoint_paths["Steady"] != steady_checkpoint:
        raise RuntimeError("frozen checkpoint integrity failure")
    summary = {
        "status": "EXPANDED_S_NATIVE_FROZEN_CAPABILITY_SCAN_COMPLETE",
        "root": str(args.root),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(modal_path),
        "split_manifest_sha256": sha256(split_path),
        "acceptance_sha256": sha256(acceptance_path),
        "targets": sorted(float(value) for value in np.unique(data["Re"])),
        "split_contract": {f"{key:.2f}": value for key, value in EXPECTED_SPLITS.items()},
        "test_policy": "Final-test trajectories were evaluated mechanically with the already fixed wrapper and metrics; they were not used to select ROM nodes, windows, thresholds, or definitions.",
        "K56": K56,
        "K128": {
            "status": "UNAVAILABLE",
            "reason": "Each supplied trajectory has only 64 snapshots; after the required three-state history there is no 128-step native future truth.",
        },
        "history_contract": "For every start >=2, both specialists receive the same physical states at [t,t-1,t-2]. The archive stores history oldest-to-current, so the scanner reconstructs the vendor-required current-to-oldest order instead of copying that tensor.",
        "phase_contract": {
            "Steady": "Original indexed fallback phase (t-t0)/(t_end-t0), evaluated only at database query timestamps.",
            "Hopf": "phase-free",
        },
        "steady_ROM_contract": "Nearest frozen computed Galerkin and pressure-surrogate node, matching the vendor compatibility contract; no interpolation or refit.",
        "rom_assignments": rom_assignments,
        "checkpoint_sha256_before": hashes_before,
        "checkpoint_sha256_after": hashes_after,
        "specialists_retrained": False,
        "specialist_checkpoints_modified": False,
        "curves": curves,
        "capacity_crossings": crossings,
    }
    atomic_json(args.output_dir / "S_H_CAPABILITY_SCAN.json", summary)
    lines = [
        "# Expanded frozen S-native S/H capability scan",
        "",
        "All seven supplied Re trajectories were evaluated with six valid K56 windows each. "
        "K128 is unavailable because every trajectory contains only 64 snapshots.",
        "",
        "| Re | split | S native ROM Re | windows | S joint mean | H joint mean | S-H |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in curves:
        lines.append(
            f"| {float(row['Re']):.2f} | {row['split']} | {float(row['native_ROM_Re']):.6f} | "
            f"{row['K56_window_count']} | {float(row['S_joint_error_K56_mean']):.6g} | "
            f"{float(row['H_joint_error_K56_mean']):.6g} | "
            f"{float(row['S_minus_H_joint_K56_mean']):.6g} |"
        )
    lines += [
        "",
        "Any linear crossing is descriptive only because the frozen S contract uses nearest ROM nodes.",
    ]
    for crossing in crossings:
        lines.append(
            f"- Descriptive Re_cap ≈ {float(crossing['linear_interpolated_Re_cap']):.6f} "
            f"between {float(crossing['left_Re']):.2f} and {float(crossing['right_Re']):.2f}."
        )
    (args.output_dir / "S_H_CAPABILITY_SCAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(
        args.output_dir / "FREEZE_MANIFEST.json",
        {
            "status": "FROZEN",
            "files": {
                path.name: sha256(path)
                for path in sorted(args.output_dir.iterdir())
                if path.is_file() and path.name != "FREEZE_MANIFEST.json"
            },
            "checkpoint_sha256": hashes_after,
        },
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "Re_count": len(curves),
                "window_count": len(per_window),
                "crossings": crossings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
