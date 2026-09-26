"""Execute G_SH_native using Steady-native train/validation trajectories only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


ROOT = Path("/root/panxy/particalMOE")
RANK = 32
HORIZON = 16


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def affine_map(source: Any, target: Any, vector: bool) -> tuple[np.ndarray, np.ndarray]:
    areas = np.asarray(source["point_areas"], dtype=np.float64)
    weights = np.concatenate((areas, areas)) if vector else areas
    phi_key, mean_key = ("phi_uv", "mean_uv_regime") if vector else ("phi_p", "mean_p_regime")
    phi_source = np.asarray(source[phi_key][:RANK], dtype=np.float64)
    phi_target = np.asarray(target[phi_key][:RANK], dtype=np.float64)
    mean_source = np.asarray(source[mean_key], dtype=np.float64)
    mean_target = np.asarray(target[mean_key], dtype=np.float64)
    linear = (phi_target * weights[None, :]) @ phi_source.T
    offset = (phi_target * weights[None, :]) @ (mean_source - mean_target)
    return linear.astype(np.float32), offset.astype(np.float32)


def decode(coeff: np.ndarray, pod: Any, vector: bool) -> np.ndarray:
    phi_key, mean_key = ("phi_uv", "mean_uv_regime") if vector else ("phi_p", "mean_p_regime")
    return np.asarray(pod[mean_key], dtype=np.float64) + np.asarray(coeff, dtype=np.float64) @ np.asarray(pod[phi_key][:RANK], dtype=np.float64)


def field_relative(pred: np.ndarray, true: np.ndarray, areas: np.ndarray, vector: bool) -> float:
    weights = np.concatenate((areas, areas)) if vector else areas
    delta = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return math.sqrt(float(np.dot(delta * weights, delta)) / max(float(np.dot(true * weights, true)), 1.0e-30))


class HopfRuntime:
    def __init__(self, device: torch.device) -> None:
        root = ROOT / "Hopf/migrated_h4_expanded"
        self.baseline = load_module("top2_sh_hopf_baseline", root / "code/train_hopf_moe_expanded.py")
        h4 = load_module("top2_sh_hopf_h4", root / "code/train_h4_expanded.py")
        checkpoint_path = root / "runs/HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = SimpleNamespace(**checkpoint["args"])
        base_args = SimpleNamespace(**{
            **vars(args), "history_len": 3, "hidden_dim": 256, "expert_hidden": 1024,
            "num_blocks": 3, "experts": 6, "top_k": 2, "expert_blocks": 4,
            "quadratic_rank": 4, "dropout": 0.04, "temperature": 0.8,
            "adaptive_gate_initial_logit": 6.0, "lambda_scale_amplitude": 1.0,
            "lambda_scale_growth": 0.5, "lambda_scale_sign": 0.1, "scale_floor_quantile": 0.1,
        })
        rom_np = self.baseline.load_train_rom(SimpleNamespace(
            galerkin_path=root / "assets_r32/expanded_h4_trainonly_galerkin_r32.npz",
            pressure_path=root / "assets_r32/expanded_h4_trainonly_pressure_r32.npz",
        ))
        self.rom = {key: torch.as_tensor(value, device=device) for key, value in rom_np.items()}
        self.stats = {key: torch.as_tensor(value, device=device) for key, value in checkpoint["norm_stats"].items()}
        self.model = self.baseline.build_model(493, base_args, self.stats, device)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256(checkpoint_path)

    def rollout(self, a: np.ndarray, b: np.ndarray, ah: np.ndarray, bh: np.ndarray,
                re_value: float, dts: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        a_t = torch.as_tensor(a[None], device=self.device)
        b_t = torch.as_tensor(b[None], device=self.device)
        ah_t = torch.as_tensor(ah[None], device=self.device)
        bh_t = torch.as_tensor(bh[None], device=self.device)
        re_t = torch.tensor([re_value], dtype=torch.float32, device=self.device)
        rh_t = self.baseline.galerkin(
            ah_t.reshape(-1, RANK), bh_t.reshape(-1, RANK),
            re_t[:, None].expand(-1, ah_t.shape[1]).reshape(-1), self.rom,
        ).reshape_as(ah_t)
        pred_a, pred_b = [], []
        with torch.inference_mode():
            for dt_value in dts.tolist():
                dt_t = torch.tensor([[dt_value]], dtype=torch.float32, device=self.device)
                an, bn, galerkin_current, _rhs, _gates, _stack = self.baseline.autonomous_step(
                    self.model, a_t, b_t, re_t, dt_t, ah_t, bh_t, rh_t, self.rom, self.stats,
                )
                pred_a.append(an[0].float().cpu().numpy()); pred_b.append(bn[0].float().cpu().numpy())
                ah_t = torch.cat((an[:, None], ah_t[:, :-1]), dim=1)
                bh_t = torch.cat((bn[:, None], bh_t[:, :-1]), dim=1)
                rh_t = torch.cat((galerkin_current[:, None], rh_t[:, :-1]), dim=1)
                a_t, b_t = an, bn
        return pred_a, pred_b


def build_steady_runtime(output_dir: Path):
    specialist = ROOT / "steady_specialist_v1"
    trainer = load_module("top2_sh_steady_s4", specialist / "code/train_s4_steady.py")
    runtime = output_dir / "steady_runtime"
    sealed = os.environ.get("TOP2_STEADY_TRAINVAL_RUNTIME")
    sealed_root = Path(sealed) if sealed else None
    args = SimpleNamespace(
        trainer=str(sealed_root / "code/train_s2b_3090_trainval_only.py") if sealed_root else str(specialist / "code/train_s2b_3090.py"),
        finalizer=str(specialist / "code/finalize_s2b_3090.py"),
        s3_trainer=str(specialist / "code/train_s3.py"),
        config=str(sealed_root / "training_s2b_trainval_only.json") if sealed_root else str(specialist / "code/training_s2b_portable.json"),
        s2b_checkpoint=str(specialist / "checkpoint/frozen_s2b_validation.pt"),
        s3b_checkpoint=str(specialist / "checkpoint/frozen_s3b_contraction.pt"),
        bank=str(sealed_root / "perturbation_bank_train_validation_reindexed.npz") if sealed_root else str(specialist / "data/perturbation_bank_train_validation.npz"),
        run_dir=str(runtime), learning_rate=3.0e-5, preflight_only=False,
    )
    instance = trainer.S4(args)
    checkpoint_path = specialist / "checkpoint/frozen_s4_validation_step_1200.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    instance.exp.model.load_state_dict(checkpoint["model"], strict=True)
    instance.exp.model.eval()
    return instance.exp, checkpoint_path


def selected_starts(exp: Any, horizon: int) -> list[tuple[str, int, float]]:
    windows = exp.build_windows(horizon)
    selected = []
    for split, count in (("train", 3), ("validation", 1)):
        candidates = []
        for label, ids in windows[split].items():
            re_value = float(np.mean(exp.a["re"][exp.a["label_id"] == label]))
            candidates.append((re_value, ids))
        for re_value, ids in sorted(candidates, key=lambda item: item[0], reverse=True)[:count]:
            selected.append((split, int(ids[len(ids)//2]), re_value))
    return selected


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("TOP2_STEADY_TRAINVAL_RUNTIME"):
        raise RuntimeError("G_SH_native formal run requires TOP2_STEADY_TRAINVAL_RUNTIME to preserve the test seal")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    atomic_json(args.output_dir / "STARTED.json", {
        "gate": "G_SH_native", "source_split_access": ["train", "validation"],
        "test_physical_modal_or_metric_data_loaded": False, "horizon": HORIZON,
    })
    exp, steady_checkpoint = build_steady_runtime(args.output_dir)
    hopf = HopfRuntime(device)
    steady_root = ROOT / "steady_specialist_v1/source_artifacts/steady"
    hopf_root = ROOT / "Hopf/artifacts/hopf"
    with np.load(steady_root / "velocity_pod_steady.npz") as su, \
         np.load(steady_root / "pressure_pod_steady.npz") as sp, \
         np.load(hopf_root / "velocity_pod_hopf.npz") as hu, \
         np.load(hopf_root / "pressure_pod_hopf.npz") as hp:
        if not (np.array_equal(su["points"], hu["points"]) and np.array_equal(su["point_areas"], hu["point_areas"])):
            raise RuntimeError("S/H common mesh or area mismatch")
        gauge_s = str(np.asarray(sp["pressure_gauge"]).item()); gauge_h = str(np.asarray(hp["pressure_gauge"]).item())
        if gauge_s != gauge_h:
            raise RuntimeError("S/H pressure gauge mismatch")
        map_u, offset_u = affine_map(su, hu, True); map_p, offset_p = affine_map(sp, hp, False)
        areas = np.asarray(su["point_areas"], dtype=np.float64)
        rows = []
        for split, start, re_value in selected_starts(exp, HORIZON):
            start_t = torch.tensor([start], dtype=torch.long, device=device)
            indices = exp.indices(start_t, HORIZON)[0]
            source_times = np.asarray(exp.a["time"])[indices.cpu().numpy()]
            dts = np.diff(source_times).astype(np.float32)
            if not (np.all(dts > 0) and len(dts) == HORIZON):
                raise RuntimeError("invalid Steady native query timestamps")
            current = indices[0:1]
            hist_ids = exp.tensor("hist_idx", current, torch.long)
            a0 = exp.tensor("a", current)[0].float().cpu().numpy(); b0 = exp.tensor("b", current)[0].float().cpu().numpy()
            ah = exp.tensor("a", hist_ids)[0].float().cpu().numpy(); bh = exp.tensor("b", hist_ids)[0].float().cpu().numpy()
            ha0 = map_u @ a0 + offset_u; hb0 = map_p @ b0 + offset_p
            hah = ah @ map_u.T + offset_u; hbh = bh @ map_p.T + offset_p
            with torch.inference_mode():
                source = exp.rollout(start_t, HORIZON)
            hpa, hpb = hopf.rollout(ha0, hb0, hah, hbh, re_value, dts)
            source_pa = source["pa"][:, 0].float().cpu().numpy(); source_pb = source["pb"][:, 0].float().cpu().numpy()
            true_a = source["ta"][:, 0].float().cpu().numpy(); true_b = source["tb"][:, 0].float().cpu().numpy()
            finite_s = bool(np.isfinite(source_pa).all() and np.isfinite(source_pb).all())
            finite_h = bool(np.isfinite(hpa).all() and np.isfinite(hpb).all())
            true_u = decode(true_a[-1], su, True); true_p = decode(true_b[-1], sp, False)
            s_u = decode(source_pa[-1], su, True); s_p = decode(source_pb[-1], sp, False)
            h_u = decode(hpa[-1], hu, True); h_p = decode(hpb[-1], hp, False)
            initial_u_projection = field_relative(decode(ha0, hu, True), decode(a0, su, True), areas, True)
            initial_p_projection = field_relative(decode(hb0, hp, False), decode(b0, sp, False), areas, False)
            rows.append({
                "split": split, "Re": re_value, "start_snapshot": start, "horizon": HORIZON,
                "start_time": float(source_times[0]), "end_time": float(source_times[-1]),
                "timestamps_strictly_shared": True, "dt_min": float(dts.min()), "dt_max": float(dts.max()),
                "same_initial_three_step_history": True,
                "initial_velocity_projection_relative_l2": initial_u_projection,
                "initial_pressure_projection_relative_l2": initial_p_projection,
                "Steady_finite": finite_s, "Hopf_finite": finite_h,
                "Steady_divergent": int(not finite_s), "Hopf_divergent": int(not finite_h),
                "Steady_final_velocity_field_error": field_relative(s_u, true_u, areas, True),
                "Steady_final_pressure_field_error": field_relative(s_p, true_p, areas, False),
                "Hopf_final_velocity_field_error": field_relative(h_u, true_u, areas, True),
                "Hopf_final_pressure_field_error": field_relative(h_p, true_p, areas, False),
            })
    passed = bool(rows) and all(row["Steady_finite"] and row["Hopf_finite"] and row["timestamps_strictly_shared"] for row in rows)
    result = {
        "gate": "G_SH_native", "decision": "PASS" if passed else "FAIL_CLOSED",
        "protocol": "Steady database-native phase/time plus phase-free Hopf on mapped common physical history",
        "source_database": "Steady train/validation only", "test_physical_modal_or_metric_data_loaded": False,
        "checkpoint_sha256": {"Steady": sha256(steady_checkpoint), "Hopf": hopf.checkpoint_sha256},
        "common_grid_area_pressure_gauge": True, "history_length": 3, "horizon": HORIZON,
        "rows": rows,
        "finite_fraction": float(np.mean([row["Steady_finite"] and row["Hopf_finite"] for row in rows])),
        "divergent_windows": int(sum(row["Steady_divergent"] + row["Hopf_divergent"] for row in rows)),
        "authorized_next_action": "construct S-H train/validation native cache" if passed else "do not construct S-H cache or train",
    }
    atomic_json(args.output_dir / "G_SH_native.json", result)
    atomic_json(args.output_dir / ("PASS.json" if passed else "FAIL_CLOSED.json"), {"gate": "G_SH_native", "decision": result["decision"]})
    print(json.dumps({"decision": result["decision"], "rows": len(rows), "finite_fraction": result["finite_fraction"], "divergent_windows": result["divergent_windows"]}, indent=2))


if __name__ == "__main__":
    main()
