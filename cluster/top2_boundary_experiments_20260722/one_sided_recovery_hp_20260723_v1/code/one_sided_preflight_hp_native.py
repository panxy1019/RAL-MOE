#!/usr/bin/env python3
"""Execute the sealed-development G_HP_native preflight.

Periodic owns the database clock and indexed phase.  Hopf is phase-free and
receives the same three physical-history states projected into its native chart.
No heldout modal bundle is accepted by this program.
"""

from __future__ import annotations

import argparse
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


RANK = 32
HISTORY = 3


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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_bundle(paths: list[Path], module: Any, tensors: Any) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {}
    for path in paths:
        if "heldout" in path.name.lower() or "test" in path.name.lower():
            raise RuntimeError(f"sealed-development preflight rejects {path}")
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                chunks.setdefault(key, []).append(np.asarray(archive[key]))
    arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    order = np.lexsort((arrays["time"], arrays["Re"]))
    arrays = {key: value[order] for key, value in arrays.items()}

    labels = np.asarray(sorted(set(arrays["Re_label"].astype(str).tolist())))
    label_to_id = {label: i for i, label in enumerate(labels.tolist())}
    arrays["labels"] = labels
    arrays["label_id"] = np.asarray(
        [label_to_id[label] for label in arrays["Re_label"].astype(str)], dtype=np.int64
    )
    arrays["re"] = arrays.pop("Re").astype(np.float32)
    arrays["a"] = arrays["a"].astype(np.float32)
    arrays["b"] = arrays["b"].astype(np.float32)
    arrays["time"] = arrays["time"].astype(np.float32)
    arrays["phase"] = arrays["phase"].astype(np.float32)
    arrays["next_idx"] = module.next_index_by_sequence(arrays["re"], arrays["time"])
    arrays["prev_idx"] = module.prev_index_by_sequence(arrays["re"], arrays["time"])
    arrays["hist_idx"] = module.history_index_matrix(
        np.arange(arrays["a"].shape[0], dtype=np.int64), arrays["prev_idx"], HISTORY
    )
    arrays["rhs_g"] = module.galerkin_rhs_by_label(
        tensors,
        arrays["a"],
        arrays["b"],
        arrays["label_id"],
        arrays["labels"],
        RANK,
        RANK,
    )
    return arrays


def instantiate_periodic(module: Any, checkpoint: dict[str, Any], device: torch.device):
    args = SimpleNamespace(**checkpoint["args"])
    scalers = {
        key: module.Standardizer(
            mean=np.asarray(value["mean"], dtype=np.float32),
            scale=np.asarray(value["scale"], dtype=np.float32),
        )
        for key, value in checkpoint["scalers"].items()
    }
    in_dim = int(scalers["x"].mean.size)
    model = module.OperatorSpaceMoEROM(
        in_dim=in_dim,
        out_dim=args.r_u,
        pressure_dim=args.r_p,
        hidden_dim=args.hidden_dim,
        expert_hidden=args.expert_hidden,
        num_blocks=args.num_blocks,
        num_experts=args.num_experts,
        num_operator_spaces=args.num_shared_experts,
        num_regime_groups=args.num_regime_groups,
        experts_per_group=args.experts_per_group,
        top_k=args.top_k,
        group_top_k=args.group_top_k,
        dropout=args.dropout,
        temperature=args.temperature,
        gate_floor=args.gate_floor,
        group_temperature=args.group_temperature,
        group_gate_floor=args.group_gate_floor,
        shared_scale=args.shared_scale,
        routed_scale=args.routed_scale,
        expert_blocks=args.expert_blocks,
        quadratic_rank=args.quadratic_rank,
        quadratic_scale=args.quadratic_scale,
        phase_harmonics=args.phase_harmonics,
        closure_mode=args.closure_mode,
        pressure_base_mode=args.pressure_base_mode,
        film_base_hidden=args.film_base_hidden,
        film_base_scale=args.film_base_scale,
        attractor_conditioned=args.attractor_conditioned,
        attractor_adapter_dim=args.attractor_adapter_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if in_dim != 501 or args.history_len != HISTORY or args.phase_harmonics != 4:
        raise RuntimeError(
            f"Periodic feature contract mismatch: in={in_dim}, history={args.history_len}, "
            f"harmonics={args.phase_harmonics}"
        )
    return model, args, scalers


class HopfRuntime:
    def __init__(self, root: Path, device: torch.device) -> None:
        specialist = root / "Hopf/migrated_h4_expanded"
        self.baseline = load_module(
            "hp_native_hopf_baseline", specialist / "code/train_hopf_moe_expanded.py"
        )
        load_module("hp_native_hopf_h4", specialist / "code/train_h4_expanded.py")
        checkpoint_path = (
            specialist / "runs/HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = SimpleNamespace(**checkpoint["args"])
        base_args = SimpleNamespace(
            **{
                **vars(args),
                "history_len": 3,
                "hidden_dim": 256,
                "expert_hidden": 1024,
                "num_blocks": 3,
                "experts": 6,
                "top_k": 2,
                "expert_blocks": 4,
                "quadratic_rank": 4,
                "dropout": 0.04,
                "temperature": 0.8,
                "adaptive_gate_initial_logit": 6.0,
                "lambda_scale_amplitude": 1.0,
                "lambda_scale_growth": 0.5,
                "lambda_scale_sign": 0.1,
                "scale_floor_quantile": 0.1,
            }
        )
        rom_np = self.baseline.load_train_rom(
            SimpleNamespace(
                galerkin_path=specialist
                / "assets_r32/expanded_h4_trainonly_galerkin_r32.npz",
                pressure_path=specialist
                / "assets_r32/expanded_h4_trainonly_pressure_r32.npz",
            )
        )
        self.rom = {
            key: torch.as_tensor(value, device=device) for key, value in rom_np.items()
        }
        self.stats = {
            key: torch.as_tensor(value, device=device)
            for key, value in checkpoint["norm_stats"].items()
        }
        self.model = self.baseline.build_model(493, base_args, self.stats, device)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()
        self.device = device
        self.checkpoint_path = checkpoint_path

    def rollout(
        self,
        a: np.ndarray,
        b: np.ndarray,
        a_hist: np.ndarray,
        b_hist: np.ndarray,
        re_value: float,
        dts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        a_t = torch.as_tensor(a[None], device=self.device)
        b_t = torch.as_tensor(b[None], device=self.device)
        ah_t = torch.as_tensor(a_hist[None], device=self.device)
        bh_t = torch.as_tensor(b_hist[None], device=self.device)
        re_t = torch.tensor([re_value], dtype=torch.float32, device=self.device)
        rh_t = self.baseline.galerkin(
            ah_t.reshape(-1, RANK),
            bh_t.reshape(-1, RANK),
            re_t[:, None].expand(-1, ah_t.shape[1]).reshape(-1),
            self.rom,
        ).reshape_as(ah_t)
        predicted_a, predicted_b = [], []
        with torch.inference_mode():
            for dt_value in dts.tolist():
                dt_t = torch.tensor([[dt_value]], dtype=torch.float32, device=self.device)
                an, bn, galerkin_current, _rhs, _gates, _stack = (
                    self.baseline.autonomous_step(
                        self.model,
                        a_t,
                        b_t,
                        re_t,
                        dt_t,
                        ah_t,
                        bh_t,
                        rh_t,
                        self.rom,
                        self.stats,
                    )
                )
                predicted_a.append(an[0].float().cpu().numpy())
                predicted_b.append(bn[0].float().cpu().numpy())
                ah_t = torch.cat((an[:, None], ah_t[:, :-1]), dim=1)
                bh_t = torch.cat((bn[:, None], bh_t[:, :-1]), dim=1)
                rh_t = torch.cat((galerkin_current[:, None], rh_t[:, :-1]), dim=1)
                a_t, b_t = an, bn
        return np.asarray(predicted_a), np.asarray(predicted_b)


def affine_map(
    source_phi: np.ndarray,
    source_mean: np.ndarray,
    target_phi: np.ndarray,
    target_mean: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    linear = (target_phi * weights[None, :]) @ source_phi.T
    offset = (target_phi * weights[None, :]) @ (source_mean - target_mean)
    return linear.astype(np.float32), offset.astype(np.float32)


def decode(coeff: np.ndarray, phi: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return np.asarray(mean, dtype=np.float64) + np.asarray(coeff, dtype=np.float64) @ np.asarray(
        phi, dtype=np.float64
    )


def field_relative(
    predicted: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> float:
    delta = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return math.sqrt(
        float(np.sum(delta * delta * weights))
        / max(float(np.sum(np.asarray(truth, dtype=np.float64) ** 2 * weights)), 1.0e-30)
    )


def safe_field_relative(
    predicted: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> float | None:
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        return None
    value = field_relative(predicted, truth, weights)
    return value if math.isfinite(value) else None


def selected_starts(
    arrays: dict[str, np.ndarray], horizon: int, coverage: str
) -> list[tuple[str, int]]:
    candidates: list[tuple[float, str, int]] = []
    for label in arrays["labels"].tolist():
        ids = np.where(arrays["Re_label"].astype(str) == label)[0]
        ids = ids[np.argsort(arrays["time"][ids])]
        valid = [
            int(i)
            for i in ids[HISTORY - 1 : -horizon]
            if np.all(arrays["hist_idx"][i] >= 0)
        ]
        if not valid:
            continue
        split = "validation" if label in {
            "Re_66p970112",
            "Re_91p792204",
            "Re_121p050171",
            "Re_139p642302",
            "Re_169p244893",
            "Re_196p160723",
        } else "train"
        candidates.append((float(arrays["re"][valid[0]]), split, valid[len(valid) // 2]))
    train_all = sorted(
        (row for row in candidates if row[1] == "train"), key=lambda row: row[0]
    )
    validation_all = sorted(
        (row for row in candidates if row[1] == "validation"), key=lambda row: row[0]
    )
    if coverage == "all":
        train, validation = train_all, validation_all
    else:
        train, validation = train_all[:3], validation_all[:1]
    return [(split, start) for _re, split, start in train + validation]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--coverage", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--gate-name", default="G_HP_native")
    args = parser.parse_args()
    output_dir = args.recovery_dir / "preflight" / args.gate_name
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    atomic_json(
        output_dir / "STARTED.json",
        {
            "gate": args.gate_name,
            "source_split_access": ["train", "validation"],
            "heldout_bundle_loaded": False,
            "horizon": args.horizon,
        },
    )

    device = torch.device("cuda")
    periodic_root = args.root / "periodic_specialist_r32"
    trainer_path = periodic_root / "code/train_periodic_moe.py"
    checkpoint_path = periodic_root / "checkpoint/FINAL_PERIODIC_SPECIALIST.pt"
    trainer = load_module("hp_native_periodic_trainer", trainer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, periodic_args, scalers = instantiate_periodic(trainer, checkpoint, device)
    tensor_path = periodic_root / "assets/velocity_rom_periodic.npz"
    pressure_tensor_path = periodic_root / "assets/pressure_poisson_surrogate_periodic.npz"
    tensors = np.load(tensor_path)
    pressure_tensors = np.load(pressure_tensor_path)
    arrays = load_bundle(
        [
            args.recovery_dir / "assets/periodic_train_modal_r32.npz",
            args.recovery_dir / "assets/periodic_validation_modal_r32.npz",
        ],
        trainer,
        tensors,
    )
    hopf = HopfRuntime(args.root, device)

    with np.load(
        args.recovery_dir / "assets/periodic_projection_contract_no_coeff.npz",
        allow_pickle=False,
    ) as periodic_pod, np.load(
        args.root / "Hopf/artifacts/hopf/velocity_pod_hopf.npz", allow_pickle=False
    ) as hopf_velocity, np.load(
        args.root / "Hopf/artifacts/hopf/pressure_pod_hopf.npz", allow_pickle=False
    ) as hopf_pressure:
        p_points = np.asarray(periodic_pod["points"])
        h_points = np.asarray(hopf_velocity["points"])
        p_area = np.asarray(periodic_pod["point_areas"], dtype=np.float64)
        h_area = np.asarray(hopf_velocity["point_areas"], dtype=np.float64)
        if not np.array_equal(p_points, h_points) or not np.array_equal(p_area, h_area):
            raise RuntimeError("P/H common grid or area weights mismatch")
        p_gauge = str(np.asarray(periodic_pod["pressure_gauge"]).item())
        h_gauge = str(np.asarray(hopf_pressure["pressure_gauge"]).item())
        if p_gauge != h_gauge:
            raise RuntimeError(f"P/H pressure gauge mismatch: {p_gauge} != {h_gauge}")

        p_phi_u = np.asarray(periodic_pod["phi_uv"][:RANK])
        p_mu_u = np.asarray(periodic_pod["mean_uv_regime"])
        p_phi_p = np.asarray(periodic_pod["phi_p"][:RANK])
        p_mu_p = np.asarray(periodic_pod["mean_p_regime"])
        h_phi_u = np.asarray(hopf_velocity["phi_uv"][:RANK])
        h_mu_u = np.asarray(hopf_velocity["mean_uv_regime"])
        h_phi_p = np.asarray(hopf_pressure["phi_p"][:RANK])
        h_mu_p = np.asarray(hopf_pressure["mean_p_regime"])
        uv_weights = np.concatenate((p_area, p_area))
        map_u, offset_u = affine_map(p_phi_u, p_mu_u, h_phi_u, h_mu_u, uv_weights)
        map_p, offset_p = affine_map(p_phi_p, p_mu_p, h_phi_p, h_mu_p, p_area)

        rows = []
        for split, start in selected_starts(arrays, args.horizon, args.coverage):
            sequence = [start]
            for _ in range(args.horizon):
                nxt = int(arrays["next_idx"][sequence[-1]])
                if nxt < 0:
                    raise RuntimeError("window exceeded Periodic native trajectory")
                sequence.append(nxt)
            sequence_array = np.asarray(sequence, dtype=np.int64)
            times = arrays["time"][sequence_array]
            dts = np.diff(times).astype(np.float32)
            if not (np.all(dts > 0.0) and np.all(np.isfinite(dts))):
                raise RuntimeError("invalid Periodic native timestamps")

            a0 = arrays["a"][start].copy()
            b0 = arrays["b"][start].copy()
            a_hist, b_hist, rhs_hist = trainer.init_history_states_np(start, arrays)
            p_pred_a, p_pred_b = [], []
            current = start
            current_a, current_b = a0, b0
            with torch.inference_mode():
                for dt in dts.tolist():
                    an, bn, rhs_g = trainer.integrate_autonomous_step_np(
                        model,
                        current_a,
                        current_b,
                        current,
                        float(dt),
                        a_hist,
                        b_hist,
                        rhs_hist,
                        arrays,
                        scalers,
                        tensors,
                        pressure_tensors,
                        periodic_args,
                        device,
                    )
                    p_pred_a.append(an)
                    p_pred_b.append(bn)
                    a_hist = np.concatenate((an[None, None], a_hist[:, :-1]), axis=1)
                    b_hist = np.concatenate((bn[None, None], b_hist[:, :-1]), axis=1)
                    rhs_hist = np.concatenate((rhs_g[None, None], rhs_hist[:, :-1]), axis=1)
                    current_a, current_b = an, bn
                    current = int(arrays["next_idx"][current])
            p_pred_a = np.asarray(p_pred_a)
            p_pred_b = np.asarray(p_pred_b)

            native_hist = arrays["hist_idx"][start]
            h_a0 = map_u @ a0 + offset_u
            h_b0 = map_p @ b0 + offset_p
            h_a_hist = arrays["a"][native_hist] @ map_u.T + offset_u
            h_b_hist = arrays["b"][native_hist] @ map_p.T + offset_p
            h_pred_a, h_pred_b = hopf.rollout(
                h_a0,
                h_b0,
                h_a_hist,
                h_b_hist,
                float(arrays["re"][start]),
                dts,
            )

            true_a = arrays["a"][sequence_array[1:]]
            true_b = arrays["b"][sequence_array[1:]]
            true_u = decode(true_a[-1], p_phi_u, p_mu_u)
            true_p = decode(true_b[-1], p_phi_p, p_mu_p)
            p_u = decode(p_pred_a[-1], p_phi_u, p_mu_u)
            p_p = decode(p_pred_b[-1], p_phi_p, p_mu_p)
            h_u = decode(h_pred_a[-1], h_phi_u, h_mu_u)
            h_p = decode(h_pred_b[-1], h_phi_p, h_mu_p)
            initial_p_u = decode(a0, p_phi_u, p_mu_u)
            initial_p_p = decode(b0, p_phi_p, p_mu_p)
            initial_h_u = decode(h_a0, h_phi_u, h_mu_u)
            initial_h_p = decode(h_b0, h_phi_p, h_mu_p)
            finite_p = bool(np.isfinite(p_pred_a).all() and np.isfinite(p_pred_b).all())
            finite_h = bool(np.isfinite(h_pred_a).all() and np.isfinite(h_pred_b).all())
            true_a_scale = max(float(np.max(np.linalg.norm(true_a, axis=1))), 1.0e-12)
            true_b_scale = max(float(np.max(np.linalg.norm(true_b, axis=1))), 1.0e-12)
            divergent_p = (not finite_p) or bool(
                np.max(np.linalg.norm(p_pred_a, axis=1)) > 10.0 * true_a_scale
                or np.max(np.linalg.norm(p_pred_b, axis=1)) > 10.0 * true_b_scale
            )
            divergent_h = (not finite_h) or bool(
                np.max(np.linalg.norm(h_pred_a, axis=1)) > 10.0 * true_a_scale
                or np.max(np.linalg.norm(h_pred_b, axis=1)) > 10.0 * true_b_scale
            )
            rows.append(
                {
                    "split": split,
                    "Re": float(arrays["re"][start]),
                    "Re_label": str(arrays["Re_label"][start]),
                    "start_snapshot": int(start),
                    "horizon": args.horizon,
                    "start_time": float(times[0]),
                    "end_time": float(times[-1]),
                    "dt_min": float(dts.min()),
                    "dt_max": float(dts.max()),
                    "timestamps_strictly_shared": True,
                    "same_initial_three_step_physical_history": True,
                    "Periodic_phase_source": "native indexed time/estimated_period",
                    "Hopf_phase_source": "phase-free",
                    "future_truth_used_by_router_or_rollout": False,
                    "initial_velocity_projection_relative_l2": field_relative(
                        initial_h_u, initial_p_u, uv_weights
                    ),
                    "initial_pressure_projection_relative_l2": field_relative(
                        initial_h_p, initial_p_p, p_area
                    ),
                    "Periodic_finite": finite_p,
                    "Hopf_finite": finite_h,
                    "Periodic_divergent": divergent_p,
                    "Hopf_divergent": divergent_h,
                    "Periodic_final_velocity_field_error": safe_field_relative(
                        p_u, true_u, uv_weights
                    ),
                    "Periodic_final_pressure_field_error": safe_field_relative(
                        p_p, true_p, p_area
                    ),
                    "Hopf_final_velocity_field_error": safe_field_relative(
                        h_u, true_u, uv_weights
                    ),
                    "Hopf_final_pressure_field_error": safe_field_relative(
                        h_p, true_p, p_area
                    ),
                }
            )

    passed = bool(rows) and all(
        row["Periodic_finite"]
        and row["Hopf_finite"]
        and not row["Periodic_divergent"]
        and not row["Hopf_divergent"]
        and row["timestamps_strictly_shared"]
        and row["same_initial_three_step_physical_history"]
        for row in rows
    )
    result = {
        "gate": args.gate_name,
        "decision": "PASS" if passed else "FAIL_CLOSED",
        "protocol": "Periodic database-native indexed phase/time plus phase-free Hopf",
        "source_database": "Periodic train/validation-only sealed modal bundle",
        "heldout_modal_or_metric_data_loaded": False,
        "checkpoint_sha256": {
            "Periodic": sha256(checkpoint_path),
            "Hopf": sha256(hopf.checkpoint_path),
        },
        "asset_sha256": {
            "train_bundle": sha256(
                args.recovery_dir / "assets/periodic_train_modal_r32.npz"
            ),
            "validation_bundle": sha256(
                args.recovery_dir / "assets/periodic_validation_modal_r32.npz"
            ),
            "projection_contract_no_coeff": sha256(
                args.recovery_dir / "assets/periodic_projection_contract_no_coeff.npz"
            ),
        },
        "common_grid_area_pressure_gauge": True,
        "history_length": HISTORY,
        "horizon": args.horizon,
        "rows": rows,
        "finite_fraction": float(
            np.mean([row["Periodic_finite"] and row["Hopf_finite"] for row in rows])
        ),
        "divergent_windows": int(
            sum(row["Periodic_divergent"] + row["Hopf_divergent"] for row in rows)
        ),
        "authorized_next_action": (
            "construct H-P train/validation native cache"
            if passed
            else "do not construct H-P cache or train"
        ),
    }
    atomic_json(output_dir / "G_HP_native.json", result)
    atomic_json(
        output_dir / ("PASS.json" if passed else "FAIL_CLOSED.json"),
        {"gate": args.gate_name, "decision": result["decision"]},
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "rows": len(rows),
                "finite_fraction": result["finite_fraction"],
                "divergent_windows": result["divergent_windows"],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
