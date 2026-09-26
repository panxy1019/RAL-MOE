#!/usr/bin/env python3
"""Restore each frozen specialist and run audit-only step-halving checks."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch

from rhs_only_wrappers import RHSOnlyResult, SpecialistOperatorContract


ROOT = Path("/root/panxy/particalMOE")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rel_l2(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1.0e-30))


def physical_error(
    one: np.ndarray,
    half: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray,
    weights: np.ndarray,
) -> float:
    x = mean.astype(np.float64) + one.astype(np.float64) @ basis.astype(np.float64)
    y = mean.astype(np.float64) + half.astype(np.float64) @ basis.astype(np.float64)
    d = x - y
    return float(np.sqrt(np.sum(weights * d * d)) / max(np.sqrt(np.sum(weights * x * x)), 1.0e-30))


def dt_grid(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    # Include the lower native-time tail so a non-vanishing algebraic pressure
    # discrepancy cannot be mistaken for a large-step RK truncation effect.
    return [float(x) for x in np.unique(np.quantile(finite, [0.01, 0.05, 0.15, 0.35, 0.55]))]


def convergence_orders(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    out: list[float | None] = []
    for left, right in zip(rows[:-1], rows[1:]):
        e0, e1 = float(left[key]), float(right[key])
        h0, h1 = float(left["dt"]), float(right["dt"])
        if e0 <= 0 or e1 <= 0 or h0 == h1:
            out.append(None)
        else:
            out.append(float(math.log(e1 / e0) / math.log(h1 / h0)))
    return out


def evaluate(
    name: str,
    contract: SpecialistOperatorContract,
    step: Callable[[torch.Tensor, torch.Tensor, float], tuple[torch.Tensor, torch.Tensor, RHSOnlyResult]],
    a0: torch.Tensor,
    b0: torch.Tensor,
    dts: list[float],
    geometry: dict[str, np.ndarray],
    restore: dict[str, Any],
    native_integrator: str,
) -> dict[str, Any]:
    rows = []
    with torch.no_grad():
        for dt in sorted(dts):
            one_a, one_b, rhs = step(a0, b0, dt)
            half_a, half_b, _ = step(a0, b0, dt / 2.0)
            two_a, two_b, _ = step(half_a, half_b, dt / 2.0)
            oa, ob = one_a[0].float().cpu().numpy(), one_b[0].float().cpu().numpy()
            ta, tb = two_a[0].float().cpu().numpy(), two_b[0].float().cpu().numpy()
            rows.append(
                {
                    "dt": dt,
                    "velocity_state_relative_l2": rel_l2(oa, ta),
                    "pressure_state_relative_l2": rel_l2(ob, tb),
                    "velocity_field_relative_l2": physical_error(
                        oa,
                        ta,
                        geometry["phi_u"],
                        geometry["mean_u"],
                        np.concatenate([geometry["areas"], geometry["areas"]]),
                    ),
                    "pressure_field_relative_l2": physical_error(
                        ob,
                        tb,
                        geometry["phi_p"],
                        geometry["mean_p"],
                        geometry["areas"],
                    ),
                    "velocity_rhs_norm": float(torch.linalg.norm(rhs.velocity_rhs).cpu()),
                    "pressure_closure_norm": float(torch.linalg.norm(rhs.pressure_closure_output).cpu()),
                }
            )
    orders = {
        key: convergence_orders(rows, key)
        for key in (
            "velocity_state_relative_l2",
            "pressure_state_relative_l2",
            "velocity_field_relative_l2",
            "pressure_field_relative_l2",
        )
    }
    return {
        "specialist": name,
        "restore": restore,
        "operator_contract": asdict(contract),
        "native_integrator": native_integrator,
        "history_differences_divided_by_dt": False,
        "step_halving": rows,
        "observed_orders": orders,
    }


def run_steady(device: torch.device) -> dict[str, Any]:
    mod = load_module("steady_s2b_audit", ROOT / "steady_specialist_v1/code/train_s2b_3090.py")
    cfg = json.loads((ROOT / "steady_specialist_v1/code/training_s2b_portable.json").read_text())
    cfg["checkpoint_root"] = "/tmp/continuous_rhs_audit_s/checkpoints"
    exp = mod.Experiment(SimpleNamespace(run_dir="/tmp/continuous_rhs_audit_s/run", resume=None), cfg)
    checkpoint = torch.load(
        ROOT / "steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt",
        map_location="cpu",
        weights_only=False,
    )
    exp.model.load_state_dict(checkpoint["model"], strict=True)
    exp.model.eval().requires_grad_(False)
    starts = np.concatenate([np.asarray(v) for v in exp.build_windows(1)["validation"].values()])
    sid = int(starts[len(starts) // 2])
    cur = torch.tensor([sid], dtype=torch.long, device=device)
    a0, b0 = exp.cache["a"][cur], exp.cache["b"][cur]
    label = exp.cache["label_id"][cur]
    hist = exp.cache["hist_idx"][cur]
    ah, bh, rh = exp.cache["a"][hist], exp.cache["b"][hist], exp.cache["rhs_g"][hist]
    re = exp.cache["re"][cur]
    phase = exp.cache["phase"][cur]

    def rhs(a: torch.Tensor, b: torch.Tensor) -> RHSOnlyResult:
        g = exp.vendor.galerkin_rhs_torch(a, b, label, exp.gal)
        base = exp.vendor.make_features_torch(a, b, g, re, phase, cfg["model"]["phase_harmonics"])
        features = exp.vendor.make_history_features_from_states_torch(base, a, b, g, ah, bh, rh)
        raw_u, raw_p = exp._model_call((features - exp.xmt) / exp.xst, 1)
        velocity_rhs = g + raw_u.float() * exp.rst + exp.rmt
        pressure_residual = raw_p.float() * exp.pst + exp.pmt
        return RHSOnlyResult(
            velocity_rhs=velocity_rhs,
            pressure_closure_output=pressure_residual,
            galerkin_rhs=g,
            diagnostics={"phase": float(phase[0].cpu())},
        )

    def step(a: torch.Tensor, b: torch.Tensor, dt: float):
        out = rhs(a, b)
        an = a + float(dt) * out.velocity_rhs
        bn = exp.vendor.pressure_surrogate_torch(an, label, exp.sur) + out.pressure_closure_output
        return an, bn, out

    with np.load(ROOT / "steady_specialist_v1/data/steady_pod_runtime_r32.npz") as z:
        geometry = {
            "phi_u": z["velocity_basis"],
            "phi_p": z["pressure_basis"],
            "mean_u": z["velocity_mean"],
            "mean_p": z["pressure_mean"],
            "areas": z["point_areas"],
        }
    report = evaluate(
        "Steady",
        SpecialistOperatorContract("Steady"),
        step,
        a0,
        b0,
        dt_grid(exp.a["dt_next"]),
        geometry,
        {"strict_load_state_dict": True, "checkpoint_step": int(checkpoint["step"]), "sample_id": sid},
        "Euler in the final S2/S3/S4 wrapper (vendor RK4 is bypassed)",
    )
    del exp, checkpoint
    return report


def periodic_model(device: torch.device):
    trainer = load_module("periodic_trainer_audit", ROOT / "periodic_specialist_r32/code/train_periodic_moe.py")
    checkpoint = torch.load(
        ROOT / "periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt",
        map_location="cpu",
        weights_only=False,
    )
    args = SimpleNamespace(**dict(checkpoint["args"]))
    args.data_root = ROOT / "periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2"
    args.tensor_path = ROOT / "periodic_specialist_r32/assets/velocity_rom_periodic.npz"
    args.pressure_surrogate_path = ROOT / "periodic_specialist_r32/assets/pressure_poisson_surrogate_periodic.npz"
    if isinstance(getattr(args, "regime_rom_root", None), str):
        args.regime_rom_root = Path(args.regime_rom_root)
    arrays, _ = trainer.build_arrays(args)
    model = trainer.OperatorSpaceMoEROM(
        in_dim=arrays["x"].shape[1], out_dim=args.r_u, pressure_dim=args.r_p,
        hidden_dim=args.hidden_dim, expert_hidden=args.expert_hidden,
        num_blocks=args.num_blocks, num_experts=args.num_experts,
        num_operator_spaces=args.num_shared_experts, num_regime_groups=args.num_regime_groups,
        experts_per_group=args.experts_per_group, top_k=args.top_k, group_top_k=args.group_top_k,
        dropout=args.dropout, temperature=args.temperature, gate_floor=args.gate_floor,
        group_temperature=args.group_temperature, group_gate_floor=args.group_gate_floor,
        shared_scale=args.shared_scale, routed_scale=args.routed_scale,
        expert_blocks=args.expert_blocks, quadratic_rank=args.quadratic_rank,
        quadratic_scale=args.quadratic_scale, phase_harmonics=args.phase_harmonics,
        closure_mode=args.closure_mode, pressure_base_mode=args.pressure_base_mode,
        film_base_hidden=args.film_base_hidden, film_base_scale=args.film_base_scale,
        attractor_conditioned=args.attractor_conditioned,
        attractor_adapter_dim=args.attractor_adapter_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().requires_grad_(False)
    tensors = np.load(args.tensor_path)
    pressure_tensors = np.load(args.pressure_surrogate_path)
    gal = trainer.build_galerkin_torch(tensors, arrays["labels"], args.r_u, args.r_p, device)
    sur = trainer.build_pressure_surrogate_torch(pressure_tensors, arrays["labels"], args.r_u, args.r_p, device)
    base_bundle = trainer.build_base_bundle(args, arrays, device)
    arrays_t = {}
    for key in ("a", "b", "rhs_g", "re", "phase", "time", "label_id", "hist_idx", "next_idx"):
        value = arrays[key]
        dtype = torch.long if np.issubdtype(value.dtype, np.integer) else torch.float32
        arrays_t[key] = torch.as_tensor(value, dtype=dtype, device=device)
    saved = checkpoint["scalers"]
    scalers_t = {
        "x_mean": torch.as_tensor(saved["x"]["mean"], device=device),
        "x_scale": torch.as_tensor(saved["x"]["scale"], device=device),
        "rhs_op_mean": torch.as_tensor(saved["rhs_operator"]["mean"], device=device),
        "rhs_op_scale": torch.as_tensor(saved["rhs_operator"]["scale"], device=device),
        "pressure_mean": torch.as_tensor(saved["pressure_next"]["mean"], device=device),
        "pressure_scale": torch.as_tensor(saved["pressure_next"]["scale"], device=device),
        "pressure_state_scale": torch.as_tensor(saved["pressure_state"]["scale"], device=device),
        "pressure_input_mean": torch.zeros(args.r_u + args.r_p, device=device),
        "pressure_input_scale": torch.ones(args.r_u + args.r_p, device=device),
    }
    return trainer, checkpoint, args, arrays, arrays_t, model, gal, sur, base_bundle, scalers_t


def run_periodic(device: torch.device) -> dict[str, Any]:
    trainer, checkpoint, args, arrays, arrays_t, model, gal, sur, base_bundle, scalers_t = periodic_model(device)
    validation_labels = set(checkpoint["split_stats"]["validation_labels"])
    ids = [int(i) for i in arrays["sample_ids"] if str(arrays["labels"][arrays["label_id"][i]]) in validation_labels]
    sid = ids[len(ids) // 2]
    current = torch.tensor([sid], dtype=torch.long, device=device)
    a0, b0 = arrays_t["a"][current], arrays_t["b"][current]
    hist = arrays_t["hist_idx"][current]
    ah, bh, rh = arrays_t["a"][hist], arrays_t["b"][hist], arrays_t["rhs_g"][hist]

    def rhs(a: torch.Tensor, b: torch.Tensor) -> RHSOnlyResult:
        vel, pressure, g, closure = trainer.model_outputs_from_states_torch(
            model, a, b, current, ah, bh, rh, arrays_t, scalers_t, gal, base_bundle, args
        )
        return RHSOnlyResult(vel, pressure, g, {"closure": closure, "phase": float(arrays_t["phase"][current][0].cpu())})

    def step(a: torch.Tensor, b: torch.Tensor, dt: float):
        dt_t = torch.full((1, 1), float(dt), device=device)
        an, bn, _, aux = trainer.integrate_autonomous_step_torch(
            model, a, b, current, dt_t, ah, bh, rh,
            arrays_t, scalers_t, gal, sur, base_bundle, args, return_aux=True,
        )
        out = rhs(a, b)
        return an, bn, RHSOnlyResult(out.velocity_rhs, aux["pressure_residual"], out.galerkin_rhs, out.diagnostics)

    pod = args.data_root
    with np.load(pod / "global_velocity_pod_area_weighted_l2.npz") as u, np.load(pod / "global_pressure_pod_area_weighted_l2.npz") as p:
        geometry = {"phi_u": u["phi_uv"], "phi_p": p["phi_p"], "mean_u": u["mean_uv_regime"], "mean_p": p["mean_p_regime"], "areas": u["point_areas"]}
    report = evaluate(
        "Periodic", SpecialistOperatorContract("Periodic"), step, a0, b0,
        dt_grid(arrays["dt_next"]), geometry,
        {"strict_load_state_dict": True, "best_epoch": int(checkpoint["best_epoch"]), "sample_id": sid},
        "RK4 for velocity; one algebraic pressure closure after the velocity macro-step",
    )
    report["phase_contract"] = {
        "harmonics": int(args.phase_harmonics),
        "source": "CSV has no phase/period; loader uses normalized full-trajectory database time",
        "dt_is_network_input": False,
    }
    return report


def run_hopf(device: torch.device) -> dict[str, Any]:
    work = ROOT / "Hopf/migrated_h4_expanded"
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))
    base = load_module("hopf_base_audit", work / "code/train_hopf_moe_expanded.py")
    h4 = load_module("hopf_h4_audit", work / "code/train_h4_expanded.py")
    checkpoint = torch.load(
        work / "final_evaluation/20260722_h4_expanded_final/HopfExpanded34_H4_NormalFormRadial_r32/final.pt",
        map_location="cpu", weights_only=False,
    )
    outer = SimpleNamespace(**dict(checkpoint["args"]))
    outer.baseline_trainer = work / "code/train_hopf_moe_expanded.py"
    outer.coefficient_view = work / "assets_r32/expanded_h4_trainval_r32.npz"
    outer.galerkin_path = work / "assets_r32/expanded_h4_trainonly_galerkin_r32.npz"
    outer.pressure_path = work / "assets_r32/expanded_h4_trainonly_pressure_r32.npz"
    outer.contract = work / "trainonly_contract/trainonly_fluctuation_contract.npz"
    base_args = SimpleNamespace(**{
        **vars(outer), "history_len": 3, "hidden_dim": 256, "expert_hidden": 1024,
        "num_blocks": 3, "experts": 6, "top_k": 2, "expert_blocks": 4,
        "quadratic_rank": 4, "dropout": 0.04, "temperature": 0.8,
        "adaptive_gate_initial_logit": 6.0, "lambda_scale_amplitude": 1.0,
        "lambda_scale_growth": 0.5, "lambda_scale_sign": 0.1, "scale_floor_quantile": 0.1,
    })
    data = base.load_coefficients(SimpleNamespace(coefficient_view=outer.coefficient_view, history_len=3))
    rom_np = base.load_train_rom(SimpleNamespace(galerkin_path=outer.galerkin_path, pressure_path=outer.pressure_path))
    rom = {k: torch.as_tensor(v, device=device) for k, v in rom_np.items()}
    stats = {k: torch.as_tensor(v, device=device) for k, v in checkpoint["norm_stats"].items()}
    fc = h4.FluctuationContract(outer.contract, device)
    # The expanded H4 trainer deliberately keeps the physical backbone on the
    # native 493-D feature contract.  Fluctuation coordinates and the normal
    # form head are loss-only; they are not concatenated into model.forward.
    state = base.state_features
    probe = base.batch_from_ids(data, data["train_ids"][:2], device)
    prh = base.galerkin(
        probe["ah"].reshape(-1, 32), probe["bh"].reshape(-1, 32),
        probe["re"][:, None].expand(-1, 3).reshape(-1), rom,
    ).reshape_as(probe["ah"])
    in_dim = state(probe["a"], probe["b"], probe["re"], probe["ah"], probe["bh"], prh, rom, stats)[0].shape[1]
    model = base.build_model(in_dim, base_args, stats, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().requires_grad_(False)
    outputs, _ = h4.make_ops(base, fc, state)
    sid = int(data["val_ids"][len(data["val_ids"]) // 2])
    q = base.batch_from_ids(data, np.asarray([sid]), device)
    a0, b0, re, ah, bh = q["a"], q["b"], q["re"], q["ah"], q["bh"]
    rh = base.galerkin(ah.reshape(-1, 32), bh.reshape(-1, 32), re[:, None].expand(-1, 3).reshape(-1), rom).reshape_as(ah)

    def rhs(a: torch.Tensor, b: torch.Tensor) -> RHSOnlyResult:
        vel, pressure, _, _, closure, g = outputs(model, a, b, re, ah, bh, rh, rom, stats, False)
        return RHSOnlyResult(vel, pressure, g, {"closure": closure, "phase_free": True})

    def step(a: torch.Tensor, b: torch.Tensor, dt: float):
        dt_t = torch.full((1, 1), float(dt), device=device)
        k1, pres, _, _, closure, g = outputs(model, a, b, re, ah, bh, rh, rom, stats, False)
        k2 = outputs(model, a + 0.5 * dt_t * k1, b, re, ah, bh, rh, rom, stats, False)[0]
        k3 = outputs(model, a + 0.5 * dt_t * k2, b, re, ah, bh, rh, rom, stats, False)[0]
        k4 = outputs(model, a + dt_t * k3, b, re, ah, bh, rh, rom, stats, False)[0]
        an = a + dt_t / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        bn = base.v16.closure_components_torch("adaptive_gate", base.pressure_base(an, re, rom), pres, closure)[0]
        return an, bn, RHSOnlyResult(k1, pres, g, {"closure": closure, "phase_free": True})

    with np.load(outer.coefficient_view) as z:
        geometry = {"phi_u": z["phi_uv"], "phi_p": z["phi_p"], "mean_u": z["mean_uv_train"], "mean_p": z["mean_p_train"], "areas": z["point_areas"]}
        times = z["time"]
        res = z["Re"]
    dt_values = np.concatenate([np.diff(times[res == rv]) for rv in np.unique(res)])
    report = evaluate(
        "Hopf", SpecialistOperatorContract("Hopf"), step, a0, b0,
        dt_grid(dt_values), geometry,
        {"strict_load_state_dict": True, "best_step": int(checkpoint["best_step"]), "sample_id": sid},
        "RK4 for velocity; one algebraic pressure closure after the velocity macro-step",
    )
    report["phase_contract"] = {"harmonics": 0, "source": "phase-free", "dt_is_network_input": False}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist", choices=("steady", "hopf", "periodic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the frozen checkpoint contract")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"steady": run_steady, "hopf": run_hopf, "periodic": run_periodic}[args.specialist](device)
    report["full_state_continuous_rhs"] = False
    report["decision"] = "FAIL_FULL_STATE_RHS_CONTRACT"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"specialist": report["specialist"], "decision": report["decision"], "restore": report["restore"]}))
    del report
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
