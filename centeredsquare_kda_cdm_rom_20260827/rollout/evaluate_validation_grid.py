#!/usr/bin/env python3
"""GPU-vectorized M0/M1/M3 periodic validation with stage-consistent coupled RK4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--fvm-tensors", type=Path, required=True)
    parser.add_argument("--pressure-closure", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--memory-fit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-experiment", default="cdm-grom-validation-grid")
    parser.add_argument("--swanlab-mode", choices=["online", "cloud", "local", "offline"], default="online")
    args = parser.parse_args()
    swanlab = None
    if args.swanlab_project:
        import swanlab as swanlab_module
        swanlab = swanlab_module
        swanlab.init(
            project=args.swanlab_project, experiment_name=args.swanlab_experiment,
            mode=args.swanlab_mode,
            config={"max_step": args.max_step, "device": args.device, "heldout_loaded": False},
        )
    dtype, device = torch.float64, torch.device(args.device)
    phase = json.loads(args.phase_summary.read_text())
    pair = (int(phase["phase_mode_p"]), int(phase["phase_mode_q"]))
    norm = phase["normalization"]
    fit = json.loads((args.memory_fit_root / "TEACHER_MEMORY_FIT.json").read_text())

    with np.load(args.velocity_pod, allow_pickle=False) as source:
        mean = np.asarray(source["mean"], dtype=np.float64)
        weights = np.asarray(source["weights"], dtype=np.float64)
        weighted_modes = np.asarray(source["weighted_modes"], dtype=np.float64)[:28]
    with np.load(args.fvm_tensors, allow_pickle=False) as source:
        tensor_np = {key: np.asarray(source[key], dtype=np.float64) for key in source.files}
    with np.load(args.pressure_closure, allow_pickle=False) as source:
        feature_scale_np = np.asarray(source["feature_scale"], dtype=np.float64)
        pressure_weights_np = np.asarray(source["weights"], dtype=np.float64)
    tensor = {key: torch.as_tensor(value, dtype=dtype, device=device) for key, value in tensor_np.items()}
    feature_scale = torch.as_tensor(feature_scale_np, dtype=dtype, device=device)
    pressure_weights = torch.as_tensor(pressure_weights_np, dtype=dtype, device=device)

    records = [{"name": "M0_frozen", "kind": 0}]
    with np.load(args.memory_fit_root / "markov_phase_fit.npz", allow_pickle=False) as source:
        for ridge, coefficient in zip(source["ridge_candidates"], source["readout_coefficients"]):
            records.append({"name": f"M1_ridge_{ridge:.8g}", "kind": 1, "ridge": float(ridge), "coefficient": coefficient})
    for candidate in fit["candidates"]:
        with np.load(args.memory_fit_root / candidate["asset"], allow_pickle=False) as source:
            for ridge, coefficient in zip(source["ridge_candidates"], source["readout_coefficients"]):
                records.append({
                    "name": f"M3_gamma_{candidate['gamma']:.8g}_eta_{candidate['eta']:.8g}_ridge_{ridge:.8g}",
                    "kind": 2, "gamma": float(candidate["gamma"]), "eta": float(candidate["eta"]),
                    "ridge": float(ridge), "coefficient": coefficient,
                })
    model_count = len(records)
    kinds = torch.as_tensor([item["kind"] for item in records], device=device)
    gamma = torch.as_tensor([item.get("gamma", 0.0) for item in records], dtype=dtype, device=device)
    eta = torch.as_tensor([item.get("eta", 0.0) for item in records], dtype=dtype, device=device)
    m1_coeff = torch.zeros((model_count, 7), dtype=dtype, device=device)
    m3_coeff = torch.zeros((model_count, 4), dtype=dtype, device=device)
    for index, item in enumerate(records):
        if item["kind"] == 1:
            m1_coeff[index] = torch.as_tensor(item["coefficient"], dtype=dtype, device=device)
        elif item["kind"] == 2:
            m3_coeff[index] = torch.as_tensor(item["coefficient"], dtype=dtype, device=device)

    def baseline_rhs(state, viscosity):
        pressure_feature = torch.cat([
            torch.ones((len(state), 1), dtype=dtype, device=device),
            torch.full((len(state), 1), viscosity, dtype=dtype, device=device),
            state, viscosity * state,
        ], dim=1)
        pressure = (pressure_feature / feature_scale) @ pressure_weights
        return (
            tensor["c_conv"] + viscosity * tensor["c_diff"] + tensor["c_pressure"]
            + state @ (tensor["A_conv"] + viscosity * tensor["A_diff"]).T
            + torch.einsum("ijk,bj,bk->bi", tensor["H_conv"], state, state)
            + pressure @ tensor["P"].T
        )

    re_mean, re_scale = norm["Re"]["mean"], norm["Re"]["scale"]
    rho_mean, rho_scale = norm["rho"]["mean"], norm["rho"]["scale"]
    omega_mean, omega_scale = norm["omega_fvm"]["mean"], norm["omega_fvm"]["scale"]
    rhodot_mean, rhodot_scale = norm["rho_dot_fvm"]["mean"], norm["rho_dot_fvm"]["scale"]

    def joint_rhs(state, memory, re_value):
        viscosity = 1.0 / re_value
        fvm = baseline_rhs(state, viscosity)
        ap, aq = state[:, pair[0]], state[:, pair[1]]
        fp, fq = fvm[:, pair[0]], fvm[:, pair[1]]
        rho = torch.sqrt(ap * ap + aq * aq).clamp_min(1e-12)
        theta = torch.atan2(aq, ap)
        omega = (ap * fq - aq * fp) / (rho * rho + 1e-12)
        rho_dot = (ap * fp + aq * fq) / (rho + 1e-12)
        phi = torch.stack([
            torch.ones_like(theta), torch.cos(theta), torch.sin(theta), torch.cos(2.0 * theta),
            torch.sin(2.0 * theta), (rho - rho_mean) / rho_scale,
            torch.full_like(theta, (re_value - re_mean) / re_scale),
        ], dim=1)
        key = phi / torch.linalg.vector_norm(phi, dim=1, keepdim=True).clamp_min(1e-12)
        value = torch.stack([
            (omega - omega_mean) / omega_scale, (rho_dot - rhodot_mean) / rhodot_scale,
            (rho - rho_mean) / rho_scale, torch.ones_like(theta),
        ], dim=1)
        memory_read = torch.einsum("bkv,bk->bv", memory, key)
        delta = torch.zeros(model_count, dtype=dtype, device=device)
        markov = kinds == 1
        kda = kinds == 2
        delta[markov] = torch.sum(m1_coeff[markov] * phi[markov], dim=1)
        delta[kda] = torch.sum(m3_coeff[kda] * memory_read[kda], dim=1)
        correction = torch.zeros_like(state)
        correction[:, pair[0]] = -delta * aq
        correction[:, pair[1]] = delta * ap
        memory_derivative = -gamma[:, None, None] * memory + eta[:, None, None] * torch.einsum(
            "bk,bv->bkv", key, value - memory_read
        )
        memory_derivative[kinds != 2] = 0.0
        return fvm + correction, memory_derivative

    def advance(state, memory, duration, re_value):
        count = max(1, int(np.ceil(duration / args.max_step)))
        step = duration / count
        for _ in range(count):
            k1a, k1s = joint_rhs(state, memory, re_value)
            k2a, k2s = joint_rhs(state + 0.5 * step * k1a, memory + 0.5 * step * k1s, re_value)
            k3a, k3s = joint_rhs(state + 0.5 * step * k2a, memory + 0.5 * step * k2s, re_value)
            k4a, k4s = joint_rhs(state + step * k3a, memory + step * k3s, re_value)
            state = state + step * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0
            memory = memory + step * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0
        return state, memory

    validation = json.loads(args.validation_manifest.read_text())
    per_case = []
    for entry in validation["cases"]:
        validation_path = Path(entry["local_path"])
        if args.validation_root is not None:
            validation_path = args.validation_root / validation_path.name
        with np.load(validation_path, allow_pickle=False) as source:
            times = np.asarray(source["times"], dtype=np.float64)
            velocity = np.asarray(source["U"], dtype=np.float64)
            re_value = float(source["Re"])
        truth = ((velocity - mean[None]).reshape(len(times), -1) * weights[None]) @ weighted_modes.T
        fluctuation_energy = np.sum(((velocity - mean[None]).reshape(len(times), -1) * weights[None]) ** 2, axis=1)
        unresolved = np.maximum(0.0, fluctuation_energy - np.sum(truth**2, axis=1))
        denominator = float(np.sum((velocity.reshape(len(times), -1) * weights[None]) ** 2))
        state = torch.as_tensor(np.repeat(truth[0][None], model_count, axis=0), dtype=dtype, device=device)
        memory = torch.zeros((model_count, 7, 4), dtype=dtype, device=device)
        trajectory = np.empty((model_count, len(times), 28), dtype=np.float64)
        memory_norm = np.empty((model_count, len(times)), dtype=np.float64)
        memory_error = np.empty((model_count, len(times)), dtype=np.float64)
        for time_index in range(len(times)):
            trajectory[:, time_index] = state.detach().cpu().numpy()
            memory_norm[:, time_index] = torch.linalg.vector_norm(memory.reshape(model_count, -1), dim=1).cpu().numpy()
            with torch.no_grad():
                fvm_now = baseline_rhs(state, 1.0 / re_value)
                ap_now, aq_now = state[:, pair[0]], state[:, pair[1]]
                fp_now, fq_now = fvm_now[:, pair[0]], fvm_now[:, pair[1]]
                rho_now = torch.sqrt(ap_now * ap_now + aq_now * aq_now).clamp_min(1e-12)
                theta_now = torch.atan2(aq_now, ap_now)
                omega_now = (ap_now * fq_now - aq_now * fp_now) / (rho_now * rho_now + 1e-12)
                rho_dot_now = (ap_now * fp_now + aq_now * fq_now) / (rho_now + 1e-12)
                phi_now = torch.stack([
                    torch.ones_like(theta_now), torch.cos(theta_now), torch.sin(theta_now),
                    torch.cos(2.0 * theta_now), torch.sin(2.0 * theta_now),
                    (rho_now - rho_mean) / rho_scale,
                    torch.full_like(theta_now, (re_value - re_mean) / re_scale),
                ], dim=1)
                key_now = phi_now / torch.linalg.vector_norm(phi_now, dim=1, keepdim=True).clamp_min(1e-12)
                value_now = torch.stack([
                    (omega_now - omega_mean) / omega_scale,
                    (rho_dot_now - rhodot_mean) / rhodot_scale,
                    (rho_now - rho_mean) / rho_scale, torch.ones_like(theta_now),
                ], dim=1)
                read_now = torch.einsum("bkv,bk->bv", memory, key_now)
                memory_error[:, time_index] = torch.linalg.vector_norm(value_now - read_now, dim=1).cpu().numpy()
            if time_index + 1 < len(times):
                state, memory = advance(state, memory, float(times[time_index + 1] - times[time_index]), re_value)
        case_metrics = []
        truth_phase = np.unwrap(np.arctan2(truth[:, pair[1]], truth[:, pair[0]]))
        truth_frequency = float(np.polyfit(times, truth_phase, 1)[0] / (2.0 * np.pi))
        for index, item in enumerate(records):
            predicted = trajectory[index]
            numerator = float(np.sum((predicted - truth) ** 2) + np.sum(unresolved))
            predicted_phase = np.unwrap(np.arctan2(predicted[:, pair[1]], predicted[:, pair[0]]))
            finite = bool(np.all(np.isfinite(predicted)) and np.all(np.isfinite(memory_norm[index])))
            case_metrics.append({
                "name": item["name"], "J500": float(np.sqrt(numerator / denominator)),
                "finite": finite, "coefficient_norm_max": float(np.max(np.linalg.norm(predicted, axis=1))),
                "memory_norm_max": float(np.max(memory_norm[index])),
                "memory_prediction_error_mean": float(np.mean(memory_error[index])),
                "phase_error_final_abs": float(abs(predicted_phase[-1] - truth_phase[-1])),
                "frequency_relative_error": float(abs(np.polyfit(times, predicted_phase, 1)[0] / (2.0 * np.pi) - truth_frequency) / max(abs(truth_frequency), 1e-14)),
                "amplitude_relative_error": float(abs(np.mean(np.hypot(predicted[:, pair[0]], predicted[:, pair[1]])) - np.mean(np.hypot(truth[:, pair[0]], truth[:, pair[1]]))) / max(np.mean(np.hypot(truth[:, pair[0]], truth[:, pair[1]])), 1e-14)),
            })
        mean_numerator = float(np.sum(truth**2) + np.sum(unresolved))
        per_case.append({"Re": re_value, "truth_frequency": truth_frequency, "mean_flow_J500": float(np.sqrt(mean_numerator / denominator)), "models": case_metrics})
        if swanlab is not None:
            swanlab.log({
                "validation/Re": re_value,
                "validation/M0_J500": case_metrics[0]["J500"],
                "validation/completed_cases": len(per_case),
            })

    aggregate = []
    for index, item in enumerate(records):
        values = [case["models"][index]["J500"] for case in per_case]
        stable = all(case["models"][index]["finite"] and case["models"][index]["coefficient_norm_max"] < 1e6 and case["models"][index]["memory_norm_max"] < 1e6 for case in per_case)
        aggregate.append({
            "name": item["name"], "kind": item["kind"], "stable_6_of_6": stable,
            "J500_mean": float(np.mean(values)), "J500_median": float(np.median(values)), "J500_max": float(np.max(values)),
        })
    best = {}
    for kind, label in ((0, "M0"), (1, "M1"), (2, "M3")):
        eligible = [item for item in aggregate if item["kind"] == kind and item["stable_6_of_6"]]
        best[label] = min(eligible, key=lambda item: item["J500_mean"]) if eligible else None
    payload = {
        "schema_version": 1, "device": str(device), "dtype": "float64", "model_count": model_count,
        "phase_pair": list(pair), "initialization": "truth_a_at_t0_and_zero_memory",
        "per_case": per_case, "aggregate": aggregate, "best": best,
        "mean_flow_J500_mean": float(np.mean([case["mean_flow_J500"] for case in per_case])),
        "heldout_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if swanlab is not None:
        final_log = {"summary/mean_flow_J500_mean": payload["mean_flow_J500_mean"]}
        for label in ("M0", "M1", "M3"):
            if best[label] is not None:
                final_log[f"summary/{label}_J500_mean"] = best[label]["J500_mean"]
        swanlab.log(final_log)
        swanlab.finish()
    print(json.dumps({"best": best, "mean_flow_J500_mean": payload["mean_flow_J500_mean"]}, indent=2))


if __name__ == "__main__":
    main()
