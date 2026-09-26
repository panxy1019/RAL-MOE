#!/usr/bin/env python3
"""Independent physical-field evaluation for the CenteredSquare periodic MoE.

This evaluates autonomous coefficient rollouts against the full held-out
snapshots (not merely against POD coefficients).  The POD reconstruction floor
is reported separately, so model error and representation truncation are not
conflated.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def load_trainer(path: Path):
    spec = importlib.util.spec_from_file_location("square_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def relative_l2(diff, truth, weights):
    numerator = float(np.sum(diff * diff * weights))
    denominator = float(np.sum(truth * truth * weights))
    return float(np.sqrt(numerator / max(denominator, 1e-30))), numerator, denominator


def pressure_gauge(p, volume):
    return p - np.sum(p * volume[None, :], axis=1, keepdims=True) / np.sum(volume)


def raw_file(raw_dir: Path, label: str) -> Path:
    path = raw_dir / f"snapshots_{label}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_model(tr, arrays, args, device):
    return tr.OperatorSpaceMoEROM(
        in_dim=arrays["x"].shape[1], out_dim=args.r_u, pressure_dim=args.r_p,
        hidden_dim=args.hidden_dim, expert_hidden=args.expert_hidden,
        num_blocks=args.num_blocks, num_experts=args.num_experts,
        num_operator_spaces=args.num_shared_experts,
        num_regime_groups=args.num_regime_groups,
        experts_per_group=args.experts_per_group, top_k=args.top_k,
        group_top_k=args.group_top_k, dropout=args.dropout,
        temperature=args.temperature, gate_floor=args.gate_floor,
        group_temperature=args.group_temperature, group_gate_floor=args.group_gate_floor,
        shared_scale=args.shared_scale, routed_scale=args.routed_scale,
        expert_blocks=args.expert_blocks, quadratic_rank=args.quadratic_rank,
        quadratic_scale=args.quadratic_scale, phase_harmonics=args.phase_harmonics,
        closure_mode=args.closure_mode, pressure_base_mode=args.pressure_base_mode,
        film_base_hidden=args.film_base_hidden, film_base_scale=args.film_base_scale,
        attractor_conditioned=args.attractor_conditioned,
        attractor_adapter_dim=args.attractor_adapter_dim,
        batched_experts=bool(getattr(args, "batched_experts", False)),
        fixed_regime_group=int(getattr(args, "fixed_regime_group", -1)),
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16, 24])
    parser.add_argument("--stride", type=int, default=0,
                        help="rollout-start stride; 0 reuses the checkpoint rollout_steps")
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    tr = load_trainer(cli.trainer)
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    args = argparse.Namespace(**checkpoint["args"])
    # Explicitly preserve the stable integration used by the final checkpoint.
    args.max_integrator_dt = 0.5
    rollout_stride = int(cli.stride) if cli.stride > 0 else int(args.rollout_steps)
    arrays, _ = tr.build_arrays(args)
    snapshot_index = tr.load_snapshot_index(args.data_root / "pod_snapshot_index.csv")
    arrays["local_snapshot_index"] = snapshot_index["local_snapshot_index"].astype(np.int64)
    scalers = {
        key: tr.Standardizer(mean=np.asarray(value["mean"], dtype=np.float32),
                             scale=np.asarray(value["scale"], dtype=np.float32))
        for key, value in checkpoint["scalers"].items()
    }
    model = build_model(tr, arrays, args, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tensors = np.load(args.tensor_path)
    pressure_tensors = np.load(args.pressure_surrogate_path)
    vel_pod = np.load(args.data_root / "global_velocity_pod_area_weighted_l2.npz")
    pre_pod = np.load(args.data_root / "global_pressure_pod_area_weighted_l2.npz")
    phi_u = vel_pod["phi_uv"][:args.r_u].astype(np.float32)
    mean_u = vel_pod["mean_uv_regime"].astype(np.float32)
    phi_p = pre_pod["phi_p"][:args.r_p].astype(np.float32)
    mean_p = pre_pod["mean_p_regime"].astype(np.float32)
    volume = vel_pod["point_areas"].astype(np.float64)
    n_cells = len(volume)
    w_u = volume[None, :, None]
    w_p = volume[None, :]
    heldout = [str(x) for x in checkpoint["split_stats"]["holdout_labels"]]
    valid_samples = set(arrays["sample_ids"].tolist())
    result = {
        "checkpoint": str(cli.checkpoint), "epoch": int(checkpoint["epoch"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "integrator": args.integrator, "max_integrator_dt": args.max_integrator_dt,
        "external_dt": 4.0, "comparison": "full raw snapshots; pressure volume-mean removed per frame",
        "horizons": {},
    }
    for horizon in cli.horizons:
        aggregate = {key: [0.0, 0.0] for key in ("velocity", "pressure", "velocity_pod_floor", "pressure_pod_floor")}
        per_re = {}
        total_windows = 0
        nonfinite = 0
        for label in heldout:
            label_id = int(np.where(arrays["labels"].astype(str) == label)[0][0])
            ids = np.where(arrays["label_id"] == label_id)[0]
            ids = ids[np.argsort(arrays["time"][ids])]
            raw = np.load(raw_file(cli.raw_dir, label))
            raw_u = raw["U"].astype(np.float32)
            raw_p = pressure_gauge(raw["p"].astype(np.float32), volume).astype(np.float32)
            sums = {key: [0.0, 0.0] for key in aggregate}
            windows = 0
            for pos in range(0, len(ids) - horizon, rollout_stride):
                start = int(ids[pos])
                if start not in valid_samples:
                    continue
                chain = ids[pos:pos + horizon + 1]
                if not np.all(arrays["next_idx"][chain[:-1]] == chain[1:]):
                    continue
                a_state, b_state = arrays["a"][start].copy(), arrays["b"][start].copy()
                a_hist, b_hist, rhs_hist = tr.init_history_states_np(start, arrays)
                predicted_a, predicted_b = [], []
                finite = True
                for step in range(horizon):
                    cur, nxt = int(chain[step]), int(chain[step + 1])
                    dt = float(arrays["time"][nxt] - arrays["time"][cur])
                    a_state, b_state, rhs = tr.integrate_autonomous_step_np(
                        model, a_state, b_state, cur, dt, a_hist, b_hist, rhs_hist,
                        arrays, scalers, tensors, pressure_tensors, args, device)
                    if not (np.all(np.isfinite(a_state)) and np.all(np.isfinite(b_state))):
                        finite = False
                        break
                    predicted_a.append(a_state.copy()); predicted_b.append(b_state.copy())
                    a_hist = np.concatenate([a_state[None, None, :], a_hist[:, :-1, :]], axis=1)
                    b_hist = np.concatenate([b_state[None, None, :], b_hist[:, :-1, :]], axis=1)
                    rhs_hist = np.concatenate([rhs[None, None, :], rhs_hist[:, :-1, :]], axis=1)
                if not finite:
                    nonfinite += 1
                    continue
                local_idx = arrays["local_snapshot_index"][chain[1:]].astype(int)
                truth_u = raw_u[local_idx]
                truth_p = raw_p[local_idx]
                coeff_u = arrays["a"][chain[1:]]
                coeff_p = arrays["b"][chain[1:]]
                pred_u = (mean_u + np.asarray(predicted_a) @ phi_u).reshape(-1, n_cells, 2)
                pred_p = mean_p + np.asarray(predicted_b) @ phi_p
                oracle_u = (mean_u + coeff_u @ phi_u).reshape(-1, n_cells, 2)
                oracle_p = mean_p + coeff_p @ phi_p
                pred_p = pressure_gauge(pred_p, volume)
                oracle_p = pressure_gauge(oracle_p, volume)
                pairs = {"velocity": (pred_u - truth_u, truth_u, w_u),
                         "pressure": (pred_p - truth_p, truth_p, w_p),
                         "velocity_pod_floor": (oracle_u - truth_u, truth_u, w_u),
                         "pressure_pod_floor": (oracle_p - truth_p, truth_p, w_p)}
                for key, (diff, truth, weight) in pairs.items():
                    _, num, den = relative_l2(diff, truth, weight)
                    sums[key][0] += num; sums[key][1] += den
                    aggregate[key][0] += num; aggregate[key][1] += den
                windows += 1; total_windows += 1
            per_re[label] = {key: float(np.sqrt(val[0] / max(val[1], 1e-30))) for key, val in sums.items()}
            per_re[label]["windows"] = windows
        result["horizons"][f"K{horizon}"] = {
            "physical_time": float(horizon * 4.0), "rollout_start_stride": rollout_stride,
            "windows": total_windows,
            "nonfinite_windows": nonfinite,
            "aggregate": {key: float(np.sqrt(val[0] / max(val[1], 1e-30))) for key, val in aggregate.items()},
            "per_Re": per_re,
        }
        print(f"K={horizon}: {result['horizons'][f'K{horizon}']['aggregate']}", flush=True)
    out = cli.output_dir / "physical_field_multihorizon.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
