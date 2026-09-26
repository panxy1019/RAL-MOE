from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


EPS = 1e-8
HORIZONS = (1, 4, 8, 16, 56)
MODELS = ("S2-B", "S3-A", "S3-B", "S3-C")
METRICS = (
    "normalized_state",
    "normalized_velocity",
    "normalized_pressure",
    "raw_state",
    "raw_velocity",
    "raw_pressure",
    "physical_velocity",
    "physical_pressure_gauged",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def function_locations(path: Path, names: list[str]) -> list[dict]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found[node.name] = {"function": node.name, "start_line": node.lineno, "end_line": node.end_lineno}
    return [{"file": str(path), **found[name]} for name in names if name in found]


def tensor_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.float(), dim=-1)


def rms_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value.float() ** 2, dim=-1))


def gauged_pressure(value: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    mean = torch.sum(value * area, dim=-1, keepdim=True) / torch.sum(area)
    return value - mean


def weighted_norm(value: torch.Tensor, sqrt_weight: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.float() * sqrt_weight, dim=-1)


def safe_metric(initial: torch.Tensor, distances: torch.Tensor) -> dict:
    initial_value = float(initial.float().cpu())
    terminal = float(distances[-1].float().cpu()) if torch.isfinite(distances[-1]) else None
    finite = bool(torch.isfinite(distances).all().cpu())
    if initial_value <= 1e-30 or not math.isfinite(initial_value):
        terminal_gain = path_max = None
        denominator_zero = True
    else:
        gain = distances / initial
        terminal_gain = float(gain[-1].float().cpu()) if torch.isfinite(gain[-1]) else None
        path_max = float(gain[torch.isfinite(gain)].max().float().cpu()) if torch.isfinite(gain).any() else None
        denominator_zero = False
    return {
        "initial_actual_norm": initial_value,
        "terminal_distance": terminal,
        "gain_terminal": terminal_gain,
        "gain_path_max_1_to_k": path_max,
        "distance_finite_all_steps": finite,
        "denominator_zero": denominator_zero,
    }


def quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "worst": float(np.max(array)),
    }


def summarize_records(records: list[dict]) -> tuple[dict, dict, dict]:
    summary, grouped, denominators = {}, {}, {}
    for model in MODELS:
        model_rows = [row for row in records if row["model"] == model]
        summary[model], grouped[model] = {}, {}
        for k in HORIZONS:
            rows = [row for row in model_rows if row["horizon"] == k]
            summary[model][f"k{k}"] = {}
            for metric in METRICS:
                valid = [row for row in rows if row["metrics"][metric]["gain_terminal"] is not None]
                values = [row["metrics"][metric]["gain_terminal"] for row in valid]
                if not values:
                    summary[model][f"k{k}"][metric] = {"count": 0}
                    continue
                stats = quantiles(values)
                worst_row = max(valid, key=lambda row: row["metrics"][metric]["gain_terminal"])
                stats["worst_case"] = {
                    key: worst_row[key]
                    for key in ("re", "scale", "direction_id", "direction_category", "window_id")
                }
                summary[model][f"k{k}"][metric] = stats
            grouped[model][f"k{k}"] = {}
            for re_name in sorted(set(row["re"] for row in rows)):
                grouped[model][f"k{k}"][re_name] = {}
                for scale in sorted(set(row["scale"] for row in rows if row["re"] == re_name)):
                    cell = [row for row in rows if row["re"] == re_name and row["scale"] == scale]
                    grouped[model][f"k{k}"][re_name][str(scale)] = {}
                    for metric in METRICS:
                        values = [row["metrics"][metric]["gain_terminal"] for row in cell if row["metrics"][metric]["gain_terminal"] is not None]
                        grouped[model][f"k{k}"][re_name][str(scale)][metric] = quantiles(values) if values else {"count": 0}
    for metric in METRICS:
        items = [
            row["metrics"][metric]["initial_actual_norm"]
            for row in records
            if row["model"] == "S2-B" and row["horizon"] == 1 and not row["metrics"][metric]["denominator_zero"]
        ]
        denominators[metric] = quantiles(items) if items else {"count": 0}
    return summary, grouped, denominators


def agreement(records: list[dict], left: str, right: str) -> dict:
    relative_differences = []
    ratios = []
    for row in records:
        a = row["metrics"][left]["gain_terminal"]
        b = row["metrics"][right]["gain_terminal"]
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            continue
        relative_differences.append(abs(a - b) / max(abs(a), abs(b), 1e-30))
        ratios.append(a / max(b, 1e-30))
    return {"relative_difference": quantiles(relative_differences), "ratio_left_over_right": quantiles(ratios)}


def scaler_stats(value: torch.Tensor) -> dict:
    array = value.detach().float().cpu().numpy()
    order = np.argsort(array)
    return {
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "max_over_min": float(array.max() / array.min()),
        "five_smallest_modes_zero_based": [{"mode": int(i), "std": float(array[i])} for i in order[:5]],
    }


def basis_audit(s3) -> dict:
    exp = s3.exp
    velocity_basis = exp.cache["velocity_pod_basis"].float()
    pressure_basis = exp.cache["pressure_pod_basis"].float()
    with np.load(exp.paths["velocity_pod"]) as velocity:
        sqrt_area = torch.as_tensor(velocity["sqrt_point_areas"], dtype=torch.float32, device=exp.device)
    velocity_weight = torch.cat((sqrt_area, sqrt_area))
    area = sqrt_area ** 2
    velocity_gram = (velocity_basis * velocity_weight.square()) @ velocity_basis.T
    pressure_gram = (pressure_basis * area) @ pressure_basis.T
    identity_u = torch.eye(exp.ru, device=exp.device)
    identity_p = torch.eye(exp.rp, device=exp.device)
    pressure_area_means = torch.sum(pressure_basis * area, dim=1) / torch.sum(area)
    return {
        "velocity_weighted_gram_max_abs_error": float(torch.max(torch.abs(velocity_gram - identity_u)).cpu()),
        "pressure_weighted_gram_max_abs_error": float(torch.max(torch.abs(pressure_gram - identity_p)).cpu()),
        "pressure_basis_area_mean_max_abs": float(torch.max(torch.abs(pressure_area_means)).cpu()),
        "pressure_gauge": "subtract_area_mean_per_snapshot",
    }


def repair_terminal_dt(s3, fixed_ids: np.ndarray) -> None:
    next_idx = np.asarray(s3.exp.a["next_idx"])
    source_dt = np.asarray(s3.exp.a["dt_next"])
    for fixed_id in fixed_ids.tolist():
        if torch.isfinite(s3.dt[fixed_id]):
            continue
        predecessors = np.flatnonzero(next_idx == fixed_id)
        if len(predecessors):
            value = float(source_dt[predecessors[-1]])
        else:
            label = s3.exp.a["label_id"][fixed_id]
            value = float(np.nanmedian(source_dt[s3.exp.a["label_id"] == label]))
        s3.dt[fixed_id] = value
    ids = torch.as_tensor(fixed_ids, device=s3.exp.device)
    if not torch.isfinite(s3.dt[ids]).all():
        raise AssertionError("non-finite fixed-point dt")


def calculate_metrics(s3, clean_a, clean_b, pert_a, pert_b, da, db, horizons, area, velocity_weight):
    exp = s3.exp
    velocity_basis = exp.cache["velocity_pod_basis"].float()
    pressure_basis = exp.cache["pressure_pod_basis"].float()
    initial = {
        "normalized_state": torch.sqrt(torch.mean((da / exp.avt) ** 2, 1) + torch.mean((db / exp.bvt) ** 2, 1)),
        "normalized_velocity": rms_norm(da / exp.avt),
        "normalized_pressure": rms_norm(db / exp.bvt),
        "raw_state": torch.sqrt(torch.sum(da.float() ** 2, 1) + torch.sum(db.float() ** 2, 1)),
        "raw_velocity": tensor_norm(da),
        "raw_pressure": tensor_norm(db),
    }
    initial_velocity_field = da.float() @ velocity_basis
    initial_pressure_field = gauged_pressure(db.float() @ pressure_basis, area)
    initial["physical_velocity"] = weighted_norm(initial_velocity_field, velocity_weight)
    initial["physical_pressure_gauged"] = weighted_norm(initial_pressure_field, torch.sqrt(area))
    result = []
    delta_a = pert_a.float() - clean_a.float()
    delta_b = pert_b.float() - clean_b.float()
    distances = {
        "normalized_state": torch.sqrt(torch.mean((delta_a / exp.avt) ** 2, 2) + torch.mean((delta_b / exp.bvt) ** 2, 2)),
        "normalized_velocity": rms_norm(delta_a / exp.avt),
        "normalized_pressure": rms_norm(delta_b / exp.bvt),
        "raw_state": torch.sqrt(torch.sum(delta_a ** 2, 2) + torch.sum(delta_b ** 2, 2)),
        "raw_velocity": tensor_norm(delta_a),
        "raw_pressure": tensor_norm(delta_b),
    }
    velocity_field = torch.matmul(delta_a, velocity_basis)
    pressure_field = gauged_pressure(torch.matmul(delta_b, pressure_basis), area)
    distances["physical_velocity"] = weighted_norm(velocity_field, velocity_weight)
    distances["physical_pressure_gauged"] = weighted_norm(pressure_field, torch.sqrt(area))
    for sample in range(da.shape[0]):
        sample_metrics = {}
        for metric in METRICS:
            sample_metrics[metric] = {}
            for k in horizons:
                sample_metrics[metric][k] = safe_metric(initial[metric][sample], distances[metric][:k, sample])
        legacy_den = torch.sqrt(torch.mean((da[sample] / exp.avt) ** 2) + torch.mean((db[sample] / exp.bvt) ** 2) + EPS)
        legacy_num = torch.sqrt(torch.mean((delta_a[:, sample] / exp.avt) ** 2, 1) + torch.mean((delta_b[:, sample] / exp.bvt) ** 2, 1) + EPS)
        state0 = torch.sqrt(torch.mean((clean_a[0, sample] / exp.avt) ** 2) + torch.mean((clean_b[0, sample] / exp.bvt) ** 2) + EPS)
        state_clean = torch.sqrt(torch.mean((clean_a[:, sample] / exp.avt) ** 2, 1) + torch.mean((clean_b[:, sample] / exp.bvt) ** 2, 1) + EPS)
        state_pert = torch.sqrt(torch.mean((pert_a[:, sample] / exp.avt) ** 2, 1) + torch.mean((pert_b[:, sample] / exp.bvt) ** 2, 1) + EPS)
        abnormal = (~torch.isfinite(state_clean)) | (~torch.isfinite(state_pert)) | (state_clean / state0 > 20) | (state_pert / state0 > 20)
        location = torch.nonzero(abnormal, as_tuple=False)
        top_modes = {}
        for k in horizons:
            cu = ((delta_a[k - 1, sample] / exp.avt) ** 2 / exp.ru).float()
            cp = ((delta_b[k - 1, sample] / exp.bvt) ** 2 / exp.rp).float()
            combined = torch.cat((cu, cp))
            total = torch.sum(combined)
            values, indices = torch.topk(combined, k=min(5, combined.numel()))
            top_modes[k] = [
                {
                    "channel": "velocity" if int(index) < exp.ru else "pressure",
                    "mode_zero_based": int(index) if int(index) < exp.ru else int(index) - exp.ru,
                    "squared_contribution": float(value.cpu()),
                    "fraction_of_normalized_distance_squared": float((value / (total + 1e-30)).cpu()),
                }
                for value, index in zip(values, indices)
            ]
        result.append({
            "metrics": sample_metrics,
            "legacy_eps_gain_by_step": (legacy_num / legacy_den).float().cpu().tolist(),
            "normalized_terminal_top_modes": top_modes,
            "finite_all_steps": bool((torch.isfinite(clean_a[:, sample]).all() & torch.isfinite(clean_b[:, sample]).all() & torch.isfinite(pert_a[:, sample]).all() & torch.isfinite(pert_b[:, sample]).all()).cpu()),
            "first_abnormal_step": int(location[0, 0].cpu()) + 1 if location.numel() else None,
        })
    return result


def flatten_sample(sample_metrics: dict, metadata: dict, model: str, k: int) -> dict:
    return {
        "model": model,
        "horizon": k,
        **metadata,
        "finite_all_steps_to_horizon": sample_metrics["finite_all_steps"],
        "first_abnormal_step": sample_metrics["first_abnormal_step"] if sample_metrics["first_abnormal_step"] and sample_metrics["first_abnormal_step"] <= k else None,
        "legacy_normalized_gain_with_eps_terminal": sample_metrics["legacy_eps_gain_by_step"][k - 1],
        "normalized_terminal_top_modes": sample_metrics["normalized_terminal_top_modes"][k],
        "metrics": {metric: sample_metrics["metrics"][metric][k] for metric in METRICS},
    }


def top_mode_diagnostics(s3, records: list[dict]) -> dict:
    result = {}
    for model in sorted(set(row["model"] for row in records), key=MODELS.index):
        rows = [row for row in records if row["model"] == model and row["horizon"] == 16]
        worst = max(rows, key=lambda row: row["metrics"]["normalized_state"]["gain_terminal"])
        result[model] = {
            "worst_case": {key: worst[key] for key in ("re", "scale", "direction_id", "direction_category", "window_id")},
            "normalized_state_gain": worst["metrics"]["normalized_state"]["gain_terminal"],
            "raw_velocity_gain": worst["metrics"]["raw_velocity"]["gain_terminal"],
            "raw_pressure_gain": worst["metrics"]["raw_pressure"]["gain_terminal"],
            "physical_velocity_gain": worst["metrics"]["physical_velocity"]["gain_terminal"],
            "physical_pressure_gain": worst["metrics"]["physical_pressure_gauged"]["gain_terminal"],
            "normalized_terminal_top_modes": worst["normalized_terminal_top_modes"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--s3-trainer", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--s3-run-root", type=Path, required=True)
    parser.add_argument("--training-bank", type=Path, required=True)
    parser.add_argument("--heldout-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = args.output_dir / "AUDIT_DONE.json"
    if done.exists():
        raise RuntimeError("audit already completed; refusing to overwrite")
    bank_sha_before = sha256(args.heldout_bank)
    with np.load(args.heldout_bank) as loaded:
        bank = {key: loaded[key] for key in loaded.files}
    checkpoint_paths = {
        "S2-B": args.source_checkpoint,
        "S3-A": args.s3_run_root / "S3-A/checkpoints/best_contraction.pt",
        "S3-B": args.s3_run_root / "S3-B/checkpoints/best_contraction.pt",
        "S3-C": args.s3_run_root / "S3-C/checkpoints/best_contraction.pt",
    }
    checkpoint_audit = {name: {"path": str(path), "sha256": sha256(path)} for name, path in checkpoint_paths.items()}
    atomic_json(args.output_dir / "AUDIT_STARTED.json", {
        "started_unix": time.time(), "read_only_inputs": True, "heldout_bank_sha256": bank_sha_before,
        "checkpoints": checkpoint_audit,
    })
    s3_module = load_module("metric_audit_s3", args.s3_trainer)
    init_args = SimpleNamespace(
        experiment="S3-A", run_dir=str(args.output_dir / "runtime"), config=str(args.config), trainer=str(args.trainer),
        finalizer=str(args.finalizer), checkpoint=str(args.source_checkpoint), bank=str(args.training_bank),
        validation_lock=str(args.output_dir / "unused.lock"), learning_rate=args.learning_rate,
    )
    s3 = s3_module.S3(init_args)
    s3.init_eval_assets()
    fixed_ids = np.unique(bank["fixed_id"].astype(np.int64))
    repair_terminal_dt(s3, fixed_ids)
    with np.load(s3.exp.paths["velocity_pod"]) as velocity:
        sqrt_area = torch.as_tensor(velocity["sqrt_point_areas"], dtype=torch.float32, device=s3.exp.device)
    area = sqrt_area.square()
    velocity_weight = torch.cat((sqrt_area, sqrt_area))
    records = []
    model_limit = 1 if args.smoke else len(MODELS)
    group_limit = 1 if args.smoke else None
    for model in MODELS[:model_limit]:
        checkpoint = torch.load(checkpoint_paths[model], map_location="cpu", weights_only=False)
        s3.exp.model.load_state_dict(checkpoint["model"], strict=True)
        s3.exp.model.eval()
        groups = []
        for re_name in sorted(set(bank["re_name"].astype(str).tolist())):
            for scale in sorted(set(bank["scale"][bank["re_name"].astype(str) == re_name].astype(float).tolist())):
                groups.append((re_name, scale))
        if group_limit is not None:
            groups = groups[:group_limit]
        for re_name, scale in groups:
            mask = (bank["re_name"].astype(str) == re_name) & np.isclose(bank["scale"].astype(float), scale)
            fixed_values = bank["fixed_id"][mask].astype(np.int64)
            if len(set(fixed_values.tolist())) != 1:
                raise AssertionError("a Re/scale group must share one start window")
            fixed_id = int(fixed_values[0])
            count = int(mask.sum())
            fixed = torch.full((count,), fixed_id, dtype=torch.long, device=s3.exp.device)
            a0, b0 = s3.exp.tensor("a", fixed), s3.exp.tensor("b", fixed)
            du = torch.as_tensor(bank["du"][mask], dtype=torch.float32, device=s3.exp.device)
            dp = torch.as_tensor(bank["dp"][mask], dtype=torch.float32, device=s3.exp.device)
            da, db = float(scale) * s3.exp.avt * du, float(scale) * s3.exp.bvt * dp
            with torch.inference_mode(), torch.autocast("cuda", dtype=s3.exp.amp_dtype):
                clean_a, clean_b = s3.free_rollout(fixed, a0, b0, 56)
                pert_a, pert_b = s3.free_rollout(fixed, a0 + da, b0 + db, 56)
            calculated = calculate_metrics(s3, clean_a, clean_b, pert_a, pert_b, da, db, HORIZONS, area, velocity_weight)
            categories = bank["category"][mask].astype(str)
            original_indices = np.flatnonzero(mask)
            for local_id, sample in enumerate(calculated):
                metadata = {
                    "re": re_name,
                    "scale": float(scale),
                    "direction_id": int(local_id),
                    "bank_entry_id": int(original_indices[local_id]),
                    "direction_category": str(categories[local_id]),
                    "window_id": fixed_id,
                }
                for k in HORIZONS:
                    records.append(flatten_sample(sample, metadata, model, k))
        print(json.dumps({"event": "model_audit_complete", "model": model, "record_count": len(records)}), flush=True)
    summary, grouped, denominator_stats = summarize_records(records)
    metric_audit = {
        "schema_version": 1,
        "primary_aggregation": "terminal gain at K",
        "secondary_aggregation": "maximum strict gain over steps 1..K",
        "epsilon_policy": {
            "strict_recomputation": "no epsilon in numerator or denominator; zero same-space denominator is reported undefined",
            "legacy_reproduction": "EPS=1e-8 inside both normalized numerator and denominator square roots",
        },
        "checkpoint_audit": checkpoint_audit,
        "heldout_bank": {"path": str(args.heldout_bank), "sha256_before": bank_sha_before, "sha256_after": sha256(args.heldout_bank)},
        "scalers": {"velocity_coeff_std": scaler_stats(s3.exp.avt), "pressure_coeff_std": scaler_stats(s3.exp.bvt), "fit_split": "train", "floor": 1e-7},
        "basis_and_gauge": basis_audit(s3),
        "records": records,
        "summary": summary,
        "grouped_summary": grouped,
        "denominator_distribution": denominator_stats,
        "agreement": {
            "raw_velocity_vs_physical_velocity": agreement(records, "raw_velocity", "physical_velocity"),
            "raw_pressure_vs_gauged_physical_pressure": agreement(records, "raw_pressure", "physical_pressure_gauged"),
            "normalized_velocity_vs_raw_velocity": agreement(records, "normalized_velocity", "raw_velocity"),
            "normalized_pressure_vs_raw_pressure": agreement(records, "normalized_pressure", "raw_pressure"),
        },
        "worst_k16_diagnostics": top_mode_diagnostics(s3, records),
    }
    if bank_sha_before != sha256(args.heldout_bank):
        raise AssertionError("heldout perturbation bank changed during read-only audit")
    output = args.output_dir / ("smoke_metrics.json" if args.smoke else "paired_gain_metric_audit_raw.json")
    atomic_json(output, metric_audit)
    if args.smoke:
        print(json.dumps({"status": "PASS", "records": len(records), "output": str(output)}), flush=True)
        return
    locations = []
    locations += function_locations(args.s3_trainer, ["free_rollout", "fixed_inputs", "pair_loss", "fixed_validation"])
    locations += function_locations(args.trainer, ["rollout", "controlled_contraction"])
    locations += function_locations(args.finalizer, ["perturbation_evolution", "legacy_contraction"])
    locations += function_locations(args.vendor, ["make_features_torch", "make_history_features_from_states_torch", "build_arrays"])
    formulas = {
        "current_s3_terminal_normalized_gain": "sqrt(mean_i(((aK_pert-aK_clean)/sigma_a_i)^2)+mean_j(((bK_pert-bK_clean)/sigma_b_j)^2)+EPS) / sqrt(mean_i((delta_a0/sigma_a_i)^2)+mean_j((delta_b0/sigma_b_j)^2)+EPS)",
        "strict_normalized_gain": "same expression without EPS, using the actual nonzero initial norm",
        "strict_raw_velocity_gain": "||aK_pert-aK_clean||_2 / ||delta_a0||_2",
        "strict_raw_pressure_gain": "||bK_pert-bK_clean||_2 / ||delta_b0||_2",
        "strict_physical_velocity_gain": "||(aK_pert-aK_clean) Phi_u||_M_u / ||delta_a0 Phi_u||_M_u",
        "strict_physical_pressure_gain": "||gauge((bK_pert-bK_clean) Phi_p)||_M_p / ||gauge(delta_b0 Phi_p)||_M_p",
    }
    atomic_json(args.output_dir / "formula_and_code_locations.json", {"formulas": formulas, "locations": locations})
    atomic_json(done, {
        "finished_unix": time.time(), "status": "PASS", "record_count": len(records),
        "heldout_bank_sha256": bank_sha_before, "raw_metrics_sha256": sha256(output),
    })
    del s3
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
