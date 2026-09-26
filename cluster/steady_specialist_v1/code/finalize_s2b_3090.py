from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


EPS = 1e-8
HORIZONS = (1, 4, 8, 16, 56, 128)
PERTURBATION_SCALES = (0.001, 0.005, 0.01, 0.02, 0.05)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trainer(path: Path):
    spec = importlib.util.spec_from_file_location("s2b_trainer_for_final_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_validation_history(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint_is_admissible(validation: dict) -> bool:
    return all(
        validation[f"k{k}"]["finite_fraction"] == 1.0
        and validation[f"k{k}"]["divergent_windows"] == 0
        for k in (1, 4, 8, 16)
    )


def selection_key(validation: dict) -> tuple[float, float, float, float]:
    report = validation["k16"]
    return (
        float(report["worst_pressure"]),
        float(report["worst_velocity"]),
        float(report["pressure_drift"]),
        float(report["fixed_point_residual"]),
    )


def select_checkpoint(checkpoint_dir: Path, output_dir: Path) -> tuple[Path, dict]:
    candidates = []
    for path in sorted(checkpoint_dir.glob("validation_step_*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        validation = checkpoint.get("validation", {})
        if not validation or "k16" not in validation:
            continue
        candidates.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "step": int(checkpoint["step"]),
                "admissible": checkpoint_is_admissible(validation),
                "selection_key": list(selection_key(validation)),
                "validation": validation,
            }
        )
    admissible = [item for item in candidates if item["admissible"]]
    if not admissible:
        raise RuntimeError("no finite, zero-divergence validation checkpoint is available")
    selected = min(admissible, key=lambda item: tuple(item["selection_key"]))
    source = Path(selected["path"])
    frozen = output_dir / "frozen_best_validation.pt"
    shutil.copy2(source, frozen)
    frozen.chmod(0o444)
    manifest = {
        "schema_version": 1,
        "selection_contract": {
            "admissibility": "K1/K4/K8/K16 finite_fraction=1 and divergent_windows=0",
            "ordered_key": [
                "K16 worst validation pressure relative L2",
                "K16 worst validation velocity relative L2",
                "K16 pressure drift",
                "K16 fixed-point residual",
            ],
            "contraction_ratio_used_for_selection": False,
            "final_step_weight_used_automatically": False,
        },
        "candidates": candidates,
        "selected_step": selected["step"],
        "selected_source": str(source),
        "selected_source_sha256": selected["sha256"],
        "frozen_checkpoint": str(frozen),
        "frozen_checkpoint_sha256": sha256(frozen),
    }
    atomic_json(output_dir / "checkpoint_selection_manifest.json", manifest)
    return frozen, manifest


def field_sums(coeff_pred: torch.Tensor, coeff_true: torch.Tensor, basis: torch.Tensor, mean: torch.Tensor, sqrt_weight: torch.Tensor) -> tuple[float, float]:
    pred = torch.matmul(coeff_pred, basis) + mean
    true = torch.matmul(coeff_true, basis) + mean
    difference = (pred - true) * sqrt_weight
    weighted_true = true * sqrt_weight
    return float(torch.sum(difference.float() ** 2).cpu()), float(torch.sum(weighted_true.float() ** 2).cpu())


def relative(num: float, den: float) -> float:
    return math.sqrt(num / max(den, EPS))


def evaluate_horizon(exp, split: str, k: int) -> dict:
    windows = exp.build_windows(k)[split]
    if not windows:
        return {"available": False, "reason": f"no {split} sequence has {k} successors", "k": k}
    velocity_basis = exp.cache["velocity_pod_basis"]
    pressure_basis = exp.cache["pressure_pod_basis"]
    velocity_mean = exp.cache["velocity_mean"]
    pressure_mean = exp.cache["pressure_mean"]
    with np.load(exp.paths["velocity_pod"]) as pod:
        sqrt_area = torch.as_tensor(pod["sqrt_point_areas"], device=exp.device)
    velocity_sqrt_weight = torch.cat((sqrt_area, sqrt_area))
    pressure_sqrt_weight = sqrt_area
    by_re = {}
    total_windows = finite_windows = divergent_windows = 0
    first_divergent = None
    exp.model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=exp.amp_dtype):
        for label, all_ids in windows.items():
            sums = {name: 0.0 for name in ("cu_n", "cu_d", "cp_n", "cp_d", "fu_n", "fu_d", "fp_n", "fp_d")}
            re_total = re_finite = re_divergent = 0
            re_first = None
            max_state_multiple = max_pressure_drift = max_fixed = 0.0
            for offset in range(0, len(all_ids), min(16, exp.e["validation_batch"])):
                ids = torch.as_tensor(all_ids[offset : offset + 16], dtype=torch.long, device=exp.device)
                out = exp.rollout(ids, k)
                pa, pb, ta, tb = [out[name].float() for name in ("pa", "pb", "ta", "tb")]
                batch = pa.shape[1]
                total_windows += batch
                re_total += batch
                finite = torch.isfinite(pa).all((0, 2)) & torch.isfinite(pb).all((0, 2))
                finite_count = int(finite.sum().cpu())
                finite_windows += finite_count
                re_finite += finite_count
                ra = torch.linalg.vector_norm(pa, dim=2) / (torch.linalg.vector_norm(ta, dim=2) + 1e-6)
                rb = torch.linalg.vector_norm(pb, dim=2) / (torch.linalg.vector_norm(tb, dim=2) + 1e-6)
                bad_steps = (~torch.isfinite(pa).all(2)) | (~torch.isfinite(pb).all(2)) | (ra > 20) | (rb > 20)
                bad_count = int(bad_steps.any(0).sum().cpu())
                divergent_windows += bad_count
                re_divergent += bad_count
                locations = torch.nonzero(bad_steps, as_tuple=False)
                if locations.numel():
                    first = int(locations[:, 0].min().cpu()) + 1
                    first_divergent = first if first_divergent is None else min(first_divergent, first)
                    re_first = first if re_first is None else min(re_first, first)
                sums["cu_n"] += float(torch.sum((pa - ta) ** 2).cpu())
                sums["cu_d"] += float(torch.sum(ta ** 2).cpu())
                sums["cp_n"] += float(torch.sum((pb - tb) ** 2).cpu())
                sums["cp_d"] += float(torch.sum(tb ** 2).cpu())
                value = field_sums(pa, ta, velocity_basis, velocity_mean, velocity_sqrt_weight)
                sums["fu_n"] += value[0]
                sums["fu_d"] += value[1]
                value = field_sums(pb, tb, pressure_basis, pressure_mean, pressure_sqrt_weight)
                sums["fp_n"] += value[0]
                sums["fp_d"] += value[1]
                a0, b0 = out["a0"].float(), out["b0"].float()
                initial_norm = torch.sqrt(torch.mean((a0 / exp.avt) ** 2, 1) + torch.mean((b0 / exp.bvt) ** 2, 1))
                state_norm = torch.sqrt(torch.mean((pa / exp.avt) ** 2, 2) + torch.mean((pb / exp.bvt) ** 2, 2))
                max_state_multiple = max(max_state_multiple, float(torch.max(state_norm / (initial_norm.unsqueeze(0) + EPS)).cpu()))
                drift = torch.linalg.vector_norm(pb[-1] - b0, dim=1) / (torch.linalg.vector_norm(b0, dim=1) + EPS)
                fixed = torch.sqrt(torch.sum((pb - b0.unsqueeze(0)) ** 2, dim=-1) / (torch.sum(b0 ** 2, dim=-1).unsqueeze(0) + EPS))
                max_pressure_drift = max(max_pressure_drift, float(drift.max().cpu()))
                max_fixed = max(max_fixed, float(fixed.max().cpu()))
            by_re[str(exp.a["labels"][label])] = {
                "window_count": re_total,
                "coefficient_space": {
                    "velocity_relative_l2": relative(sums["cu_n"], sums["cu_d"]),
                    "pressure_relative_l2": relative(sums["cp_n"], sums["cp_d"]),
                },
                "physical_reconstruction_area_weighted": {
                    "velocity_relative_l2": relative(sums["fu_n"], sums["fu_d"]),
                    "pressure_relative_l2": relative(sums["fp_n"], sums["fp_d"]),
                },
                "finite_fraction": re_finite / max(re_total, 1),
                "divergent_windows": re_divergent,
                "first_divergent_step": re_first,
                "state_norm_max_multiple": max_state_multiple,
                "pressure_drift": max_pressure_drift,
                "fixed_point_residual": max_fixed,
            }
    worst = {}
    for space in ("coefficient_space", "physical_reconstruction_area_weighted"):
        worst[space] = {
            field: max(item[space][field] for item in by_re.values())
            for field in ("velocity_relative_l2", "pressure_relative_l2")
        }
    worst.update(
        {
            "finite_fraction": finite_windows / max(total_windows, 1),
            "divergent_windows": divergent_windows,
            "first_divergent_step": first_divergent,
            "state_norm_max_multiple": max(item["state_norm_max_multiple"] for item in by_re.values()),
            "pressure_drift": max(item["pressure_drift"] for item in by_re.values()),
            "fixed_point_residual": max(item["fixed_point_residual"] for item in by_re.values()),
        }
    )
    return {"available": True, "k": k, "by_re": by_re, "overall_worst": worst}


def perturbation_evolution(exp, split: str) -> dict:
    max_k = 128 if exp.build_windows(128)[split] else 56
    windows = exp.build_windows(max_k)[split]
    if not windows:
        return {"available": False, "reason": "no 56-step heldout sequence"}
    result = {"available": True, "horizon": max_k, "definition": "distance between perturbed and unperturbed autonomous predictions, normalized by train coefficient standard deviations", "by_re": {}}
    signs_u = torch.where(torch.arange(exp.ru, device=exp.device) % 2 == 0, 1.0, -1.0)
    signs_p = -torch.where(torch.arange(exp.rp, device=exp.device) % 2 == 0, 1.0, -1.0)
    exp.model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=exp.amp_dtype):
        for label, ids in windows.items():
            start = torch.as_tensor(ids[:1], dtype=torch.long, device=exp.device)
            a0, b0 = exp.tensor("a", start), exp.tensor("b", start)
            base = exp.rollout(start, max_k, initial_a=a0, initial_b=b0)
            scales = {}
            for fraction in PERTURBATION_SCALES:
                da = fraction * exp.avt * signs_u
                db = fraction * exp.bvt * signs_p
                perturbed = exp.rollout(start, max_k, initial_a=a0 + da, initial_b=b0 + db)
                distance = torch.sqrt(
                    torch.mean(((perturbed["pa"] - base["pa"]) / exp.avt) ** 2, (1, 2))
                    + torch.mean(((perturbed["pb"] - base["pb"]) / exp.bvt) ** 2, (1, 2))
                )
                initial = torch.sqrt(torch.mean((da / exp.avt) ** 2) + torch.mean((db / exp.bvt) ** 2))
                ratio = distance / (initial + EPS)
                scales[f"{100*fraction:g}%"] = {
                    "normalized_error_by_step": distance.float().cpu().tolist(),
                    "amplification_ratio_by_step": ratio.float().cpu().tolist(),
                    "final_amplification_ratio": float(ratio[-1].cpu()),
                    "max_amplification_ratio": float(ratio.max().cpu()),
                    "finite_fraction": float(torch.isfinite(distance).float().mean().cpu()),
                }
            result["by_re"][str(exp.a["labels"][label])] = scales
    return result


def legacy_contraction(exp, split: str) -> dict:
    windows = exp.build_windows(16)[split]
    values = {}
    exp.model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=exp.amp_dtype):
        for label, ids in windows.items():
            start = torch.as_tensor(ids[:1], dtype=torch.long, device=exp.device)
            a0, b0 = exp.tensor("a", start), exp.tensor("b", start)
            signs = torch.where(torch.arange(exp.ru, device=exp.device) % 2 == 0, 1.0, -1.0)
            da, db = 0.01 * exp.avt * signs, -0.01 * exp.bvt * signs
            out = exp.rollout(start, 16, initial_a=a0 + da, initial_b=b0 + db)
            numerator = torch.sqrt(torch.mean(((out["pa"][-1] - a0) / exp.avt) ** 2) + torch.mean(((out["pb"][-1] - b0) / exp.bvt) ** 2))
            denominator = torch.sqrt(torch.mean((da / exp.avt) ** 2) + torch.mean((db / exp.bvt) ** 2))
            values[str(exp.a["labels"][label])] = float((numerator / denominator).float().cpu())
    return {"definition": "legacy fixed-reference controlled perturbation ratio at K16", "by_re": values, "worst": max(values.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-step", type=int, required=True)
    parser.add_argument("--stop-reason", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = read_validation_history(args.run_dir / "validation_history.jsonl")
    atomic_json(args.output_dir / "validation.json", {"history": history, "stop_step": args.stop_step, "stop_reason": args.stop_reason})
    frozen, selection = select_checkpoint(args.checkpoint_dir, args.output_dir)
    cfg = json.loads(args.config.read_text())
    trainer = load_trainer(args.trainer)
    cli = SimpleNamespace(run_dir=str(args.output_dir / "evaluation_runtime"), resume=None)
    exp = trainer.Experiment(cli, cfg)
    checkpoint = torch.load(frozen, map_location="cpu", weights_only=False)
    exp.model.load_state_dict(checkpoint["model"])
    exp.model.eval()
    if args.smoke_only:
        smoke = evaluate_horizon(exp, "validation", 1)
        print(json.dumps({"status": "PASS", "checkpoint_step": int(checkpoint["step"]), "validation_k1_available": smoke["available"]}, sort_keys=True))
        return
    marker = args.output_dir / "HELDOUT_EVALUATION_STARTED.json"
    if marker.exists():
        raise RuntimeError("heldout evaluation marker already exists; refusing a second final evaluation")
    atomic_json(marker, {"started_unix": time.time(), "checkpoint": str(frozen), "checkpoint_sha256": sha256(frozen)})
    heldout = {"checkpoint_step": int(checkpoint["step"]), "checkpoint_sha256": sha256(frozen), "horizons": {}}
    for k in HORIZONS:
        heldout["horizons"][f"k{k}"] = evaluate_horizon(exp, "heldout", k)
    heldout["controlled_perturbation_contraction_ratio"] = legacy_contraction(exp, "heldout")
    heldout["multi_scale_perturbation_evolution"] = perturbation_evolution(exp, "heldout")
    atomic_json(args.output_dir / "heldout_metrics.json", heldout)
    audit = trainer.startup_manifest(exp)
    fairness = {
        "schema_version": 1,
        "metric_contract": {
            "relative_errors_stored_as_fraction": True,
            "report_display_multiplier": 100,
            "coefficient_space": "Euclidean relative L2 over all predicted snapshots and retained coefficients",
            "physical_space": "area-weighted relative L2 after POD mean-plus-basis reconstruction",
            "divergence_threshold": "non-finite or predicted/true coefficient norm ratio > 20",
        },
        "current_run": {
            "initial_model_sha256": checkpoint["initial_model_sha256"],
            "selected_checkpoint_sha256": sha256(frozen),
            "selected_checkpoint_step": int(checkpoint["step"]),
            "stop_step": args.stop_step,
            "stop_reason": args.stop_reason,
            "asset_audit": audit,
        },
        "comparators": {
            "steady_v1": {"status": "PENDING_SAME-CONTRACT_ARTIFACT_DISCOVERY"},
            "original_a40_s2b": {"status": "PENDING_SAME-CONTRACT_ARTIFACT_DISCOVERY"},
        },
    }
    atomic_json(args.output_dir / "fairness_manifest.json", fairness)
    raw = {"selection": selection, "validation": history, "heldout": heldout, "fairness": fairness}
    atomic_json(args.output_dir / "raw_metrics.json", raw)
    atomic_json(args.output_dir / "HELDOUT_EVALUATION_DONE.json", {"finished_unix": time.time(), "checkpoint_step": int(checkpoint["step"]), "checkpoint_sha256": sha256(frozen)})


if __name__ == "__main__":
    main()
