from __future__ import annotations

import argparse
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
HORIZONS = (1, 4, 8, 16, 56, 128)
PERTURBATION_SCALES = (0.001, 0.005, 0.01, 0.02, 0.05)
EXPERIMENTS = ("S3-A", "S3-B", "S3-C")


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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalized(u: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.sqrt(np.mean(u * u) + np.mean(p * p))
    return u / max(norm, 1e-12), p / max(norm, 1e-12)


def directions(rng: np.random.Generator, ru: int, rp: int):
    result = []
    for _ in range(2):
        result.append(("velocity_only", *normalized(rng.normal(size=ru), np.zeros(rp))))
    for _ in range(7):
        result.append(("pressure_only", *normalized(np.zeros(ru), rng.normal(size=rp))))
    for _ in range(6):
        result.append(("joint", *normalized(rng.normal(size=ru), rng.normal(size=rp))))
    for mode in (0, 1):
        u, p = np.zeros(ru), np.zeros(rp)
        (u if mode == 0 else p)[mode] = 1.0
        result.append(("leading_pod", *normalized(u, p)))
    for _ in range(3):
        result.append(("random_normalized", *normalized(rng.normal(size=ru), rng.normal(size=rp))))
    return result


def heldout_fixed_points(exp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(exp.paths["velocity_pod"]) as pod:
        split = pod["snapshot_splits"].astype(str)
    labels, fixed = [], []
    for label in sorted(set(exp.a["label_id"][split == "heldout"].tolist())):
        ids = np.flatnonzero((exp.a["label_id"] == label) & (split == "heldout"))
        terminal = ids[exp.a["next_idx"][ids] < 0]
        labels.append(label)
        fixed.append(int(terminal[-1] if len(terminal) else ids[-1]))
    if len(labels) != 4:
        raise AssertionError(f"expected four heldout Re, found {len(labels)}")
    names = np.asarray([str(exp.a["labels"][label]) for label in labels])
    return np.asarray(labels, np.int64), np.asarray(fixed, np.int64), names


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
        raise AssertionError("non-finite heldout fixed-point dt")


def build_heldout_bank(exp, seed: int, output: Path) -> dict[str, np.ndarray]:
    labels, fixed_ids, names = heldout_fixed_points(exp)
    rng = np.random.default_rng(seed)
    out = {key: [] for key in ("label_id", "fixed_id", "re_name", "scale", "category", "du", "dp")}
    for label, fixed_id, name in zip(labels.tolist(), fixed_ids.tolist(), names.tolist()):
        for scale in PERTURBATION_SCALES:
            for category, du, dp in directions(rng, exp.ru, exp.rp):
                out["label_id"].append(label)
                out["fixed_id"].append(fixed_id)
                out["re_name"].append(name)
                out["scale"].append(scale)
                out["category"].append(category)
                out["du"].append(du)
                out["dp"].append(dp)
    bank = {
        key: np.asarray(value, dtype=np.float32 if key in ("scale", "du", "dp") else None)
        for key, value in out.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **bank)
    return bank


def mean_worst(values: torch.Tensor) -> dict[str, float]:
    finite = torch.isfinite(values)
    return {
        "mean": float(values[finite].mean().cpu()) if finite.any() else math.inf,
        "worst": float(values[finite].max().cpu()) if finite.any() else math.inf,
        "finite_fraction": float(finite.float().mean().cpu()),
    }


def physical_relative(finalizer, pred, true, basis, mean, weight) -> float:
    numerator, denominator = finalizer.field_sums(pred, true, basis, mean, weight)
    return math.sqrt(numerator / max(denominator, EPS))


def evaluate_attractivity(s3, bank: dict[str, np.ndarray], max_k: int) -> dict:
    exp = s3.exp
    selected_horizons = [k for k in HORIZONS if k <= max_k]
    result = {
        "available": True,
        "max_horizon": max_k,
        "definition": {
            "paired_gain": "distance between perturbed and clean autonomous trajectories divided by initial normalized perturbation",
            "recovery_to_truth": "distance between perturbed trajectory and heldout terminal CFD state divided by initial normalized perturbation",
            "fixed_point_residual": "clean autonomous trajectory relative to heldout terminal CFD state",
        },
        "by_re": {},
    }
    device = exp.device
    for re_name in sorted(set(bank["re_name"].tolist())):
        re_mask = bank["re_name"] == re_name
        fixed_id = int(bank["fixed_id"][np.flatnonzero(re_mask)[0]])
        fixed = torch.tensor([fixed_id], dtype=torch.long, device=device)
        a_star, b_star = exp.tensor("a", fixed), exp.tensor("b", fixed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=exp.amp_dtype):
            clean_a, clean_b = s3.free_rollout(fixed, a_star, b_star, max_k)
        fixed_metrics = {}
        for k in selected_horizons:
            pa, pb = clean_a[k - 1 : k], clean_b[k - 1 : k]
            modal_u = float((torch.linalg.vector_norm(pa - a_star) / (torch.linalg.vector_norm(a_star) + EPS)).cpu())
            modal_p = float((torch.linalg.vector_norm(pb - b_star) / (torch.linalg.vector_norm(b_star) + EPS)).cpu())
            field_u = physical_relative(s3.finalizer, pa, a_star[None], exp.cache["velocity_pod_basis"], exp.cache["velocity_mean"], s3.velocity_weight)
            field_p = physical_relative(s3.finalizer, pb, b_star[None], exp.cache["pressure_pod_basis"], exp.cache["pressure_mean"], s3.pressure_weight)
            fixed_metrics[f"k{k}"] = {
                "coefficient_space": {"velocity_relative_l2": modal_u, "pressure_relative_l2": modal_p},
                "physical_reconstruction_area_weighted": {"velocity_relative_l2": field_u, "pressure_relative_l2": field_p},
            }
        scales = {}
        for scale in PERTURBATION_SCALES:
            mask = re_mask & np.isclose(bank["scale"], scale)
            du = torch.as_tensor(bank["du"][mask], dtype=torch.float32, device=device)
            dp = torch.as_tensor(bank["dp"][mask], dtype=torch.float32, device=device)
            categories = bank["category"][mask].astype(str)
            count = len(du)
            fixed_batch = fixed.repeat(count)
            aa, bb = a_star.repeat(count, 1), b_star.repeat(count, 1)
            da, db = scale * exp.avt * du, scale * exp.bvt * dp
            with torch.inference_mode(), torch.autocast("cuda", dtype=exp.amp_dtype):
                clean_pa, clean_pb = s3.free_rollout(fixed_batch, aa, bb, max_k)
                pert_pa, pert_pb = s3.free_rollout(fixed_batch, aa + da, bb + db, max_k)
            clean_pa, clean_pb, pert_pa, pert_pb = [x.float() for x in (clean_pa, clean_pb, pert_pa, pert_pb)]
            initial = torch.sqrt(torch.mean((da / exp.avt) ** 2, 1) + torch.mean((db / exp.bvt) ** 2, 1) + EPS)
            paired = torch.sqrt(torch.mean(((pert_pa - clean_pa) / exp.avt) ** 2, 2) + torch.mean(((pert_pb - clean_pb) / exp.bvt) ** 2, 2) + EPS) / initial
            recovery = torch.sqrt(torch.mean(((pert_pa - aa) / exp.avt) ** 2, 2) + torch.mean(((pert_pb - bb) / exp.bvt) ** 2, 2) + EPS) / initial
            initial_state = torch.sqrt(torch.mean((aa / exp.avt) ** 2, 1) + torch.mean((bb / exp.bvt) ** 2, 1) + EPS)
            state_multiple = torch.sqrt(torch.mean((pert_pa / exp.avt) ** 2, 2) + torch.mean((pert_pb / exp.bvt) ** 2, 2) + EPS) / initial_state
            finite = torch.isfinite(pert_pa).all(2) & torch.isfinite(pert_pb).all(2)
            divergent = (~finite) | (state_multiple > 20)
            by_horizon = {}
            for k in selected_horizons:
                by_category = {}
                for category in sorted(set(categories.tolist())):
                    category_mask = torch.as_tensor(categories == category, device=device)
                    by_category[category] = {
                        "paired_gain": mean_worst(paired[k - 1, category_mask]),
                        "recovery_to_truth": mean_worst(recovery[k - 1, category_mask]),
                    }
                by_horizon[f"k{k}"] = {
                    "paired_gain": mean_worst(paired[k - 1]),
                    "recovery_to_truth": mean_worst(recovery[k - 1]),
                    "by_category": by_category,
                }
            first_bad = torch.nonzero(divergent, as_tuple=False)
            scales[f"{100 * scale:g}%"] = {
                "sample_count": count,
                "by_horizon": by_horizon,
                "paired_gain_worst_by_step": paired.max(1).values.cpu().tolist(),
                "recovery_to_truth_worst_by_step": recovery.max(1).values.cpu().tolist(),
                "finite_fraction": float(finite.float().mean().cpu()),
                "divergent_windows": int(divergent.any(0).sum().cpu()),
                "first_divergent_step": int(first_bad[:, 0].min().cpu()) + 1 if first_bad.numel() else None,
                "state_norm_max_multiple": float(state_multiple.max().cpu()),
            }
        result["by_re"][re_name] = {"fixed_point_residual": fixed_metrics, "perturbations": scales}
    summary = {}
    for k in selected_horizons:
        paired_values, recovery_values, fixed_u, fixed_p = [], [], [], []
        finite, divergent, multiples = [], [], []
        for re_item in result["by_re"].values():
            fp = re_item["fixed_point_residual"][f"k{k}"]["physical_reconstruction_area_weighted"]
            fixed_u.append(fp["velocity_relative_l2"])
            fixed_p.append(fp["pressure_relative_l2"])
            for scale, scale_item in re_item["perturbations"].items():
                if scale == "0.1%" and k == 16:
                    pass
                metric = scale_item["by_horizon"][f"k{k}"]
                paired_values.append(metric["paired_gain"]["worst"])
                recovery_values.append(metric["recovery_to_truth"]["worst"])
                finite.append(scale_item["finite_fraction"])
                divergent.append(scale_item["divergent_windows"])
                multiples.append(scale_item["state_norm_max_multiple"])
        summary[f"k{k}"] = {
            "fixed_point_physical_velocity_worst": max(fixed_u),
            "fixed_point_physical_pressure_worst": max(fixed_p),
            "paired_gain_worst_all_scales": max(paired_values),
            "recovery_to_truth_worst_all_scales": max(recovery_values),
            "finite_fraction_min": min(finite),
            "divergent_windows_sum": sum(divergent),
            "state_norm_max_multiple": max(multiples),
        }
    result["overall_worst"] = summary
    result["qualification_k16"] = {
        "scales": ["0.5%", "1%", "2%", "5%"],
        "paired_gain_worst": max(
            re_item["perturbations"][scale]["by_horizon"]["k16"]["paired_gain"]["worst"]
            for re_item in result["by_re"].values()
            for scale in ("0.5%", "1%", "2%", "5%")
        ),
        "fixed_point_physical_pressure_worst": max(
            re_item["fixed_point_residual"]["k16"]["physical_reconstruction_area_weighted"]["pressure_relative_l2"]
            for re_item in result["by_re"].values()
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--s3-trainer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--training-bank", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = args.output_dir / "HELDOUT_EVALUATION_DONE.json"
    if done.exists():
        raise RuntimeError("heldout has already been evaluated; refusing to run twice")
    selection = json.loads(args.selection_manifest.read_text())
    selected = {}
    for experiment in EXPERIMENTS:
        item = selection["experiments"][experiment]
        if item["status"] != "NO_QUALIFIED_CHECKPOINT":
            raise AssertionError("this evaluator expects the preregistered no-qualified fallback")
        checkpoint = args.run_root / experiment / "checkpoints" / "best_contraction.pt"
        selected[experiment] = {
            "selection_role": "relative_best_diagnostic_checkpoint",
            "selection_metric": "validation paired contraction gain",
            "step": int(item["best_contraction_step"]),
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
        }
    atomic_json(args.output_dir / "HELDOUT_EVALUATION_STARTED.json", {
        "started_unix": time.time(),
        "selection_completed_before_heldout": True,
        "selected_checkpoints": selected,
    })
    config = json.loads(args.config.read_text())
    s3_module = load_module("s3_heldout_driver", args.s3_trainer)
    results = {
        "schema_version": 1,
        "heldout_used_for_selection": False,
        "checkpoint_role": "relative best diagnostic checkpoint; no experiment had a qualified checkpoint",
        "selected_checkpoints": selected,
        "experiments": {},
    }
    heldout_bank_path = args.output_dir / "heldout_perturbation_bank.npz"
    bank = None
    for index, experiment in enumerate(EXPERIMENTS):
        runtime = args.output_dir / "evaluation_runtime" / experiment
        runtime.mkdir(parents=True, exist_ok=True)
        init_args = SimpleNamespace(
            experiment=experiment,
            run_dir=str(runtime),
            config=str(args.config),
            trainer=str(args.trainer),
            finalizer=str(args.finalizer),
            checkpoint=str(args.source_checkpoint),
            bank=str(args.training_bank),
            validation_lock=str(args.output_dir / "heldout.lock"),
            learning_rate=args.learning_rate,
        )
        s3 = s3_module.S3(init_args)
        s3.init_eval_assets()
        checkpoint = torch.load(selected[experiment]["path"], map_location="cpu", weights_only=False)
        if int(checkpoint["step"]) != selected[experiment]["step"]:
            raise AssertionError(f"{experiment} checkpoint step mismatch")
        s3.exp.model.load_state_dict(checkpoint["model"], strict=True)
        s3.exp.model.eval()
        if bank is None:
            bank = build_heldout_bank(s3.exp, config["seed"] + 3004, heldout_bank_path)
        labels, fixed_ids, _ = heldout_fixed_points(s3.exp)
        repair_terminal_dt(s3, fixed_ids)
        if sorted(set(bank["label_id"].tolist())) != labels.tolist():
            raise AssertionError("heldout bank labels differ from evaluation dataset")
        horizon_metrics = {}
        for k in HORIZONS:
            horizon_metrics[f"k{k}"] = s3.finalizer.evaluate_horizon(s3.exp, "heldout", k)
        max_k = 128 if horizon_metrics["k128"]["available"] else 56
        attractivity = evaluate_attractivity(s3, bank, max_k)
        results["experiments"][experiment] = {
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_sha256": selected[experiment]["sha256"],
            "horizons": horizon_metrics,
            "attractivity": attractivity,
        }
        atomic_json(args.output_dir / f"heldout_metrics_{experiment}.json", results["experiments"][experiment])
        print(json.dumps({"event": "heldout_experiment_complete", "experiment": experiment, "checkpoint_step": int(checkpoint["step"])}), flush=True)
        del s3
        gc.collect()
        torch.cuda.empty_cache()
    results["heldout_perturbation_bank"] = {
        "path": str(heldout_bank_path),
        "sha256": sha256(heldout_bank_path),
        "seed": config["seed"] + 3004,
        "directions_per_re_scale": 20,
        "scales": list(PERTURBATION_SCALES),
        "heldout_re_count": 4,
    }
    atomic_json(args.output_dir / "heldout_metrics.json", results)
    atomic_json(done, {
        "finished_unix": time.time(),
        "heldout_used_for_selection": False,
        "selected_checkpoints": selected,
        "heldout_metrics_sha256": sha256(args.output_dir / "heldout_metrics.json"),
    })


if __name__ == "__main__":
    main()
