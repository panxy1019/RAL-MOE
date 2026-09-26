from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


CLASS_NAMES = ("Steady", "Hopf", "Periodic")
EXPECTED_E2_SHA256 = "f0286a9974f16b513bb42cd94232c397123dcd3e0a01eab475ff1593f7881dcd"
VALIDATION_RE = {
    "Steady": [28.695137758, 43.7974019747],
    "Hopf": [46.7, 56.5432463134],
    "Periodic": [66.9701121204, 91.792204085, 121.05017082, 139.642302099, 169.244893107, 196.160723205],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    def sanitize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): sanitize(subvalue) for key, subvalue in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(subvalue) for subvalue in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(sanitize(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class E2Router(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(1, 24), torch.nn.Tanh(), torch.nn.Linear(24, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def route_validation(checkpoint_path: Path) -> dict[str, Any]:
    actual_sha = sha256(checkpoint_path)
    if actual_sha != EXPECTED_E2_SHA256:
        raise RuntimeError(f"E2 SHA mismatch: {actual_sha}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["experiment"] != "E2" or checkpoint["input_dim"] != 1:
        raise RuntimeError("frozen checkpoint is not the certified E2 Re-only Router")
    model = E2Router()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(1, 1)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(1, 1)
    rows = []
    with torch.inference_mode():
        for oracle, name in enumerate(CLASS_NAMES):
            for re_value in VALIDATION_RE[name]:
                feature = (np.asarray([[re_value]], dtype=np.float32) - mean) / std
                probabilities = torch.softmax(model(torch.from_numpy(feature)), dim=-1)[0].numpy()
                selected = int(np.argmax(probabilities))
                rows.append({
                    "Re": re_value,
                    "oracle_class": name,
                    "oracle_id": oracle,
                    "selected_class": CLASS_NAMES[selected],
                    "selected_id": selected,
                    "probabilities": {CLASS_NAMES[i]: float(probabilities[i]) for i in range(3)},
                    "oracle_match": selected == oracle,
                })
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_sha,
        "checkpoint_step": int(checkpoint["step"]),
        "rows": rows,
        "trajectory_count": len(rows),
        "oracle_match_count": sum(row["oracle_match"] for row in rows),
        "all_oracle_match": all(row["oracle_match"] for row in rows),
    }


def steady_native(root: Path, output_dir: Path) -> dict[str, Any]:
    specialist = root / "steady_specialist_v1"
    trainer = load_module("e2_smoke_steady_s4", specialist / "code/train_s4_steady.py")
    runtime = output_dir / "steady_runtime"
    args = SimpleNamespace(
        trainer=str(specialist / "code/train_s2b_3090.py"),
        finalizer=str(specialist / "code/finalize_s2b_3090.py"),
        s3_trainer=str(specialist / "code/train_s3.py"),
        config=str(specialist / "code/training_s2b_portable.json"),
        s2b_checkpoint=str(specialist / "checkpoint/frozen_s2b_validation.pt"),
        s3b_checkpoint=str(specialist / "checkpoint/frozen_s3b_contraction.pt"),
        bank=str(specialist / "data/perturbation_bank_train_validation.npz"),
        run_dir=str(runtime),
        learning_rate=3.0e-5,
        preflight_only=False,
    )
    instance = trainer.S4(args)
    checkpoint_path = specialist / "checkpoint/frozen_s4_validation_step_1200.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    instance.exp.model.load_state_dict(checkpoint["model"], strict=True)
    instance.exp.model.eval()
    with torch.inference_mode():
        native = instance.base.finalizer.evaluate_horizon(instance.exp, "validation", 4)
    by_re = native["by_re"]
    expected_tokens = ("28p695138", "43p797402")
    if not all(any(token in key for key in by_re) for token in expected_tokens):
        raise RuntimeError(f"Steady native validation Re mismatch: {list(by_re)}")
    return {
        "class": "Steady",
        "native_entrypoint": "finalize_s4_one_time.evaluate_horizon(exp, 'validation', 4)",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "horizon": 4,
        "validation_Re_count": len(by_re),
        "finite_fraction": float(native["overall_worst"]["finite_fraction"]),
        "divergent_windows": int(native["overall_worst"]["divergent_windows"]),
        "output_keys": sorted(native.keys()),
        "by_re_keys": sorted(by_re.keys()),
        "native_result": native,
    }


def hopf_native(root: Path) -> dict[str, Any]:
    specialist = root / "Hopf/migrated_h4_expanded"
    h4 = load_module("e2_smoke_hopf_h4", specialist / "code/train_h4_expanded.py")
    baseline = load_module("e2_smoke_hopf_baseline", specialist / "code/train_hopf_moe_expanded.py")
    checkpoint_path = specialist / "runs/HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**checkpoint["args"])
    args.baseline_trainer = specialist / "code/train_hopf_moe_expanded.py"
    args.coefficient_view = specialist / "assets_r32/expanded_h4_trainval_r32.npz"
    args.galerkin_path = specialist / "assets_r32/expanded_h4_trainonly_galerkin_r32.npz"
    args.pressure_path = specialist / "assets_r32/expanded_h4_trainonly_pressure_r32.npz"
    args.asset_manifest = specialist / "assets_r32/TRAINING_ASSET_MANIFEST.json"
    args.contract = specialist / "trainonly_contract/trainonly_fluctuation_contract.npz"
    args.validation_windows_per_re = 1
    device = torch.device("cuda")
    h4.configure_math(args.allow_tf32)
    data = baseline.load_coefficients(SimpleNamespace(coefficient_view=args.coefficient_view, history_len=3))
    rom_np = baseline.load_train_rom(SimpleNamespace(galerkin_path=args.galerkin_path, pressure_path=args.pressure_path))
    base_args = SimpleNamespace(**{
        **vars(args), "history_len": 3, "hidden_dim": 256, "expert_hidden": 1024,
        "num_blocks": 3, "experts": 6, "top_k": 2, "expert_blocks": 4,
        "quadratic_rank": 4, "dropout": 0.04, "temperature": 0.8,
        "adaptive_gate_initial_logit": 6.0, "lambda_scale_amplitude": 1.0,
        "lambda_scale_growth": 0.5, "lambda_scale_sign": 0.1, "scale_floor_quantile": 0.1,
    })
    norms, _ = baseline.fit_stats(data, rom_np, base_args)
    stats = {key: torch.as_tensor(value, device=device) for key, value in vars(norms).items()}
    rom = {key: torch.as_tensor(value, device=device) for key, value in rom_np.items()}
    contract = h4.FluctuationContract(args.contract, device)
    probe = baseline.batch_from_ids(data, data["train_ids"][:2], device)
    predicted_rhs_history = baseline.galerkin(
        probe["ah"].reshape(-1, 32), probe["bh"].reshape(-1, 32),
        probe["re"][:, None].expand(-1, 3).reshape(-1), rom,
    ).reshape_as(probe["ah"])
    in_dim = baseline.state_features(
        probe["a"], probe["b"], probe["re"], probe["ah"], probe["bh"],
        predicted_rhs_history, rom, stats,
    )[0].shape[1]
    model = baseline.build_model(in_dim, base_args, stats, device)
    head = h4.NormalForm(contract.nodes).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    head.load_state_dict(checkpoint["normal_form_state"], strict=True)
    model.eval(); head.eval()
    _, rollout = h4.make_ops(baseline, contract, baseline.state_features)
    velocity_geometry, pressure_geometry = h4.geometry(data, device)
    native = h4.validate(
        baseline, contract, model, rollout, data, rom, stats,
        velocity_geometry, pressure_geometry, device, args, 1,
    )
    rows = [row for per_re in native["by_re"].values() for row in per_re.values()]
    finite = [float(row["finite_fraction"]) for row in rows]
    divergent = [int(row["divergent_windows"]) for row in rows]
    return {
        "class": "Hopf",
        "native_entrypoint": "train_h4_expanded.validate(...)",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "horizons": sorted({int(key) for per_re in native["by_re"].values() for key in per_re}),
        "validation_Re_count": len(native["by_re"]),
        "finite_fraction": min(finite),
        "divergent_windows": sum(divergent),
        "output_keys": sorted(native.keys()),
        "by_re_keys": sorted(native["by_re"].keys()),
        "native_result": native,
    }


def periodic_native(root: Path) -> dict[str, Any]:
    specialist = root / "periodic_specialist_r32"
    trainer = load_module("e2_smoke_periodic_trainer", specialist / "code/train_periodic_moe.py")
    portable = load_module("e2_smoke_periodic_portable", specialist / "code/evaluate_periodic_r32_portable.py")
    checkpoint_path = specialist / "checkpoint/FINAL_PERIODIC_SPECIALIST.pt"
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = SimpleNamespace(**checkpoint["args"])
    args.device = "cuda"
    args.data_root = specialist / "assets/Global_POD_AreaWeighted_L2"
    args.tensor_path = specialist / "assets/velocity_rom_periodic.npz"
    args.pressure_surrogate_path = specialist / "assets/pressure_poisson_surrogate_periodic.npz"
    arrays, _ = trainer.build_arrays(args)
    tensors = np.load(args.tensor_path)
    pressure_tensors = np.load(args.pressure_surrogate_path)
    scalers = {
        key: trainer.Standardizer(mean=np.asarray(value["mean"], dtype=np.float32), scale=np.asarray(value["scale"], dtype=np.float32))
        for key, value in checkpoint["scalers"].items()
    }
    model = portable.instantiate_model(trainer, args, arrays, checkpoint["model_state"], device)
    results = {}
    finite_values = []
    divergent = 0
    valid_samples = set(arrays["sample_ids"].tolist())
    for requested_re in VALIDATION_RE["Periodic"]:
        label_re = np.asarray([
            arrays["re"][np.where(arrays["label_id"] == label)[0][0]]
            for label in range(len(arrays["labels"]))
        ])
        label_id = int(np.argmin(np.abs(label_re - requested_re)))
        actual_re = float(label_re[label_id])
        if abs(actual_re - requested_re) > 5e-5:
            raise RuntimeError(f"Periodic validation Re unavailable: requested={requested_re}, actual={actual_re}")
        ids = np.where(arrays["label_id"] == label_id)[0]
        ids = ids[np.argsort(arrays["time"][ids])]
        legal_positions = [position for position in range(1, len(ids) - 5) if int(ids[position]) in valid_samples]
        if not legal_positions:
            raise RuntimeError(f"no native Periodic validation start for Re={actual_re}")
        position = legal_positions[len(legal_positions) // 2]
        current = int(ids[position])
        a_state = arrays["a"][current].copy(); b_state = arrays["b"][current].copy()
        a_history, b_history, rhs_history = trainer.init_history_states_np(current, arrays)
        true_a = []; true_b = []; pred_a = []; pred_b = []
        first_bad = None
        for step in range(1, 5):
            nxt = int(arrays["next_idx"][current])
            dt = float(arrays["time"][nxt] - arrays["time"][current])
            a_next, b_next, rhs_g = trainer.integrate_autonomous_step_np(
                model, a_state, b_state, current, dt, a_history, b_history, rhs_history,
                arrays, scalers, tensors, pressure_tensors, args, device,
            )
            if not (np.isfinite(a_next).all() and np.isfinite(b_next).all()):
                first_bad = step; break
            pred_a.append(a_next.copy()); pred_b.append(b_next.copy())
            true_a.append(arrays["a"][nxt].copy()); true_b.append(arrays["b"][nxt].copy())
            a_history = np.concatenate([a_next[None, None, :], a_history[:, :-1, :]], axis=1)
            b_history = np.concatenate([b_next[None, None, :], b_history[:, :-1, :]], axis=1)
            rhs_history = np.concatenate([rhs_g[None, None, :], rhs_history[:, :-1, :]], axis=1)
            a_state, b_state, current = a_next, b_next, nxt
        is_finite = first_bad is None and len(pred_a) == 4
        finite_values.append(float(is_finite)); divergent += int(not is_finite)
        results[f"{actual_re:.10f}"] = {
            "horizon": 4, "finite_fraction": float(is_finite),
            "divergent_windows": int(not is_finite), "first_divergent_step": first_bad,
            "velocity_coefficient_relative_l2": float(np.linalg.norm(np.asarray(pred_a) - np.asarray(true_a)) / max(np.linalg.norm(true_a), 1e-12)) if is_finite else None,
            "pressure_coefficient_relative_l2": float(np.linalg.norm(np.asarray(pred_b) - np.asarray(true_b)) / max(np.linalg.norm(true_b), 1e-12)) if is_finite else None,
        }
    return {
        "class": "Periodic",
        "native_entrypoint": "train_periodic_moe.integrate_autonomous_step_np(...), K4",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "horizon": 4,
        "validation_Re_count": len(results),
        "finite_fraction": min(finite_values),
        "divergent_windows": divergent,
        "output_keys": ["by_re", "finite_fraction", "divergent_windows"],
        "by_re_keys": sorted(results),
        "native_result": {"by_re": results},
    }


def native_worker(class_name: str, root: Path, output_dir: Path) -> None:
    started = time.time()
    if class_name == "Steady":
        result = steady_native(root, output_dir)
    elif class_name == "Hopf":
        result = hopf_native(root)
    elif class_name == "Periodic":
        result = periodic_native(root)
    else:
        raise ValueError(class_name)
    result["elapsed_seconds"] = time.time() - started
    result["pass"] = result["finite_fraction"] == 1.0 and result["divergent_windows"] == 0
    atomic_json(output_dir / f"native_{class_name.lower()}.json", result)
    print(json.dumps({key: result[key] for key in ("class", "pass", "finite_fraction", "divergent_windows", "elapsed_seconds")}), flush=True)


def render_report(result: dict[str, Any]) -> str:
    native = result["native_execution"]
    lines = [
        "# E2 Unified Validation Top-1 Smoke",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Frozen E2 SHA256: `{result['router']['checkpoint_sha256']}`",
        f"- Validation routes matching oracle: {result['router']['oracle_match_count']}/{result['router']['trajectory_count']}",
        f"- Specialists actually invoked: {', '.join(result['actual_native_invocations'])}",
        "- Contract: one trajectory-level Top-1 decision; selected native specialist retains its own scaler, feature/history, integrator and pressure closure.",
        "",
        "| Specialist | Native entrypoint | Validation Re | Finite | Divergent | PASS |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in CLASS_NAMES:
        item = native[name]
        lines.append(f"| {name} | `{item['native_entrypoint']}` | {item['validation_Re_count']} | {item['finite_fraction']:.6f} | {item['divergent_windows']} | {item['pass']} |")
    lines += [
        "",
        "## Scope",
        "",
        "This is a validation smoke, not a new scientific benchmark. It proves that the frozen E2 route can dispatch each validation trajectory class to the corresponding native rollout implementation and receive finite, structurally valid outputs without substituting E0 hashes for execution.",
        "",
        "No Top-2 path was executed.",
    ]
    return "\n".join(lines) + "\n"


def orchestrate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "STARTED.json", {"status": "STARTED", "time": time.time(), "pid": os.getpid()})
    router = route_validation(args.e2_checkpoint)
    atomic_json(args.output_dir / "router_validation_routes.json", router)
    native = {}
    for class_name in CLASS_NAMES:
        command = [sys.executable, str(Path(__file__).resolve()), "--native-only", class_name, "--root", str(args.root), "--output-dir", str(args.output_dir)]
        log_path = args.output_dir / f"native_{class_name.lower()}.log"
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"{class_name} native worker failed; see {log_path}")
        native[class_name] = json.loads((args.output_dir / f"native_{class_name.lower()}.json").read_text())
    structural = all(
        item["validation_Re_count"] == len(VALIDATION_RE[name]) and item["by_re_keys"] and item["output_keys"]
        for name, item in native.items()
    )
    passed = router["all_oracle_match"] and all(item["pass"] for item in native.values()) and structural
    result = {
        "schema": "E2_UNIFIED_VALIDATION_TOP1_SMOKE_V1",
        "status": "COMPLETE",
        "decision": "PASS" if passed else "FAIL",
        "router": router,
        "actual_native_invocations": list(CLASS_NAMES),
        "native_execution": native,
        "structural_contract_pass": structural,
        "top2_executed": False,
        "finished_time": time.time(),
    }
    atomic_json(args.output_dir / "E2_UNIFIED_VALIDATION_SMOKE.json", result)
    (args.output_dir / "E2_UNIFIED_VALIDATION_SMOKE.md").write_text(render_report(result), encoding="utf-8")
    atomic_json(args.output_dir / ("PASS.json" if passed else "FAIL.json"), {"decision": result["decision"], "time": time.time()})
    print(json.dumps({"decision": result["decision"], "routes": f"{router['oracle_match_count']}/{router['trajectory_count']}", "native": {key: value["pass"] for key, value in native.items()}}))
    if not passed:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/panxy/particalMOE"))
    parser.add_argument("--e2-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-only", choices=CLASS_NAMES)
    values = parser.parse_args()
    if values.e2_checkpoint is None:
        values.e2_checkpoint = values.root / "trajectory_router_e1_e2_e3_20260722/frozen_E2_baseline_candidate/best.pt"
    return values


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.native_only:
        native_worker(arguments.native_only, arguments.root, arguments.output_dir)
    else:
        orchestrate(arguments)
