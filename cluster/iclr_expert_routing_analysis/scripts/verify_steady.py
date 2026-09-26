"""Reproduce exact S4 K56 errors with the original, unmodified runtime."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from inspect_checkpoints import CHECKPOINTS, sha256


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def relocate(value, root):
    if isinstance(value, dict):
        return {k: relocate(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, root) for v in value]
    if isinstance(value, str):
        return value.replace("/root/panxy/particalMOE", str(root))
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    analysis = a.root / "iclr_expert_routing_analysis"
    runtime = analysis / "raw/steady_verification_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    steady = a.root / "steady_specialist_v1"
    source = steady / "code/train_s2b_3090.py"
    evaluator = steady / "code/finalize_s2b_3090.py"
    original_cfg = steady / "code/training_s2b_portable.json"
    cfg = relocate(json.loads(original_cfg.read_text()), a.root)
    cfg["checkpoint_root"] = str(runtime / "unused_checkpoint_directory")
    (runtime / "relocated_config.json").write_text(json.dumps(cfg, indent=2))
    trainer = load("routing_original_steady", source)
    finalizer = load("routing_original_steady_finalizer", evaluator)
    start_time = time.monotonic()
    exp = trainer.Experiment(SimpleNamespace(run_dir=str(runtime), resume=None), cfg)
    path = a.root / CHECKPOINTS["steady"]
    before = sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = exp.model.load_state_dict(checkpoint["model"], strict=True)
    exp.model.eval()
    exp.model.requires_grad_(False)
    model_before = trainer.model_digest(exp.model)
    windows = exp.build_windows(56)["heldout"]
    shapes = []
    def observe(module, args, output):
        shapes.append({"input": list(args[0].shape), "outputs":
                       [list(v.shape) if torch.is_tensor(v) else type(v).__name__ for v in output]})
    hook = exp.model.register_forward_hook(observe)
    first = next(iter(windows.values()))
    with torch.inference_mode(), torch.autocast("cuda", dtype=exp.amp_dtype):
        smoke = exp.rollout(torch.as_tensor(first[:2], device=exp.device), 3)
    hook.remove()
    assert torch.isfinite(smoke["pa"]).all() and torch.isfinite(smoke["pb"]).all()
    print(json.dumps({"event": "steady_smoke_pass", "strict_load": str(incompatible),
                      "router_outputs": shapes[0], "windows": {str(k): len(v) for k, v in windows.items()}}), flush=True)
    metrics = finalizer.evaluate_horizon(exp, "heldout", 56)
    assert trainer.model_digest(exp.model) == model_before
    assert sha256(path) == before
    u = [100 * row["physical_reconstruction_area_weighted"]["velocity_relative_l2"] for row in metrics["by_re"].values()]
    pp = [100 * row["physical_reconstruction_area_weighted"]["pressure_relative_l2"] for row in metrics["by_re"].values()]
    report = {"checkpoint": str(path), "sha256_before_and_after": before,
              "model_state_unchanged": True, "model_class": type(exp.model).__name__,
              "strict_load": {"missing": incompatible.missing_keys, "unexpected": incompatible.unexpected_keys},
              "original_runtime": str(source), "original_evaluator": str(evaluator),
              "asset_hashes": {str(p): sha256(p) for p in [source, evaluator, original_cfg, *exp.paths.values()]},
              "precision": str(exp.amp_dtype), "tf32": torch.backends.cuda.matmul.allow_tf32,
              "integrator": "Original Experiment.rollout: forward Euler velocity; additive algebraic pressure closure",
              "routing_config": cfg["model"], "smoke_forward_shapes": shapes,
              "metrics": metrics, "observed_Eu_percent_range": [min(u), max(u)],
              "observed_Ep_percent_range": [min(pp), max(pp)],
              "requested_paper_Eu_percent_range": [0.0330, 0.1723],
              "requested_paper_Ep_percent_range": [0.2500, 1.3249],
              "paper_range_reproduced": False, "elapsed_seconds": time.monotonic() - start_time}
    (analysis / "logs/steady_reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "steady_reproduction_complete", "u_percent": report["observed_Eu_percent_range"],
                      "p_percent": report["observed_Ep_percent_range"], "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
