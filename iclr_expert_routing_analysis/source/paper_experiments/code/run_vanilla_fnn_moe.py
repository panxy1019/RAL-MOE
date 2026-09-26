#!/usr/bin/env python3
"""Launch an original specialist trainer with only its expert class replaced."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class _InitializationCompatibilityStub:
    """Non-trainable attribute used only by the frozen zero-output initializer."""

    def __init__(self, shape: tuple[int, ...]):
        self.weight = torch.zeros(shape)


def target_parameter_count(
    h_dim: int,
    state_dim: int,
    out_dim: int,
    expert_hidden: int,
    blocks: int,
    quadratic_rank: int,
) -> int:
    input_norm = 2 * (h_dim + state_dim)
    input_linear = (h_dim + state_dim) * h_dim + h_dim
    block = 2 * h_dim + h_dim * expert_hidden + expert_hidden + expert_hidden * h_dim + h_dim + 1
    head = 2 * h_dim + h_dim * out_dim + out_dim
    linear = state_dim * out_dim
    quadratic = 2 * out_dim * max(0, quadratic_rank) * state_dim
    return input_norm + input_linear + max(1, blocks) * block + head + linear + quadratic


def matched_width(input_dim: int, output_dim: int, layers: int, target: int) -> int:
    hidden_layers = max(0, layers - 1)
    # hidden_layers*w^2 + (input+output+layers)*w + output ~= target
    linear = input_dim + output_dim + layers
    if hidden_layers:
        width = int(
            round(
                (-linear + math.sqrt(linear * linear + 4 * hidden_layers * target))
                / (2 * hidden_layers)
            )
        )
    else:
        width = max(1, int(round((target - output_dim) / max(linear, 1))))
    return max(32, width)


class VanillaFNNExpert(nn.Module):
    """Plain GELU FNN; no residual block, explicit linear branch, or quadratic branch."""

    def __init__(
        self,
        h_dim: int,
        state_dim: int,
        out_dim: int,
        expert_hidden: int,
        num_blocks: int,
        quadratic_rank: int,
        quadratic_scale: float,
        dropout: float,
    ):
        super().__init__()
        del quadratic_scale
        self.state_dim = state_dim
        self.out_dim = out_dim
        input_dim = h_dim + state_dim
        layers = max(2, num_blocks + 1)
        target = target_parameter_count(
            h_dim,
            state_dim,
            out_dim,
            expert_hidden,
            num_blocks,
            quadratic_rank,
        )
        width = matched_width(input_dim, out_dim, layers, target)
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            modules.extend(
                [nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)]
            )
            current = width
        final = nn.Linear(current, out_dim)
        modules.append(final)
        self.net = nn.Sequential(*modules)
        # The original physical-zero initializer addresses these names. They
        # are compatibility aliases only and are not used by forward().
        self.linear = _InitializationCompatibilityStub((out_dim, state_dim))
        self.mlp_head = [final]
        self.quad_left = None
        self.quad_right = None
        self.target_parameter_count = target
        self.actual_parameter_count = sum(p.numel() for p in self.parameters())

    def forward(self, h: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, state], dim=1))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def patch_vendor(vendor: Any) -> Any:
    if not hasattr(vendor, "PhysicsAwareExpert"):
        return vendor
    if vendor.PhysicsAwareExpert is VanillaFNNExpert:
        return vendor
    original = vendor.PhysicsAwareExpert
    vendor.OriginalPhysicsAwareExpert = original
    vendor.PhysicsAwareExpert = VanillaFNNExpert
    return vendor


def patch_nested_loaders(module: Any) -> Any:
    patch_vendor(module)
    if hasattr(module, "v16"):
        patch_vendor(module.v16)
    if hasattr(module, "load_vendor"):
        original_vendor_loader = module.load_vendor

        def patched_vendor_loader(path: Path) -> Any:
            return patch_vendor(original_vendor_loader(path))

        module.load_vendor = patched_vendor_loader
    if hasattr(module, "load_module"):
        original_module_loader = module.load_module

        def patched_module_loader(*loader_args: Any) -> Any:
            child = original_module_loader(*loader_args)
            return patch_nested_loaders(child)

        module.load_module = patched_module_loader
    if hasattr(module, "load_training_module"):
        original_training_loader = module.load_training_module

        def patched_training_loader(*loader_args: Any) -> Any:
            child = original_training_loader(*loader_args)
            return patch_nested_loaders(child)

        module.load_training_module = patched_training_loader
    return module


def patch_rng_state_compat(module: Any) -> None:
    """Normalize legacy serialized RNG arrays without changing RNG values."""

    if not hasattr(module, "restore_rng_state"):
        return
    original = module.restore_rng_state

    def compatible_restore(state: dict[str, Any]) -> None:
        normalized = dict(state)
        normalized["torch"] = torch.as_tensor(
            normalized["torch"], dtype=torch.uint8, device="cpu"
        )
        cuda_states = normalized.get("cuda")
        if cuda_states is not None:
            normalized["cuda"] = [
                torch.as_tensor(value, dtype=torch.uint8, device="cpu")
                for value in cuda_states
            ]
        original(normalized)

    module.restore_rng_state = compatible_restore


def patch_hopf_safety_allowlist(trainer: Any) -> None:
    """Downgrade only the legacy magnitude gate; non-finite rollouts still fail."""

    if not hasattr(trainer, "rollout_safety"):
        raise RuntimeError("Hopf trainer does not expose rollout_safety")
    original = trainer.rollout_safety

    def finite_only_safety(output: dict[str, Any], *args: Any, **kwargs: Any):
        legacy_bad, diagnostics = original(output, *args, **kwargs)
        finite_bad = torch.zeros_like(legacy_bad, dtype=torch.bool)
        for key in ("pred_a", "pred_b"):
            values = output.get(key, ())
            if isinstance(values, torch.Tensor):
                values = (values,)
            for value in values:
                finite_bad |= ~torch.isfinite(value).reshape(value.shape[0], -1).all(1)
        diagnostics = dict(diagnostics)
        diagnostics["legacy_magnitude_bad_windows"] = int(legacy_bad.sum().detach().cpu())
        diagnostics["ablation_allowlist"] = "vanilla-fnn:finite-only-training-gate"
        if finite_bad.any():
            print(
                json.dumps(
                    {
                        "event": "vanilla_fnn_nonfinite_gate",
                        "nonfinite_windows": int(finite_bad.sum().detach().cpu()),
                        "legacy_magnitude_bad_windows": diagnostics[
                            "legacy_magnitude_bad_windows"
                        ],
                    }
                ),
                flush=True,
            )
        return finite_bad, diagnostics

    trainer.rollout_safety = finite_only_safety


def run_hopf_validation_smoke(
    trainer: Any, forwarded: list[str], output: Path
) -> None:
    """Exercise the native validation rollout at K1/K4/K8/K16 before training."""

    previous_argv = sys.argv
    sys.argv = [str(trainer.__file__), *forwarded]
    try:
        args = trainer.parse()
    finally:
        sys.argv = previous_argv
    baseline = trainer.load_module(args.baseline_trainer)
    trainer.seed_all(args.seed)
    device = torch.device(args.device)
    trainer.configure_math(args.allow_tf32)
    if args.gpu_memory_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    baseline.audit_assets(
        SimpleNamespace(
            coefficient_view=args.coefficient_view,
            galerkin_path=args.galerkin_path,
            pressure_path=args.pressure_path,
            asset_manifest=args.asset_manifest,
        )
    )
    data = baseline.load_coefficients(
        SimpleNamespace(coefficient_view=args.coefficient_view, history_len=3)
    )
    rom_numpy = baseline.load_train_rom(
        SimpleNamespace(
            galerkin_path=args.galerkin_path, pressure_path=args.pressure_path
        )
    )
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
    norms, scale = baseline.fit_stats(data, rom_numpy, base_args)
    del scale
    stats = {
        key: torch.as_tensor(value, device=device)
        for key, value in asdict(norms).items()
    }
    rom = {
        key: torch.as_tensor(value, device=device)
        for key, value in rom_numpy.items()
    }
    contract = trainer.FluctuationContract(args.contract, device)
    probe = baseline.batch_from_ids(data, data["train_ids"][:2], device)
    history_rhs = baseline.galerkin(
        probe["ah"].reshape(-1, 32),
        probe["bh"].reshape(-1, 32),
        probe["re"][:, None].expand(-1, 3).reshape(-1),
        rom,
    ).reshape_as(probe["ah"])
    in_dim = baseline.state_features(
        probe["a"],
        probe["b"],
        probe["re"],
        probe["ah"],
        probe["bh"],
        history_rhs,
        rom,
        stats,
    )[0].shape[1]
    model = baseline.build_model(in_dim, base_args, stats, device)
    _outputs, rollout = trainer.make_ops(
        baseline, contract, baseline.state_features
    )
    velocity_geometry, pressure_geometry = trainer.geometry(data, device)
    report = trainer.validate(
        baseline,
        contract,
        model,
        rollout,
        data,
        rom,
        stats,
        velocity_geometry,
        pressure_geometry,
        device,
        args,
        0,
    )
    required = ("1", "4", "8", "16")
    failures: list[str] = []
    for reynolds, rows in report["by_re"].items():
        for horizon in required:
            row = rows[horizon]
            if row["finite_fraction"] != 1.0 or row["divergent_windows"] != 0:
                failures.append(f"{reynolds}:K{horizon}")
    payload = {
        "schema": "hopf_vanilla_native_validation_smoke/v1",
        "status": "PASS" if not failures else "FAILED",
        "required_horizons": [1, 4, 8, 16],
        "failures": failures,
        "report": report,
        "test_truth_read": False,
    }
    atomic_json(output, payload)
    if failures:
        raise RuntimeError(f"Hopf validation smoke failed: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--ablation-mode", choices=("vanilla-fnn",), required=True
    )
    parser.add_argument("--regime", choices=("steady", "hopf", "periodic"), required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--saved-args-checkpoint", type=Path)
    parser.add_argument("--saved-data-root", type=Path)
    parser.add_argument("--saved-tensor-path", type=Path)
    parser.add_argument("--saved-pressure-surrogate-path", type=Path)
    parser.add_argument("--saved-output-dir", type=Path)
    parser.add_argument("--saved-experiment-name")
    parser.add_argument("--saved-seed", type=int)
    parser.add_argument("--saved-epochs", type=int)
    parser.add_argument("--saved-swanlab-project")
    parser.add_argument("--saved-swanlab-group")
    parser.add_argument("--saved-swanlab-run-name")
    parser.add_argument("--saved-resume-checkpoint", type=Path)
    parser.add_argument("--evaluation-horizons")
    parser.add_argument("--evaluation-re-values")
    parser.add_argument("--hopf-validation-smoke-output", type=Path)
    parser.add_argument("--hopf-preflight-center-window", action="store_true")
    args, forwarded = parser.parse_known_args()
    trainer = load_module(
        f"vanilla_{args.regime}_trainer", args.entrypoint.resolve()
    )
    if args.evaluation_horizons and hasattr(trainer, "HORIZONS"):
        trainer.HORIZONS = tuple(
            int(value) for value in args.evaluation_horizons.split(",")
        )
    if args.evaluation_re_values and hasattr(trainer, "HELDOUT_RE"):
        trainer.HELDOUT_RE = tuple(
            float(value) for value in args.evaluation_re_values.split(",")
        )
    if args.regime == "steady":
        patch_nested_loaders(trainer)
    elif args.regime == "hopf":
        if hasattr(trainer, "rollout_safety"):
            patch_hopf_safety_allowlist(trainer)
        if hasattr(trainer, "v16"):
            patch_vendor(trainer.v16)
        else:
            original_loader = trainer.load_module

            def patched_hopf_loader(path: Path) -> Any:
                baseline = original_loader(path)
                patch_vendor(baseline.v16)
                if args.hopf_preflight_center_window:
                    original_legal_starts = baseline.legal_starts

                    def center_first_legal_starts(*loader_args: Any, **kwargs: Any):
                        starts = original_legal_starts(*loader_args, **kwargs)
                        if len(starts) > 1:
                            starts = np.roll(starts, -(len(starts) // 2))
                        return starts

                    baseline.legal_starts = center_first_legal_starts
                return baseline

            trainer.load_module = patched_hopf_loader
    else:
        patch_nested_loaders(trainer)
        patch_rng_state_compat(trainer)
    atomic_json(
        args.contract_output,
        {
            "schema": "specialist_ablation_allowlist/v2",
            "ablation_mode": args.ablation_mode,
            "regime": args.regime,
            "entrypoint": str(args.entrypoint.resolve()),
            "entrypoint_sha256": sha256(args.entrypoint.resolve()),
            "allowed_changes": [
                "PhysicsAwareExpert implementation -> parameter-matched plain GELU FNN",
                (
                    "Hopf legacy magnitude-only pre-optimizer gate is diagnostic; "
                    "NaN/Inf remains a hard failure"
                    if args.regime == "hopf"
                    else "none"
                ),
            ],
            "preserved_contract": [
                "POD/mean/scaler and ru=rp=32",
                "complete-Re split",
                "feature builder and history length",
                "MoE routers and expert count",
                "RK4/Galerkin/pressure closure",
                "loss/curriculum/optimizer budget/validation selector",
            ],
            "test_truth_read": False,
        },
    )
    if args.hopf_validation_smoke_output is not None:
        if args.regime != "hopf":
            raise ValueError("Hopf validation smoke is only valid for regime=hopf")
        run_hopf_validation_smoke(
            trainer, forwarded, args.hopf_validation_smoke_output
        )
        return 0
    if args.saved_args_checkpoint is None:
        sys.argv = [str(args.entrypoint), *forwarded]
        trainer.main()
        return 0

    if args.regime != "periodic":
        raise ValueError("--saved-args-checkpoint is currently certified only for Periodic")
    if forwarded:
        raise ValueError(
            "Unrecognized arguments cannot be combined with --saved-args-checkpoint: "
            + " ".join(forwarded)
        )
    checkpoint = torch.load(
        args.saved_args_checkpoint, map_location="cpu", weights_only=False
    )
    saved_args = checkpoint["args"]
    values = dict(
        vars(saved_args) if hasattr(saved_args, "__dict__") else saved_args
    )
    overrides = {
        "data_root": args.saved_data_root,
        "tensor_path": args.saved_tensor_path,
        "pressure_surrogate_path": args.saved_pressure_surrogate_path,
        "output_dir": args.saved_output_dir,
        "experiment_name": args.saved_experiment_name,
        "seed": args.saved_seed,
        "epochs": args.saved_epochs,
        "swanlab_project": args.saved_swanlab_project,
        "swanlab_group": args.saved_swanlab_group,
        "swanlab_run_name": args.saved_swanlab_run_name,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    values.update(
        resume_checkpoint=args.saved_resume_checkpoint,
        eval_only_checkpoint=None,
        swanlab_mode="online",
        swanlab_required=True,
        swanlab_log_dir=args.saved_output_dir / "swanlog",
    )
    original_parse_args = argparse.ArgumentParser.parse_args

    def use_saved_args(_parser: argparse.ArgumentParser, *unused: Any, **kwargs: Any):
        del unused, kwargs
        argparse.ArgumentParser.parse_args = original_parse_args
        return argparse.Namespace(**values)

    argparse.ArgumentParser.parse_args = use_saved_args
    try:
        trainer.main()
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
