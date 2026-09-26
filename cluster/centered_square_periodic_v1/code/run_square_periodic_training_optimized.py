#!/usr/bin/env python3
"""Optimized CenteredSquare periodic-specialist training launcher."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("square_periodic_trainer_optimized", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--rollout-batch", type=int, default=56)
    parser.add_argument("--rollout-batch-by-stage", default="56,56,40,28")
    parser.add_argument("--rollout-updates-per-epoch", type=int, default=2)
    parser.add_argument("--curriculum-steps", default="4,8,12,16")
    parser.add_argument("--curriculum-stage-epochs", default="120,160,160,200")
    parser.add_argument("--amp-mode", choices=["off", "bf16"], default="bf16")
    parser.add_argument(
        "--dense-moe-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--batched-experts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fixed-regime-group", type=int, default=1)
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.95)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--swanlab-mode", choices=["online", "local", "offline"], default="online")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    cli = parser.parse_args()

    checkpoint = torch.load(cli.source_checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    values = dict(vars(saved_args) if hasattr(saved_args, "__dict__") else saved_args)
    values.update(
        data_root=cli.data_root,
        tensor_path=cli.data_root / "velocity_rom_periodic.npz",
        pressure_surrogate_path=cli.data_root / "pressure_poisson_surrogate_periodic.npz",
        output_dir=cli.output_dir,
        r_u=28,
        r_p=26,
        max_integrator_dt=0.5,
        fixed_integrator_substeps=8,
        amp_mode=cli.amp_mode,
        dense_moe_training=cli.dense_moe_training,
        batched_experts=cli.batched_experts,
        fixed_regime_group=cli.fixed_regime_group,
        compile_model=cli.compile_model,
        gpu_memory_fraction=cli.gpu_memory_fraction,
        batch_size=cli.batch_size,
        rollout_batch=cli.rollout_batch,
        rollout_batch_by_stage=cli.rollout_batch_by_stage,
        rollout_updates_per_epoch=cli.rollout_updates_per_epoch,
        rollout_every_batches=8,
        curriculum_steps=cli.curriculum_steps,
        curriculum_stage_epochs=cli.curriculum_stage_epochs,
        validation_re_values=[
            99.0, 101.5, 110.344827586207, 125.862068965517,
            141.379310344828, 150.0,
        ],
        test_re_selection="values",
        test_re_values=[100.5, 102.0, 120.689655172414, 144.827586206897],
        epochs=cli.epochs,
        min_epochs=min(max(1, int(cli.epochs * 0.78)), cli.epochs),
        patience=cli.patience if cli.patience is not None else max(80, int(cli.epochs * 0.25)),
        eval_every=cli.eval_every,
        resume_checkpoint=cli.resume_checkpoint,
        eval_only_checkpoint=None,
        experiment_name="centered_square_periodic_optimized_r28_p26",
        experiment_tag="CenteredSquare Periodic optimized long curriculum",
        swanlab_mode=cli.swanlab_mode,
        swanlab_required=True,
        swanlab_project="V17SquarePeriodicMOE",
        swanlab_group="Periodic-native-rank999-optimized-long",
        swanlab_log_dir=cli.output_dir / "swanlog",
        swanlab_run_name=(
            "CenteredSquare-Periodic-optimized-long-resume"
            if cli.resume_checkpoint is not None
            else "CenteredSquare-Periodic-optimized-long"
        ),
    )

    trainer = load_module(cli.trainer)
    original_parse_args = argparse.ArgumentParser.parse_args

    def use_saved_args(_parser, *args, **kwargs):
        argparse.ArgumentParser.parse_args = original_parse_args
        return argparse.Namespace(**values)

    argparse.ArgumentParser.parse_args = use_saved_args
    try:
        trainer.main()
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args


if __name__ == "__main__":
    main()
