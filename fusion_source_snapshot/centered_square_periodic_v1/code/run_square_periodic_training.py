#!/usr/bin/env python3
"""Run the finalized Periodic HPRS-MoE configuration on CenteredSquare data."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("square_periodic_trainer", path)
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
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--swanlab-mode", choices=["online", "local", "offline"], default="online")
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
        rollout_every_batches=8,
        validation_re_values=[99.0, 101.5, 110.344827586207, 125.862068965517, 141.379310344828, 150.0],
        test_re_selection="values",
        test_re_values=[100.5, 102.0, 120.689655172414, 144.827586206897],
        epochs=cli.epochs,
        min_epochs=min(180, cli.epochs),
        resume_checkpoint=None,
        eval_only_checkpoint=None,
        experiment_name="centered_square_periodic_v16_public_r28_p26",
        experiment_tag="CenteredSquare Periodic-only V16 public-loss native rank999",
        swanlab_mode=cli.swanlab_mode,
        swanlab_required=True,
        swanlab_project="V17SquarePeriodicMOE",
        swanlab_group="Periodic-native-rank999-from-scratch",
        swanlab_log_dir=cli.output_dir / "swanlog",
        swanlab_run_name="CenteredSquare-Periodic-ru28-rp26",
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
