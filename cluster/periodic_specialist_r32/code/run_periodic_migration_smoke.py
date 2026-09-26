#!/usr/bin/env python3
"""Run one from-scratch epoch with the finalized Periodic configuration."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("periodic_trainer_smoke", path)
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
    parser.add_argument("--tensor-path", type=Path, required=True)
    parser.add_argument("--pressure-surrogate-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    cli = parser.parse_args()

    checkpoint = torch.load(cli.source_checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    values = dict(vars(saved_args) if hasattr(saved_args, "__dict__") else saved_args)
    values.update(
        data_root=cli.data_root,
        tensor_path=cli.tensor_path,
        pressure_surrogate_path=cli.pressure_surrogate_path,
        output_dir=cli.output_dir,
        epochs=1,
        min_epochs=1,
        eval_every=1,
        resume_checkpoint=None,
        eval_only_checkpoint=None,
        experiment_name="periodic_v16_public_r32_migration_smoke",
        experiment_tag="strict migration smoke: finalized config, one epoch from scratch",
        swanlab_mode="disabled",
        swanlab_required=False,
        swanlab_log_dir=cli.output_dir / "swanlog",
        swanlab_run_name="V17-Periodic-r32-migration-smoke",
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
