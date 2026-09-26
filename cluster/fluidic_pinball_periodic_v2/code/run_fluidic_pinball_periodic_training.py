#!/usr/bin/env python3
"""Launch the Fluidic Pinball V2 periodic-specialist curriculum."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("fluidic_pinball_periodic_trainer", path)
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
    parser.add_argument("--epochs", type=int, default=720)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rollout-batch", type=int, default=56)
    parser.add_argument("--rollout-batch-by-stage", default="56,56,40,28")
    parser.add_argument("--rollout-updates-per-epoch", type=int, default=2)
    parser.add_argument("--curriculum-steps", default="4,8,12,16")
    parser.add_argument("--curriculum-stage-epochs", default="120,160,160,280")
    parser.add_argument("--amp-mode", choices=["off", "bf16"], default="bf16")
    parser.add_argument("--dense-moe-training", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batched-experts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.90)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--swanlab-mode", choices=["online", "local", "offline"], default="online")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    cli = parser.parse_args()

    contract = json.loads((cli.data_root / "dataset_contract.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(cli.source_checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    values = dict(vars(saved_args) if hasattr(saved_args, "__dict__") else saved_args)
    values.update(
        data_root=cli.data_root,
        tensor_path=cli.data_root / "velocity_rom_periodic.npz",
        pressure_surrogate_path=cli.data_root / "pressure_poisson_surrogate_periodic.npz",
        output_dir=cli.output_dir,
        r_u=17,
        r_p=16,
        max_integrator_dt=0.03125,
        fixed_integrator_substeps=8,
        amp_mode=cli.amp_mode,
        dense_moe_training=cli.dense_moe_training,
        batched_experts=cli.batched_experts,
        fixed_regime_group=1,
        compile_model=cli.compile_model,
        gpu_memory_fraction=cli.gpu_memory_fraction,
        batch_size=cli.batch_size,
        rollout_batch=cli.rollout_batch,
        rollout_batch_by_stage=cli.rollout_batch_by_stage,
        rollout_updates_per_epoch=cli.rollout_updates_per_epoch,
        rollout_every_batches=8,
        curriculum_steps=cli.curriculum_steps,
        curriculum_stage_epochs=cli.curriculum_stage_epochs,
        validation_re_values=contract["validation_re_values"],
        test_re_selection="values",
        test_re_values=contract["test_re_values"],
        epochs=cli.epochs,
        min_epochs=min(max(1, int(cli.epochs * 0.78)), cli.epochs),
        patience=cli.patience if cli.patience is not None else max(80, int(cli.epochs * 0.25)),
        eval_every=cli.eval_every,
        resume_checkpoint=cli.resume_checkpoint,
        eval_only_checkpoint=None,
        experiment_name="fluidic_pinball_periodic_v2_rank999_ru17_rp16",
        experiment_tag="Fluidic Pinball V2 periodic train-only POD curriculum",
        swanlab_mode=cli.swanlab_mode,
        swanlab_required=True,
        swanlab_project="FluidicPinballPeriodicV2",
        swanlab_group="periodic-rank999-long-curriculum",
        swanlab_log_dir=cli.output_dir / "swanlog",
        swanlab_run_name=(
            "FluidicPinball-Periodic-V2-long-resume"
            if cli.resume_checkpoint is not None
            else "FluidicPinball-Periodic-V2-long"
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
