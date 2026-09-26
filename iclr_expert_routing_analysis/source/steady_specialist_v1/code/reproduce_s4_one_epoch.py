"""Run one effective data epoch through the frozen S4 training path.

The original S4 optimizer is step based and samples with replacement.  With
894 train snapshots and effective batch 64, one epoch-equivalent is ceil(894
/ 64) = 14 optimizer steps.  This driver calls the unmodified S4 train_step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("portable_s4", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run = root / "portability_runs" / "one_epoch_reproduction_torch211"
    run.mkdir(parents=True, exist_ok=True)
    code = root / "code"
    s4_module = load_module(code / "train_s4_steady.py")
    runtime_args = SimpleNamespace(
        trainer=str(code / "train_s2b_3090.py"),
        finalizer=str(code / "finalize_s2b_3090.py"),
        s3_trainer=str(code / "train_s3.py"),
        config=str(root / "code" / "training_s2b_portable.json"),
        s2b_checkpoint=str(root / "checkpoint" / "frozen_s2b_validation.pt"),
        s3b_checkpoint=str(root / "checkpoint" / "frozen_s3b_contraction.pt"),
        bank=str(root / "data" / "perturbation_bank_train_validation.npz"),
        run_dir=str(run), learning_rate=1.5516372391099407e-5, preflight_only=False,
    )
    started = time.time()
    trainer = s4_module.S4(runtime_args)
    train_snapshot_count = int((trainer.exp.split == "train").sum())
    assert train_snapshot_count == 894
    assert not set(trainer.exp.windows["train"]) & set(trainer.exp.windows["heldout"])
    epoch_steps = math.ceil(train_snapshot_count / 64)
    assert epoch_steps == 14
    calibration = trainer.calibrate_anchor()
    trainer.restore_initial()
    trainer.baseline = trainer.validate(0)
    trainer.preregister(calibration)
    source_model_digest = trainer.base.trainer.model_digest(trainer.exp.model)
    records = []
    for step in range(1, epoch_steps + 1):
        metrics = trainer.train_step(step)
        records.append({"step": step, **metrics})
        print(json.dumps(records[-1]), flush=True)
    post_model_digest = trainer.base.trainer.model_digest(trainer.exp.model)
    validation = trainer.validate(epoch_steps)
    protection = trainer.protection(validation)
    checkpoint = run / "one_epoch_step_0014.pt"
    torch.save(trainer.checkpoint_payload(epoch_steps, validation, protection), checkpoint)
    result = {
        "status": "PASS",
        "definition": "ceil(894 train snapshots / effective batch 64) optimizer steps",
        "epoch_steps": epoch_steps,
        "train_snapshot_count": train_snapshot_count,
        "effective_batch": 64,
        "heldout_used_for_training": False,
        "source_s3b_sha256": digest(Path(runtime_args.s3b_checkpoint)),
        "source_model_digest": source_model_digest,
        "post_model_digest": post_model_digest,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "protection": protection,
        "steps": records,
        "elapsed_seconds": time.time() - started,
    }
    write_json(run / "ONE_EPOCH_REPRODUCTION.json", result)
    print(json.dumps({k: result[k] for k in ("status", "epoch_steps", "train_snapshot_count", "heldout_used_for_training", "checkpoint_sha256", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
