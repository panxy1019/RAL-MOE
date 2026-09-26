#!/usr/bin/env python3
"""Freeze the validation-selected expanded-data H4 checkpoint before final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch


EXPERIMENTS = ("HopfExpanded34_H4_NormalFormRadial_r32",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite finalization directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    manifest = {
        "schema_version": 1,
        "selection_split": "validation_only",
        "heldout_consulted_for_selection": False,
        "eligibility": "step>=6400, at least six K8 validations, hard_gate=true",
        "score": "trainer-registered worst-Re K8/K16 Hopf validation score",
        "experiments": {},
    }
    for name in EXPERIMENTS:
        source = args.training_run / name
        history_obj = json.loads((source / "validation_history.json").read_text())
        history = history_obj["history"] if isinstance(history_obj, dict) else history_obj
        eligible = [row for row in history if row["step"] >= 6400 and row["hard_gate"]]
        if not eligible:
            raise RuntimeError(f"{name}: no validation-admissible checkpoint")
        selected = min(eligible, key=lambda row: row["score"])
        checkpoint_path = source / "best_validation.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint["optimizer_step"]) != int(selected["step"]):
            raise RuntimeError(f"{name}: checkpoint step disagrees with validation recomputation")
        if abs(float(checkpoint["best_score"]) - float(selected["score"])) > 1e-12:
            raise RuntimeError(f"{name}: checkpoint score disagrees with validation recomputation")
        if int(checkpoint["best_step"]) != int(selected["step"]):
            raise RuntimeError(f"{name}: checkpoint best_step is inconsistent")

        target = args.output_dir / name
        target.mkdir()
        frozen = target / "final.pt"
        shutil.copy2(checkpoint_path, frozen)
        frozen.chmod(0o444)
        for filename in (
            "config.json", "trainonly_manifest.json", "validation_history.json",
            "throughput.json", "gradient_audits.json", "normal_form_train_nodes.json",
        ):
            item = source / filename
            if item.exists():
                shutil.copy2(item, target / filename)
        manifest["experiments"][name] = {
            "source_checkpoint": str(checkpoint_path),
            "frozen_checkpoint": str(frozen),
            "sha256": sha256(frozen),
            "optimizer_step": int(selected["step"]),
            "validation_score": float(selected["score"]),
            "hard_gate": bool(selected["hard_gate"]),
            "validation_record": selected,
            "training_history_records": len(history),
        }
        print(json.dumps({"event": "checkpoint_frozen", "experiment": name,
                          "step": selected["step"], "sha256": manifest["experiments"][name]["sha256"]}), flush=True)
        del checkpoint

    atomic_json(manifest, args.output_dir / "checkpoint_selection_manifest.json")
    atomic_json({"status": "CHECKPOINTS_FROZEN", "heldout_started": False},
                args.output_dir / "FREEZE_DONE.json")


if __name__ == "__main__":
    main()
