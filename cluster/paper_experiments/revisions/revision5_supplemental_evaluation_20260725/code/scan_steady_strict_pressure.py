#!/usr/bin/env python3
"""Scan frozen S3-B on every Steady Re using a frozen K56 pressure gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


EPS = 1.0e-12
PRESSURE_FIXED_POINT_MAX = 0.05
PRESSURE_DRIFT_MAX = 0.05


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_relative(finalizer, pred, true, basis, mean, weight) -> float:
    numerator, denominator = finalizer.field_sums(
        pred, true, basis, mean, weight
    )
    return math.sqrt(numerator / max(denominator, EPS))


def terminal_ids(exp, split: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(exp.paths["velocity_pod"]) as pod:
        snapshot_split = np.char.lower(pod["snapshot_splits"].astype(str))
    labels = []
    fixed = []
    for label in sorted(set(exp.a["label_id"][snapshot_split == split].tolist())):
        ids = np.flatnonzero(
            (exp.a["label_id"] == label) & (snapshot_split == split)
        )
        terminal = ids[exp.a["next_idx"][ids] < 0]
        labels.append(int(label))
        fixed.append(int(terminal[-1] if len(terminal) else ids[-1]))
    return np.asarray(labels, np.int64), np.asarray(fixed, np.int64)


def repair_dt(driver, fixed_ids: np.ndarray) -> None:
    nxt = np.asarray(driver.exp.a["next_idx"])
    source_dt = np.asarray(driver.exp.a["dt_next"])
    for fixed_id in fixed_ids.tolist():
        if torch.isfinite(driver.dt[fixed_id]):
            continue
        predecessors = np.flatnonzero(nxt == fixed_id)
        if len(predecessors):
            value = float(source_dt[predecessors[-1]])
        else:
            label = driver.exp.a["label_id"][fixed_id]
            value = float(
                np.nanmedian(source_dt[driver.exp.a["label_id"] == label])
            )
        driver.dt[fixed_id] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    steady = args.root / "steady_specialist_v1"
    code = steady / "code"
    config_source = code / "training_s2b_portable.json"
    config = copy.deepcopy(json.loads(config_source.read_text()))
    config["vendor_trainer"] = str(code / "train_v16_4_v2_r32_compat.py")
    config["artifact_dir"] = str(steady / "source_artifacts/steady")
    config["checkpoint_root"] = str(args.output_dir / "runtime_checkpoints")
    config_path = args.output_dir / "resolved_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    s3_module = load_module("revision5_s3_driver", code / "train_s3.py")
    init_args = SimpleNamespace(
        experiment="S3-B",
        run_dir=str(args.output_dir / "runtime"),
        config=str(config_path),
        trainer=str(code / "train_s2b_3090.py"),
        finalizer=str(code / "finalize_s2b_3090.py"),
        checkpoint=str(steady / "checkpoint/frozen_s2b_validation.pt"),
        bank=str(steady / "data/perturbation_bank_train_validation.npz"),
        validation_lock=str(args.output_dir / "evaluation.lock"),
        learning_rate=float(config["training"]["learning_rate"]),
    )
    driver = s3_module.S3(init_args)
    driver.init_eval_assets()
    checkpoint_path = steady / "checkpoint/frozen_s3b_contraction.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    driver.exp.model.load_state_dict(checkpoint["model"], strict=True)
    driver.exp.model.eval()

    output = {
        "schema": "steady_strict_pressure_scan/v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "horizon": 56,
        "gate_frozen_before_scan": {
            "finite_fraction": 1.0,
            "divergent_windows": 0,
            "pressure_fixed_point_physical_relative_l2_max": PRESSURE_FIXED_POINT_MAX,
            "pressure_drift_coefficient_relative_l2_max": PRESSURE_DRIFT_MAX,
            "interpretation": (
                "Supplementary strict-pressure diagnostic gate; it is not the "
                "original S3 checkpoint-selection gate."
            ),
        },
        "splits": {},
    }
    all_fixed = []
    split_terminals = {}
    for split in ("train", "validation", "heldout"):
        labels, fixed = terminal_ids(driver.exp, split)
        split_terminals[split] = (labels, fixed)
        all_fixed.extend(fixed.tolist())
    repair_dt(driver, np.asarray(all_fixed, np.int64))

    for split, (labels, fixed_ids) in split_terminals.items():
        clean = driver.finalizer.evaluate_horizon(driver.exp, split, 56)
        fixed_tensor = torch.as_tensor(
            fixed_ids, dtype=torch.long, device=driver.exp.device
        )
        a_star = driver.exp.tensor("a", fixed_tensor)
        b_star = driver.exp.tensor("b", fixed_tensor)
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=driver.exp.amp_dtype
        ):
            pred_a, pred_b = driver.free_rollout(
                fixed_tensor, a_star, b_star, 56
            )
        rows = []
        for column, label in enumerate(labels.tolist()):
            name = str(driver.exp.a["labels"][label])
            clean_row = clean["by_re"][name]
            fixed_pressure = physical_relative(
                driver.finalizer,
                pred_b[-1:, column : column + 1].float(),
                b_star[column : column + 1][None].float(),
                driver.exp.cache["pressure_pod_basis"],
                driver.exp.cache["pressure_mean"],
                driver.pressure_weight,
            )
            finite = bool(
                torch.isfinite(pred_a[:, column]).all()
                and torch.isfinite(pred_b[:, column]).all()
            )
            passed = (
                finite
                and clean_row["finite_fraction"] == 1.0
                and clean_row["divergent_windows"] == 0
                and fixed_pressure <= PRESSURE_FIXED_POINT_MAX
                and clean_row["pressure_drift"] <= PRESSURE_DRIFT_MAX
            )
            rows.append(
                {
                    "Re_label": name,
                    "Re": float(driver.exp.a["re"][fixed_ids[column]]),
                    "split": split,
                    "pressure_fixed_point_physical_relative_l2": fixed_pressure,
                    "pressure_drift_coefficient_relative_l2": float(
                        clean_row["pressure_drift"]
                    ),
                    "K56_pressure_physical_relative_l2": float(
                        clean_row["physical_reconstruction_area_weighted"][
                            "pressure_relative_l2"
                        ]
                    ),
                    "K56_velocity_physical_relative_l2": float(
                        clean_row["physical_reconstruction_area_weighted"][
                            "velocity_relative_l2"
                        ]
                    ),
                    "finite_fraction": float(clean_row["finite_fraction"]),
                    "divergent_windows": int(clean_row["divergent_windows"]),
                    "terminal_rollout_finite": finite,
                    "strict_pressure_pass": passed,
                }
            )
        output["splits"][split] = {
            "rows": rows,
            "pass_Re": [row["Re"] for row in rows if row["strict_pressure_pass"]],
            "pass_count": sum(row["strict_pressure_pass"] for row in rows),
            "total_Re": len(rows),
        }
        print(
            json.dumps(
                {
                    "event": "split_complete",
                    "split": split,
                    "pass_Re": output["splits"][split]["pass_Re"],
                }
            ),
            flush=True,
        )

    result_path = args.output_dir / "STEADY_PROPOSED_STRICT_PRESSURE_SCAN.json"
    result_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "complete", "output": str(result_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
