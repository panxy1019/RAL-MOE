#!/usr/bin/env python3
"""Run DataOnly and Vanilla-FNN ablations sequentially for one native regime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SEEDS_BY_REGIME = {
    "steady": (202607251, 202607351, 202607451),
    "hopf": (1248, 1348, 1448),
    "periodic": (1600, 1700, 1800),
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        status_path = log_path.with_suffix(".running.json")
        atomic_json(
            status_path,
            {"status": "RUNNING", "pid": process.pid, "command": command},
        )
        return_code = process.wait()
        status_path.unlink(missing_ok=True)
    if return_code:
        raise RuntimeError(f"command failed with code {return_code}: {command}")


def dataonly_command(
    python: str, root: Path, code_root: Path, output: Path, regime: str, seed: int
) -> list[str]:
    budgets = {"steady": 11000, "hopf": 8000, "periodic": 10080}
    curricula = {
        "steady": "1,4,8,16",
        "hopf": "1,2,4,8",
        "periodic": "4,8,12,16",
    }
    return [
        python,
        str(code_root / "train_dataonly_moe.py"),
        "--root",
        str(root),
        "--regime",
        regime,
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
        "--max-steps",
        str(budgets[regime]),
        "--curriculum-horizons",
        curricula[regime],
        "--micro-batch",
        "8",
        "--effective-batch",
        "24",
        "--validation-every",
        "200",
        "--swanlab-project",
        "PaperSpecialistAblations20260723",
    ]


def write_steady_config(root: Path, output: Path, seed: int) -> Path:
    source = root / "steady_specialist_v1/code/training_s2b_3090.json"
    config = json.loads(source.read_text())
    config["vendor_trainer"] = str(
        root / "steady_specialist_v1/code/train_v16_4_v2_r32_compat.py"
    )
    config["artifact_dir"] = str(root / "steady_specialist_v1/source_artifacts/steady")
    config["checkpoint_root"] = str(output / "s2/checkpoints")
    config["seed"] = seed
    config["training"]["benchmark_candidates"] = [
        {"micro_batch": 8, "grad_accum": 8}
    ]
    config["training"]["gpu_test_seconds"] = 15
    config["swanlab"] = {
        "project": "PaperSpecialistAblations20260723",
        "group": "Vanilla-FNN-steady",
        "experiment": f"Vanilla-FNN-steady-seed{seed}",
    }
    path = output / "steady_vanilla_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def run_vanilla_steady(
    python: str, root: Path, code_root: Path, output: Path, seed: int
) -> None:
    wrapper = code_root / "run_vanilla_fnn_moe.py"
    steady = root / "steady_specialist_v1"
    config = write_steady_config(root, output, seed)
    base = [
        python,
        str(wrapper),
        "--regime",
        "steady",
        "--trainer",
        str(steady / "code/train_s2b_3090.py"),
        "--config",
        str(config),
        "--run-dir",
        str(output / "s2/runtime"),
    ]
    run(base + ["--mode", "preflight"], output / "s2/preflight.log", steady)
    run(base + ["--mode", "train"], output / "s2/train.log", steady)
    s2_checkpoint = output / "s2/checkpoints/best-qualified.pt"
    if not s2_checkpoint.is_file():
        s2_checkpoint = output / "s2/checkpoints/final.pt"
    s3_output = output / "s3"
    s3_command = [
        python,
        str(wrapper),
        "--regime",
        "steady",
        "--trainer",
        str(steady / "code/train_s3.py"),
        "--experiment",
        "S3-B",
        "--trainer",
        str(steady / "code/train_s2b_3090.py"),
        "--finalizer",
        str(steady / "code/finalize_s2b_3090.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(s2_checkpoint),
        "--bank",
        str(steady / "data/perturbation_bank_train_validation.npz"),
        "--run-dir",
        str(s3_output),
        "--validation-lock",
        str(output / "validation.lock"),
        "--learning-rate",
        "1.5516372391099407e-5",
    ]
    run(s3_command, output / "s3/train.log", steady)
    s3_checkpoint = s3_output / "checkpoints/best_contraction.pt"
    if not s3_checkpoint.is_file():
        raise FileNotFoundError(s3_checkpoint)
    s4_command = [
        python,
        str(wrapper),
        "--regime",
        "steady",
        "--trainer",
        str(steady / "code/train_s4_steady.py"),
        "--trainer",
        str(steady / "code/train_s2b_3090.py"),
        "--finalizer",
        str(steady / "code/finalize_s2b_3090.py"),
        "--s3-trainer",
        str(steady / "code/train_s3.py"),
        "--config",
        str(config),
        "--s2b-checkpoint",
        str(s2_checkpoint),
        "--s3b-checkpoint",
        str(s3_checkpoint),
        "--bank",
        str(steady / "data/perturbation_bank_train_validation.npz"),
        "--run-dir",
        str(output / "s4"),
        "--learning-rate",
        "1.5516372391099407e-5",
        "--micro-batch",
        "8",
        "--grad-accum",
        "8",
    ]
    run(s4_command, output / "s4/train.log", steady)


def run_vanilla_hopf(
    python: str, root: Path, code_root: Path, output: Path, seed: int
) -> None:
    base = root / "Hopf/migrated_h4_expanded"
    command = [
        python,
        str(code_root / "run_vanilla_fnn_moe.py"),
        "--regime",
        "hopf",
        "--trainer",
        str(base / "code/train_h4_expanded.py"),
        "--variant",
        "h4",
        "--baseline-trainer",
        str(base / "code/train_hopf_moe_expanded.py"),
        "--coefficient-view",
        str(base / "assets_r32/expanded_h4_trainval_r32.npz"),
        "--galerkin-path",
        str(base / "assets_r32/expanded_h4_trainonly_galerkin_r32.npz"),
        "--pressure-path",
        str(base / "assets_r32/expanded_h4_trainonly_pressure_r32.npz"),
        "--asset-manifest",
        str(base / "assets_r32/TRAINING_ASSET_MANIFEST.json"),
        "--contract",
        str(base / "trainonly_contract/trainonly_fluctuation_contract.npz"),
        "--output-root",
        str(output),
        "--experiment-name",
        "HopfExpanded34_H4_NormalFormRadial_r32",
        "--seed",
        str(seed),
        "--max-steps",
        "8000",
        "--gpu-memory-fraction",
        "0.42",
        "--swanlab-mode",
        "online",
        "--swanlab-project",
        "PaperSpecialistAblations20260723",
        "--swanlab-group",
        "Vanilla-FNN-hopf",
        "--validation-lock",
        str(output / "validation.lock"),
    ]
    run(command, output / "train.log", base)


def run_vanilla_periodic(
    python: str, root: Path, code_root: Path, output: Path, seed: int
) -> None:
    base = root / "periodic_specialist_r32"
    command = [
        python,
        str(code_root / "run_vanilla_fnn_moe.py"),
        "--regime",
        "periodic",
        "--trainer",
        str(base / "code/train_periodic_moe.py"),
        "--saved-args-checkpoint",
        str(base / "checkpoint/FINAL_PERIODIC_SPECIALIST.pt"),
        "--saved-data-root",
        str(base / "assets/Global_POD_AreaWeighted_L2"),
        "--saved-tensor-path",
        str(base / "assets/velocity_rom_periodic.npz"),
        "--saved-pressure-surrogate-path",
        str(base / "assets/pressure_poisson_surrogate_periodic.npz"),
        "--saved-output-dir",
        str(output),
        "--saved-experiment-name",
        f"VanillaFNN_periodic_seed{seed}",
        "--saved-seed",
        str(seed),
        "--saved-epochs",
        "240",
        "--saved-swanlab-project",
        "PaperSpecialistAblations20260723",
        "--saved-swanlab-group",
        "Vanilla-FNN-periodic",
        "--saved-swanlab-run-name",
        f"Vanilla-FNN-periodic-seed{seed}",
    ]
    run(command, output / "train.log", base)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--regime", choices=("steady", "hopf", "periodic"), required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = args.root.resolve()
    experiment_root = args.experiment_root.resolve()
    chain = experiment_root / args.regime
    if chain.exists():
        raise FileExistsError(chain)
    chain.mkdir(parents=True)
    code_root = root / "paper_experiments/code"
    seeds = SEEDS_BY_REGIME[args.regime]
    started = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "regime": args.regime,
        "seeds": list(seeds),
        "methods": ["DataOnly-MoE", "Vanilla-FNN-MoE"],
        "started_unix": time.time(),
    }
    atomic_json(chain / "CHAIN_STARTED.json", started)
    try:
        if args.regime == "hopf":
            for seed in seeds:
                output = chain / "dataonly" / f"seed_{seed}"
                run(
                    dataonly_command(
                        args.python, root, code_root, output, args.regime, seed
                    ),
                    chain / "dataonly" / f"seed_{seed}.log",
                    root,
                )
            atomic_json(
                chain / "VANILLA_PREFLIGHT_BLOCKED.json",
                {
                    "status": "BLOCKED",
                    "reason": "the unmodified H4 control trainer and Vanilla wrapper both fail the current physical-backbone safety gate at sampler starts 928 and 1250 before optimizer step 1",
                    "test_truth_used": False,
                    "dataonly_completed_before_block": True,
                },
            )
            atomic_json(
                chain / "CHAIN_PARTIAL.json",
                {**started, "status": "PARTIAL", "finished_unix": time.time()},
            )
            return 0
        for seed in seeds:
            vanilla_output = chain / "vanilla_fnn" / f"seed_{seed}"
            if args.regime == "steady":
                run_vanilla_steady(
                    args.python, root, code_root, vanilla_output, seed
                )
            else:
                run_vanilla_periodic(
                    args.python, root, code_root, vanilla_output, seed
                )
            dataonly_output = chain / "dataonly" / f"seed_{seed}"
            run(
                dataonly_command(
                    args.python,
                    root,
                    code_root,
                    dataonly_output,
                    args.regime,
                    seed,
                ),
                chain / "dataonly" / f"seed_{seed}.log",
                root,
            )
        atomic_json(
            chain / "CHAIN_COMPLETE.json",
            {**started, "status": "COMPLETE", "finished_unix": time.time()},
        )
    except Exception as error:
        atomic_json(
            chain / "CHAIN_FAILED.json",
            {
                **started,
                "status": "FAILED",
                "failed_unix": time.time(),
                "error": repr(error),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
