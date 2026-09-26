#!/usr/bin/env python3
"""Single-seed, fail-closed specialist ablation state machine for one regime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SEEDS = {"steady": 202607251, "hopf": 1248, "periodic": 1600}
DATAONLY_STEPS = {"steady": 11000, "hopf": 8000, "periodic": 10080}
DATAONLY_CURRICULUM = {
    "steady": "1,4,8,16",
    "hopf": "1,2,4,8",
    "periodic": "4,8,12,16",
}
PROPOSED_SHA256 = {
    "steady": "bcab5661af39e3262103ed887ce9846415467452b453f3dcc14e133f0f8bd80d",
    "hopf": "02148741ed8bc9fbec88709f69b10492e263514b2edcee76652485b399962235",
    "periodic": "b2052a746fdaab60f83e65d1b91d038a9dded8255ce74f41e3f6fca4c9e717c5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.root.resolve()
        self.run = args.run_root.resolve() / args.regime
        self.code = self.root / "paper_experiments/code"
        self.seed = SEEDS[args.regime]
        self.return_codes: dict[str, int] = {}
        if self.run.exists():
            raise FileExistsError(self.run)
        self.run.mkdir(parents=True)
        atomic_json(
            self.run / "REGIME_STARTED.json",
            {
                "status": "RUNNING",
                "regime": args.regime,
                "seed": self.seed,
                "pid": os.getpid(),
                "methods": ["vanilla-fnn", "data-only"],
                "repetitions": 1,
                "started_unix": time.time(),
            },
        )

    def command(self, name: str, command: list[str], cwd: Path) -> None:
        log = self.run / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        status = self.run / "commands" / f"{name}.json"
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            atomic_json(
                status,
                {
                    "status": "RUNNING",
                    "pid": process.pid,
                    "command": command,
                    "started_unix": time.time(),
                },
            )
            code = process.wait()
        self.return_codes[name] = code
        atomic_json(
            status,
            {
                "status": "DONE" if code == 0 else "FAILED",
                "return_code": code,
                "command": command,
                "finished_unix": time.time(),
            },
        )
        atomic_json(self.run / "RETURN_CODES.json", self.return_codes)
        if code:
            raise RuntimeError(f"{name} returned {code}")

    def checkpoint_manifest(
        self, method: str, checkpoint: Path, selection: dict[str, Any]
    ) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        atomic_json(
            self.run / method / "FROZEN_SELECTION.json",
            {
                "status": "FROZEN",
                "selection_source": "validation only",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "selection": selection,
                "test_read_before_freeze": False,
            },
        )

    def verify_proposed_checkpoint(self) -> None:
        paths = {
            "steady": self.root
            / "steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt",
            "hopf": self.root
            / "Hopf/migrated_h4_expanded/runs/"
            "HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt",
            "periodic": self.root
            / "periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt",
        }
        path = paths[self.args.regime]
        observed = sha256(path)
        if observed != PROPOSED_SHA256[self.args.regime]:
            raise RuntimeError(
                f"frozen Proposed checkpoint changed: {observed} != "
                f"{PROPOSED_SHA256[self.args.regime]}"
            )
        atomic_json(
            self.run / "STATIC_CONTRACT_PASS.json",
            {
                "status": "PASS",
                "proposed_checkpoint": str(path),
                "proposed_checkpoint_sha256": observed,
                "ru": 32,
                "rp": 32,
                "history_length": 3,
                "test_truth_read": False,
            },
        )

    def vanilla_base(self, trainer: Path, contract: Path) -> list[str]:
        return [
            self.args.python,
            str(self.code / "run_vanilla_fnn_moe.py"),
            "--ablation-mode",
            "vanilla-fnn",
            "--regime",
            self.args.regime,
            "--entrypoint",
            str(trainer),
            "--contract-output",
            str(contract),
        ]

    def write_steady_config(self, task: Path) -> Path:
        source = self.root / "steady_specialist_v1/code/training_s2b_3090.json"
        config = json.loads(source.read_text())
        config["vendor_trainer"] = str(
            self.root / "steady_specialist_v1/code/train_v16_4_v2_r32_compat.py"
        )
        config["artifact_dir"] = str(
            self.root / "steady_specialist_v1/source_artifacts/steady"
        )
        config["checkpoint_root"] = str(task / "training/s2/checkpoints")
        config["seed"] = self.seed
        config["training"]["benchmark_candidates"] = [
            {"micro_batch": 8, "grad_accum": 8}
        ]
        config["training"]["gpu_test_seconds"] = 15
        config["swanlab"] = {
            "project": "PaperSpecialistAblationsSingleSeed20260724",
            "group": "Vanilla-FNN-steady",
            "experiment": f"Vanilla-FNN-steady-seed{self.seed}",
        }
        path = task / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        (task / "config.sha256").write_text(
            f"{sha256(path)}  config.json\n", encoding="utf-8"
        )
        return path

    def vanilla_steady(self) -> Path:
        task = self.run / "vanilla-fnn"
        steady = self.root / "steady_specialist_v1"
        config = self.write_steady_config(task)
        trainer = steady / "code/train_s2b_3090.py"
        resume = (
            self.root
            / "paper_experiments/runs/specialist_ablations_20260723_v2/steady/"
            "vanilla_fnn/seed_202607251/s2/checkpoints/latest.pt"
        )
        preflight = self.vanilla_base(
            trainer, task / "allowlist_preflight.json"
        ) + [
            "--config",
            str(config),
            "--run-dir",
            str(task / "preflight"),
            "--mode",
            "preflight",
        ]
        self.command("vanilla_preflight", preflight, steady)
        training_runtime = task / "training/s2/runtime"
        training_runtime.mkdir(parents=True, exist_ok=True)
        for name in ("preflight_report.json", "benchmark.json", "calibration.json"):
            source = task / "preflight" / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, training_runtime / name)
        train = self.vanilla_base(trainer, task / "allowlist_train.json") + [
            "--config",
            str(config),
            "--run-dir",
            str(training_runtime),
            "--resume",
            str(resume),
            "--mode",
            "train",
        ]
        self.command("vanilla_train_s2", train, steady)
        s2 = task / "training/s2/checkpoints/best-qualified.pt"
        if not s2.is_file():
            s2 = task / "training/s2/checkpoints/final.pt"
        s3 = task / "training/s3"
        s3_command = self.vanilla_base(
            steady / "code/train_s3.py", task / "allowlist_s3.json"
        ) + [
            "--experiment",
            "S3-B",
            "--trainer",
            str(trainer),
            "--finalizer",
            str(steady / "code/finalize_s2b_3090.py"),
            "--config",
            str(config),
            "--checkpoint",
            str(s2),
            "--bank",
            str(steady / "data/perturbation_bank_train_validation.npz"),
            "--run-dir",
            str(s3),
            "--validation-lock",
            str(task / "validation.lock"),
            "--learning-rate",
            "1.5516372391099407e-5",
        ]
        self.command("vanilla_train_s3", s3_command, steady)
        s3_best = s3 / "checkpoints/best_contraction.pt"
        s4 = task / "training/s4"
        s4_command = self.vanilla_base(
            steady / "code/train_s4_steady.py", task / "allowlist_s4.json"
        ) + [
            "--trainer",
            str(trainer),
            "--finalizer",
            str(steady / "code/finalize_s2b_3090.py"),
            "--s3-trainer",
            str(steady / "code/train_s3.py"),
            "--config",
            str(config),
            "--s2b-checkpoint",
            str(s2),
            "--s3b-checkpoint",
            str(s3_best),
            "--bank",
            str(steady / "data/perturbation_bank_train_validation.npz"),
            "--run-dir",
            str(s4),
            "--learning-rate",
            "1.5516372391099407e-5",
            "--micro-batch",
            "8",
            "--grad-accum",
            "8",
        ]
        self.command("vanilla_train_s4", s4_command, steady)
        selected = s4 / "checkpoints/best_protected.pt"
        if not selected.is_file():
            selected = s3_best
        self.checkpoint_manifest(
            "vanilla-fnn", selected, {"stage": "S4" if selected.parent == s4 / "checkpoints" else "S3-B"}
        )
        return selected

    def hopf_command(self, output: Path, extra: list[str], contract: Path) -> list[str]:
        base = self.root / "Hopf/migrated_h4_expanded"
        return self.vanilla_base(
            base / "code/train_h4_expanded.py", contract
        ) + [
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
            str(self.seed),
            "--max-steps",
            "8000",
            "--no-amp",
            "--gpu-memory-fraction",
            "0.32",
            "--swanlab-project",
            "PaperSpecialistAblationsSingleSeed20260724",
            "--swanlab-group",
            "Vanilla-FNN-hopf",
            "--validation-lock",
            str(self.run / "vanilla-fnn/validation.lock"),
            *extra,
        ]

    def vanilla_hopf(self) -> Path:
        task = self.run / "vanilla-fnn"
        base = self.root / "Hopf/migrated_h4_expanded"
        self.command(
            "vanilla_preflight",
            self.hopf_command(
                task / "preflight",
                [
                    "--preflight-per-re",
                    "--preflight-windows",
                    "1",
                    "--hopf-preflight-center-window",
                    "--swanlab-mode",
                    "disabled",
                ],
                task / "allowlist_preflight.json",
            ),
            base,
        )
        self.command(
            "vanilla_validation_rollout_smoke",
            self.hopf_command(
                task / "validation_smoke_runtime",
                [
                    "--hopf-validation-smoke-output",
                    str(task / "validation_smoke/K1_K4_K8_K16.json"),
                    "--swanlab-mode",
                    "disabled",
                ],
                task / "allowlist_validation_smoke.json",
            ),
            base,
        )
        self.command(
            "vanilla_throughput",
            self.hopf_command(
                task / "throughput",
                [
                    "--benchmark-steps",
                    "20",
                    "--benchmark-horizon",
                    "1",
                    "--swanlab-mode",
                    "disabled",
                ],
                task / "allowlist_throughput.json",
            ),
            base,
        )
        self.command(
            "vanilla_train",
            self.hopf_command(
                task / "training",
                ["--swanlab-mode", "online"],
                task / "allowlist_train.json",
            ),
            base,
        )
        selected = (
            task
            / "training/HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt"
        )
        self.checkpoint_manifest("vanilla-fnn", selected, {"budget_steps": 8000})
        return selected

    def periodic_command(
        self, output: Path, epochs: int, contract: Path, resume: Path | None
    ) -> list[str]:
        base = self.root / "periodic_specialist_r32"
        command = self.vanilla_base(
            base / "code/train_periodic_moe.py", contract
        ) + [
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
            f"VanillaFNN_periodic_seed{self.seed}",
            "--saved-seed",
            str(self.seed),
            "--saved-epochs",
            str(epochs),
            "--saved-swanlab-project",
            "PaperSpecialistAblationsSingleSeed20260724",
            "--saved-swanlab-group",
            "Vanilla-FNN-periodic",
            "--saved-swanlab-run-name",
            f"Vanilla-FNN-periodic-seed{self.seed}",
        ]
        if resume is not None:
            command.extend(["--saved-resume-checkpoint", str(resume)])
        return command

    def vanilla_periodic(self) -> Path:
        task = self.run / "vanilla-fnn"
        base = self.root / "periodic_specialist_r32"
        self.command(
            "vanilla_preflight",
            self.periodic_command(
                task / "preflight", 1, task / "allowlist_preflight.json", None
            ),
            base,
        )
        self.command(
            "vanilla_preflight_rollout_smoke",
            [
                self.args.python,
                str(self.code / "evaluate_specialist_ablation.py"),
                "--root",
                str(self.root),
                "--regime",
                "periodic",
                "--method",
                "vanilla-fnn",
                "--checkpoint",
                str(task / "preflight/best_validation.pt"),
                "--split",
                "validation",
                "--output-dir",
                str(task / "preflight_rollout_smoke"),
            ],
            self.root,
        )
        resume = (
            self.root
            / "paper_experiments/runs/specialist_ablations_20260723_periodic_v3/"
            "periodic/vanilla_fnn/seed_1600/latest.pt"
        )
        self.command(
            "vanilla_train",
            self.periodic_command(
                task / "training",
                240,
                task / "allowlist_train.json",
                resume,
            ),
            base,
        )
        selected = task / "training/best_validation.pt"
        self.checkpoint_manifest(
            "vanilla-fnn", selected, {"budget_epochs": 240}
        )
        return selected

    def evaluate(self, method: str, checkpoint: Path) -> None:
        for split in ("validation", "test"):
            if split == "test":
                validation = self.run / method / "evaluation/validation/PASS.json"
                if not validation.is_file():
                    raise RuntimeError("test remains sealed because validation evaluation did not pass")
                atomic_json(
                    self.run / method / "TEST_UNSEALED_ONCE.json",
                    {
                        "reason": "validation unified evaluator passed",
                        "test_runs_allowed": 1,
                        "unsealed_unix": time.time(),
                    },
                )
            command = [
                self.args.python,
                str(self.code / "evaluate_specialist_ablation.py"),
                "--root",
                str(self.root),
                "--regime",
                self.args.regime,
                "--method",
                method,
                "--checkpoint",
                str(checkpoint),
                "--split",
                split,
                "--output-dir",
                str(self.run / method / "evaluation" / split),
            ]
            self.command(f"{method}_evaluate_{split}", command, self.root)

    def data_only(self) -> Path:
        task = self.run / "data-only"
        if self.args.regime == "hopf":
            source = (
                self.root
                / "paper_experiments/runs/specialist_ablations_20260723_v2/hopf/"
                "dataonly/seed_1248/best_validation.pt"
            )
            self.checkpoint_manifest(
                "data-only",
                source,
                {
                    "adopted_completed_training": True,
                    "best_step": 7600,
                    "final_step": 8000,
                },
            )
            return source
        smoke = task / "smoke"
        common = [
            self.args.python,
            str(self.code / "train_dataonly_moe.py"),
            "--ablation-mode",
            "data-only",
            "--root",
            str(self.root),
            "--regime",
            self.args.regime,
            "--seed",
            str(self.seed),
            "--max-steps",
            str(DATAONLY_STEPS[self.args.regime]),
            "--curriculum-horizons",
            DATAONLY_CURRICULUM[self.args.regime],
            "--micro-batch",
            "8",
            "--effective-batch",
            "24",
            "--validation-every",
            "200",
            "--swanlab-project",
            "PaperSpecialistAblationsSingleSeed20260724",
        ]
        self.command(
            "data_only_smoke",
            [*common, "--output-dir", str(smoke), "--smoke-only"],
            self.root,
        )
        formal = task / "training"
        self.command(
            "data_only_train",
            [*common, "--output-dir", str(formal)],
            self.root,
        )
        selected = formal / "best_validation.pt"
        self.checkpoint_manifest(
            "data-only",
            selected,
            {"budget_steps": DATAONLY_STEPS[self.args.regime]},
        )
        return selected

    def run_all(self) -> None:
        errors: dict[str, str] = {}
        try:
            self.verify_proposed_checkpoint()
            if self.args.regime == "steady":
                vanilla = self.vanilla_steady()
            elif self.args.regime == "hopf":
                vanilla = self.vanilla_hopf()
            else:
                vanilla = self.vanilla_periodic()
            self.evaluate("vanilla-fnn", vanilla)
            atomic_json(
                self.run / "vanilla-fnn/DONE.json",
                {"status": "DONE", "checkpoint": str(vanilla)},
            )
        except Exception as error:
            errors["vanilla-fnn"] = repr(error)
            atomic_json(
                self.run / "vanilla-fnn/FAILED.json",
                {
                    "status": "FAILED",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
        atomic_json(
            self.run / "VANILLA_TERMINAL.json",
            {
                "status": "DONE" if "vanilla-fnn" not in errors else "FAILED",
                "finished_unix": time.time(),
            },
        )
        if self.args.vanilla_only:
            atomic_json(
                self.run / "REGIME_DONE.json",
                {
                    "status": "DONE" if not errors else "FAILED",
                    "regime": self.args.regime,
                    "methods": ["vanilla-fnn"],
                    "return_codes": self.return_codes,
                    "errors": errors,
                    "finished_unix": time.time(),
                },
            )
            if errors:
                raise RuntimeError(f"terminal method failures: {errors}")
            return
        barrier = self.args.run_root.resolve() / "START_DATAONLY"
        while not barrier.is_file():
            time.sleep(10)
        try:
            data_only = self.data_only()
            self.evaluate("data-only", data_only)
            atomic_json(
                self.run / "data-only/DONE.json",
                {"status": "DONE", "checkpoint": str(data_only)},
            )
        except Exception as error:
            errors["data-only"] = repr(error)
            atomic_json(
                self.run / "data-only/FAILED.json",
                {
                    "status": "FAILED",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
        atomic_json(
            self.run / "REGIME_DONE.json",
            {
                "status": "DONE" if not errors else "PARTIAL",
                "regime": self.args.regime,
                "return_codes": self.return_codes,
                "errors": errors,
                "finished_unix": time.time(),
            },
        )
        if errors:
            raise RuntimeError(f"terminal method failures: {errors}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--regime", choices=("steady", "hopf", "periodic"), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vanilla-only", action="store_true")
    args = parser.parse_args()
    runner: Runner | None = None
    try:
        runner = Runner(args)
        runner.run_all()
        return 0
    except Exception as error:
        run = (
            runner.run
            if runner is not None
            else args.run_root.resolve() / args.regime
        )
        atomic_json(
            run / "REGIME_FAILED.json",
            {
                "status": "FAILED",
                "regime": args.regime,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "return_codes": {} if runner is None else runner.return_codes,
                "finished_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
