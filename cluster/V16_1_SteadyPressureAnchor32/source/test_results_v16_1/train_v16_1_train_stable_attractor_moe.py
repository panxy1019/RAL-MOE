#!/usr/bin/env python3
"""V16 AttractorMoE-ROM for the Physics-Generalizable Attractor Database.

V16 keeps the V15/V15_3 semi-intrusive Shared Encoder + Physics-aware Experts +
Galerkin + RK4 backbone and reinterprets the Re=20--200 retained data as an
attractor database. The retained windows cover steady fixed points, weak Hopf
attractors, developing periodic attractors, and mature periodic limit cycles.

    [a_t, b_t, Re, phase, physical descriptors, history]
      -> shared encoder h_t
      -> group router pi^g and in-group velocity/pressure routers pi^u, pi^p
      -> group-shared expert + group-local Top-2 routed experts
      -> learned closure c_u, c_p with W a + low-rank a^T Q a + residual FFN
      -> f = Galerkin RHS + c_u
      -> RK4 velocity advance + pressure branch
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1.0e-12
RE_FEATURE_CENTER = 110.0
RE_FEATURE_SCALE = 90.0
RE_FEATURE_REF = 200.0
CLOSURE_MODES = (
    "baseline",
    "residual_scaling",
    "base_scaling",
    "dual_adaptive",
    "adaptive_gate",
)
PRESSURE_BASE_MODES = ("static", "film_base", "regime_aware_rom")
REGIME_ROM_NAMES = ("steady", "hopf", "periodic")
ATTRACTOR_NAMES = ("steady", "hopf", "periodic")
STEADY_REGIMES = frozenset(("steady_wake", "pre_hopf_steady"))
HOPF_REGIMES = frozenset(("hopf_transition",))
PERIODIC_REGIMES = frozenset(
    (
        "developing_periodic_shedding",
        "mature_periodic_shedding",
        "high_re_2d_periodic_near_modeA",
    )
)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0).astype(np.float32)
        scale = x.std(axis=0).astype(np.float32)
        scale[scale < 1.0e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/root/moe/ROM_PhysicsGeneralizable/data/Global_POD_AreaWeighted_L2"
        ),
    )
    parser.add_argument(
        "--tensor-path",
        type=Path,
        default=Path(
            "/root/moe/ROM_PhysicsGeneralizable/data/"
            "semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz"
        ),
    )
    parser.add_argument(
        "--pressure-surrogate-path",
        type=Path,
        default=Path(
            "/root/moe/ROM_PhysicsGeneralizable/data/"
            "pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz"
        ),
    )
    parser.add_argument(
        "--pressure-base-mode",
        choices=list(PRESSURE_BASE_MODES),
        default="static",
        help=(
            "V15_1 pressure/ROM base evolution mode. static keeps V15_Base; "
            "film_base adds a FiLM-calibrated pressure base; regime_aware_rom "
            "mixes regime-specific velocity Galerkin and pressure Poisson bases."
        ),
    )
    parser.add_argument(
        "--regime-rom-root",
        type=Path,
        default=Path("/root/moe/ROM_PhysicsGeneralizable/Regime_ROM_Library"),
    )
    parser.add_argument("--film-base-hidden", type=int, default=64)
    parser.add_argument("--film-base-scale", type=float, default=0.20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/moe/V16_AttractorMoEROM/test_results_v16/results"),
    )
    parser.add_argument("--experiment-name", default="v16_attractor_moe_rom")
    parser.add_argument("--r-u", type=int, default=16)
    parser.add_argument("--r-p", type=int, default=16)
    parser.add_argument("--test-re-indices", type=int, nargs="+", default=[10, 59, 99])
    parser.add_argument(
        "--test-re-values",
        type=float,
        nargs="*",
        default=None,
        help="Select held-out Re values by nearest available database Reynolds number.",
    )
    parser.add_argument(
        "--test-re-selection",
        choices=["explicit", "uniform", "values", "regime_default"],
        default="explicit",
        help=(
            "Use explicit indices, uniformly selected Re values, user-provided "
            "--test-re-values, or the V15 steady/Hopf/periodic default set."
        ),
    )
    parser.add_argument("--num-uniform-test-re", type=int, default=10)
    parser.add_argument(
        "--train-time-stride",
        type=int,
        default=1,
        help="Keep one training sample every N time steps within each training Re.",
    )
    parser.add_argument(
        "--train-time-offset",
        type=int,
        default=0,
        help="Offset for sparse time sampling within each training Re.",
    )
    parser.add_argument(
        "--train-re-stride",
        type=int,
        default=1,
        help="Keep one training Reynolds number every N non-test Re values.",
    )
    parser.add_argument(
        "--train-re-offset",
        type=int,
        default=0,
        help="Offset for sparse Reynolds sampling among non-test Re values.",
    )
    parser.add_argument(
        "--regime-balanced-sampling",
        action="store_true",
        help="Use inverse-frequency regime weights when drawing mini-batches.",
    )
    parser.add_argument(
        "--experiment-tag",
        default="V16_AttractorMoEROM",
        help="Human-readable experiment name for logs, reports, and SwanLab.",
    )
    parser.add_argument("--phase-harmonics", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument("--num-shared-experts", type=int, default=1)
    parser.add_argument("--num-regime-groups", type=int, default=3)
    parser.add_argument("--experts-per-group", type=int, default=6)
    parser.add_argument("--group-top-k", type=int, default=1)
    parser.add_argument("--group-temperature", type=float, default=0.9)
    parser.add_argument("--group-gate-floor", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=1024)
    parser.add_argument("--expert-blocks", type=int, default=4)
    parser.add_argument("--quadratic-rank", type=int, default=4)
    parser.add_argument("--quadratic-scale", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gate-floor", type=float, default=0.0)
    parser.add_argument("--shared-scale", type=float, default=1.0)
    parser.add_argument("--routed-scale", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=90)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1.0e-3)
    parser.add_argument(
        "--eval-only-checkpoint",
        type=Path,
        default=None,
        help="Load a saved checkpoint and run the final evaluator without training.",
    )
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--train-rollout-steps", type=int, default=16)
    parser.add_argument("--rollout-batch", type=int, default=24)
    parser.add_argument("--rollout-every-batches", type=int, default=1)
    parser.add_argument("--recon-dim", type=int, default=4096)
    parser.add_argument("--lambda-coeff", type=float, default=1.0)
    parser.add_argument("--lambda-dyn", type=float, default=1.0)
    parser.add_argument("--lambda-pressure", type=float, default=0.55)
    parser.add_argument("--lambda-recon", type=float, default=0.08)
    parser.add_argument("--lambda-rollout", type=float, default=0.30)
    parser.add_argument("--lambda-pressure-rollout", type=float, default=0.25)
    parser.add_argument("--lambda-consistency", type=float, default=0.15)
    parser.add_argument("--lambda-router-balance", type=float, default=0.02)
    parser.add_argument("--lambda-router-entropy", type=float, default=0.002)
    parser.add_argument("--lambda-group-balance", type=float, default=0.04)
    parser.add_argument("--lambda-group-entropy", type=float, default=0.001)
    parser.add_argument("--lambda-group-supervision", type=float, default=0.03)
    parser.add_argument("--lambda-router-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-expert-diversity", type=float, default=0.01)
    parser.add_argument("--lambda-regime-router", type=float, default=0.005)
    parser.add_argument("--lambda-energy", type=float, default=0.04)
    parser.add_argument("--lambda-trajectory-consistency", type=float, default=0.10)
    parser.add_argument("--lambda-alpha-rel", type=float, default=0.20)
    parser.add_argument("--lambda-rhs-rel", type=float, default=0.20)
    parser.add_argument("--lambda-pressure-rel", type=float, default=0.60)
    parser.add_argument("--hopf-pair", default="0,1")
    parser.add_argument("--hopf-re-min", type=float, default=46.0)
    parser.add_argument("--hopf-re-max", type=float, default=55.0)
    parser.add_argument("--hopf-focus-re", type=float, default=51.78645)
    parser.add_argument("--hopf-focus-width", type=float, default=0.85)
    parser.add_argument("--lambda-hopf-log-amp", type=float, default=0.0)
    parser.add_argument("--lambda-hopf-energy", type=float, default=0.0)
    parser.add_argument("--lambda-hopf-overshoot", type=float, default=0.0)
    parser.add_argument("--hopf-loss-warmup-epochs", type=int, default=30)
    parser.add_argument("--hopf-overshoot-ratio", type=float, default=2.0)
    parser.add_argument("--hopf-sample-weight", type=float, default=1.0)
    parser.add_argument("--hopf-focus-sample-weight", type=float, default=1.0)
    parser.add_argument("--hopf-rollout-weight", type=float, default=1.0)
    parser.add_argument("--hopf-diagnostic-horizons", default="8,16,24,32")
    parser.add_argument(
        "--attractor-balanced-sampling",
        action="store_true",
        help="Use inverse-frequency Steady/Hopf/Periodic attractor weights.",
    )
    parser.add_argument(
        "--attractor-conditioned",
        action="store_true",
        help="Enable explicit Attractor Router and lightweight latent adapters.",
    )
    parser.add_argument("--attractor-adapter-dim", type=int, default=64)
    parser.add_argument("--lambda-attractor-ce", type=float, default=0.0)
    parser.add_argument("--lambda-steady-rhs", type=float, default=0.0)
    parser.add_argument("--lambda-steady-state", type=float, default=0.0)
    parser.add_argument("--lambda-attractor-hopf-radius", type=float, default=0.0)
    parser.add_argument("--lambda-attractor-hopf-overshoot", type=float, default=0.0)
    parser.add_argument("--lambda-attractor-hopf-onset", type=float, default=0.0)
    parser.add_argument("--v16-1-hopf-floor-quantile", type=float, default=10.0)
    parser.add_argument("--v16-1-loss-warmup-epochs", type=int, default=0)
    parser.add_argument("--lambda-v16-1-hopf-growth", type=float, default=0.0)
    parser.add_argument("--lambda-v16-1-hopf-false-growth", type=float, default=0.0)
    parser.add_argument("--lambda-v16-1-hopf-floor-rel", type=float, default=0.0)
    parser.add_argument("--lambda-v16-1-steady-p-state", type=float, default=0.0)
    parser.add_argument("--lambda-v16-1-steady-p-mean", type=float, default=0.0)
    parser.add_argument("--lambda-v16-1-steady-p-delta", type=float, default=0.0)
    parser.add_argument("--lambda-v16-1-steady-residual-damp", type=float, default=0.0)
    parser.add_argument("--lambda-v16-1-steady-p-energy", type=float, default=0.0)
    parser.add_argument("--lambda-periodic-energy", type=float, default=0.0)
    parser.add_argument("--lambda-periodic-radius", type=float, default=0.0)
    parser.add_argument("--lambda-proto-energy", type=float, default=0.0)
    parser.add_argument("--lambda-proto-radius", type=float, default=0.0)
    parser.add_argument(
        "--attractor-r-floor",
        type=float,
        default=0.0,
        help="Hopf overshoot radius floor. <=0 estimates from training Hopf radii.",
    )
    parser.add_argument("--attractor-onset-threshold", type=float, default=5.0e-5)
    parser.add_argument("--pressure-target", choices=["closure", "state"], default="closure")
    parser.add_argument(
        "--closure-mode",
        choices=list(CLOSURE_MODES),
        default="baseline",
        help=(
            "Pressure closure fusion. baseline uses b_base+r; residual_scaling uses "
            "b_base+alpha*r; base_scaling uses alpha*b_base+r; dual_adaptive uses "
            "(1+beta)*b_base+alpha*r."
        ),
    )
    parser.add_argument(
        "--pressure-input-mode",
        choices=["pressure_only", "velocity_only", "hybrid"],
        default="pressure_only",
        help=(
            "Pressure expert state ablation. pressure_only keeps the V14 baseline "
            "current [a_t,b_t] state; velocity_only uses [a_next,0]; hybrid uses "
            "[a_next,b_base]."
        ),
    )
    parser.add_argument("--rhs-target", choices=["full", "residual"], default="residual")
    parser.add_argument("--pressure-amplitude-weight-power", type=float, default=0.0)
    parser.add_argument("--pressure-amplitude-weight-max", type=float, default=6.0)
    parser.add_argument("--rollout-relative-mix", type=float, default=0.60)
    parser.add_argument("--relative-floor-frac", type=float, default=0.02)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--curriculum-steps", default="2,4,8,16")
    parser.add_argument("--scheduled-sampling-start", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling-end", type=float, default=0.75)
    parser.add_argument("--scheduled-sampling-warmup-frac", type=float, default=0.70)
    parser.add_argument("--min-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--integrator", choices=["euler", "rk4"], default="rk4")
    parser.add_argument("--analysis-bins", type=int, default=4)
    parser.add_argument("--eval-routing-every", type=int, default=0)
    parser.add_argument("--eval-routing-max-samples", type=int, default=2048)
    parser.add_argument("--skip-expert-error-analysis", action="store_true", default=True)
    parser.add_argument(
        "--run-expert-error-analysis",
        action="store_false",
        dest="skip_expert_error_analysis",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=["disabled", "online", "local", "offline"],
        default="disabled",
        help="Enable SwanLab experiment tracking.",
    )
    parser.add_argument("--swanlab-project", default="V16_AttractorMoEROM")
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument("--swanlab-group", default="adaptive-pressure-closure")
    parser.add_argument("--swanlab-run-name", default=None)
    parser.add_argument("--swanlab-log-dir", type=Path, default=None)
    parser.add_argument(
        "--swanlab-required",
        action="store_true",
        help="Fail training if SwanLab cannot initialize.",
    )
    parser.add_argument("--allow-tf32", action="store_true", default=True)
    parser.add_argument("--no-allow-tf32", action="store_false", dest="allow_tf32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-full-json", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def swanlab_active(args: argparse.Namespace) -> bool:
    return str(getattr(args, "swanlab_mode", "disabled")) != "disabled"


def swanlab_init_run(
    args: argparse.Namespace,
    split_stats: Dict[str, object],
) -> object | None:
    if not swanlab_active(args):
        return None
    try:
        import swanlab

        api_key = os.environ.get("SWANLAB_API_KEY", "")
        if api_key and args.swanlab_mode == "online":
            swanlab.login(api_key=api_key, relogin=False, save=False, timeout=20)
        elif args.swanlab_mode == "online" and args.swanlab_required:
            raise RuntimeError("SWANLAB_API_KEY is required for online SwanLab logging.")

        config = {
            "experiment_name": args.experiment_name,
            "experiment_tag": args.experiment_tag,
            "closure_mode": args.closure_mode,
            "pressure_base_mode": args.pressure_base_mode,
            "regime_rom_root": str(args.regime_rom_root),
            "film_base_hidden": args.film_base_hidden,
            "film_base_scale": args.film_base_scale,
            "pressure_target": args.pressure_target,
            "pressure_input_mode": args.pressure_input_mode,
            "rhs_target": args.rhs_target,
            "r_u": args.r_u,
            "r_p": args.r_p,
            "hidden_dim": args.hidden_dim,
            "expert_hidden": args.expert_hidden,
            "num_regime_groups": args.num_regime_groups,
            "experts_per_group": args.experts_per_group,
            "top_k": args.top_k,
            "group_top_k": args.group_top_k,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "rollout_steps": args.rollout_steps,
            "train_rollout_steps": args.train_rollout_steps,
            "curriculum_steps": args.curriculum_steps,
            "scheduled_sampling_end": args.scheduled_sampling_end,
            "hopf_pair": args.hopf_pair,
            "hopf_re_min": args.hopf_re_min,
            "hopf_re_max": args.hopf_re_max,
            "hopf_focus_re": args.hopf_focus_re,
            "lambda_hopf_log_amp": args.lambda_hopf_log_amp,
            "lambda_hopf_energy": args.lambda_hopf_energy,
            "lambda_hopf_overshoot": args.lambda_hopf_overshoot,
            "hopf_sample_weight": args.hopf_sample_weight,
            "hopf_focus_sample_weight": args.hopf_focus_sample_weight,
            "hopf_rollout_weight": args.hopf_rollout_weight,
            "attractor_balanced_sampling": bool(args.attractor_balanced_sampling),
            "attractor_conditioned": bool(args.attractor_conditioned),
            "attractor_adapter_dim": args.attractor_adapter_dim,
            "lambda_attractor_ce": args.lambda_attractor_ce,
            "lambda_steady_rhs": args.lambda_steady_rhs,
            "lambda_steady_state": args.lambda_steady_state,
            "lambda_attractor_hopf_radius": args.lambda_attractor_hopf_radius,
            "lambda_attractor_hopf_overshoot": args.lambda_attractor_hopf_overshoot,
            "lambda_attractor_hopf_onset": args.lambda_attractor_hopf_onset,
            "lambda_periodic_energy": args.lambda_periodic_energy,
            "lambda_periodic_radius": args.lambda_periodic_radius,
            "lambda_proto_energy": args.lambda_proto_energy,
            "lambda_proto_radius": args.lambda_proto_radius,
            "train_time_stride": args.train_time_stride,
            "train_re_stride": args.train_re_stride,
            "regime_balanced_sampling": bool(args.regime_balanced_sampling),
            "test_re_selection": args.test_re_selection,
            "test_re_values": args.test_re_values,
            "num_uniform_test_re": args.num_uniform_test_re,
            "train_re_count": split_stats.get("train_re_count"),
            "test_re_count": split_stats.get("test_re_count"),
            "holdout_Re": split_stats.get("holdout_Re"),
            "kept_train_samples": split_stats.get("kept_train_samples"),
            "compression_ratio_vs_dense_train": split_stats.get(
                "compression_ratio_vs_dense_train"
            ),
        }
        run_name = args.swanlab_run_name or args.experiment_name
        if len(run_name) > 60:
            run_name = f"v15_{args.experiment_tag.lower()}"
        kwargs = {
            "project": args.swanlab_project,
            "name": run_name,
            "group": args.swanlab_group,
            "mode": args.swanlab_mode,
            "config": config,
            "tags": [
                "V16_Attractor",
                str(args.experiment_tag)[:20],
                str(args.closure_mode),
                "HPRS_MoE",
                "RK4",
            ],
            "reinit": True,
            "parallel": "shared",
        }
        if args.swanlab_workspace:
            kwargs["workspace"] = args.swanlab_workspace
        if args.swanlab_log_dir:
            kwargs["log_dir"] = str(args.swanlab_log_dir)
        return swanlab.init(**kwargs)
    except Exception as exc:
        if args.swanlab_required:
            raise
        print(
            json.dumps({"event": "swanlab_init_failed", "error": repr(exc)}),
            flush=True,
        )
        return None


def swanlab_log(run: object | None, metrics: Dict[str, object], step: int | None = None) -> None:
    if run is None:
        return
    clean: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, float)):
            val = float(value)
            if math.isfinite(val):
                clean[key] = val
        elif isinstance(value, (np.integer, int)):
            clean[key] = int(value)
    if not clean:
        return
    try:
        import swanlab

        if step is None:
            swanlab.log(clean)
        else:
            swanlab.log(clean, step=step)
    except Exception as exc:
        print(json.dumps({"event": "swanlab_log_failed", "error": repr(exc)}), flush=True)


def swanlab_finish(run: object | None, state: str = "success") -> None:
    if run is None:
        return
    try:
        import swanlab

        swanlab.finish(state=state)
    except Exception as exc:
        print(json.dumps({"event": "swanlab_finish_failed", "error": repr(exc)}), flush=True)


def swanlab_epoch_payload(
    batch_losses: List[Dict[str, float]],
    val: Dict[str, object],
    val_score: float,
    epoch: int,
    args: argparse.Namespace,
    lr: float,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "epoch": int(epoch),
        "train/lr": float(lr),
        "train/scheduled_sampling": float(args.scheduled_sampling_probability),
        "train/curriculum_rollout_steps": int(curriculum_step_for_epoch(args, epoch)),
        "train/hopf_loss_weight": float(hopf_loss_warmup(args, epoch)),
        "val/score": float(val_score),
        "val/rhs_relative_l2": float(val["rhs_relative_l2"]),
        "val/velocity_one_step_relative_l2": float(val["alpha_head_relative_l2"]),
        "val/pressure_relative_l2": float(val["pressure_head_relative_l2"]),
        "val/pressure_energy_relative_error": float(
            val.get("pressure_energy_relative_error", float("nan"))
        ),
        "closure/alpha_mean": float(val.get("closure_alpha_mean", float("nan"))),
        "closure/alpha_std": float(val.get("closure_alpha_std", float("nan"))),
        "closure/beta_mean": float(val.get("closure_beta_mean", float("nan"))),
        "closure/beta_std": float(val.get("closure_beta_std", float("nan"))),
        "closure/base_scale_mean": float(val.get("closure_base_scale_mean", float("nan"))),
        "closure/residual_scale_mean": float(
            val.get("closure_residual_scale_mean", float("nan"))
        ),
        "closure/base_error_mean": float(val.get("closure_base_error_mean", float("nan"))),
        "closure/residual_magnitude_mean": float(
            val.get("closure_residual_magnitude_mean", float("nan"))
        ),
        "closure/base_contribution_ratio_mean": float(
            val.get("closure_base_contribution_ratio_mean", float("nan"))
        ),
        "closure/residual_contribution_ratio_mean": float(
            val.get("closure_residual_contribution_ratio_mean", float("nan"))
        ),
        "closure/alpha_base_error_corr": float(
            val.get("closure_alpha_base_error_corr", float("nan"))
        ),
        "closure/alpha_residual_magnitude_corr": float(
            val.get("closure_alpha_residual_magnitude_corr", float("nan"))
        ),
    }
    if batch_losses:
        keys = sorted({key for row in batch_losses for key in row.keys()})
        for key in keys:
            vals = [
                float(row[key])
                for row in batch_losses
                if key in row and math.isfinite(float(row[key]))
            ]
            if vals:
                payload[f"train/{key}"] = float(np.mean(vals))
    return payload


def load_snapshot_index(path: Path) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda row: int(row["snapshot_id"]))
    has_phase = bool(rows) and "phase" in rows[0]
    has_period = bool(rows) and ("period" in rows[0] or "estimated_period" in rows[0])
    time = np.asarray([float(r["time"]) for r in rows], dtype=np.float32)
    re_label = np.asarray([r["Re_label"] for r in rows])
    if has_phase:
        phase = np.asarray([float(r["phase"]) % 1.0 for r in rows], dtype=np.float32)
    else:
        phase = np.zeros(len(rows), dtype=np.float32)
        for label in sorted(set(re_label.tolist())):
            mask = re_label == label
            local_t = time[mask]
            period_values: List[float] = []
            for row in np.asarray(rows, dtype=object)[mask].tolist():
                raw = row.get("period", "") or row.get("estimated_period", "")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = 0.0
                if value > 0.0 and math.isfinite(value):
                    period_values.append(value)
            if period_values:
                period = float(np.median(period_values))
                phase[mask] = ((local_t - float(local_t[0])) / period % 1.0).astype(np.float32)
            else:
                denom = max(float(local_t[-1] - local_t[0]), EPS)
                phase[mask] = ((local_t - float(local_t[0])) / denom).astype(np.float32)
    if has_period:
        periods = []
        for r in rows:
            raw = r.get("period", "") or r.get("estimated_period", "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            periods.append(value if math.isfinite(value) else 0.0)
        period_arr = np.asarray(periods, dtype=np.float32)
    else:
        period_arr = np.zeros(len(rows), dtype=np.float32)
    regimes = np.asarray([r.get("regime", "unknown") or "unknown" for r in rows])
    return {
        "snapshot_id": np.asarray([int(r["snapshot_id"]) for r in rows], dtype=np.int64),
        "Re": np.asarray([float(r["Re"]) for r in rows], dtype=np.float32),
        "Re_label": re_label,
        "regime": regimes,
        "time": time,
        "period": period_arr,
        "phase": phase,
        "local_snapshot_index": np.asarray(
            [int(float(r["local_snapshot_index"])) for r in rows], dtype=np.int64
        ),
    }


def centered_time_derivative(
    coeff: np.ndarray, re_values: np.ndarray, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    deriv = np.full_like(coeff, np.nan, dtype=np.float32)
    valid: List[int] = []
    for re in sorted(np.unique(re_values).tolist()):
        idx = np.where(re_values == re)[0]
        idx = idx[np.argsort(times[idx])]
        if len(idx) < 3:
            continue
        dt = (times[idx[2:]] - times[idx[:-2]]).astype(np.float32)
        deriv[idx[1:-1]] = (coeff[idx[2:]] - coeff[idx[:-2]]) / dt[:, None]
        valid.extend(idx[1:-1].tolist())
    return deriv, np.asarray(sorted(valid), dtype=np.int64)


def next_index_by_sequence(re_values: np.ndarray, times: np.ndarray) -> np.ndarray:
    nxt = np.full(re_values.shape[0], -1, dtype=np.int64)
    for re in sorted(np.unique(re_values).tolist()):
        idx = np.where(re_values == re)[0]
        idx = idx[np.argsort(times[idx])]
        nxt[idx[:-1]] = idx[1:]
    return nxt


def prev_index_by_sequence(re_values: np.ndarray, times: np.ndarray) -> np.ndarray:
    prv = np.full(re_values.shape[0], -1, dtype=np.int64)
    for re in sorted(np.unique(re_values).tolist()):
        idx = np.where(re_values == re)[0]
        idx = idx[np.argsort(times[idx])]
        prv[idx[1:]] = idx[:-1]
    return prv


def galerkin_rhs_by_label(
    tensors: np.lib.npyio.NpzFile,
    a: np.ndarray,
    b: np.ndarray,
    label_ids: np.ndarray,
    labels: np.ndarray,
    r_u: int,
    r_p: int,
) -> np.ndarray:
    rhs = np.empty((a.shape[0], r_u), dtype=np.float32)
    first_prefix = str(labels[int(label_ids[0])])
    H_key = "H" if "H" in tensors.files else f"{first_prefix}_H"
    P_key = "P" if "P" in tensors.files else f"{first_prefix}_P"
    H = tensors[H_key][:r_u, :r_u, :r_u].astype(np.float32)
    P = tensors[P_key][:r_u, :r_p].astype(np.float32)
    label_to_tensor_row = {}
    if "c_all" in tensors.files and "A_all" in tensors.files:
        computed_labels = tensors["Re_labels_computed"].astype(str)
        label_to_tensor_row = {
            str(label): i for i, label in enumerate(computed_labels.tolist())
        }
    for label_id in sorted(np.unique(label_ids).tolist()):
        row = np.where(label_ids == label_id)[0]
        prefix = str(labels[int(label_id)])
        if label_to_tensor_row:
            tensor_row = label_to_tensor_row[prefix]
            c = tensors["c_all"][tensor_row, :r_u].astype(np.float32)
            A = tensors["A_all"][tensor_row, :r_u, :r_u].astype(np.float32)
        else:
            c = tensors[f"{prefix}_c"][:r_u].astype(np.float32)
            A = tensors[f"{prefix}_A"][:r_u, :r_u].astype(np.float32)
        ar = a[row]
        br = b[row]
        rhs[row] = (
            c[None, :]
            + ar @ A.T
            + np.einsum("ijk,nj,nk->ni", H, ar, ar, optimize=True)
            + br @ P.T
        )
    return rhs.astype(np.float32)


def pressure_surrogate_by_label(
    tensors: np.lib.npyio.NpzFile,
    a_next: np.ndarray,
    label_ids: np.ndarray,
    labels: np.ndarray,
    r_u: int,
    r_p: int,
) -> np.ndarray:
    out = np.empty((a_next.shape[0], r_p), dtype=np.float32)
    H_tilde = tensors["H_tilde"][:r_p, :r_u, :r_u].astype(np.float32)
    for label_id in sorted(np.unique(label_ids).tolist()):
        row = np.where(label_ids == label_id)[0]
        prefix = str(labels[int(label_id)])
        c_tilde = tensors[f"{prefix}_c_tilde"][:r_p].astype(np.float32)
        A_tilde = tensors[f"{prefix}_A_tilde"][:r_p, :r_u].astype(np.float32)
        ar = a_next[row]
        out[row] = (
            c_tilde[None, :]
            + ar @ A_tilde.T
            + np.einsum("pij,bi,bj->bp", H_tilde, ar, ar, optimize=True)
        )
    return out.astype(np.float32)


def pressure_input_raw_np(
    mode: str,
    a_next: np.ndarray,
    pressure_base: np.ndarray,
) -> np.ndarray | None:
    if mode == "pressure_only":
        return None
    if mode == "velocity_only":
        return np.concatenate([a_next, np.zeros_like(pressure_base)], axis=1).astype(np.float32)
    if mode == "hybrid":
        return np.concatenate([a_next, pressure_base], axis=1).astype(np.float32)
    raise ValueError(f"Unknown pressure input mode: {mode}")


def pressure_input_override_np(
    mode: str,
    a_next: np.ndarray,
    pressure_base: np.ndarray,
    scalers: Dict[str, Standardizer],
) -> np.ndarray | None:
    raw = pressure_input_raw_np(mode, a_next, pressure_base)
    if raw is None:
        return None
    return scalers["pressure_input"].transform(raw)


def pressure_input_override_torch(
    mode: str,
    a_next: torch.Tensor,
    pressure_base: torch.Tensor,
    scalers_t: Dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if mode == "pressure_only":
        return None
    if mode == "velocity_only":
        raw = torch.cat([a_next, torch.zeros_like(pressure_base)], dim=1)
    elif mode == "hybrid":
        raw = torch.cat([a_next, pressure_base], dim=1)
    else:
        raise ValueError(f"Unknown pressure input mode: {mode}")
    return (raw - scalers_t["pressure_input_mean"]) / scalers_t["pressure_input_scale"]


def regime_to_attractor_name(regime: str) -> str:
    name = str(regime)
    if name in STEADY_REGIMES:
        return "steady"
    if name in HOPF_REGIMES:
        return "hopf"
    if name in PERIODIC_REGIMES:
        return "periodic"
    return "periodic"


def attractor_ids_from_regimes(regimes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vocab = np.asarray(ATTRACTOR_NAMES)
    to_id = {name: i for i, name in enumerate(vocab.tolist())}
    names = np.asarray([regime_to_attractor_name(str(name)) for name in regimes], dtype=object)
    ids = np.asarray([to_id[str(name)] for name in names], dtype=np.int64)
    return names.astype(str), ids


def closure_components_torch(
    mode: str,
    b_base: torch.Tensor,
    pressure_residual: torch.Tensor,
    closure_params: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha = closure_params["alpha"]
    beta = closure_params["beta"]
    if alpha.shape[1] not in (1, b_base.shape[1]):
        raise ValueError(
            f"alpha has incompatible shape {tuple(alpha.shape)} for pressure dim {b_base.shape[1]}"
        )
    if beta.shape[1] not in (1, b_base.shape[1]):
        raise ValueError(
            f"beta has incompatible shape {tuple(beta.shape)} for pressure dim {b_base.shape[1]}"
        )
    ones = torch.ones_like(alpha)
    if mode == "baseline":
        base_scale = ones
        residual_scale = ones
    elif mode == "residual_scaling":
        base_scale = ones
        residual_scale = alpha
    elif mode == "base_scaling":
        base_scale = alpha
        residual_scale = ones
    elif mode == "dual_adaptive":
        base_scale = 1.0 + beta
        residual_scale = alpha
    elif mode == "adaptive_gate":
        base_scale = alpha
        residual_scale = ones
    else:
        raise ValueError(f"Unknown closure mode: {mode}")
    base_component = base_scale * b_base
    residual_component = residual_scale * pressure_residual
    return (
        base_component + residual_component,
        base_component,
        residual_component,
        base_scale,
        residual_scale,
    )


def closure_components_np(
    mode: str,
    b_base: np.ndarray,
    pressure_residual: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def scale_array(value: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((b_base.shape[0], 1), float(arr), dtype=np.float32)
        elif arr.ndim == 1:
            if arr.shape[0] == b_base.shape[0]:
                arr = arr.reshape(-1, 1)
            elif arr.shape[0] == b_base.shape[1]:
                arr = np.broadcast_to(arr.reshape(1, -1), b_base.shape).astype(np.float32)
            else:
                raise ValueError(f"{name} has incompatible shape {arr.shape}")
        elif arr.ndim == 2:
            if arr.shape[0] != b_base.shape[0]:
                raise ValueError(f"{name} has incompatible shape {arr.shape}")
            if arr.shape[1] not in (1, b_base.shape[1]):
                raise ValueError(f"{name} has incompatible shape {arr.shape}")
        else:
            raise ValueError(f"{name} has incompatible shape {arr.shape}")
        return arr.astype(np.float32)

    alpha = scale_array(alpha, "alpha")
    beta = scale_array(beta, "beta")
    ones = np.ones_like(alpha, dtype=np.float32)
    if mode == "baseline":
        base_scale = ones
        residual_scale = ones
    elif mode == "residual_scaling":
        base_scale = ones
        residual_scale = alpha
    elif mode == "base_scaling":
        base_scale = alpha
        residual_scale = ones
    elif mode == "dual_adaptive":
        base_scale = 1.0 + beta
        residual_scale = alpha
    elif mode == "adaptive_gate":
        base_scale = alpha
        residual_scale = ones
    else:
        raise ValueError(f"Unknown closure mode: {mode}")
    base_component = base_scale * b_base
    residual_component = residual_scale * pressure_residual
    return (
        base_component + residual_component,
        base_component,
        residual_component,
        base_scale,
        residual_scale,
    )


def finite_corr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if float(np.std(xv)) < 1.0e-12 or float(np.std(yv)) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def summarize_closure_np(
    mode: str,
    y_pressure: np.ndarray,
    b_base: np.ndarray,
    pressure_residual: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    re_values: np.ndarray,
    time_values: np.ndarray,
) -> Dict[str, object]:
    pressure_pred, base_component, residual_component, base_scale, residual_scale = (
        closure_components_np(mode, b_base, pressure_residual, alpha, beta)
    )
    y_norm = np.linalg.norm(y_pressure, axis=1) + EPS
    pred_norm = np.linalg.norm(pressure_pred, axis=1) + EPS
    base_error = np.linalg.norm(y_pressure - b_base, axis=1) / y_norm
    closure_error = np.linalg.norm(y_pressure - pressure_pred, axis=1) / y_norm
    residual_mag = np.linalg.norm(pressure_residual, axis=1) / y_norm
    effective_residual_mag = np.linalg.norm(residual_component, axis=1) / y_norm
    base_contrib = np.linalg.norm(base_component, axis=1) / pred_norm
    residual_contrib = np.linalg.norm(residual_component, axis=1) / pred_norm
    alpha_arr = np.asarray(alpha, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)
    base_scale_arr = np.asarray(base_scale, dtype=np.float64)
    residual_scale_arr = np.asarray(residual_scale, dtype=np.float64)
    if alpha_arr.ndim == 1:
        alpha_arr = alpha_arr.reshape(-1, 1)
    if beta_arr.ndim == 1:
        beta_arr = beta_arr.reshape(-1, 1)
    if base_scale_arr.ndim == 1:
        base_scale_arr = base_scale_arr.reshape(-1, 1)
    if residual_scale_arr.ndim == 1:
        residual_scale_arr = residual_scale_arr.reshape(-1, 1)
    alpha_sample = alpha_arr.mean(axis=1)
    beta_sample = beta_arr.mean(axis=1)
    base_scale_sample = base_scale_arr.mean(axis=1)
    residual_scale_sample = residual_scale_arr.mean(axis=1)
    order = np.argsort(time_values)
    time_series = []
    for idx in order.tolist():
        time_series.append(
            {
                "time": float(time_values[idx]),
                "Re": float(re_values[idx]),
                "alpha": float(alpha_sample[idx]),
                "beta": float(beta_sample[idx]),
                "base_scale": float(base_scale_sample[idx]),
                "residual_scale": float(residual_scale_sample[idx]),
                "base_error": float(base_error[idx]),
                "closure_error": float(closure_error[idx]),
                "residual_magnitude": float(residual_mag[idx]),
                "effective_residual_magnitude": float(effective_residual_mag[idx]),
                "base_contribution_ratio": float(base_contrib[idx]),
                "residual_contribution_ratio": float(residual_contrib[idx]),
            }
        )

    def stats(values: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "alpha": stats(alpha_sample),
        "beta": stats(beta_sample),
        "base_scale": stats(base_scale_sample),
        "residual_scale": stats(residual_scale_sample),
        "alpha_per_mode_mean": [float(v) for v in alpha_arr.mean(axis=0).tolist()],
        "alpha_per_mode_std": [float(v) for v in alpha_arr.std(axis=0).tolist()],
        "base_scale_per_mode_mean": [
            float(v) for v in base_scale_arr.mean(axis=0).tolist()
        ],
        "base_scale_per_mode_std": [
            float(v) for v in base_scale_arr.std(axis=0).tolist()
        ],
        "base_error": stats(base_error),
        "closure_error": stats(closure_error),
        "residual_magnitude": stats(residual_mag),
        "effective_residual_magnitude": stats(effective_residual_mag),
        "base_contribution_ratio": stats(base_contrib),
        "residual_contribution_ratio": stats(residual_contrib),
        "alpha_base_error_corr": finite_corr_np(alpha_sample, base_error),
        "alpha_residual_magnitude_corr": finite_corr_np(alpha_sample, residual_mag),
        "beta_base_error_corr": finite_corr_np(beta_sample, base_error),
        "beta_residual_magnitude_corr": finite_corr_np(beta_sample, residual_mag),
        "time_series": time_series,
    }


def make_features_np(
    a: np.ndarray,
    b: np.ndarray,
    rhs: np.ndarray,
    re_values: np.ndarray,
    phase: np.ndarray,
    harmonics: int,
) -> np.ndarray:
    re = re_values.astype(np.float32)
    re_norm = ((re - RE_FEATURE_CENTER) / RE_FEATURE_SCALE)[:, None]
    inv_re = (RE_FEATURE_REF / np.maximum(re, EPS))[:, None]
    theta = (2.0 * np.pi * phase.astype(np.float32))[:, None]
    cols = [re_norm, inv_re]
    for k in range(1, harmonics + 1):
        cols.append(np.sin(k * theta))
        cols.append(np.cos(k * theta))

    r_u = a.shape[1]
    low = min(4, r_u)
    split = min(12, r_u)
    e_low = np.linalg.norm(a[:, :low], axis=1, keepdims=True)
    e_mid = np.linalg.norm(a[:, low:split], axis=1, keepdims=True)
    e_high = np.linalg.norm(a[:, split:], axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    rhs_norm = np.linalg.norm(rhs, axis=1, keepdims=True)
    a_energy = np.sum(a * a, axis=1, keepdims=True)
    b_energy = np.sum(b * b, axis=1, keepdims=True)
    total_energy = a_energy + b_energy
    low_fraction = (e_low * e_low) / (a_energy + EPS)
    high_fraction = (e_high * e_high) / (a_energy + EPS)
    pressure_velocity_ratio = b_norm / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)
    cols.extend(
        [
            a,
            b,
            rhs,
            e_low,
            e_mid,
            e_high,
            b_norm,
            rhs_norm,
            a_energy,
            b_energy,
            total_energy,
            low_fraction,
            high_fraction,
            pressure_velocity_ratio,
        ]
    )
    return np.hstack(cols).astype(np.float32)


def history_index_matrix(
    sample_ids: np.ndarray,
    prev_idx: np.ndarray,
    history_len: int,
) -> np.ndarray:
    hist = np.full((len(sample_ids), max(history_len, 1)), -1, dtype=np.int64)
    for row, sid in enumerate(sample_ids.tolist()):
        cur = int(sid)
        for h in range(max(history_len, 1)):
            if cur < 0:
                break
            hist[row, h] = cur
            cur = int(prev_idx[cur])
    return hist


def make_history_features_np(
    base_x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    rhs: np.ndarray,
    hist_idx: np.ndarray,
) -> np.ndarray:
    current = hist_idx[:, 0]
    cols = [base_x[current]]
    for h in range(1, hist_idx.shape[1]):
        idx = hist_idx[:, h]
        valid = idx >= 0
        a_hist = np.zeros((hist_idx.shape[0], a.shape[1]), dtype=np.float32)
        b_hist = np.zeros((hist_idx.shape[0], b.shape[1]), dtype=np.float32)
        rhs_hist = np.zeros((hist_idx.shape[0], rhs.shape[1]), dtype=np.float32)
        da = np.zeros_like(a_hist)
        db = np.zeros_like(b_hist)
        drhs = np.zeros_like(rhs_hist)
        if np.any(valid):
            a_hist[valid] = a[idx[valid]]
            b_hist[valid] = b[idx[valid]]
            rhs_hist[valid] = rhs[idx[valid]]
            da[valid] = a[current[valid]] - a[idx[valid]]
            db[valid] = b[current[valid]] - b[idx[valid]]
            drhs[valid] = rhs[current[valid]] - rhs[idx[valid]]
        cols.extend([a_hist, b_hist, rhs_hist, da, db, drhs])
    return np.hstack(cols).astype(np.float32)


def make_history_features_from_current_np(
    base_current: np.ndarray,
    a_current: np.ndarray,
    b_current: np.ndarray,
    rhs_current: np.ndarray,
    hist_idx: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    cols = [base_current]
    for h in range(1, hist_idx.shape[1]):
        idx = hist_idx[:, h]
        valid = idx >= 0
        a_hist = np.zeros((hist_idx.shape[0], a.shape[1]), dtype=np.float32)
        b_hist = np.zeros((hist_idx.shape[0], b.shape[1]), dtype=np.float32)
        rhs_hist = np.zeros((hist_idx.shape[0], rhs.shape[1]), dtype=np.float32)
        da = np.zeros_like(a_hist)
        db = np.zeros_like(b_hist)
        drhs = np.zeros_like(rhs_hist)
        if np.any(valid):
            a_hist[valid] = a[idx[valid]]
            b_hist[valid] = b[idx[valid]]
            rhs_hist[valid] = rhs[idx[valid]]
            da[valid] = a_current[valid] - a_hist[valid]
            db[valid] = b_current[valid] - b_hist[valid]
            drhs[valid] = rhs_current[valid] - rhs_hist[valid]
        cols.extend([a_hist, b_hist, rhs_hist, da, db, drhs])
    return np.hstack(cols).astype(np.float32)


def make_history_features_from_states_np(
    base_current: np.ndarray,
    a_current: np.ndarray,
    b_current: np.ndarray,
    rhs_current: np.ndarray,
    a_hist: np.ndarray,
    b_hist: np.ndarray,
    rhs_hist: np.ndarray,
) -> np.ndarray:
    cols = [base_current]
    for h in range(1, a_hist.shape[1]):
        ah = a_hist[:, h]
        bh = b_hist[:, h]
        rh = rhs_hist[:, h]
        cols.extend([ah, bh, rh, a_current - ah, b_current - bh, rhs_current - rh])
    return np.hstack(cols).astype(np.float32)


def shift_history_for_current_np(hist_row: np.ndarray, current_id: int) -> np.ndarray:
    shifted = hist_row.copy()
    if shifted.shape[0] > 1:
        shifted[1:] = shifted[:-1]
    shifted[0] = current_id
    return shifted


def make_features_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    rhs: torch.Tensor,
    re_values: torch.Tensor,
    phase: torch.Tensor,
    harmonics: int,
) -> torch.Tensor:
    re = re_values.float()
    cols = [
        ((re - RE_FEATURE_CENTER) / RE_FEATURE_SCALE).unsqueeze(1),
        (RE_FEATURE_REF / torch.clamp(re, min=EPS)).unsqueeze(1),
    ]
    theta = 2.0 * math.pi * phase.float()
    for k in range(1, harmonics + 1):
        cols.append(torch.sin(k * theta).unsqueeze(1))
        cols.append(torch.cos(k * theta).unsqueeze(1))
    r_u = a.shape[1]
    low = min(4, r_u)
    split = min(12, r_u)
    e_low = torch.linalg.norm(a[:, :low], dim=1, keepdim=True)
    e_mid = torch.linalg.norm(a[:, low:split], dim=1, keepdim=True)
    e_high = torch.linalg.norm(a[:, split:], dim=1, keepdim=True)
    a_norm = torch.linalg.norm(a, dim=1, keepdim=True)
    b_norm = torch.linalg.norm(b, dim=1, keepdim=True)
    rhs_norm = torch.linalg.norm(rhs, dim=1, keepdim=True)
    a_energy = torch.sum(a * a, dim=1, keepdim=True)
    b_energy = torch.sum(b * b, dim=1, keepdim=True)
    cols.extend(
        [
            a,
            b,
            rhs,
            e_low,
            e_mid,
            e_high,
            b_norm,
            rhs_norm,
            a_energy,
            b_energy,
            a_energy + b_energy,
            (e_low * e_low) / (a_energy + EPS),
            (e_high * e_high) / (a_energy + EPS),
            b_norm / (a_norm + EPS),
        ]
    )
    return torch.cat(cols, dim=1)


def make_history_features_torch(
    base_current: torch.Tensor,
    a_current: torch.Tensor,
    b_current: torch.Tensor,
    rhs_current: torch.Tensor,
    hist_ids: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
) -> torch.Tensor:
    cols = [base_current]
    for h in range(1, hist_ids.shape[1]):
        idx = hist_ids[:, h]
        valid = idx >= 0
        safe_idx = torch.clamp(idx, min=0)
        a_hist = torch.where(valid[:, None], arrays_t["a"][safe_idx], torch.zeros_like(a_current))
        b_hist = torch.where(valid[:, None], arrays_t["b"][safe_idx], torch.zeros_like(b_current))
        rhs_hist = torch.where(valid[:, None], arrays_t["rhs_g"][safe_idx], torch.zeros_like(rhs_current))
        da = torch.where(valid[:, None], a_current - a_hist, torch.zeros_like(a_current))
        db = torch.where(valid[:, None], b_current - b_hist, torch.zeros_like(b_current))
        drhs = torch.where(valid[:, None], rhs_current - rhs_hist, torch.zeros_like(rhs_current))
        cols.extend([a_hist, b_hist, rhs_hist, da, db, drhs])
    return torch.cat(cols, dim=1)


def make_history_features_from_states_torch(
    base_current: torch.Tensor,
    a_current: torch.Tensor,
    b_current: torch.Tensor,
    rhs_current: torch.Tensor,
    a_hist: torch.Tensor,
    b_hist: torch.Tensor,
    rhs_hist: torch.Tensor,
) -> torch.Tensor:
    cols = [base_current]
    for h in range(1, a_hist.shape[1]):
        ah = a_hist[:, h]
        bh = b_hist[:, h]
        rh = rhs_hist[:, h]
        cols.extend([ah, bh, rh, a_current - ah, b_current - bh, rhs_current - rh])
    return torch.cat(cols, dim=1)


def shifted_history_ids_torch(current: torch.Tensor, arrays_t: Dict[str, torch.Tensor]) -> torch.Tensor:
    hist = arrays_t["hist_idx"][current].clone()
    if hist.shape[1] > 1:
        hist[:, 1:] = hist[:, :-1].clone()
    hist[:, 0] = current
    return hist


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicalContextEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def re_fourier_features_torch(re_values: torch.Tensor, num_freq: int = 4) -> torch.Tensor:
    re_norm = ((re_values.float() - RE_FEATURE_CENTER) / RE_FEATURE_SCALE).unsqueeze(1)
    inv_re = (RE_FEATURE_REF / torch.clamp(re_values.float(), min=EPS)).unsqueeze(1)
    cols = [re_norm, inv_re]
    for k in range(1, num_freq + 1):
        angle = k * math.pi * re_norm
        cols.extend([torch.sin(angle), torch.cos(angle)])
    return torch.cat(cols, dim=1)


class FiLMPressureBase(nn.Module):
    """Lightweight FiLM calibration for the Poisson pressure base."""

    def __init__(self, r_u: int, r_p: int, hidden_dim: int, scale: float):
        super().__init__()
        self.r_u = int(r_u)
        self.r_p = int(r_p)
        self.scale = float(scale)
        re_dim = 2 + 2 * 4
        in_dim = r_u + 2 * r_p + re_dim
        self.base_proj = nn.Linear(in_dim, hidden_dim)
        self.re_film = nn.Sequential(
            nn.Linear(re_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, r_p),
        )
        nn.init.xavier_uniform_(self.base_proj.weight, gain=0.15)
        nn.init.zeros_(self.base_proj.bias)
        film_final = self.re_film[-1]
        if isinstance(film_final, nn.Linear):
            nn.init.zeros_(film_final.weight)
            nn.init.zeros_(film_final.bias)
        out_final = self.out[-1]
        if isinstance(out_final, nn.Linear):
            nn.init.zeros_(out_final.weight)
            nn.init.zeros_(out_final.bias)

    def forward(
        self,
        a_next: torch.Tensor,
        static_base: torch.Tensor,
        re_values: torch.Tensor,
    ) -> torch.Tensor:
        re_feat = re_fourier_features_torch(re_values)
        z = self.base_proj(torch.cat([a_next, static_base, static_base * static_base, re_feat], dim=1))
        gamma_beta = self.re_film(re_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        z = (1.0 + gamma) * z + beta
        return static_base + self.scale * self.out(z)


def topk_router_from_logits(
    logits: torch.Tensor,
    top_k: int,
    gate_floor: float,
    temperature: float,
) -> torch.Tensor:
    probs = torch.softmax(logits / max(temperature, 1.0e-4), dim=-1)
    num_experts = probs.shape[1]
    if 0 < top_k < num_experts:
        top_val, top_idx = torch.topk(probs, top_k, dim=-1)
        gate = torch.zeros_like(probs)
        gate.scatter_(1, top_idx, top_val)
        gate = gate / (gate.sum(dim=-1, keepdim=True) + EPS)
        if top_k == 1:
            gate = gate - probs.detach() + probs
    else:
        gate = probs
    floor = max(0.0, min(gate_floor, 0.45))
    if floor > 0.0:
        gate = (1.0 - floor) * gate + floor / num_experts
        gate = gate / (gate.sum(dim=-1, keepdim=True) + EPS)
    return gate


class ExpandedFFNBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.ffn(self.norm(x))


class PhysicsAwareExpert(nn.Module):
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
        self.state_dim = state_dim
        self.out_dim = out_dim
        self.quadratic_rank = max(0, quadratic_rank)
        self.quadratic_scale = float(quadratic_scale)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(h_dim + state_dim),
            nn.Linear(h_dim + state_dim, h_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ExpandedFFNBlock(h_dim, expert_hidden, dropout)
                for _ in range(max(1, num_blocks))
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, out_dim),
        )
        self.linear = nn.Linear(state_dim, out_dim, bias=False)
        if self.quadratic_rank > 0:
            self.quad_left = nn.Parameter(
                torch.empty(out_dim, self.quadratic_rank, state_dim)
            )
            self.quad_right = nn.Parameter(
                torch.empty(out_dim, self.quadratic_rank, state_dim)
            )
        else:
            self.register_parameter("quad_left", None)
            self.register_parameter("quad_right", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight, gain=0.25)
        head = self.mlp_head[-1]
        if isinstance(head, nn.Linear):
            nn.init.xavier_uniform_(head.weight, gain=0.25)
            nn.init.zeros_(head.bias)
        if self.quadratic_rank > 0:
            nn.init.normal_(self.quad_left, mean=0.0, std=0.015)
            nn.init.normal_(self.quad_right, mean=0.0, std=0.015)

    def forward(self, h: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(torch.cat([h, state], dim=1))
        for block in self.blocks:
            z = block(z)
        out = self.mlp_head(z) + self.linear(state)
        if self.quadratic_rank > 0:
            left = torch.einsum("bs,ors->bor", state, self.quad_left)
            right = torch.einsum("bs,ors->bor", state, self.quad_right)
            quad = torch.sum(left * right, dim=2)
            out = out + self.quadratic_scale * quad
        return out


class OperatorSpaceMoEROM(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        pressure_dim: int,
        hidden_dim: int,
        expert_hidden: int,
        num_blocks: int,
        num_experts: int,
        num_operator_spaces: int,
        num_regime_groups: int,
        experts_per_group: int,
        top_k: int,
        group_top_k: int,
        dropout: float,
        temperature: float,
        gate_floor: float,
        group_temperature: float,
        group_gate_floor: float,
        shared_scale: float,
        routed_scale: float,
        expert_blocks: int,
        quadratic_rank: int,
        quadratic_scale: float,
        phase_harmonics: int,
        closure_mode: str,
        pressure_base_mode: str,
        film_base_hidden: int,
        film_base_scale: float,
        attractor_conditioned: bool = False,
        attractor_adapter_dim: int = 64,
        diversity_samples: int = 32,
    ):
        super().__init__()
        if closure_mode not in CLOSURE_MODES:
            raise ValueError(f"Unknown closure mode: {closure_mode}")
        if pressure_base_mode not in PRESSURE_BASE_MODES:
            raise ValueError(f"Unknown pressure base mode: {pressure_base_mode}")
        self.out_dim = out_dim
        self.pressure_dim = pressure_dim
        self.closure_mode = closure_mode
        self.pressure_base_mode = pressure_base_mode
        self.num_regime_groups = max(1, int(num_regime_groups))
        self.experts_per_group = max(1, int(experts_per_group or num_experts))
        self.shared_per_group = 1
        self.num_operator_spaces = self.num_regime_groups
        self.num_shared_experts = self.num_regime_groups * self.shared_per_group
        self.num_experts = self.num_regime_groups * (
            self.experts_per_group + self.shared_per_group
        )
        self.top_k = min(max(1, int(top_k)), self.experts_per_group)
        self.group_top_k = min(max(1, int(group_top_k)), self.num_regime_groups)
        self.temperature = temperature
        self.gate_floor = gate_floor
        self.group_temperature = group_temperature
        self.group_gate_floor = group_gate_floor
        self.shared_scale = shared_scale
        self.routed_scale = routed_scale
        self.expert_blocks = max(1, expert_blocks)
        self.quadratic_rank = max(0, quadratic_rank)
        self.quadratic_scale = quadratic_scale
        self.diversity_samples = max(0, diversity_samples)
        self.attractor_conditioned = bool(attractor_conditioned)
        self.attractor_adapter_dim = max(1, int(attractor_adapter_dim))
        self.a_start = 2 + 2 * int(phase_harmonics)
        self.b_start = self.a_start + out_dim
        self.encoder = PhysicalContextEncoder(in_dim, hidden_dim, dropout)
        self.refine_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(max(0, num_blocks - 1))
            ]
        )
        self.attractor_router_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, len(ATTRACTOR_NAMES)),
            )
            if self.attractor_conditioned
            else None
        )
        self.attractor_adapters = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, self.attractor_adapter_dim),
                        nn.GELU(),
                        nn.Linear(self.attractor_adapter_dim, hidden_dim),
                    )
                    for _ in ATTRACTOR_NAMES
                ]
            )
            if self.attractor_conditioned
            else nn.ModuleList()
        )
        router_dim = hidden_dim + in_dim
        self.group_router = nn.Sequential(
            nn.LayerNorm(router_dim),
            nn.Linear(router_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_regime_groups),
        )
        self.velocity_group_routers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(router_dim),
                    nn.Linear(router_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.experts_per_group),
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.pressure_group_routers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(router_dim),
                    nn.Linear(router_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.experts_per_group),
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.velocity_expert_groups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PhysicsAwareExpert(
                            hidden_dim,
                            out_dim,
                            out_dim,
                            expert_hidden,
                            self.expert_blocks,
                            self.quadratic_rank,
                            self.quadratic_scale,
                            dropout,
                        )
                        for _ in range(self.experts_per_group)
                    ]
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.pressure_expert_groups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PhysicsAwareExpert(
                            hidden_dim,
                            out_dim + pressure_dim,
                            pressure_dim,
                            expert_hidden,
                            self.expert_blocks,
                            self.quadratic_rank,
                            self.quadratic_scale,
                            dropout,
                        )
                        for _ in range(self.experts_per_group)
                    ]
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.velocity_shared_experts = nn.ModuleList(
            [
                PhysicsAwareExpert(
                    hidden_dim,
                    out_dim,
                    out_dim,
                    expert_hidden,
                    self.expert_blocks,
                    self.quadratic_rank,
                    self.quadratic_scale,
                    dropout,
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.pressure_shared_experts = nn.ModuleList(
            [
                PhysicsAwareExpert(
                    hidden_dim,
                    out_dim + pressure_dim,
                    pressure_dim,
                    expert_hidden,
                    self.expert_blocks,
                    self.quadratic_rank,
                    self.quadratic_scale,
                    dropout,
                )
                for _ in range(self.num_regime_groups)
            ]
        )
        self.velocity_shared_mixer = None
        self.pressure_shared_mixer = None
        confidence_hidden = 64 if closure_mode == "adaptive_gate" else max(16, min(64, hidden_dim // 4))
        confidence_out = pressure_dim if closure_mode == "adaptive_gate" else 2
        self.closure_confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, confidence_hidden),
            nn.GELU() if closure_mode == "adaptive_gate" else nn.SiLU(),
            nn.Linear(confidence_hidden, confidence_out),
        )
        final_conf = self.closure_confidence_head[-1]
        if isinstance(final_conf, nn.Linear):
            nn.init.xavier_uniform_(final_conf.weight, gain=0.05)
            nn.init.zeros_(final_conf.bias)
            if closure_mode != "adaptive_gate":
                final_conf.bias.data[0] = 4.0
        self.pressure_base_film = (
            FiLMPressureBase(out_dim, pressure_dim, max(16, int(film_base_hidden)), film_base_scale)
            if pressure_base_mode == "film_base"
            else None
        )
        self.rom_regime_gate_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, len(REGIME_ROM_NAMES)),
            )
            if pressure_base_mode == "regime_aware_rom"
            else None
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for block in self.refine_blocks:
            h = h + block(h)
        return h

    def _condition_attractor(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.attractor_router_head is None:
            return h, {}
        logits = self.attractor_router_head(h)
        pi = torch.softmax(logits, dim=1)
        adapter_stack = torch.stack([adapter(h) for adapter in self.attractor_adapters], dim=1)
        delta = torch.sum(pi[:, :, None] * adapter_stack, dim=1)
        h_attr = h + delta
        return h_attr, {
            "attr_logits": logits,
            "attr_pi": pi,
            "attr_adapter_delta_norm": torch.linalg.norm(delta, dim=1, keepdim=True),
            "attr_adapter_h_norm": torch.linalg.norm(h, dim=1, keepdim=True),
        }

    def _closure_params(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        ones = torch.ones((h.shape[0], 1), dtype=h.dtype, device=h.device)
        zeros = torch.zeros_like(ones)
        if self.closure_mode == "baseline":
            return {"alpha": ones, "beta": zeros}
        raw = self.closure_confidence_head(h)
        if self.closure_mode == "adaptive_gate":
            alpha = torch.sigmoid(raw)
            return {"alpha": alpha, "beta": torch.zeros_like(alpha)}
        alpha = torch.sigmoid(raw[:, 0:1])
        beta = zeros
        if self.closure_mode == "dual_adaptive":
            beta = 0.5 * torch.tanh(raw[:, 1:2])
        return {"alpha": alpha, "beta": beta}

    def pressure_base_from_static(
        self,
        a_next: torch.Tensor,
        static_base: torch.Tensor,
        re_values: torch.Tensor,
    ) -> torch.Tensor:
        if self.pressure_base_film is None:
            return static_base
        return self.pressure_base_film(a_next, static_base, re_values)

    def rom_regime_gate(self, h: torch.Tensor) -> torch.Tensor | None:
        if self.rom_regime_gate_head is None:
            return None
        return torch.softmax(self.rom_regime_gate_head(h), dim=1)

    def _state_slices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_state = x[:, self.a_start : self.a_start + self.out_dim]
        b_state = x[:, self.b_start : self.b_start + self.pressure_dim]
        return a_state, torch.cat([a_state, b_state], dim=1)

    def _sparse_mix(
        self,
        experts: nn.ModuleList,
        h: torch.Tensor,
        state: torch.Tensor,
        gate: torch.Tensor,
        return_stack: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_dim = experts[0].out_dim if len(experts) else 0
        mixed = torch.zeros((h.shape[0], out_dim), dtype=h.dtype, device=h.device)
        for expert_id, expert in enumerate(experts):
            weights = gate[:, expert_id]
            active = weights > 0.0
            if bool(torch.any(active)):
                mixed[active] = mixed[active] + weights[active, None] * expert(
                    h[active], state[active]
                )
        if return_stack and self.diversity_samples > 0:
            n = min(int(self.diversity_samples), h.shape[0])
            stack = torch.stack([expert(h[:n], state[:n]) for expert in experts], dim=1)
        else:
            stack = torch.empty((0, len(experts), out_dim), dtype=h.dtype, device=h.device)
        return mixed, stack

    def _group_mix(
        self,
        expert_groups: nn.ModuleList,
        shared_experts: nn.ModuleList,
        local_routers: nn.ModuleList,
        h: torch.Tensor,
        state: torch.Tensor,
        router_in: torch.Tensor,
        group_gate: torch.Tensor,
        return_stack: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_dim = shared_experts[0].out_dim
        mixed = torch.zeros((h.shape[0], out_dim), dtype=h.dtype, device=h.device)
        combined_gates = []
        stack_items = []
        denom = max(float(self.shared_scale + self.routed_scale), 1.0e-6)
        shared_part = float(self.shared_scale) / denom
        routed_part = float(self.routed_scale) / denom
        for group_id in range(self.num_regime_groups):
            logits = local_routers[group_id](router_in)
            local_gate = topk_router_from_logits(
                logits, self.top_k, self.gate_floor, self.temperature
            )
            group_weight = group_gate[:, group_id : group_id + 1]
            active = group_weight.squeeze(1) > 0.0
            if bool(torch.any(active)):
                routed, _ = self._sparse_mix(
                    expert_groups[group_id],
                    h[active],
                    state[active],
                    local_gate[active],
                    return_stack=False,
                )
                shared = shared_experts[group_id](h[active], state[active])
                group_out = shared_part * shared + routed_part * routed
                mixed[active] = mixed[active] + group_weight[active] * group_out
            combined_gates.append(
                torch.cat(
                    [
                        shared_part * group_weight,
                        routed_part * group_weight * local_gate,
                    ],
                    dim=1,
                )
            )
            if return_stack and self.diversity_samples > 0:
                n = min(int(self.diversity_samples), h.shape[0])
                stack_items.extend(
                    [expert(h[:n], state[:n]) for expert in expert_groups[group_id]]
                )
        combined_gate = torch.cat(combined_gates, dim=1)
        if return_stack and stack_items:
            stack = torch.stack(stack_items, dim=1)
        else:
            stack = torch.empty(
                (0, self.num_regime_groups * self.experts_per_group, out_dim),
                dtype=h.dtype,
                device=h.device,
            )
        return mixed, combined_gate, stack

    def group_router_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._encode(x)
        h, _ = self._condition_attractor(h)
        router_in = torch.cat([h, x], dim=1)
        logits = self.group_router(router_in)
        gate = topk_router_from_logits(
            logits, self.group_top_k, self.group_gate_floor, self.group_temperature
        )
        return gate, logits

    def forward(
        self,
        x: torch.Tensor,
        return_expert_stack: bool = True,
        pressure_state_override: torch.Tensor | None = None,
        return_closure_params: bool = False,
    ):
        h = self._encode(x)
        h, attr_params = self._condition_attractor(h)
        a_state, pressure_state = self._state_slices(x)
        if pressure_state_override is not None:
            pressure_state = pressure_state_override
        closure_params = self._closure_params(h)
        closure_params.update(attr_params)
        rom_gate = self.rom_regime_gate(h)
        if rom_gate is not None:
            closure_params["rom_gate"] = rom_gate
        router_in = torch.cat([h, x], dim=1)
        group_logits = self.group_router(router_in)
        group_gate = topk_router_from_logits(
            group_logits,
            self.group_top_k,
            self.group_gate_floor,
            self.group_temperature,
        )
        rhs_op, vel_gate, velocity_stack = self._group_mix(
            self.velocity_expert_groups,
            self.velocity_shared_experts,
            self.velocity_group_routers,
            h,
            a_state,
            router_in,
            group_gate,
            return_stack=return_expert_stack,
        )
        pressure_op, pressure_gate, _ = self._group_mix(
            self.pressure_expert_groups,
            self.pressure_shared_experts,
            self.pressure_group_routers,
            h,
            pressure_state,
            router_in,
            group_gate,
            return_stack=False,
        )
        base_out = (rhs_op, pressure_op, [vel_gate, pressure_gate], velocity_stack)
        if return_closure_params:
            return (*base_out, closure_params)
        return base_out


def router_regularization(gates: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if not gates:
        zero = torch.tensor(0.0)
        return zero, zero, {"balance": 0.0, "entropy": 0.0, "utilization": [1.0]}

    balance = torch.tensor(0.0, device=gates[0].device)
    entropy = torch.tensor(0.0, device=gates[0].device)
    utilization = []
    for gate in gates:
        mean_gate = gate.mean(dim=0)
        target = torch.full_like(mean_gate, 1.0 / gate.shape[1])
        balance = balance + torch.mean((mean_gate - target) ** 2)
        ent = -torch.sum(gate * torch.log(gate + EPS), dim=1).mean()
        entropy = entropy + ent
        utilization.append(mean_gate.detach().cpu().numpy())
    balance = balance / max(len(gates), 1)
    entropy = entropy / max(len(gates), 1)
    util = np.mean(np.stack(utilization, axis=0), axis=0) if utilization else np.zeros(1)
    return balance, entropy, {
        "balance": float(balance.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "utilization": [float(v) for v in util.tolist()],
    }


def gate_smoothness_loss(
    gates: List[torch.Tensor],
    prev_gates: List[torch.Tensor],
) -> torch.Tensor:
    if not gates or not prev_gates:
        return torch.tensor(0.0, device=gates[0].device if gates else "cpu")
    losses = []
    for gate, prev_gate in zip(gates, prev_gates):
        losses.append(F.mse_loss(gate, prev_gate))
    return torch.stack(losses).mean()


def expert_diversity_loss(expert_stack: torch.Tensor) -> torch.Tensor:
    if expert_stack.numel() == 0 or expert_stack.shape[1] <= 1:
        return torch.tensor(0.0, device=expert_stack.device)
    # Compare expert functions over the batch in operator-output space.
    expert_vectors = expert_stack.permute(1, 0, 2).reshape(expert_stack.shape[1], -1)
    expert_vectors = expert_vectors - expert_vectors.mean(dim=1, keepdim=True)
    expert_vectors = F.normalize(expert_vectors, dim=1, eps=1.0e-8)
    sim = expert_vectors @ expert_vectors.T
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return torch.mean(sim[~eye] ** 2)


def regime_router_loss(gates: List[torch.Tensor], re_values: torch.Tensor) -> torch.Tensor:
    if not gates:
        return torch.tensor(0.0, device=re_values.device)
    gate = torch.stack(gates, dim=0).mean(dim=0)
    masks = [re_values < 80.0, (re_values >= 80.0) & (re_values < 160.0), re_values >= 160.0]
    means = []
    for mask in masks:
        if bool(torch.sum(mask) >= 4):
            means.append(gate[mask].mean(dim=0))
    if len(means) <= 1:
        return torch.tensor(0.0, device=gate.device)
    sims = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            sims.append(
                F.cosine_similarity(
                    means[i].unsqueeze(0), means[j].unsqueeze(0), dim=1
                ).squeeze(0)
            )
    return torch.stack(sims).mean()


def re_regime_targets_torch(re_values: torch.Tensor, num_groups: int) -> torch.Tensor:
    targets = torch.zeros_like(re_values, dtype=torch.long)
    if num_groups <= 1:
        return targets
    if num_groups == 2:
        targets = torch.where(re_values >= 160.0, torch.ones_like(targets), targets)
        return targets
    targets = torch.where(re_values >= 80.0, torch.ones_like(targets), targets)
    high_value = torch.full_like(targets, min(num_groups - 1, 2))
    targets = torch.where(re_values >= 160.0, high_value, targets)
    return torch.clamp(targets, min=0, max=num_groups - 1)


def group_router_regularization(
    group_gate: torch.Tensor,
    group_logits: torch.Tensor,
    re_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean_gate = group_gate.mean(dim=0)
    target = torch.full_like(mean_gate, 1.0 / group_gate.shape[1])
    balance = torch.mean((mean_gate - target) ** 2)
    entropy = -torch.sum(group_gate * torch.log(group_gate + EPS), dim=1).mean()
    targets = re_regime_targets_torch(re_values, group_logits.shape[1])
    supervision = F.cross_entropy(group_logits, targets)
    return balance, entropy, supervision


def scheduled_sampling_probability(args: argparse.Namespace, epoch: int) -> float:
    start = float(np.clip(args.scheduled_sampling_start, 0.0, 1.0))
    end = float(np.clip(args.scheduled_sampling_end, 0.0, 1.0))
    warmup = float(np.clip(args.scheduled_sampling_warmup_frac, 1.0e-6, 1.0))
    denom = max(1.0, args.epochs * warmup)
    progress = float(np.clip((epoch - 1) / denom, 0.0, 1.0))
    return start + (end - start) * progress


def relative_l2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + EPS))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def relative_vector_loss_torch(
    diff: torch.Tensor, target: torch.Tensor, floor: torch.Tensor
) -> torch.Tensor:
    numerator = torch.sum(diff * diff, dim=1)
    denominator = torch.sum(target * target, dim=1) + torch.clamp(floor, min=EPS)
    return torch.mean(numerator / denominator)


def relative_vector_loss_per_sample_torch(
    diff: torch.Tensor, target: torch.Tensor, floor: torch.Tensor
) -> torch.Tensor:
    numerator = torch.sum(diff * diff, dim=1)
    denominator = torch.sum(target * target, dim=1) + torch.clamp(floor, min=EPS)
    return numerator / denominator


def weighted_mean_torch(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return torch.mean(values)
    w = weights.to(values.device, dtype=values.dtype)
    return torch.sum(values * w) / torch.clamp(torch.sum(w), min=EPS)


def parse_hopf_pair(args: argparse.Namespace) -> Tuple[int, int]:
    parts = [part.strip() for part in str(args.hopf_pair).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"--hopf-pair must contain two comma-separated indices, got {args.hopf_pair!r}")
    return int(parts[0]), int(parts[1])


def hopf_mask_torch(re_values: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    return (re_values >= float(args.hopf_re_min)) & (re_values <= float(args.hopf_re_max))


def hopf_focus_mask_torch(re_values: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    return torch.abs(re_values - float(args.hopf_focus_re)) <= float(args.hopf_focus_width)


def hopf_loss_warmup(args: argparse.Namespace, epoch: int) -> float:
    warmup = max(0, int(args.hopf_loss_warmup_epochs))
    if warmup <= 0:
        return 1.0
    return float(np.clip(epoch / max(1, warmup), 0.0, 1.0))


def v16_1_loss_warmup(args: argparse.Namespace, epoch: int) -> float:
    warmup = max(0, int(getattr(args, "v16_1_loss_warmup_epochs", 0)))
    if warmup <= 0:
        return 1.0
    return float(np.clip(epoch / max(1, warmup), 0.0, 1.0))


def hopf_envelope_loss_torch(
    pred_a: torch.Tensor,
    true_a: torch.Tensor,
    re_values: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
    weights: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    i0, i1 = parse_hopf_pair(args)
    if pred_a.shape[1] <= max(i0, i1):
        zero = torch.tensor(0.0, dtype=pred_a.dtype, device=pred_a.device)
        return zero, {"log_amp": zero, "energy": zero, "overshoot": zero, "overshoot_ratio": zero}
    active = hopf_mask_torch(re_values, args)
    if not bool(torch.any(active)):
        zero = torch.tensor(0.0, dtype=pred_a.dtype, device=pred_a.device)
        return zero, {"log_amp": zero, "energy": zero, "overshoot": zero, "overshoot_ratio": zero}
    eps = torch.tensor(1.0e-8, dtype=pred_a.dtype, device=pred_a.device)
    pred_pair = pred_a[active][:, [i0, i1]]
    true_pair = true_a[active][:, [i0, i1]]
    r_pred = torch.sqrt(torch.sum(pred_pair * pred_pair, dim=1) + eps)
    r_true = torch.sqrt(torch.sum(true_pair * true_pair, dim=1) + eps)
    log_amp_diff = torch.log(r_pred + eps) - torch.log(r_true + eps)
    log_energy_diff = torch.log(r_pred * r_pred + eps) - torch.log(r_true * r_true + eps)
    overshoot_margin = torch.log(torch.tensor(float(args.hopf_overshoot_ratio), dtype=pred_a.dtype, device=pred_a.device))
    overshoot = torch.relu(log_amp_diff - overshoot_margin) ** 2
    local_weights = weights[active] if weights is not None else None
    log_amp = weighted_mean_torch(F.huber_loss(log_amp_diff, torch.zeros_like(log_amp_diff), reduction="none"), local_weights)
    energy = weighted_mean_torch(F.huber_loss(log_energy_diff, torch.zeros_like(log_energy_diff), reduction="none"), local_weights)
    over = weighted_mean_torch(overshoot, local_weights)
    scale = hopf_loss_warmup(args, epoch)
    total = scale * (
        float(args.lambda_hopf_log_amp) * log_amp
        + float(args.lambda_hopf_energy) * energy
        + float(args.lambda_hopf_overshoot) * over
    )
    ratio = torch.mean(r_pred / torch.clamp(r_true, min=eps))
    return total, {
        "log_amp": log_amp.detach(),
        "energy": energy.detach(),
        "overshoot": over.detach(),
        "overshoot_ratio": ratio.detach(),
    }


def _attractor_weights(
    attractor_id: torch.Tensor,
    attractor_index: int,
    attr_pi: torch.Tensor | None,
) -> torch.Tensor:
    if attr_pi is not None:
        return torch.clamp(attr_pi[:, attractor_index], min=0.0)
    return (attractor_id == int(attractor_index)).to(dtype=torch.float32)


def attractor_specific_loss_torch(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    true_a: torch.Tensor,
    true_b: torch.Tensor,
    current_a: torch.Tensor,
    current_b: torch.Tensor,
    rhs_pred: torch.Tensor,
    attractor_id: torch.Tensor,
    args: argparse.Namespace,
    attr_pi: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    zero = torch.tensor(0.0, dtype=pred_a.dtype, device=pred_a.device)
    i0, i1 = parse_hopf_pair(args)
    if pred_a.shape[1] <= max(i0, i1):
        return zero, {
            "steady_rhs": zero,
            "steady_state": zero,
            "hopf_radius": zero,
            "hopf_overshoot": zero,
            "hopf_onset": zero,
            "periodic_energy": zero,
            "periodic_radius": zero,
        }

    steady_w = _attractor_weights(attractor_id, 0, attr_pi)
    hopf_w = _attractor_weights(attractor_id, 1, attr_pi)
    periodic_w = _attractor_weights(attractor_id, 2, attr_pi)

    steady_rhs = weighted_mean_torch(torch.sum(rhs_pred * rhs_pred, dim=1), steady_w)
    steady_state_raw = torch.sum((pred_a - current_a) ** 2, dim=1) + torch.sum(
        (pred_b - current_b) ** 2, dim=1
    )
    steady_state = weighted_mean_torch(steady_state_raw, steady_w)

    pred_pair = pred_a[:, [i0, i1]]
    true_pair = true_a[:, [i0, i1]]
    r_pred = torch.sqrt(torch.sum(pred_pair * pred_pair, dim=1) + EPS)
    r_true = torch.sqrt(torch.sum(true_pair * true_pair, dim=1) + EPS)
    radius_diff = F.huber_loss(r_pred, r_true, reduction="none")
    hopf_radius = weighted_mean_torch(radius_diff, hopf_w)
    r_floor = float(getattr(args, "attractor_r_floor_effective", args.attractor_r_floor))
    if not math.isfinite(r_floor) or r_floor <= 0.0:
        r_floor = 1.0e-5
    overshoot = torch.relu(r_pred - 2.0 * r_true - r_floor) ** 2
    hopf_overshoot = weighted_mean_torch(overshoot, hopf_w)
    onset_mask = (r_true < float(args.attractor_onset_threshold)).to(dtype=pred_a.dtype)
    hopf_onset = weighted_mean_torch(r_pred * r_pred, hopf_w * onset_mask)

    eu_pred = torch.sum(pred_a * pred_a, dim=1)
    eu_true = torch.sum(true_a * true_a, dim=1)
    ep_pred = torch.sum(pred_b * pred_b, dim=1)
    ep_true = torch.sum(true_b * true_b, dim=1)
    periodic_energy_raw = F.huber_loss(eu_pred, eu_true, reduction="none") + F.huber_loss(
        ep_pred, ep_true, reduction="none"
    )
    periodic_energy = weighted_mean_torch(periodic_energy_raw, periodic_w)
    periodic_radius = weighted_mean_torch(radius_diff, periodic_w)

    total = (
        float(args.lambda_steady_rhs) * steady_rhs
        + float(args.lambda_steady_state) * steady_state
        + float(args.lambda_attractor_hopf_radius) * hopf_radius
        + float(args.lambda_attractor_hopf_overshoot) * hopf_overshoot
        + float(args.lambda_attractor_hopf_onset) * hopf_onset
        + float(args.lambda_periodic_energy) * periodic_energy
        + float(args.lambda_periodic_radius) * periodic_radius
    )
    return total, {
        "steady_rhs": steady_rhs.detach(),
        "steady_state": steady_state.detach(),
        "hopf_radius": hopf_radius.detach(),
        "hopf_overshoot": hopf_overshoot.detach(),
        "hopf_onset": hopf_onset.detach(),
        "periodic_energy": periodic_energy.detach(),
        "periodic_radius": periodic_radius.detach(),
    }


def prototype_loss_torch(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    label_ids: torch.Tensor,
    proto: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    zero = torch.tensor(0.0, dtype=pred_a.dtype, device=pred_a.device)
    if not proto or (
        float(args.lambda_proto_energy) <= 0.0 and float(args.lambda_proto_radius) <= 0.0
    ):
        return zero, {"energy": zero, "radius": zero}
    i0, i1 = parse_hopf_pair(args)
    eu = torch.sum(pred_a * pred_a, dim=1)
    ep = torch.sum(pred_b * pred_b, dim=1)
    r = torch.sqrt(torch.sum(pred_a[:, [i0, i1]] ** 2, dim=1) + EPS)
    eu_mean = proto["velocity_energy_mean"][label_ids]
    ep_mean = proto["pressure_energy_mean"][label_ids]
    r_mean = proto["radius_mean"][label_ids]
    eu_std = proto["velocity_energy_std"][label_ids]
    ep_std = proto["pressure_energy_std"][label_ids]
    r_std = proto["radius_std"][label_ids]
    energy_raw = F.huber_loss(
        (eu - eu_mean) / torch.clamp(eu_std, min=1.0e-6),
        torch.zeros_like(eu),
        reduction="none",
    ) + F.huber_loss(
        (ep - ep_mean) / torch.clamp(ep_std, min=1.0e-6),
        torch.zeros_like(ep),
        reduction="none",
    )
    radius_raw = F.huber_loss(
        (r - r_mean) / torch.clamp(r_std, min=1.0e-6),
        torch.zeros_like(r),
        reduction="none",
    )
    energy = torch.mean(energy_raw)
    radius = torch.mean(radius_raw)
    total = float(args.lambda_proto_energy) * energy + float(args.lambda_proto_radius) * radius
    return total, {"energy": energy.detach(), "radius": radius.detach()}


def centered_r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.sum((y_pred - y_true) ** 2))
    cen = y_true - y_true.mean(axis=0, keepdims=True)
    return float(1.0 - sse / (float(np.sum(cen * cen)) + EPS))


def energy_relative_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_energy = np.sum(y_true.astype(np.float64) ** 2)
    pred_energy = np.sum(y_pred.astype(np.float64) ** 2)
    return float(abs(pred_energy - true_energy) / (true_energy + EPS))


def energy_mean_np(x: np.ndarray) -> float:
    return float(np.mean(np.sum(x.astype(np.float64) ** 2, axis=1)))


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def hopf_amplitude_metrics_np(
    true_a: np.ndarray,
    pred_a: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, object]:
    i0, i1 = parse_hopf_pair(args)
    true = np.asarray(true_a, dtype=np.float64)
    pred = np.asarray(pred_a, dtype=np.float64)
    if true.shape[-1] <= max(i0, i1) or pred.shape[-1] <= max(i0, i1):
        return {}
    eps = 1.0e-12
    true_pair = true[..., [i0, i1]]
    pred_pair = pred[..., [i0, i1]]
    r_true = np.sqrt(np.sum(true_pair * true_pair, axis=-1) + eps)
    r_pred = np.sqrt(np.sum(pred_pair * pred_pair, axis=-1) + eps)
    ratio = r_pred / np.maximum(r_true, eps)
    theta_true = np.arctan2(true_pair[..., 1], true_pair[..., 0])
    theta_pred = np.arctan2(pred_pair[..., 1], pred_pair[..., 0])
    phase_abs = np.abs(_wrap_angle_np(theta_pred - theta_true))
    out: Dict[str, object] = {
        "hopf_pair": [int(i0), int(i1)],
        "r_true_mean": float(np.mean(r_true)),
        "r_pred_mean": float(np.mean(r_pred)),
        "r_true_max": float(np.max(r_true)),
        "r_pred_max": float(np.max(r_pred)),
        "amplitude_relative_l2": float(np.linalg.norm(r_pred - r_true) / (np.linalg.norm(r_true) + eps)),
        "log_amplitude_mae": float(np.mean(np.abs(np.log(r_pred + eps) - np.log(r_true + eps)))),
        "energy_log_mae": float(np.mean(np.abs(np.log(r_pred * r_pred + eps) - np.log(r_true * r_true + eps)))),
        "overshoot_ratio_mean": float(np.mean(ratio)),
        "overshoot_ratio_max": float(np.max(ratio)),
        "overshoot_gt2_fraction": float(np.mean(ratio > float(args.hopf_overshoot_ratio))),
        "phase_abs_error_mean": float(np.mean(phase_abs)),
        "phase_abs_error_median": float(np.median(phase_abs)),
    }
    if true.ndim == 3 and true.shape[1] >= 2:
        theta_true_u = np.unwrap(theta_true, axis=1)
        theta_pred_u = np.unwrap(theta_pred, axis=1)
        dtheta_true = np.diff(theta_true_u, axis=1)
        dtheta_pred = np.diff(theta_pred_u, axis=1)
        out["frequency_increment_mae"] = float(np.mean(np.abs(dtheta_pred - dtheta_true)))
        out["growth_curve"] = {
            "r_true_mean": [float(v) for v in np.mean(r_true, axis=0).tolist()],
            "r_pred_mean": [float(v) for v in np.mean(r_pred, axis=0).tolist()],
            "overshoot_ratio_mean": [float(v) for v in np.mean(ratio, axis=0).tolist()],
        }
    else:
        out["frequency_increment_mae"] = float("nan")
    return out


def build_galerkin_torch(
    tensors: np.lib.npyio.NpzFile,
    labels: Iterable[str],
    r_u: int,
    r_p: int,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    labels_list = [str(label) for label in labels]
    first_prefix = labels_list[0]
    H_key = "H" if "H" in tensors.files else f"{first_prefix}_H"
    P_key = "P" if "P" in tensors.files else f"{first_prefix}_P"
    H = torch.tensor(tensors[H_key][:r_u, :r_u, :r_u], dtype=torch.float32, device=device)
    P = torch.tensor(tensors[P_key][:r_u, :r_p], dtype=torch.float32, device=device)
    label_to_tensor_row = {}
    if "c_all" in tensors.files and "A_all" in tensors.files:
        computed_labels = tensors["Re_labels_computed"].astype(str)
        label_to_tensor_row = {
            str(label): i for i, label in enumerate(computed_labels.tolist())
        }
    for label_id, label in enumerate(labels_list):
        prefix = str(label)
        if label_to_tensor_row:
            tensor_row = label_to_tensor_row[prefix]
            c_np = tensors["c_all"][tensor_row, :r_u]
            A_np = tensors["A_all"][tensor_row, :r_u, :r_u]
        else:
            c_np = tensors[f"{prefix}_c"][:r_u]
            A_np = tensors[f"{prefix}_A"][:r_u, :r_u]
        out[label_id] = {
            "c": torch.tensor(c_np, dtype=torch.float32, device=device),
            "A": torch.tensor(A_np, dtype=torch.float32, device=device),
            "H": H,
            "P": P,
        }
    return out


def build_pressure_surrogate_torch(
    tensors: np.lib.npyio.NpzFile,
    labels: Iterable[str],
    r_u: int,
    r_p: int,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    H_tilde = torch.tensor(
        tensors["H_tilde"][:r_p, :r_u, :r_u], dtype=torch.float32, device=device
    )
    for label_id, label in enumerate(labels):
        prefix = str(label)
        out[label_id] = {
            "c_tilde": torch.tensor(
                tensors[f"{prefix}_c_tilde"][:r_p], dtype=torch.float32, device=device
            ),
            "A_tilde": torch.tensor(
                tensors[f"{prefix}_A_tilde"][:r_p, :r_u],
                dtype=torch.float32,
                device=device,
            ),
            "H_tilde": H_tilde,
        }
    return out


def pressure_surrogate_torch(
    a_next: torch.Tensor,
    label_ids: torch.Tensor,
    sur: Dict[int, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    r_p = next(iter(sur.values()))["c_tilde"].shape[0]
    out = torch.empty((a_next.shape[0], r_p), dtype=a_next.dtype, device=a_next.device)
    for label_id in torch.unique(label_ids).detach().cpu().tolist():
        label_int = int(label_id)
        mask = label_ids == label_int
        ar = a_next[mask]
        item = sur[label_int]
        out[mask] = (
            item["c_tilde"].unsqueeze(0)
            + ar @ item["A_tilde"].T
            + torch.einsum("pij,bi,bj->bp", item["H_tilde"], ar, ar)
        )
    return out


def galerkin_rhs_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    label_ids: torch.Tensor,
    gal: Dict[int, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    rhs = torch.empty_like(a)
    for label_id in torch.unique(label_ids).detach().cpu().tolist():
        label_int = int(label_id)
        mask = label_ids == label_int
        ar = a[mask]
        br = b[mask]
        item = gal[label_int]
        rhs[mask] = (
            item["c"].unsqueeze(0)
            + ar @ item["A"].T
            + torch.einsum("ijk,bj,bk->bi", item["H"], ar, ar)
            + br @ item["P"].T
        )
    return rhs


def _nearest_indices(source_re: np.ndarray, target_re: np.ndarray) -> np.ndarray:
    source = np.asarray(source_re, dtype=np.float64)
    target = np.asarray(target_re, dtype=np.float64)
    return np.asarray([int(np.argmin(np.abs(source - value))) for value in target], dtype=np.int64)


def _velocity_sqrt_weights(pod: np.lib.npyio.NpzFile) -> np.ndarray:
    sqrt_w = pod["sqrt_point_areas"].astype(np.float32)
    return np.concatenate([sqrt_w, sqrt_w], axis=0)


def _pressure_sqrt_weights(pod: np.lib.npyio.NpzFile) -> np.ndarray:
    return pod["sqrt_point_areas"].astype(np.float32)


def _projection_offsets(
    global_pod: np.lib.npyio.NpzFile,
    regime_pod: np.lib.npyio.NpzFile,
    global_mode_key: str,
    regime_mode_key: str,
    global_mean_key: str,
    regime_mean_key: str,
    global_re_values: np.ndarray,
    regime_re_values: np.ndarray,
    label_to_regime_mean: np.ndarray,
    r: int,
    sqrt_weight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global_modes_w = global_pod[global_mode_key][:r].astype(np.float32)
    regime_modes_w = regime_pod[regime_mode_key][:r].astype(np.float32)
    to_regime = regime_modes_w @ global_modes_w.T
    to_global = global_modes_w @ regime_modes_w.T
    offset_to_regime = np.zeros((len(global_re_values), r), dtype=np.float32)
    offset_to_global = np.zeros((len(global_re_values), r), dtype=np.float32)
    global_means = global_pod[global_mean_key].astype(np.float32)
    regime_means = regime_pod[regime_mean_key].astype(np.float32)
    for label_id, regime_row in enumerate(label_to_regime_mean.tolist()):
        diff_global_minus_regime = global_means[label_id] - regime_means[int(regime_row)]
        diff_weighted = diff_global_minus_regime * sqrt_weight
        offset_to_regime[label_id] = diff_weighted @ regime_modes_w.T
        offset_to_global[label_id] = (-diff_weighted) @ global_modes_w.T
    return (
        to_regime.astype(np.float32),
        to_global.astype(np.float32),
        offset_to_regime.astype(np.float32),
        offset_to_global.astype(np.float32),
    )


def build_nearest_galerkin_torch(
    tensors: np.lib.npyio.NpzFile,
    global_re_values: np.ndarray,
    r_u: int,
    r_p: int,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    labels_available = tensors["Re_labels_computed"].astype(str)
    re_available = (
        tensors["Re_values_computed"].astype(np.float64)
        if "Re_values_computed" in tensors.files
        else np.asarray([float(str(label).split("_", 1)[1].replace("p", ".")) for label in labels_available])
    )
    nearest = _nearest_indices(re_available, global_re_values)
    H = torch.tensor(tensors["H"][:r_u, :r_u, :r_u], dtype=torch.float32, device=device)
    P = torch.tensor(tensors["P"][:r_u, :r_p], dtype=torch.float32, device=device)
    c_all = tensors["c_all"][:, :r_u].astype(np.float32)
    A_all = tensors["A_all"][:, :r_u, :r_u].astype(np.float32)
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for label_id, row in enumerate(nearest.tolist()):
        out[label_id] = {
            "c": torch.tensor(c_all[row], dtype=torch.float32, device=device),
            "A": torch.tensor(A_all[row], dtype=torch.float32, device=device),
            "H": H,
            "P": P,
        }
    return out


def build_nearest_pressure_surrogate_torch(
    tensors: np.lib.npyio.NpzFile,
    global_re_values: np.ndarray,
    r_u: int,
    r_p: int,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    labels_available = tensors["Re_labels_computed"].astype(str)
    re_available = (
        tensors["Re_values_computed"].astype(np.float64)
        if "Re_values_computed" in tensors.files
        else np.asarray([float(str(label).split("_", 1)[1].replace("p", ".")) for label in labels_available])
    )
    nearest = _nearest_indices(re_available, global_re_values)
    H_tilde = torch.tensor(
        tensors["H_tilde"][:r_p, :r_u, :r_u], dtype=torch.float32, device=device
    )
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for label_id, row in enumerate(nearest.tolist()):
        prefix = str(labels_available[int(row)])
        out[label_id] = {
            "c_tilde": torch.tensor(
                tensors[f"{prefix}_c_tilde"][:r_p], dtype=torch.float32, device=device
            ),
            "A_tilde": torch.tensor(
                tensors[f"{prefix}_A_tilde"][:r_p, :r_u],
                dtype=torch.float32,
                device=device,
            ),
            "H_tilde": H_tilde,
        }
    return out


def build_base_bundle(
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, object]:
    bundle: Dict[str, object] = {"mode": args.pressure_base_mode}
    if args.pressure_base_mode != "regime_aware_rom":
        return bundle
    root = args.regime_rom_root
    if not root.exists():
        raise FileNotFoundError(f"Regime ROM root not found: {root}")
    global_vel = np.load(args.data_root / "global_velocity_pod_area_weighted_l2.npz")
    global_pre = np.load(args.data_root / "global_pressure_pod_area_weighted_l2.npz")
    global_re = np.asarray(
        [float(np.mean(arrays["re"][arrays["label_id"] == i])) for i in range(len(arrays["labels"]))],
        dtype=np.float64,
    )
    regime_items = []
    for regime_name in REGIME_ROM_NAMES:
        vel_rom = np.load(root / f"velocity_rom_{regime_name}.npz")
        pre_rom = np.load(root / f"pressure_poisson_surrogate_{regime_name}.npz")
        regime_pod_dir = root / regime_name / "Global_POD_AreaWeighted_L2"
        regime_vel = np.load(regime_pod_dir / "global_velocity_pod_area_weighted_l2.npz")
        regime_pre = np.load(regime_pod_dir / "global_pressure_pod_area_weighted_l2.npz")
        regime_vel_re = regime_vel["Re_values"].astype(np.float64)
        regime_pre_re = regime_pre["Re_values"].astype(np.float64)
        vel_mean_rows = _nearest_indices(regime_vel_re, global_re)
        pre_mean_rows = _nearest_indices(regime_pre_re, global_re)
        u_to_reg, u_to_global, u_off_reg, _u_off_global = _projection_offsets(
            global_vel,
            regime_vel,
            "phi_uv_weighted",
            "phi_uv_weighted",
            "mean_uv_by_Re",
            "mean_uv_by_Re",
            global_re,
            regime_vel_re,
            vel_mean_rows,
            args.r_u,
            _velocity_sqrt_weights(global_vel),
        )
        p_to_reg, p_to_global, p_off_reg, p_off_global = _projection_offsets(
            global_pre,
            regime_pre,
            "phi_p_weighted",
            "phi_p_weighted",
            "mean_p_by_Re",
            "mean_p_by_Re",
            global_re,
            regime_pre_re,
            pre_mean_rows,
            args.r_p,
            _pressure_sqrt_weights(global_pre),
        )
        regime_items.append(
            {
                "name": regime_name,
                "gal": build_nearest_galerkin_torch(vel_rom, global_re, args.r_u, args.r_p, device),
                "sur": build_nearest_pressure_surrogate_torch(
                    pre_rom, global_re, args.r_u, args.r_p, device
                ),
                "u_to_reg": torch.tensor(u_to_reg, dtype=torch.float32, device=device),
                "u_to_global": torch.tensor(u_to_global, dtype=torch.float32, device=device),
                "u_offset_to_reg": torch.tensor(u_off_reg, dtype=torch.float32, device=device),
                "p_to_reg": torch.tensor(p_to_reg, dtype=torch.float32, device=device),
                "p_to_global": torch.tensor(p_to_global, dtype=torch.float32, device=device),
                "p_offset_to_reg": torch.tensor(p_off_reg, dtype=torch.float32, device=device),
                "p_offset_to_global": torch.tensor(p_off_global, dtype=torch.float32, device=device),
            }
        )
    bundle["regimes"] = regime_items
    bundle["regime_names"] = list(REGIME_ROM_NAMES)
    return bundle


def velocity_base_torch(
    model: OperatorSpaceMoEROM,
    a_state: torch.Tensor,
    b_state: torch.Tensor,
    current: torch.Tensor,
    gal: Dict[int, Dict[str, torch.Tensor]],
    base_bundle: Dict[str, object],
    closure_params: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    label_ids = current if current.dtype == torch.long and current.ndim == 1 else current.long()
    label_ids = label_ids.new_tensor(label_ids)
    # current stores sample ids in all call sites; convert to label ids in the wrapper below.
    raise RuntimeError("velocity_base_torch must be called through velocity_base_from_sample_torch")


def velocity_base_from_sample_torch(
    model: OperatorSpaceMoEROM,
    a_state: torch.Tensor,
    b_state: torch.Tensor,
    sample_ids: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    base_bundle: Dict[str, object],
    closure_params: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    label_ids = arrays_t["label_id"][sample_ids]
    if base_bundle.get("mode") != "regime_aware_rom":
        return galerkin_rhs_torch(a_state, b_state, label_ids, gal)
    rom_gate = None if closure_params is None else closure_params.get("rom_gate")
    if rom_gate is None:
        rom_gate = torch.full(
            (a_state.shape[0], len(REGIME_ROM_NAMES)),
            1.0 / len(REGIME_ROM_NAMES),
            dtype=a_state.dtype,
            device=a_state.device,
        )
    mixed = torch.zeros_like(a_state)
    for regime_id, item in enumerate(base_bundle["regimes"]):
        a_reg = a_state @ item["u_to_reg"].T + item["u_offset_to_reg"][label_ids]
        b_reg = b_state @ item["p_to_reg"].T + item["p_offset_to_reg"][label_ids]
        rhs_reg = galerkin_rhs_torch(a_reg, b_reg, label_ids, item["gal"])
        rhs_global = rhs_reg @ item["u_to_global"].T
        mixed = mixed + rom_gate[:, regime_id : regime_id + 1] * rhs_global
    return mixed


def pressure_base_from_sample_torch(
    model: OperatorSpaceMoEROM,
    a_next: torch.Tensor,
    sample_ids: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    sur: Dict[int, Dict[str, torch.Tensor]],
    base_bundle: Dict[str, object],
    closure_params: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    label_ids = arrays_t["label_id"][sample_ids]
    re_values = arrays_t["re"][sample_ids]
    if base_bundle.get("mode") == "regime_aware_rom":
        rom_gate = None if closure_params is None else closure_params.get("rom_gate")
        if rom_gate is None:
            rom_gate = torch.full(
                (a_next.shape[0], len(REGIME_ROM_NAMES)),
                1.0 / len(REGIME_ROM_NAMES),
                dtype=a_next.dtype,
                device=a_next.device,
            )
        mixed = torch.zeros((a_next.shape[0], model.pressure_dim), dtype=a_next.dtype, device=a_next.device)
        for regime_id, item in enumerate(base_bundle["regimes"]):
            a_reg = a_next @ item["u_to_reg"].T + item["u_offset_to_reg"][label_ids]
            b_reg = pressure_surrogate_torch(a_reg, label_ids, item["sur"])
            b_global = b_reg @ item["p_to_global"].T + item["p_offset_to_global"][label_ids]
            mixed = mixed + rom_gate[:, regime_id : regime_id + 1] * b_global
        return mixed
    static_base = pressure_surrogate_torch(a_next, label_ids, sur)
    return model.pressure_base_from_static(a_next, static_base, re_values)


def parse_curriculum_steps(text: str) -> List[int]:
    steps = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not steps:
        return [1]
    return steps


def curriculum_step_for_epoch(args: argparse.Namespace, epoch: int) -> int:
    steps = parse_curriculum_steps(args.curriculum_steps)
    stage_len = max(1, args.epochs // len(steps))
    stage = min(len(steps) - 1, (epoch - 1) // stage_len)
    return int(steps[stage])


def evaluate_model(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    arrays_t: Dict[str, torch.Tensor] | None = None,
    gal: Dict[int, Dict[str, torch.Tensor]] | None = None,
    sur: Dict[int, Dict[str, torch.Tensor]] | None = None,
    base_bundle: Dict[str, object] | None = None,
) -> Dict[str, object]:
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    pressure_state_override = None
    pressure_input = pressure_input_override_np(
        args.pressure_input_mode,
        arrays["a_next"][sample_ids],
        arrays["pressure_base_next"][sample_ids],
        scalers,
    )
    if pressure_input is not None:
        pressure_state_override = torch.tensor(
            pressure_input, dtype=torch.float32, device=device
        )
    with torch.no_grad():
        rhs_std, pressure_std, gates, _, closure_params_t = model(
            x,
            return_expert_stack=False,
            pressure_state_override=pressure_state_override,
            return_closure_params=True,
        )
        sample_ids_t = torch.tensor(sample_ids, dtype=torch.long, device=device)
        if arrays_t is not None and gal is not None and base_bundle is not None:
            rhs_base_t = velocity_base_from_sample_torch(
                model,
                arrays_t["a"][sample_ids_t],
                arrays_t["b"][sample_ids_t],
                sample_ids_t,
                arrays_t,
                gal,
                base_bundle,
                closure_params_t,
            )
        else:
            rhs_base_t = torch.tensor(arrays["rhs_g"][sample_ids], dtype=torch.float32, device=device)
        if arrays_t is not None and sur is not None and base_bundle is not None:
            pressure_base_t = pressure_base_from_sample_torch(
                model,
                arrays_t["a_next"][sample_ids_t],
                sample_ids_t,
                arrays_t,
                sur,
                base_bundle,
                closure_params_t,
            )
        else:
            pressure_base_t = torch.tensor(
                arrays["pressure_base_next"][sample_ids], dtype=torch.float32, device=device
            )
    rhs_pred = (
        rhs_std.detach().cpu().numpy() * scalers["rhs_operator"].scale
        + scalers["rhs_operator"].mean
    )
    if args.rhs_target == "residual":
        rhs_pred = rhs_base_t.detach().cpu().numpy() + rhs_pred
    pressure_op = (
        pressure_std.detach().cpu().numpy() * scalers["pressure_next"].scale
        + scalers["pressure_next"].mean
    )
    alpha = closure_params_t["alpha"].detach().cpu().numpy()
    beta = closure_params_t["beta"].detach().cpu().numpy()
    pressure_base_eval = pressure_base_t.detach().cpu().numpy()
    if args.pressure_target == "state":
        pressure_pred = pressure_op
        pressure_delta = pressure_pred - pressure_base_eval
        residual_component = pressure_delta
        base_component = pressure_base_eval
        closure_diag = summarize_closure_np(
            "baseline",
            arrays["b_next"][sample_ids],
            pressure_base_eval,
            pressure_delta,
            np.ones((len(sample_ids), 1), dtype=np.float32),
            np.zeros((len(sample_ids), 1), dtype=np.float32),
            arrays["re"][sample_ids],
            arrays["time"][sample_ids],
        )
    else:
        pressure_delta = pressure_op
        (
            pressure_pred,
            base_component,
            residual_component,
            _base_scale,
            _residual_scale,
        ) = closure_components_np(
            args.closure_mode,
            pressure_base_eval,
            pressure_delta,
            alpha,
            beta,
        )
        closure_diag = summarize_closure_np(
            args.closure_mode,
            arrays["b_next"][sample_ids],
            pressure_base_eval,
            pressure_delta,
            alpha,
            beta,
            arrays["re"][sample_ids],
            arrays["time"][sample_ids],
        )
    y_rhs = arrays["adot"][sample_ids]
    y_alpha = arrays["a_next"][sample_ids]
    y_pressure = arrays["b_next"][sample_ids]
    pressure_residual_eval = y_pressure - pressure_base_eval
    a_curr = arrays["a"][sample_ids]
    dt = arrays["dt_next"][sample_ids][:, None]
    euler_pred = a_curr + dt * rhs_pred

    util = []
    ent = []
    for gate in gates:
        g = gate.detach().cpu().numpy()
        util.append(g.mean(axis=0))
        ent.append(float(np.mean(-np.sum(g * np.log(g + EPS), axis=1))))
    util_mean = np.mean(np.stack(util, axis=0), axis=0) if util else np.zeros(1)
    prev_ids = arrays["prev_idx"][sample_ids]
    smooth_mask = (prev_ids >= 0) & (
        arrays["label_id"][prev_ids] == arrays["label_id"][sample_ids]
    )
    gate_smooth = float("nan")
    if np.any(smooth_mask):
        px = torch.tensor(
            scalers["x"].transform(arrays["x"][prev_ids[smooth_mask]]),
            dtype=torch.float32,
            device=device,
        )
        cx = torch.tensor(
            scalers["x"].transform(arrays["x"][sample_ids[smooth_mask]]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            _, _, cg, _ = model(cx, return_expert_stack=False)
            _, _, pg, _ = model(px, return_expert_stack=False)
            gate_smooth = float(gate_smoothness_loss(cg, pg).detach().cpu())

    return {
        "rhs_relative_l2": relative_l2_np(y_rhs, rhs_pred),
        "rhs_rmse": rmse_np(y_rhs, rhs_pred),
        "rhs_centered_r2": centered_r2_np(y_rhs, rhs_pred),
        "alpha_head_relative_l2": relative_l2_np(y_alpha, euler_pred),
        "alpha_head_rmse": rmse_np(y_alpha, euler_pred),
        "operator_one_step_euler_relative_l2": relative_l2_np(y_alpha, euler_pred),
        "operator_one_step_euler_rmse": rmse_np(y_alpha, euler_pred),
        "pressure_head_relative_l2": relative_l2_np(y_pressure, pressure_pred),
        "pressure_head_rmse": rmse_np(y_pressure, pressure_pred),
        "pressure_energy_relative_error": energy_relative_error_np(
            y_pressure, pressure_pred
        ),
        "pressure_true_energy_mean": energy_mean_np(y_pressure),
        "pressure_pred_energy_mean": energy_mean_np(pressure_pred),
        "pressure_surrogate_base_relative_l2": relative_l2_np(
            y_pressure, pressure_base_eval
        ),
        "pressure_residual_only_relative_l2": relative_l2_np(y_pressure, pressure_delta),
        "pressure_residual_relative_l2": relative_l2_np(
            pressure_residual_eval, pressure_delta
        ),
        "pressure_effective_residual_relative_l2": relative_l2_np(
            pressure_residual_eval, residual_component
        ),
        "closure_alpha_mean": closure_diag["alpha"]["mean"],
        "closure_alpha_std": closure_diag["alpha"]["std"],
        "closure_beta_mean": closure_diag["beta"]["mean"],
        "closure_beta_std": closure_diag["beta"]["std"],
        "closure_base_scale_mean": closure_diag["base_scale"]["mean"],
        "closure_residual_scale_mean": closure_diag["residual_scale"]["mean"],
        "closure_base_error_mean": closure_diag["base_error"]["mean"],
        "closure_error_mean": closure_diag["closure_error"]["mean"],
        "closure_residual_magnitude_mean": closure_diag["residual_magnitude"]["mean"],
        "closure_effective_residual_magnitude_mean": closure_diag[
            "effective_residual_magnitude"
        ]["mean"],
        "closure_base_contribution_ratio_mean": closure_diag[
            "base_contribution_ratio"
        ]["mean"],
        "closure_residual_contribution_ratio_mean": closure_diag[
            "residual_contribution_ratio"
        ]["mean"],
        "closure_alpha_base_error_corr": closure_diag["alpha_base_error_corr"],
        "closure_alpha_residual_magnitude_corr": closure_diag[
            "alpha_residual_magnitude_corr"
        ],
        "closure_beta_base_error_corr": closure_diag["beta_base_error_corr"],
        "closure_beta_residual_magnitude_corr": closure_diag[
            "beta_residual_magnitude_corr"
        ],
        "closure_diagnostics": closure_diag,
        "one_step_euler_relative_l2": relative_l2_np(y_alpha, euler_pred),
        "one_step_euler_rmse": rmse_np(y_alpha, euler_pred),
        "router_entropy": float(np.mean(ent)) if ent else 0.0,
        "router_temporal_smooth_mse": gate_smooth,
        "router_utilization": [float(v) for v in util_mean.tolist()],
    }


def routing_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        _, _, gates, _ = model(x, return_expert_stack=False)
    if not gates:
        return {}
    gate_arrays = [g.detach().cpu().numpy() for g in gates]
    gate_np = np.mean(np.stack(gate_arrays, axis=0), axis=0)
    top1 = np.argmax(gate_np, axis=1)
    top1_counts = np.bincount(top1, minlength=gate_np.shape[1]).astype(float)
    top1_frac = top1_counts / max(1, len(top1))
    topk_sets: Dict[str, int] = {}
    active_counts = np.sum(gate_np > 1.0e-6, axis=1)
    for row in gate_np:
        active = np.flatnonzero(row > 1.0e-6)
        key = ",".join(str(int(v)) for v in active.tolist())
        topk_sets[key] = topk_sets.get(key, 0) + 1

    phase = arrays["phase"][sample_ids]
    phase_bins = np.linspace(0.0, 1.0, args.analysis_bins + 1)
    by_phase = []
    for i in range(args.analysis_bins):
        if i == args.analysis_bins - 1:
            mask = (phase >= phase_bins[i]) & (phase <= phase_bins[i + 1])
        else:
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
        if np.any(mask):
            load = gate_np[mask].mean(axis=0)
            ent = -np.sum(gate_np[mask] * np.log(gate_np[mask] + EPS), axis=1).mean()
            top = np.bincount(top1[mask], minlength=gate_np.shape[1]).astype(float)
            top = top / max(1, int(np.sum(mask)))
        else:
            load = np.zeros(gate_np.shape[1], dtype=float)
            ent = float("nan")
            top = np.zeros(gate_np.shape[1], dtype=float)
        by_phase.append(
            {
                "phase_range": [float(phase_bins[i]), float(phase_bins[i + 1])],
                "num_samples": int(np.sum(mask)),
                "mean_load": [float(v) for v in load.tolist()],
                "top1_fraction": [float(v) for v in top.tolist()],
                "entropy": float(ent),
            }
        )

    load = gate_np.mean(axis=0)
    entropy = -np.sum(gate_np * np.log(gate_np + EPS), axis=1)
    by_router = {}
    for name, gate_arr in zip(["velocity", "pressure"], gate_arrays):
        r_load = gate_arr.mean(axis=0)
        r_entropy = -np.sum(gate_arr * np.log(gate_arr + EPS), axis=1)
        r_top1 = np.argmax(gate_arr, axis=1)
        r_top = np.bincount(r_top1, minlength=gate_arr.shape[1]).astype(float)
        r_top = r_top / max(1, len(r_top1))
        r_active = np.sum(gate_arr > 1.0e-6, axis=1)
        by_router[name] = {
            "mean_load": [float(v) for v in r_load.tolist()],
            "top1_fraction": [float(v) for v in r_top.tolist()],
            "load_cv": float(np.std(r_load) / (np.mean(r_load) + EPS)),
            "dead_experts_threshold_1pct": int(np.sum(r_load < 0.01)),
            "entropy_mean": float(np.mean(r_entropy)),
            "active_experts_mean": float(np.mean(r_active)),
            "active_experts_std": float(np.std(r_active)),
            "active_experts_min": int(np.min(r_active)),
            "active_experts_max": int(np.max(r_active)),
        }
    return {
        "num_samples": int(len(sample_ids)),
        "num_experts": int(gate_np.shape[1]),
        "num_regime_groups": int(getattr(model, "num_regime_groups", 1)),
        "experts_per_group": int(getattr(model, "experts_per_group", gate_np.shape[1])),
        "mean_load": [float(v) for v in load.tolist()],
        "load_cv": float(np.std(load) / (np.mean(load) + EPS)),
        "dead_experts_threshold_1pct": int(np.sum(load < 0.01)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy)),
        "top1_fraction": [float(v) for v in top1_frac.tolist()],
        "active_experts_mean": float(np.mean(active_counts)),
        "active_experts_std": float(np.std(active_counts)),
        "active_experts_min": int(np.min(active_counts)),
        "active_experts_max": int(np.max(active_counts)),
        "effective_top_k_for_counts": int(np.max(active_counts)) if len(active_counts) else 0,
        "topk_set_counts": {k: int(v) for k, v in sorted(topk_sets.items())},
        "by_router": by_router,
        "by_phase_bin": by_phase,
    }


def routing_by_re_group_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        _, _, gates, _ = model(x, return_expert_stack=False)
    if not gates:
        return {}
    gate_np = np.mean(np.stack([g.detach().cpu().numpy() for g in gates], axis=0), axis=0)
    top1 = np.argmax(gate_np, axis=1)
    re = arrays["re"][sample_ids]
    groups = {
        "low_Re_lt_80": re < 80.0,
        "mid_Re_80_160": (re >= 80.0) & (re < 160.0),
        "high_Re_ge_160": re >= 160.0,
    }
    out: Dict[str, object] = {}
    for name, mask in groups.items():
        if not np.any(mask):
            out[name] = {"num_samples": 0}
            continue
        load = gate_np[mask].mean(axis=0)
        top = np.bincount(top1[mask], minlength=gate_np.shape[1]).astype(float)
        top = top / max(1, int(np.sum(mask)))
        entropy = -np.sum(gate_np[mask] * np.log(gate_np[mask] + EPS), axis=1)
        out[name] = {
            "num_samples": int(np.sum(mask)),
            "mean_load": [float(v) for v in load.tolist()],
            "top1_fraction": [float(v) for v in top.tolist()],
            "load_cv": float(np.std(load) / (np.mean(load) + EPS)),
            "entropy_mean": float(np.mean(entropy)),
        }
    return out


def routing_by_regime_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        _, _, gates, _ = model(x, return_expert_stack=False)
        group_gate, _ = model.group_router_outputs(x)
    if not gates:
        return {}
    gate_np = np.mean(np.stack([g.detach().cpu().numpy() for g in gates], axis=0), axis=0)
    group_np = group_gate.detach().cpu().numpy()
    expert_top1 = np.argmax(gate_np, axis=1)
    group_top1 = np.argmax(group_np, axis=1)
    regime_id = arrays["regime_id"][sample_ids]
    regime_vocab = arrays["regime_vocab"]
    out: Dict[str, object] = {}
    for rid, name in enumerate(regime_vocab.tolist()):
        mask = regime_id == rid
        if not np.any(mask):
            out[str(name)] = {"num_samples": 0}
            continue
        expert_top = np.bincount(expert_top1[mask], minlength=gate_np.shape[1]).astype(float)
        expert_top = expert_top / max(1, int(np.sum(mask)))
        group_top = np.bincount(group_top1[mask], minlength=group_np.shape[1]).astype(float)
        group_top = group_top / max(1, int(np.sum(mask)))
        expert_entropy = -np.sum(gate_np[mask] * np.log(gate_np[mask] + EPS), axis=1)
        group_entropy = -np.sum(group_np[mask] * np.log(group_np[mask] + EPS), axis=1)
        out[str(name)] = {
            "num_samples": int(np.sum(mask)),
            "expert_mean_load": [float(v) for v in gate_np[mask].mean(axis=0).tolist()],
            "expert_top1_fraction": [float(v) for v in expert_top.tolist()],
            "expert_entropy_mean": float(np.mean(expert_entropy)),
            "expert_load_cv": float(
                np.std(gate_np[mask].mean(axis=0)) / (np.mean(gate_np[mask].mean(axis=0)) + EPS)
            ),
            "group_mean_load": [float(v) for v in group_np[mask].mean(axis=0).tolist()],
            "group_top1_fraction": [float(v) for v in group_top.tolist()],
            "group_entropy_mean": float(np.mean(group_entropy)),
        }
    return out


def group_routing_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        group_gate, _ = model.group_router_outputs(x)
    gate_np = group_gate.detach().cpu().numpy()
    top1 = np.argmax(gate_np, axis=1)
    entropy = -np.sum(gate_np * np.log(gate_np + EPS), axis=1)
    active = np.sum(gate_np > 1.0e-6, axis=1)
    re = arrays["re"][sample_ids]
    re_groups = {
        "low_Re_lt_80": re < 80.0,
        "transition_Re_80_160": (re >= 80.0) & (re < 160.0),
        "high_Re_ge_160": re >= 160.0,
    }
    by_re: Dict[str, object] = {}
    for name, mask in re_groups.items():
        if not np.any(mask):
            by_re[name] = {"num_samples": 0}
            continue
        top = np.bincount(top1[mask], minlength=gate_np.shape[1]).astype(float)
        top = top / max(1, int(np.sum(mask)))
        by_re[name] = {
            "num_samples": int(np.sum(mask)),
            "mean_load": [float(v) for v in gate_np[mask].mean(axis=0).tolist()],
            "top1_fraction": [float(v) for v in top.tolist()],
            "entropy_mean": float(np.mean(entropy[mask])),
            "active_groups_mean": float(np.mean(active[mask])),
        }
    top_all = np.bincount(top1, minlength=gate_np.shape[1]).astype(float)
    top_all = top_all / max(1, len(top1))
    return {
        "num_samples": int(len(sample_ids)),
        "num_groups": int(gate_np.shape[1]),
        "mean_load": [float(v) for v in gate_np.mean(axis=0).tolist()],
        "top1_fraction": [float(v) for v in top_all.tolist()],
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy)),
        "active_groups_mean": float(np.mean(active)),
        "active_groups_std": float(np.std(active)),
        "by_re_group": by_re,
    }


def attractor_router_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    if not getattr(model, "attractor_conditioned", False):
        return {"enabled": False}
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        _, _, _, _, closure_params = model(
            x, return_expert_stack=False, return_closure_params=True
        )
    pi_t = closure_params.get("attr_pi")
    logits_t = closure_params.get("attr_logits")
    if pi_t is None or logits_t is None:
        return {"enabled": False}
    pi = pi_t.detach().cpu().numpy()
    logits = logits_t.detach().cpu().numpy()
    targets = arrays["attractor_id"][sample_ids].astype(np.int64)
    pred = np.argmax(pi, axis=1)
    confusion = np.zeros((len(ATTRACTOR_NAMES), len(ATTRACTOR_NAMES)), dtype=np.int64)
    for t, p in zip(targets.tolist(), pred.tolist()):
        if 0 <= t < len(ATTRACTOR_NAMES) and 0 <= p < len(ATTRACTOR_NAMES):
            confusion[t, p] += 1
    by_attr: Dict[str, object] = {}
    for aid, name in enumerate(ATTRACTOR_NAMES):
        mask = targets == aid
        if not np.any(mask):
            by_attr[str(name)] = {"num_samples": 0}
            continue
        by_attr[str(name)] = {
            "num_samples": int(np.sum(mask)),
            "pi_mean": [float(v) for v in pi[mask].mean(axis=0).tolist()],
            "accuracy": float(np.mean(pred[mask] == targets[mask])),
        }
    delta_norm = closure_params.get("attr_adapter_delta_norm")
    h_norm = closure_params.get("attr_adapter_h_norm")
    delta_np = (
        delta_norm.detach().cpu().numpy().reshape(-1)
        if delta_norm is not None
        else np.asarray([], dtype=np.float32)
    )
    h_np = (
        h_norm.detach().cpu().numpy().reshape(-1)
        if h_norm is not None
        else np.asarray([], dtype=np.float32)
    )
    ratio = delta_np / (h_np + EPS) if len(delta_np) and len(h_np) else np.asarray([])
    return {
        "enabled": True,
        "num_samples": int(len(sample_ids)),
        "pi_mean": [float(v) for v in pi.mean(axis=0).tolist()],
        "pi_std": [float(v) for v in pi.std(axis=0).tolist()],
        "accuracy": float(np.mean(pred == targets)),
        "confusion_matrix": confusion.tolist(),
        "logit_mean": [float(v) for v in logits.mean(axis=0).tolist()],
        "adapter_delta_norm_mean": float(np.mean(delta_np)) if len(delta_np) else float("nan"),
        "adapter_delta_norm_std": float(np.std(delta_np)) if len(delta_np) else float("nan"),
        "adapter_delta_to_h_norm_mean": float(np.mean(ratio)) if len(ratio) else float("nan"),
        "by_attractor": by_attr,
    }


def shared_operator_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    if model.num_shared_experts <= 0:
        return {"num_shared_experts": 0, "always_active": False}
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        group_gate, _ = model.group_router_outputs(x)
    gate_np = group_gate.detach().cpu().numpy()
    top1 = np.argmax(gate_np, axis=1)
    top = np.bincount(top1, minlength=gate_np.shape[1]).astype(float)
    top = top / max(1, len(top1))
    entropy = -np.sum(gate_np * np.log(gate_np + EPS), axis=1)
    return {
        "num_shared_experts": int(model.num_shared_experts),
        "shared_experts_per_group": int(model.shared_per_group),
        "always_active": True,
        "always_active_in_selected_group": True,
        "mean_active_shared_groups": float(np.mean(np.sum(gate_np > 1.0e-6, axis=1))),
        "mean_mixer_weight": [float(v) for v in gate_np.mean(axis=0).tolist()],
        "group_mean_load": [float(v) for v in gate_np.mean(axis=0).tolist()],
        "group_top1_fraction": [float(v) for v in top.tolist()],
        "mixer_weight_std": [float(v) for v in gate_np.std(axis=0).tolist()],
        "mixer_entropy_mean": float(np.mean(entropy)),
        "mixer_entropy_std": float(np.std(entropy)),
    }


def expert_operator_diversity_analysis(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
    max_samples: int = 1024,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    model.eval()
    chosen = sample_ids
    if len(chosen) > max_samples:
        rng = np.random.default_rng(12345)
        chosen = np.sort(rng.choice(chosen, size=max_samples, replace=False))
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][chosen]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        _, _, _, expert_stack = model(x)
    expert_vectors = expert_stack.detach().cpu().numpy().transpose(1, 0, 2)
    flat = expert_vectors.reshape(expert_vectors.shape[0], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(flat, axis=1, keepdims=True) + EPS
    sim = (flat / norm) @ (flat / norm).T
    mask = ~np.eye(sim.shape[0], dtype=bool)
    off = sim[mask]
    return {
        "num_samples": int(len(chosen)),
        "pairwise_cosine_mean": float(np.mean(off)),
        "pairwise_cosine_abs_mean": float(np.mean(np.abs(off))),
        "pairwise_cosine_max_abs": float(np.max(np.abs(off))),
        "pairwise_cosine_matrix": sim.astype(float).round(6).tolist(),
        "collapse_flag_abs_cos_gt_0p95": bool(np.max(np.abs(off)) > 0.95),
    }


def sequence_start_ids(
    sample_ids: np.ndarray,
    re_values: np.ndarray,
    next_idx: np.ndarray,
    steps: int,
    sequence_pool_ids: np.ndarray | None = None,
) -> np.ndarray:
    pool = sample_ids if sequence_pool_ids is None else sequence_pool_ids
    sample_set = set(int(i) for i in pool.tolist())
    starts = []
    for idx in sample_ids.tolist():
        cur = int(idx)
        ok = True
        re = int(re_values[cur])
        for _ in range(steps):
            nxt = int(next_idx[cur])
            if nxt < 0 or nxt not in sample_set or int(re_values[nxt]) != re:
                ok = False
                break
            cur = nxt
        if ok:
            starts.append(int(idx))
    return np.asarray(starts, dtype=np.int64)


def init_history_states_torch(
    start_ids: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hist_ids = arrays_t["hist_idx"][start_ids]
    safe = torch.clamp(hist_ids, min=0)
    valid = hist_ids >= 0
    a_hist = torch.where(valid[:, :, None], arrays_t["a"][safe], torch.zeros_like(arrays_t["a"][safe]))
    b_hist = torch.where(valid[:, :, None], arrays_t["b"][safe], torch.zeros_like(arrays_t["b"][safe]))
    rhs_hist = torch.where(
        valid[:, :, None], arrays_t["rhs_g"][safe], torch.zeros_like(arrays_t["rhs_g"][safe])
    )
    return a_hist, b_hist, rhs_hist


def model_outputs_from_states_torch(
    model: OperatorSpaceMoEROM,
    a_state: torch.Tensor,
    b_state: torch.Tensor,
    current: torch.Tensor,
    a_hist: torch.Tensor,
    b_hist: torch.Tensor,
    rhs_hist: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    base_bundle: Dict[str, object],
    args: argparse.Namespace,
    pressure_state_override: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    re = arrays_t["re"][current]
    label_id = arrays_t["label_id"][current]
    ph = arrays_t["phase"][current]
    rhs_g = galerkin_rhs_torch(a_state, b_state, label_id, gal)
    base_x = make_features_torch(a_state, b_state, rhs_g, re, ph, args.phase_harmonics)
    x = make_history_features_from_states_torch(
        base_x, a_state, b_state, rhs_g, a_hist, b_hist, rhs_hist
    )
    x = (x - scalers_t["x_mean"]) / scalers_t["x_scale"]
    rhs_std, pressure_std, _, _, closure_params = model(
        x,
        return_expert_stack=False,
        pressure_state_override=pressure_state_override,
        return_closure_params=True,
    )
    rhs_base = velocity_base_from_sample_torch(
        model, a_state, b_state, current, arrays_t, gal, base_bundle, closure_params
    )
    rhs_op = rhs_std * scalers_t["rhs_op_scale"] + scalers_t["rhs_op_mean"]
    if args.rhs_target == "residual":
        rhs_op = rhs_base + rhs_op
    pressure_op = pressure_std * scalers_t["pressure_scale"] + scalers_t["pressure_mean"]
    return rhs_op, pressure_op, rhs_base, closure_params


def integrate_autonomous_step_torch(
    model: OperatorSpaceMoEROM,
    a_state: torch.Tensor,
    b_state: torch.Tensor,
    current: torch.Tensor,
    dt: torch.Tensor,
    a_hist: torch.Tensor,
    b_hist: torch.Tensor,
    rhs_hist: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    sur: Dict[int, Dict[str, torch.Tensor]],
    base_bundle: Dict[str, object],
    args: argparse.Namespace,
    return_aux: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    k1, pressure_op, rhs_g, closure_params = model_outputs_from_states_torch(
        model, a_state, b_state, current, a_hist, b_hist, rhs_hist, arrays_t, scalers_t, gal, base_bundle, args
    )
    if args.integrator == "rk4":
        k2, _, _, _ = model_outputs_from_states_torch(
            model,
            a_state + 0.5 * dt * k1,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            base_bundle,
            args,
        )
        k3, _, _, _ = model_outputs_from_states_torch(
            model,
            a_state + 0.5 * dt * k2,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            base_bundle,
            args,
        )
        k4, _, _, _ = model_outputs_from_states_torch(
            model,
            a_state + dt * k3,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            base_bundle,
            args,
        )
        a_next = a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        a_next = a_state + dt * k1
    if args.pressure_input_mode != "pressure_only":
        b_base_for_input = pressure_base_from_sample_torch(
            model, a_next, current, arrays_t, sur, base_bundle, closure_params
        )
        pressure_state_override = pressure_input_override_torch(
            args.pressure_input_mode, a_next, b_base_for_input, scalers_t
        )
        _, pressure_op, _, closure_params = model_outputs_from_states_torch(
            model,
            a_state,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            base_bundle,
            args,
            pressure_state_override=pressure_state_override,
        )
    if args.pressure_target == "state":
        b_next = pressure_op
        b_base = pressure_base_from_sample_torch(
            model, a_next, current, arrays_t, sur, base_bundle, closure_params
        )
        base_component = b_base
        residual_component = b_next - b_base
        base_scale = torch.ones_like(b_base)
        residual_scale = torch.ones_like(b_base)
    else:
        b_base = pressure_base_from_sample_torch(
            model, a_next, current, arrays_t, sur, base_bundle, closure_params
        )
        b_next, base_component, residual_component, base_scale, residual_scale = closure_components_torch(
            args.closure_mode, b_base, pressure_op, closure_params
        )
    if return_aux:
        return a_next, b_next, rhs_g, {
            "pressure_residual": pressure_op,
            "b_base": b_base,
            "base_component": base_component,
            "residual_component": residual_component,
            "base_scale": base_scale,
            "residual_scale": residual_scale,
            "alpha": closure_params["alpha"],
            "beta": closure_params["beta"],
        }
    return a_next, b_next, rhs_g


def rollout_loss_torch(
    model: OperatorSpaceMoEROM,
    start_ids: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    sur: Dict[int, Dict[str, torch.Tensor]],
    base_bundle: Dict[str, object],
    args: argparse.Namespace,
    steps: int,
    epoch: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if start_ids.numel() == 0:
        zero = torch.tensor(0.0, device=next(model.parameters()).device)
        return zero, zero, zero, zero, zero, {
            "log_amp": zero,
            "energy": zero,
            "overshoot": zero,
            "overshoot_ratio": zero,
            "v16_1_hopf_growth": zero,
            "v16_1_hopf_false_growth": zero,
            "v16_1_hopf_floor_rel": zero,
            "v16_1_steady_p_state": zero,
            "v16_1_steady_p_mean": zero,
            "v16_1_steady_p_delta": zero,
            "v16_1_steady_residual_damp": zero,
            "v16_1_steady_p_energy": zero,
            "v16_1_loss_warmup": zero,
        }
    a_cur = arrays_t["a"][start_ids]
    b_cur = arrays_t["b"][start_ids]
    a_hist, b_hist, rhs_hist = init_history_states_torch(start_ids, arrays_t)
    current = start_ids
    losses = []
    energy_losses = []
    hopf_losses = []
    v16_1_losses = []
    hopf_infos: List[Dict[str, torch.Tensor]] = []
    v16_1_infos: List[Dict[str, torch.Tensor]] = []
    last_pred_a = a_cur
    last_pred_b = b_cur
    last_target_a = a_cur
    last_target_b = b_cur
    sampling_prob = float(np.clip(getattr(args, "scheduled_sampling_probability", 1.0), 0.0, 1.0))
    for _ in range(steps):
        nxt = arrays_t["next_idx"][current]
        dt = arrays_t["time"][nxt].unsqueeze(1) - arrays_t["time"][current].unsqueeze(1)
        a_next, b_next, rhs_g, pressure_aux = integrate_autonomous_step_torch(
            model,
            a_cur,
            b_cur,
            current,
            dt,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            sur,
            base_bundle,
            args,
            return_aux=True,
        )
        target_a = arrays_t["a"][nxt]
        target_b = arrays_t["b"][nxt]
        a_loss_std = F.mse_loss(
            (a_next - target_a) / scalers_t["alpha_scale"], torch.zeros_like(a_next), reduction="none"
        ).mean(dim=1)
        b_loss_std = F.mse_loss(
            (b_next - target_b) / scalers_t["pressure_state_scale"],
            torch.zeros_like(b_next),
            reduction="none",
        ).mean(dim=1)
        a_loss_rel = relative_vector_loss_per_sample_torch(
            a_next - target_a, target_a, scalers_t["alpha_rel_floor"]
        )
        b_loss_rel = relative_vector_loss_per_sample_torch(
            b_next - target_b, target_b, scalers_t["pressure_rel_floor"]
        )
        mix = float(np.clip(args.rollout_relative_mix, 0.0, 1.0))
        a_loss = (1.0 - mix) * a_loss_std + mix * a_loss_rel
        b_loss = (1.0 - mix) * b_loss_std + mix * b_loss_rel
        step_weights = torch.ones_like(a_loss)
        hopf_active = hopf_mask_torch(arrays_t["re"][current], args)
        focus_active = hopf_focus_mask_torch(arrays_t["re"][current], args)
        step_weights = torch.where(
            hopf_active,
            step_weights * float(args.hopf_rollout_weight),
            step_weights,
        )
        step_weights = torch.where(
            focus_active,
            step_weights * float(args.hopf_focus_sample_weight),
            step_weights,
        )
        losses.append(weighted_mean_torch(a_loss + args.lambda_pressure_rollout * b_loss, step_weights))
        hopf_loss, hopf_info = hopf_envelope_loss_torch(
            a_next, target_a, arrays_t["re"][current], args, epoch, step_weights
        )
        hopf_losses.append(hopf_loss)
        hopf_infos.append(hopf_info)

        zero_step = a_loss.mean() * 0.0
        v16_warmup = torch.tensor(
            v16_1_loss_warmup(args, epoch), dtype=a_next.dtype, device=a_next.device
        )
        i0, i1 = parse_hopf_pair(args)
        pred_pair = a_next[:, [i0, i1]]
        true_pair = target_a[:, [i0, i1]]
        prev_pred_pair = last_pred_a[:, [i0, i1]]
        prev_true_pair = last_target_a[:, [i0, i1]]
        r_pred = torch.sqrt(torch.sum(pred_pair * pred_pair, dim=1) + EPS)
        r_true = torch.sqrt(torch.sum(true_pair * true_pair, dim=1) + EPS)
        r_pred_prev = torch.sqrt(torch.sum(prev_pred_pair * prev_pred_pair, dim=1) + EPS)
        r_true_prev = torch.sqrt(torch.sum(prev_true_pair * prev_true_pair, dim=1) + EPS)
        g_pred = r_pred - r_pred_prev
        g_true = r_true - r_true_prev
        hopf_weights = torch.where(hopf_active, step_weights, torch.zeros_like(step_weights))
        hopf_growth = weighted_mean_torch(
            F.huber_loss(g_pred - g_true, torch.zeros_like(g_true), reduction="none"),
            hopf_weights,
        )
        r_floor = torch.tensor(
            float(getattr(args, "attractor_r_floor_effective", args.attractor_r_floor)),
            dtype=a_next.dtype,
            device=a_next.device,
        )
        if not bool(torch.isfinite(r_floor)) or float(r_floor.detach().cpu()) <= 0.0:
            r_floor = torch.tensor(1.0e-5, dtype=a_next.dtype, device=a_next.device)
        false_growth_mask = hopf_active & (r_true_prev < r_floor) & (g_true <= 0.0)
        false_growth_weights = torch.where(
            false_growth_mask, step_weights, torch.zeros_like(step_weights)
        )
        hopf_false_growth = weighted_mean_torch(torch.relu(g_pred) ** 2, false_growth_weights)
        a_floor = torch.tensor(
            float(getattr(args, "v16_1_a_floor_effective", 0.0)),
            dtype=a_next.dtype,
            device=a_next.device,
        )
        if not bool(torch.isfinite(a_floor)) or float(a_floor.detach().cpu()) <= 0.0:
            a_floor = torch.sqrt(torch.clamp(scalers_t["alpha_rel_floor"], min=EPS))
        diff_norm = torch.linalg.norm(a_next - target_a, dim=1)
        true_norm = torch.linalg.norm(target_a, dim=1)
        hopf_floor_rel = weighted_mean_torch(
            diff_norm / torch.clamp(true_norm, min=a_floor),
            hopf_weights,
        )

        steady_active = arrays_t["attractor_id"][current] == 0
        steady_weights = torch.where(steady_active, step_weights, torch.zeros_like(step_weights))
        label_ids = arrays_t["label_id"][current]
        if "pressure_mean_by_label" in arrays_t:
            b_mean = arrays_t["pressure_mean_by_label"][label_ids]
        else:
            b_mean = target_b
        p_scale = torch.clamp(scalers_t["pressure_state_scale"], min=EPS)
        steady_p_state = weighted_mean_torch(
            torch.mean(((b_next - target_b) / p_scale) ** 2, dim=1),
            steady_weights,
        )
        steady_p_mean = weighted_mean_torch(
            torch.mean(((b_next - b_mean) / p_scale) ** 2, dim=1),
            steady_weights,
        )
        steady_p_delta = weighted_mean_torch(
            torch.mean(((b_next - b_cur) / p_scale) ** 2, dim=1),
            steady_weights,
        )
        steady_residual_damp = weighted_mean_torch(
            torch.mean(
                (pressure_aux["pressure_residual"] / torch.clamp(scalers_t["pressure_scale"], min=EPS))
                ** 2,
                dim=1,
            ),
            steady_weights,
        )
        ep_pred = torch.sum(b_next * b_next, dim=1)
        ep_true = torch.sum(target_b * target_b, dim=1)
        steady_p_energy = weighted_mean_torch(
            F.huber_loss(
                (ep_pred - ep_true)
                / (torch.clamp(ep_true, min=EPS) + torch.clamp(scalers_t["pressure_rel_floor"], min=EPS)),
                torch.zeros_like(ep_true),
                reduction="none",
            ),
            steady_weights,
        )
        v16_1_loss = v16_warmup * (
            float(args.lambda_v16_1_hopf_growth) * hopf_growth
            + float(args.lambda_v16_1_hopf_false_growth) * hopf_false_growth
            + float(args.lambda_v16_1_hopf_floor_rel) * hopf_floor_rel
            + float(args.lambda_v16_1_steady_p_state) * steady_p_state
            + float(args.lambda_v16_1_steady_p_mean) * steady_p_mean
            + float(args.lambda_v16_1_steady_p_delta) * steady_p_delta
            + float(args.lambda_v16_1_steady_residual_damp) * steady_residual_damp
            + float(args.lambda_v16_1_steady_p_energy) * steady_p_energy
        )
        v16_1_losses.append(v16_1_loss)
        v16_1_infos.append(
            {
                "v16_1_hopf_growth": hopf_growth.detach(),
                "v16_1_hopf_false_growth": hopf_false_growth.detach(),
                "v16_1_hopf_floor_rel": hopf_floor_rel.detach(),
                "v16_1_steady_p_state": steady_p_state.detach(),
                "v16_1_steady_p_mean": steady_p_mean.detach(),
                "v16_1_steady_p_delta": steady_p_delta.detach(),
                "v16_1_steady_residual_damp": steady_residual_damp.detach(),
                "v16_1_steady_p_energy": steady_p_energy.detach(),
                "v16_1_loss_warmup": v16_warmup.detach(),
            }
        )

        pred_energy = torch.sum(a_next * a_next, dim=1)
        target_energy = torch.sum(target_a * target_a, dim=1)
        energy_losses.append(
            torch.mean(
                (pred_energy - target_energy) ** 2
                / (target_energy + torch.clamp(scalers_t["alpha_rel_floor"], min=EPS)) ** 2
            )
        )

        a_feed = sampling_prob * a_next + (1.0 - sampling_prob) * target_a
        b_feed = sampling_prob * b_next + (1.0 - sampling_prob) * target_b
        a_hist = torch.cat([a_feed[:, None, :], a_hist[:, :-1, :]], dim=1)
        b_hist = torch.cat([b_feed[:, None, :], b_hist[:, :-1, :]], dim=1)
        rhs_hist = torch.cat([rhs_g[:, None, :], rhs_hist[:, :-1, :]], dim=1)
        a_cur = a_feed
        b_cur = b_feed
        last_pred_a = a_next
        last_pred_b = b_next
        last_target_a = target_a
        last_target_b = target_b
        current = nxt
    rollout_loss = torch.stack(losses).mean()
    energy_loss = torch.stack(energy_losses).mean() if energy_losses else rollout_loss * 0.0
    traj_a = relative_vector_loss_torch(
        last_pred_a - last_target_a, last_target_a, scalers_t["alpha_rel_floor"]
    )
    traj_b = relative_vector_loss_torch(
        last_pred_b - last_target_b, last_target_b, scalers_t["pressure_rel_floor"]
    )
    trajectory_loss = traj_a + args.lambda_pressure_rollout * traj_b
    hopf_rollout_loss = torch.stack(hopf_losses).mean() if hopf_losses else rollout_loss * 0.0
    v16_1_rollout_loss = torch.stack(v16_1_losses).mean() if v16_1_losses else rollout_loss * 0.0
    if hopf_infos:
        hopf_info_mean = {
            key: torch.stack([info[key] for info in hopf_infos]).mean()
            for key in hopf_infos[0].keys()
        }
    else:
        zero = rollout_loss * 0.0
        hopf_info_mean = {"log_amp": zero, "energy": zero, "overshoot": zero, "overshoot_ratio": zero}
    if v16_1_infos:
        hopf_info_mean.update(
            {
                key: torch.stack([info[key] for info in v16_1_infos]).mean()
                for key in v16_1_infos[0].keys()
            }
        )
    else:
        zero = rollout_loss * 0.0
        hopf_info_mean.update(
            {
                "v16_1_hopf_growth": zero,
                "v16_1_hopf_false_growth": zero,
                "v16_1_hopf_floor_rel": zero,
                "v16_1_steady_p_state": zero,
                "v16_1_steady_p_mean": zero,
                "v16_1_steady_p_delta": zero,
                "v16_1_steady_residual_damp": zero,
                "v16_1_steady_p_energy": zero,
                "v16_1_loss_warmup": zero,
            }
        )
    return rollout_loss, energy_loss, trajectory_loss, hopf_rollout_loss, v16_1_rollout_loss, hopf_info_mean


def model_rhs_torch(
    model: OperatorSpaceMoEROM,
    a_state: torch.Tensor,
    current: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> torch.Tensor:
    b = arrays_t["b"][current]
    re = arrays_t["re"][current]
    label_id = arrays_t["label_id"][current]
    ph = arrays_t["phase"][current]
    rhs_g = galerkin_rhs_torch(a_state, b, label_id, gal)
    base_x = make_features_torch(a_state, b, rhs_g, re, ph, args.phase_harmonics)
    hist_ids = shifted_history_ids_torch(current, arrays_t)
    x = make_history_features_torch(base_x, a_state, b, rhs_g, hist_ids, arrays_t)
    x = (x - scalers_t["x_mean"]) / scalers_t["x_scale"]
    rhs_std, _, _, _ = model(x, return_expert_stack=False)
    rhs_op = rhs_std * scalers_t["rhs_op_scale"] + scalers_t["rhs_op_mean"]
    if args.rhs_target == "residual":
        rhs_op = rhs_g + rhs_op
    return rhs_op


def integrate_step_torch(
    model: OperatorSpaceMoEROM,
    a_state: torch.Tensor,
    current: torch.Tensor,
    dt: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.integrator == "rk4":
        k1 = model_rhs_torch(model, a_state, current, arrays_t, scalers_t, gal, args)
        k2 = model_rhs_torch(model, a_state + 0.5 * dt * k1, current, arrays_t, scalers_t, gal, args)
        k3 = model_rhs_torch(model, a_state + 0.5 * dt * k2, current, arrays_t, scalers_t, gal, args)
        k4 = model_rhs_torch(model, a_state + dt * k3, current, arrays_t, scalers_t, gal, args)
        return a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    rhs = model_rhs_torch(model, a_state, current, arrays_t, scalers_t, gal, args)
    return a_state + dt * rhs


def rollout_eval_np(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    test_label_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    idx = np.where(arrays["label_id"] == test_label_id)[0]
    idx = idx[np.argsort(arrays["time"][idx])]
    valid_sample_set = set(arrays["sample_ids"].tolist())
    rel_errors: List[float] = []
    stride = max(1, args.rollout_steps)
    model.eval()
    for start_pos in range(1, len(idx) - args.rollout_steps - 1, stride):
        start = int(idx[start_pos])
        if start not in valid_sample_set:
            continue
        a_cur = arrays["a"][start].copy()
        pred = []
        cur = start
        ok = True
        for _ in range(args.rollout_steps):
            nxt = int(arrays["next_idx"][cur])
            if nxt < 0 or int(arrays["label_id"][nxt]) != int(test_label_id):
                ok = False
                break
            dt = float(arrays["time"][nxt] - arrays["time"][cur])
            a_cur = integrate_step_np(model, a_cur, cur, dt, arrays, scalers, tensors, args, device)
            if not np.all(np.isfinite(a_cur)):
                ok = False
                break
            pred.append(a_cur.copy())
            cur = nxt
        if ok and len(pred) == args.rollout_steps:
            true = arrays["a"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            rel_errors.append(relative_l2_np(true, np.asarray(pred)))
    if not rel_errors:
        return {
            "relative_l2_mean": float("nan"),
            "relative_l2_median": float("nan"),
            "relative_l2_p90": float("nan"),
            "relative_l2_max": float("nan"),
            "num_windows": 0,
        }
    return {
        "relative_l2_mean": float(np.mean(rel_errors)),
        "relative_l2_median": float(np.median(rel_errors)),
        "relative_l2_p90": float(np.percentile(rel_errors, 90)),
        "relative_l2_max": float(np.max(rel_errors)),
        "num_windows": int(len(rel_errors)),
    }


def model_rhs_np(
    model: OperatorSpaceMoEROM,
    a_state: np.ndarray,
    cur: int,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    rhs_g = galerkin_rhs_by_label(
        tensors,
        a_state[None, :],
        arrays["b"][cur : cur + 1],
        arrays["label_id"][cur : cur + 1],
        arrays["labels"],
        args.r_u,
        args.r_p,
    )[0]
    base_x = make_features_np(
        a_state[None, :],
        arrays["b"][cur : cur + 1],
        rhs_g[None, :],
        arrays["re"][cur : cur + 1],
        arrays["phase"][cur : cur + 1],
        args.phase_harmonics,
    )
    hist_row = shift_history_for_current_np(arrays["hist_idx"][cur], cur)[None, :]
    x = make_history_features_from_current_np(
        base_x,
        a_state[None, :],
        arrays["b"][cur : cur + 1],
        rhs_g[None, :],
        hist_row,
        arrays["a"],
        arrays["b"],
        arrays["rhs_g"],
    )
    xt = torch.tensor(scalers["x"].transform(x), dtype=torch.float32, device=device)
    with torch.no_grad():
        rhs_std, _, _, _ = model(xt, return_expert_stack=False)
    rhs_op = (
        rhs_std.detach().cpu().numpy()[0] * scalers["rhs_operator"].scale
        + scalers["rhs_operator"].mean
    )
    if args.rhs_target == "residual":
        rhs_op = rhs_g + rhs_op
    return rhs_op


def integrate_step_np(
    model: OperatorSpaceMoEROM,
    a_state: np.ndarray,
    cur: int,
    dt: float,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    if args.integrator == "rk4":
        k1 = model_rhs_np(model, a_state, cur, arrays, scalers, tensors, args, device)
        k2 = model_rhs_np(model, a_state + 0.5 * dt * k1, cur, arrays, scalers, tensors, args, device)
        k3 = model_rhs_np(model, a_state + 0.5 * dt * k2, cur, arrays, scalers, tensors, args, device)
        k4 = model_rhs_np(model, a_state + dt * k3, cur, arrays, scalers, tensors, args, device)
        return a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    rhs = model_rhs_np(model, a_state, cur, arrays, scalers, tensors, args, device)
    return a_state + dt * rhs


def init_history_states_np(
    start: int,
    arrays: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist = arrays["hist_idx"][start]
    a_hist = np.zeros((1, len(hist), arrays["a"].shape[1]), dtype=np.float32)
    b_hist = np.zeros((1, len(hist), arrays["b"].shape[1]), dtype=np.float32)
    rhs_hist = np.zeros((1, len(hist), arrays["rhs_g"].shape[1]), dtype=np.float32)
    valid = hist >= 0
    if np.any(valid):
        a_hist[0, valid] = arrays["a"][hist[valid]]
        b_hist[0, valid] = arrays["b"][hist[valid]]
        rhs_hist[0, valid] = arrays["rhs_g"][hist[valid]]
    return a_hist, b_hist, rhs_hist


def model_outputs_from_states_np(
    model: OperatorSpaceMoEROM,
    a_state: np.ndarray,
    b_state: np.ndarray,
    cur: int,
    a_hist: np.ndarray,
    b_hist: np.ndarray,
    rhs_hist: np.ndarray,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    device: torch.device,
    pressure_state_override: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rhs_g = galerkin_rhs_by_label(
        tensors,
        a_state[None, :],
        b_state[None, :],
        arrays["label_id"][cur : cur + 1],
        arrays["labels"],
        args.r_u,
        args.r_p,
    )[0]
    base_x = make_features_np(
        a_state[None, :],
        b_state[None, :],
        rhs_g[None, :],
        arrays["re"][cur : cur + 1],
        arrays["phase"][cur : cur + 1],
        args.phase_harmonics,
    )
    x = make_history_features_from_states_np(
        base_x,
        a_state[None, :],
        b_state[None, :],
        rhs_g[None, :],
        a_hist,
        b_hist,
        rhs_hist,
    )
    xt = torch.tensor(scalers["x"].transform(x), dtype=torch.float32, device=device)
    pressure_override_t = None
    if pressure_state_override is not None:
        pressure_override_t = torch.tensor(
            pressure_state_override, dtype=torch.float32, device=device
        )
    with torch.no_grad():
        rhs_std, pressure_std, _, _, closure_params_t = model(
            xt,
            return_expert_stack=False,
            pressure_state_override=pressure_override_t,
            return_closure_params=True,
        )
    rhs_op = (
        rhs_std.detach().cpu().numpy()[0] * scalers["rhs_operator"].scale
        + scalers["rhs_operator"].mean
    )
    if args.rhs_target == "residual":
        rhs_op = rhs_g + rhs_op
    pressure_op = (
        pressure_std.detach().cpu().numpy()[0] * scalers["pressure_next"].scale
        + scalers["pressure_next"].mean
    )
    closure_params = {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in closure_params_t.items()
    }
    return rhs_op.astype(np.float32), pressure_op.astype(np.float32), rhs_g, closure_params


def integrate_autonomous_step_np(
    model: OperatorSpaceMoEROM,
    a_state: np.ndarray,
    b_state: np.ndarray,
    cur: int,
    dt: float,
    a_hist: np.ndarray,
    b_hist: np.ndarray,
    rhs_hist: np.ndarray,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k1, pressure_op, rhs_g, closure_params = model_outputs_from_states_np(
        model, a_state, b_state, cur, a_hist, b_hist, rhs_hist, arrays, scalers, tensors, args, device
    )
    if args.integrator == "rk4":
        k2, _, _, _ = model_outputs_from_states_np(
            model,
            a_state + 0.5 * dt * k1,
            b_state,
            cur,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        k3, _, _, _ = model_outputs_from_states_np(
            model,
            a_state + 0.5 * dt * k2,
            b_state,
            cur,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        k4, _, _, _ = model_outputs_from_states_np(
            model,
            a_state + dt * k3,
            b_state,
            cur,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        a_next = a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        a_next = a_state + dt * k1
    b_base = pressure_surrogate_by_label(
        pressure_tensors,
        a_next[None, :],
        arrays["label_id"][cur : cur + 1],
        arrays["labels"],
        args.r_u,
        args.r_p,
    )[0]
    if args.pressure_input_mode != "pressure_only":
        pressure_state_override = pressure_input_override_np(
            args.pressure_input_mode,
            a_next[None, :],
            b_base[None, :],
            scalers,
        )
        _, pressure_op, _, closure_params = model_outputs_from_states_np(
            model,
            a_state,
            b_state,
            cur,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            args,
            device,
            pressure_state_override=pressure_state_override,
        )
    if args.pressure_target == "state":
        b_next = pressure_op
    else:
        b_next, _, _, _, _ = closure_components_np(
            args.closure_mode,
            b_base[None, :],
            pressure_op[None, :],
            closure_params["alpha"],
            closure_params["beta"],
        )
        b_next = b_next[0]
    return a_next.astype(np.float32), b_next.astype(np.float32), rhs_g.astype(np.float32)


def rollout_autonomous_pressure_eval_np(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    test_label_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    idx = np.where(arrays["label_id"] == test_label_id)[0]
    idx = idx[np.argsort(arrays["time"][idx])]
    valid_sample_set = set(arrays["sample_ids"].tolist())
    a_rel_errors: List[float] = []
    b_rel_errors: List[float] = []
    all_true_a: List[np.ndarray] = []
    all_pred_a: List[np.ndarray] = []
    all_true_b: List[np.ndarray] = []
    all_pred_b: List[np.ndarray] = []
    stride = max(1, args.rollout_steps)
    model.eval()
    for start_pos in range(1, len(idx) - args.rollout_steps - 1, stride):
        start = int(idx[start_pos])
        if start not in valid_sample_set:
            continue
        a_cur = arrays["a"][start].copy()
        b_cur = arrays["b"][start].copy()
        a_hist, b_hist, rhs_hist = init_history_states_np(start, arrays)
        pred_a = []
        pred_b = []
        cur = start
        ok = True
        for _ in range(args.rollout_steps):
            nxt = int(arrays["next_idx"][cur])
            if nxt < 0 or int(arrays["label_id"][nxt]) != int(test_label_id):
                ok = False
                break
            dt = float(arrays["time"][nxt] - arrays["time"][cur])
            a_next, b_next, rhs_g = integrate_autonomous_step_np(
                model,
                a_cur,
                b_cur,
                cur,
                dt,
                a_hist,
                b_hist,
                rhs_hist,
                arrays,
                scalers,
                tensors,
                pressure_tensors,
                args,
                device,
            )
            if not (np.all(np.isfinite(a_next)) and np.all(np.isfinite(b_next))):
                ok = False
                break
            pred_a.append(a_next.copy())
            pred_b.append(b_next.copy())
            a_hist = np.concatenate([a_next[None, None, :], a_hist[:, :-1, :]], axis=1)
            b_hist = np.concatenate([b_next[None, None, :], b_hist[:, :-1, :]], axis=1)
            rhs_hist = np.concatenate([rhs_g[None, None, :], rhs_hist[:, :-1, :]], axis=1)
            a_cur = a_next
            b_cur = b_next
            cur = nxt
        if ok and len(pred_a) == args.rollout_steps:
            true_a = arrays["a"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            true_b = arrays["b"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            pred_a_arr = np.asarray(pred_a)
            pred_b_arr = np.asarray(pred_b)
            a_rel_errors.append(relative_l2_np(true_a, pred_a_arr))
            b_rel_errors.append(relative_l2_np(true_b, pred_b_arr))
            all_true_a.append(true_a)
            all_pred_a.append(pred_a_arr)
            all_true_b.append(true_b)
            all_pred_b.append(pred_b_arr)
    if not a_rel_errors:
        return {
            "a_relative_l2_mean": float("nan"),
            "a_relative_l2_median": float("nan"),
            "a_relative_l2_p90": float("nan"),
            "a_relative_l2_max": float("nan"),
            "b_relative_l2_mean": float("nan"),
            "b_relative_l2_median": float("nan"),
            "b_relative_l2_p90": float("nan"),
            "b_relative_l2_max": float("nan"),
            "b_energy_relative_error": float("nan"),
            "b_true_energy_mean": float("nan"),
            "b_pred_energy_mean": float("nan"),
            "num_windows": 0,
        }
    true_a_all_windows = np.stack(all_true_a, axis=0)
    pred_a_all_windows = np.stack(all_pred_a, axis=0)
    true_b_all = np.concatenate(all_true_b, axis=0)
    pred_b_all = np.concatenate(all_pred_b, axis=0)
    return {
        "a_relative_l2_mean": float(np.mean(a_rel_errors)),
        "a_relative_l2_median": float(np.median(a_rel_errors)),
        "a_relative_l2_p90": float(np.percentile(a_rel_errors, 90)),
        "a_relative_l2_max": float(np.max(a_rel_errors)),
        "b_relative_l2_mean": float(np.mean(b_rel_errors)),
        "b_relative_l2_median": float(np.median(b_rel_errors)),
        "b_relative_l2_p90": float(np.percentile(b_rel_errors, 90)),
        "b_relative_l2_max": float(np.max(b_rel_errors)),
        "b_energy_relative_error": energy_relative_error_np(true_b_all, pred_b_all),
        "b_true_energy_mean": energy_mean_np(true_b_all),
        "b_pred_energy_mean": energy_mean_np(pred_b_all),
        "hopf_amplitude": hopf_amplitude_metrics_np(
            true_a_all_windows, pred_a_all_windows, args
        ),
        "num_windows": int(len(a_rel_errors)),
    }


def one_step_integrator_eval_np(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    sample_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    preds = []
    targets = []
    for cur in sample_ids.tolist():
        nxt = int(arrays["next_idx"][cur])
        if nxt < 0 or int(arrays["label_id"][nxt]) != int(arrays["label_id"][cur]):
            continue
        dt = float(arrays["time"][nxt] - arrays["time"][cur])
        pred = integrate_step_np(
            model,
            arrays["a"][cur].copy(),
            int(cur),
            dt,
            arrays,
            scalers,
            tensors,
            args,
            device,
        )
        preds.append(pred)
        targets.append(arrays["a"][nxt])
    if not preds:
        return {"relative_l2": float("nan"), "rmse": float("nan"), "num_samples": 0}
    pred_arr = np.asarray(preds)
    target_arr = np.asarray(targets)
    return {
        "relative_l2": relative_l2_np(target_arr, pred_arr),
        "rmse": rmse_np(target_arr, pred_arr),
        "num_samples": int(len(preds)),
    }


def one_step_autonomous_pressure_eval_np(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    sample_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    pred_a = []
    true_a = []
    pred_b = []
    true_b = []
    for cur in sample_ids.tolist():
        nxt = int(arrays["next_idx"][cur])
        if nxt < 0 or int(arrays["label_id"][nxt]) != int(arrays["label_id"][cur]):
            continue
        dt = float(arrays["time"][nxt] - arrays["time"][cur])
        a_hist, b_hist, rhs_hist = init_history_states_np(int(cur), arrays)
        a_next, b_next, _ = integrate_autonomous_step_np(
            model,
            arrays["a"][cur].copy(),
            arrays["b"][cur].copy(),
            int(cur),
            dt,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            pressure_tensors,
            args,
            device,
        )
        pred_a.append(a_next)
        pred_b.append(b_next)
        true_a.append(arrays["a"][nxt])
        true_b.append(arrays["b"][nxt])
    if not pred_a:
        return {
            "a_relative_l2": float("nan"),
            "a_rmse": float("nan"),
            "b_relative_l2": float("nan"),
            "b_rmse": float("nan"),
            "b_energy_relative_error": float("nan"),
            "b_true_energy_mean": float("nan"),
            "b_pred_energy_mean": float("nan"),
            "num_samples": 0,
        }
    pred_a_arr = np.asarray(pred_a)
    true_a_arr = np.asarray(true_a)
    pred_b_arr = np.asarray(pred_b)
    true_b_arr = np.asarray(true_b)
    return {
        "a_relative_l2": relative_l2_np(true_a_arr, pred_a_arr),
        "a_rmse": rmse_np(true_a_arr, pred_a_arr),
        "b_relative_l2": relative_l2_np(true_b_arr, pred_b_arr),
        "b_rmse": rmse_np(true_b_arr, pred_b_arr),
        "b_energy_relative_error": energy_relative_error_np(true_b_arr, pred_b_arr),
        "b_true_energy_mean": energy_mean_np(true_b_arr),
        "b_pred_energy_mean": energy_mean_np(pred_b_arr),
        "hopf_amplitude": hopf_amplitude_metrics_np(true_a_arr, pred_a_arr, args),
        "num_samples": int(len(pred_a)),
    }


def gate_top1_for_ids_np(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if len(sample_ids) == 0:
        return np.asarray([], dtype=np.int64)
    model.eval()
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        _, _, gates, _ = model(x, return_expert_stack=False)
    gate_np = np.mean(np.stack([g.detach().cpu().numpy() for g in gates], axis=0), axis=0)
    return np.argmax(gate_np, axis=1).astype(np.int64)


def expert_error_analysis_np(
    model: OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, Standardizer],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    sample_ids: np.ndarray,
    test_label_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    if len(sample_ids) == 0:
        return {}
    top1 = gate_top1_for_ids_np(model, arrays, scalers, sample_ids, device)
    num_experts = int(max(model.num_experts, int(top1.max()) + 1 if len(top1) else 0))
    one_step = {
        e: {"pred_a": [], "true_a": [], "pred_b": [], "true_b": []}
        for e in range(num_experts)
    }
    for pos, cur in enumerate(sample_ids.tolist()):
        nxt = int(arrays["next_idx"][cur])
        if nxt < 0 or int(arrays["label_id"][nxt]) != int(arrays["label_id"][cur]):
            continue
        dt = float(arrays["time"][nxt] - arrays["time"][cur])
        a_hist, b_hist, rhs_hist = init_history_states_np(int(cur), arrays)
        a_next, b_next, _ = integrate_autonomous_step_np(
            model,
            arrays["a"][cur].copy(),
            arrays["b"][cur].copy(),
            int(cur),
            dt,
            a_hist,
            b_hist,
            rhs_hist,
            arrays,
            scalers,
            tensors,
            pressure_tensors,
            args,
            device,
        )
        item = one_step[int(top1[pos])]
        item["pred_a"].append(a_next)
        item["true_a"].append(arrays["a"][nxt])
        item["pred_b"].append(b_next)
        item["true_b"].append(arrays["b"][nxt])

    idx = np.where(arrays["label_id"] == test_label_id)[0]
    idx = idx[np.argsort(arrays["time"][idx])]
    valid_sample_set = set(arrays["sample_ids"].tolist())
    rollout = {e: {"a": [], "b": []} for e in range(num_experts)}
    stride = max(1, args.rollout_steps)
    for start_pos in range(1, len(idx) - args.rollout_steps - 1, stride):
        start = int(idx[start_pos])
        if start not in valid_sample_set:
            continue
        start_top1 = int(gate_top1_for_ids_np(model, arrays, scalers, np.asarray([start]), device)[0])
        a_cur = arrays["a"][start].copy()
        b_cur = arrays["b"][start].copy()
        a_hist, b_hist, rhs_hist = init_history_states_np(start, arrays)
        pred_a = []
        pred_b = []
        cur = start
        ok = True
        for _ in range(args.rollout_steps):
            nxt = int(arrays["next_idx"][cur])
            if nxt < 0 or int(arrays["label_id"][nxt]) != int(test_label_id):
                ok = False
                break
            dt = float(arrays["time"][nxt] - arrays["time"][cur])
            a_next, b_next, rhs_g = integrate_autonomous_step_np(
                model,
                a_cur,
                b_cur,
                cur,
                dt,
                a_hist,
                b_hist,
                rhs_hist,
                arrays,
                scalers,
                tensors,
                pressure_tensors,
                args,
                device,
            )
            if not (np.all(np.isfinite(a_next)) and np.all(np.isfinite(b_next))):
                ok = False
                break
            pred_a.append(a_next.copy())
            pred_b.append(b_next.copy())
            a_hist = np.concatenate([a_next[None, None, :], a_hist[:, :-1, :]], axis=1)
            b_hist = np.concatenate([b_next[None, None, :], b_hist[:, :-1, :]], axis=1)
            rhs_hist = np.concatenate([rhs_g[None, None, :], rhs_hist[:, :-1, :]], axis=1)
            a_cur = a_next
            b_cur = b_next
            cur = nxt
        if ok and len(pred_a) == args.rollout_steps:
            true_a = arrays["a"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            true_b = arrays["b"][idx[start_pos + 1 : start_pos + args.rollout_steps + 1]]
            rollout[start_top1]["a"].append(relative_l2_np(true_a, np.asarray(pred_a)))
            rollout[start_top1]["b"].append(relative_l2_np(true_b, np.asarray(pred_b)))

    per_expert: Dict[str, object] = {}
    for e in range(num_experts):
        item = one_step[e]
        if item["pred_a"]:
            pred_a_arr = np.asarray(item["pred_a"])
            true_a_arr = np.asarray(item["true_a"])
            pred_b_arr = np.asarray(item["pred_b"])
            true_b_arr = np.asarray(item["true_b"])
            one_a = relative_l2_np(true_a_arr, pred_a_arr)
            one_b = relative_l2_np(true_b_arr, pred_b_arr)
            count = int(len(item["pred_a"]))
        else:
            one_a = float("nan")
            one_b = float("nan")
            count = 0
        roll_a = rollout[e]["a"]
        roll_b = rollout[e]["b"]
        per_expert[str(e)] = {
            "top1_one_step_count": count,
            "one_step_a_relative_l2": float(one_a),
            "one_step_b_relative_l2": float(one_b),
            "rollout_window_count": int(len(roll_a)),
            "rollout_a_relative_l2_mean": float(np.mean(roll_a)) if roll_a else float("nan"),
            "rollout_b_relative_l2_mean": float(np.mean(roll_b)) if roll_b else float("nan"),
        }
    return per_expert


def build_arrays(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    pod_dir = args.data_root
    idx = load_snapshot_index(pod_dir / "pod_snapshot_index.csv")
    velocity_pod_path = pod_dir / "global_velocity_pod_area_weighted_l2.npz"
    pressure_pod_path = pod_dir / "global_pressure_pod_area_weighted_l2.npz"
    if not velocity_pod_path.exists():
        velocity_pod_path = pod_dir / "global_velocity_pod_weighted_l2.npz"
    if not pressure_pod_path.exists():
        pressure_pod_path = pod_dir / "global_pressure_pod_weighted_l2.npz"
    vel = np.load(velocity_pod_path)
    pre = np.load(pressure_pod_path)
    tensors = np.load(args.tensor_path)
    pressure_tensors = np.load(args.pressure_surrogate_path)

    a = vel["coeff_uv"][:, : args.r_u].astype(np.float32)
    b = pre["coeff_p"][:, : args.r_p].astype(np.float32)
    re = idx["Re"]
    labels = vel["Re_labels"].astype(str)
    label_to_id = {label: i for i, label in enumerate(labels.tolist())}
    label_id = np.asarray([label_to_id[str(label)] for label in idx["Re_label"]], dtype=np.int64)
    regime_names = np.asarray(idx.get("regime", np.asarray(["unknown"] * len(re), dtype=object))).astype(str)
    regime_vocab = np.asarray(sorted(set(regime_names.tolist())))
    regime_to_id = {name: i for i, name in enumerate(regime_vocab.tolist())}
    regime_id = np.asarray([regime_to_id[name] for name in regime_names], dtype=np.int64)
    attractor_names, attractor_id = attractor_ids_from_regimes(regime_names)
    attractor_vocab = np.asarray(ATTRACTOR_NAMES)
    phase = idx["phase"]
    time_arr = idx["time"]
    rhs_g = galerkin_rhs_by_label(tensors, a, b, label_id, labels, args.r_u, args.r_p)
    adot, valid_deriv = centered_time_derivative(a, re, time_arr)
    nxt = next_index_by_sequence(re, time_arr)
    prv = prev_index_by_sequence(re, time_arr)

    sample_ids = valid_deriv[nxt[valid_deriv] >= 0]
    sample_ids = sample_ids[label_id[nxt[sample_ids]] == label_id[sample_ids]]
    hist_idx = history_index_matrix(np.arange(a.shape[0], dtype=np.int64), prv, args.history_len)
    enough_history = np.all(hist_idx[sample_ids] >= 0, axis=1)
    sample_ids = sample_ids[enough_history]
    a_next = a[nxt]
    b_next = b[nxt]
    pressure_base_next = pressure_surrogate_by_label(
        pressure_tensors, a_next, label_id, labels, args.r_u, args.r_p
    )
    pressure_residual = (b_next - pressure_base_next).astype(np.float32)
    dt_next = np.zeros_like(time_arr, dtype=np.float32)
    ok = nxt >= 0
    dt_next[ok] = time_arr[nxt[ok]] - time_arr[ok]
    base_x = make_features_np(a, b, rhs_g, re, phase, args.phase_harmonics)
    x = make_history_features_np(base_x, a, b, rhs_g, hist_idx)
    residual = (adot - rhs_g).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    phi = vel["phi_uv"][: args.r_u].astype(np.float32)
    if args.recon_dim > 0 and args.recon_dim < phi.shape[1]:
        recon_cols = np.sort(rng.choice(phi.shape[1], size=args.recon_dim, replace=False))
        phi_recon = phi[:, recon_cols].copy()
    else:
        recon_cols = np.arange(phi.shape[1])
        phi_recon = phi

    arrays = {
        "a": a,
        "b": b,
        "rhs_g": rhs_g,
        "adot": adot,
        "residual": residual,
        "base_x": base_x,
        "x": x,
        "a_next": a_next.astype(np.float32),
        "b_next": b_next.astype(np.float32),
        "pressure_base_next": pressure_base_next.astype(np.float32),
        "pressure_residual": pressure_residual,
        "re": re,
        "label_id": label_id,
        "labels": labels,
        "regime": regime_names,
        "regime_id": regime_id,
        "regime_vocab": regime_vocab,
        "attractor": attractor_names,
        "attractor_id": attractor_id,
        "attractor_vocab": attractor_vocab,
        "stage_label": np.asarray(["attractor"] * len(re), dtype=object),
        "phase": phase,
        "time": time_arr,
        "prev_idx": prv,
        "next_idx": nxt,
        "hist_idx": hist_idx,
        "dt_next": dt_next,
        "sample_ids": sample_ids,
        "phi_recon": phi_recon.astype(np.float32),
    }
    meta = {
        "pod_energy": {
            f"velocity_first_{args.r_u}": float(vel["cumulative_energy_uv"][args.r_u - 1]),
            f"pressure_first_{args.r_p}": float(pre["cumulative_energy_p"][args.r_p - 1]),
            "velocity_first_16": float(vel["cumulative_energy_uv"][15]),
            "pressure_first_16": float(pre["cumulative_energy_p"][15]),
            "velocity_first_32": float(vel["cumulative_energy_uv"][31]),
            "pressure_first_32": float(pre["cumulative_energy_p"][31]),
            "velocity_first_80": float(vel["cumulative_energy_uv"][79]),
            "pressure_first_80": float(pre["cumulative_energy_p"][79]),
        },
        "num_re": int(len(labels)),
        "re_values": [float(v) for v in vel["Re_values"].tolist()],
        "re_labels": labels.tolist(),
        "regime_vocab": regime_vocab.tolist(),
        "attractor_vocab": attractor_vocab.tolist(),
        "stage_label": "attractor",
        "regime_by_re": [
            str(regime_names[np.where(label_id == i)[0][0]]) if np.any(label_id == i) else "unknown"
            for i in range(len(labels))
        ],
        "attractor_by_re": [
            str(attractor_names[np.where(label_id == i)[0][0]]) if np.any(label_id == i) else "unknown"
            for i in range(len(labels))
        ],
        "velocity_pod_path": str(velocity_pod_path),
        "pressure_pod_path": str(pressure_pod_path),
        "recon_columns": int(len(recon_cols)),
        "total_snapshots": int(a.shape[0]),
        "valid_samples": int(len(sample_ids)),
        "history_len": int(args.history_len),
    }
    return arrays, meta


def resolve_test_re_indices(args: argparse.Namespace, arrays: Dict[str, np.ndarray]) -> List[int]:
    labels = arrays["labels"]
    re_by_label = np.asarray(
        [float(np.mean(arrays["re"][arrays["label_id"] == i])) for i in range(len(labels))],
        dtype=np.float64,
    )
    if args.test_re_selection == "explicit":
        selected = [int(i) for i in args.test_re_indices]
    elif args.test_re_selection in {"values", "regime_default"}:
        values = args.test_re_values
        if not values:
            values = [24.0, 32.0, 40.0, 45.0, 47.0, 49.0, 52.0, 70.0, 100.0, 150.0, 190.0]
        selected = []
        used: set[int] = set()
        for target in values:
            order = np.argsort(np.abs(re_by_label - float(target)))
            for idx in order.tolist():
                if int(idx) not in used:
                    selected.append(int(idx))
                    used.add(int(idx))
                    break
        selected = sorted(selected, key=lambda i: float(re_by_label[i]))
    else:
        count = max(1, min(int(args.num_uniform_test_re), len(labels)))
        targets = np.linspace(float(np.min(re_by_label)), float(np.max(re_by_label)), count)
        selected = []
        used: set[int] = set()
        for target in targets:
            order = np.argsort(np.abs(re_by_label - target))
            for idx in order.tolist():
                if int(idx) not in used:
                    selected.append(int(idx))
                    used.add(int(idx))
                    break
        selected = sorted(selected, key=lambda i: float(re_by_label[i]))
    bad = [i for i in selected if i < 0 or i >= len(labels)]
    if bad:
        raise ValueError(f"Invalid test Re label indices: {bad}")
    return selected


def build_data_ablation_split(
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    holdout_label_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    sample_ids = arrays["sample_ids"]
    labels = arrays["labels"]
    label_ids = arrays["label_id"]
    holdout_set = set(int(v) for v in holdout_label_ids)
    all_label_ids = sorted(np.unique(label_ids[sample_ids]).astype(int).tolist())
    candidate_train_labels = [v for v in all_label_ids if v not in holdout_set]
    re_stride = max(1, int(args.train_re_stride))
    re_offset = int(args.train_re_offset) % re_stride
    selected_train_labels = [
        label_id for pos, label_id in enumerate(candidate_train_labels)
        if pos % re_stride == re_offset
    ]
    selected_set = set(int(v) for v in selected_train_labels)
    time_stride = max(1, int(args.train_time_stride))
    time_offset = int(args.train_time_offset) % time_stride

    train_ids: List[int] = []
    val_ids: List[int] = []
    rollout_pool_ids: List[int] = []
    per_re: List[Dict[str, object]] = []
    total_non_test = 0
    total_dense_train = 0
    total_sparse_train = 0
    total_val = 0
    total_test = 0
    total_excluded = 0

    for label_id_value in all_label_ids:
        ids = sample_ids[label_ids[sample_ids] == label_id_value]
        ids = ids[np.argsort(arrays["time"][ids])]
        role = "train"
        dense_train_count = 0
        kept_train_count = 0
        val_count = 0
        test_count = 0
        excluded_count = 0
        if label_id_value in holdout_set:
            role = "test"
            test_count = int(len(ids))
            total_test += test_count
        elif label_id_value not in selected_set:
            role = "excluded_by_re_sparsity"
            excluded_count = int(len(ids))
            total_excluded += excluded_count
            total_non_test += int(len(ids))
        else:
            total_non_test += int(len(ids))
            val_count = max(10, int(round(0.12 * len(ids))))
            val_count = min(max(0, len(ids) - 1), val_count)
            dense_train = ids[:-val_count] if val_count > 0 else ids
            val = ids[-val_count:] if val_count > 0 else ids[:0]
            sparse_train = dense_train[time_offset::time_stride]
            if len(sparse_train) == 0 and len(dense_train) > 0:
                sparse_train = dense_train[:1]
            train_ids.extend(sparse_train.tolist())
            val_ids.extend(val.tolist())
            rollout_pool_ids.extend(dense_train.tolist())
            dense_train_count = int(len(dense_train))
            kept_train_count = int(len(sparse_train))
            total_dense_train += dense_train_count
            total_sparse_train += kept_train_count
            total_val += int(len(val))
        per_re.append(
            {
                "label_id": int(label_id_value),
                "label": str(labels[label_id_value]),
                "Re": float(np.mean(arrays["re"][ids])) if len(ids) else float("nan"),
                "role": role,
                "total_valid_samples": int(len(ids)),
                "dense_train_samples": dense_train_count,
                "kept_train_samples": kept_train_count,
                "val_samples": int(val_count),
                "test_samples": test_count,
                "excluded_samples": excluded_count,
            }
        )

    train_arr = np.asarray(sorted(train_ids), dtype=np.int64)
    val_arr = np.asarray(sorted(val_ids), dtype=np.int64)
    rollout_pool_arr = np.asarray(sorted(rollout_pool_ids), dtype=np.int64)
    if len(train_arr) == 0:
        raise ValueError("Data ablation split produced zero training samples.")
    if len(val_arr) == 0:
        raise ValueError("Data ablation split produced zero validation samples.")
    stats = {
        "holdout_label_ids": [int(v) for v in holdout_label_ids],
        "holdout_labels": [str(labels[int(v)]) for v in holdout_label_ids],
        "holdout_Re": [
            float(np.mean(arrays["re"][sample_ids[label_ids[sample_ids] == int(v)]]))
            for v in holdout_label_ids
        ],
        "train_re_count": int(len(selected_train_labels)),
        "test_re_count": int(len(holdout_label_ids)),
        "excluded_re_count": int(len(candidate_train_labels) - len(selected_train_labels)),
        "train_re_labels": [str(labels[int(v)]) for v in selected_train_labels],
        "excluded_re_labels": [
            str(labels[int(v)]) for v in candidate_train_labels if v not in selected_set
        ],
        "train_time_stride": int(time_stride),
        "train_time_offset": int(time_offset),
        "train_re_stride": int(re_stride),
        "train_re_offset": int(re_offset),
        "total_non_test_candidate_samples": int(total_non_test),
        "dense_train_samples_before_time_sparsity": int(total_dense_train),
        "kept_train_samples": int(total_sparse_train),
        "val_samples": int(total_val),
        "test_samples": int(total_test),
        "excluded_samples_by_re_sparsity": int(total_excluded),
        "rollout_pool_samples": int(len(rollout_pool_arr)),
        "compression_ratio_vs_dense_train": float(
            total_sparse_train / max(total_dense_train, 1)
        ),
        "compression_ratio_vs_all_non_test_candidates": float(
            total_sparse_train / max(total_non_test, 1)
        ),
        "per_re": per_re,
    }
    regime_vocab = arrays.get("regime_vocab", np.asarray(["unknown"]))
    regime_id = arrays.get("regime_id")
    if regime_id is not None:
        def regime_counts(ids: np.ndarray) -> Dict[str, int]:
            if len(ids) == 0:
                return {str(name): 0 for name in regime_vocab.tolist()}
            counts = np.bincount(regime_id[ids].astype(int), minlength=len(regime_vocab))
            return {str(regime_vocab[i]): int(counts[i]) for i in range(len(regime_vocab))}

        stats["regime_sample_counts"] = {
            "train": regime_counts(train_arr),
            "val": regime_counts(val_arr),
            "rollout_pool": regime_counts(rollout_pool_arr),
            "test": regime_counts(
                sample_ids[np.isin(label_ids[sample_ids], np.asarray(holdout_label_ids))]
            ),
        }
    attractor_vocab = arrays.get("attractor_vocab", np.asarray(ATTRACTOR_NAMES))
    attractor_id = arrays.get("attractor_id")
    if attractor_id is not None:
        def attractor_counts(ids: np.ndarray) -> Dict[str, int]:
            if len(ids) == 0:
                return {str(name): 0 for name in attractor_vocab.tolist()}
            counts = np.bincount(
                attractor_id[ids].astype(int), minlength=len(attractor_vocab)
            )
            return {
                str(attractor_vocab[i]): int(counts[i])
                for i in range(len(attractor_vocab))
            }

        stats["attractor_sample_counts"] = {
            "train": attractor_counts(train_arr),
            "val": attractor_counts(val_arr),
            "rollout_pool": attractor_counts(rollout_pool_arr),
            "test": attractor_counts(
                sample_ids[np.isin(label_ids[sample_ids], np.asarray(holdout_label_ids))]
            ),
        }
    return train_arr, val_arr, rollout_pool_arr, stats


def build_attractor_prototypes_np(
    arrays: Dict[str, np.ndarray],
    train_ids: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    i0, i1 = parse_hopf_pair(args)
    n_labels = len(arrays["labels"])
    a = arrays["a"]
    b = arrays["b"]
    eu = np.sum(a * a, axis=1).astype(np.float32)
    ep = np.sum(b * b, axis=1).astype(np.float32)
    radius = np.sqrt(np.sum(a[:, [i0, i1]] ** 2, axis=1) + EPS).astype(np.float32)
    out: Dict[str, np.ndarray] = {}
    for name, values in [
        ("velocity_energy", eu),
        ("pressure_energy", ep),
        ("radius", radius),
    ]:
        mean = np.zeros(n_labels, dtype=np.float32)
        std = np.ones(n_labels, dtype=np.float32)
        for label_id in range(n_labels):
            ids = train_ids[arrays["label_id"][train_ids] == label_id]
            if len(ids) == 0:
                ids = np.where(arrays["label_id"] == label_id)[0]
            vals = values[ids]
            mean[label_id] = float(np.mean(vals)) if len(vals) else 0.0
            std_val = float(np.std(vals)) if len(vals) else 1.0
            std[label_id] = max(std_val, 1.0e-6)
        out[f"{name}_mean"] = mean
        out[f"{name}_std"] = std
    return out


def build_attractor_prototypes_torch(
    proto_np: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in proto_np.items()
    }


def train_one_split(
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, object],
    tensors: np.lib.npyio.NpzFile,
    pressure_tensors: np.lib.npyio.NpzFile,
    test_label_id: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    sample_ids = arrays["sample_ids"]
    holdout_label_ids = [int(v) for v in args.test_re_indices]
    train_ids, val_ids, rollout_pool_ids, split_stats = build_data_ablation_split(
        args, arrays, holdout_label_ids
    )
    first_test_ids = sample_ids[arrays["label_id"][sample_ids] == int(test_label_id)]
    print(
        json.dumps(
            {
                "event": "split_start",
                "experiment": args.experiment_name,
                "test_label": str(arrays["labels"][test_label_id]),
                "test_Re": float(arrays["re"][first_test_ids[0]]) if len(first_test_ids) else None,
                "holdout_labels": split_stats["holdout_labels"],
                "train_re_count": split_stats["train_re_count"],
                "test_re_count": split_stats["test_re_count"],
                "kept_train_samples": split_stats["kept_train_samples"],
                "dense_train_samples_before_time_sparsity": split_stats[
                    "dense_train_samples_before_time_sparsity"
                ],
                "compression_ratio_vs_dense_train": split_stats[
                    "compression_ratio_vs_dense_train"
                ],
            }
        ),
        flush=True,
    )
    swan_run = swanlab_init_run(args, split_stats)
    swanlab_log(
        swan_run,
        {
            "split/train_re_count": split_stats["train_re_count"],
            "split/test_re_count": split_stats["test_re_count"],
            "split/kept_train_samples": split_stats["kept_train_samples"],
            "split/dense_train_samples_before_time_sparsity": split_stats[
                "dense_train_samples_before_time_sparsity"
            ],
            "split/compression_ratio_vs_dense_train": split_stats[
                "compression_ratio_vs_dense_train"
            ],
            "meta/closure_mode": args.closure_mode,
        },
        step=0,
    )
    pressure_target_array = (
        arrays["b_next"] if args.pressure_target == "state" else arrays["pressure_residual"]
    )
    rhs_target_array = (
        arrays["residual"] if args.rhs_target == "residual" else arrays["adot"]
    )
    pressure_input_array = pressure_input_raw_np(
        args.pressure_input_mode,
        arrays["a_next"],
        arrays["pressure_base_next"],
    )

    scalers = {
        "x": Standardizer.fit(arrays["x"][train_ids]),
        "rhs_operator": Standardizer.fit(rhs_target_array[train_ids]),
        "alpha_next": Standardizer.fit(arrays["a_next"][train_ids]),
        "pressure_next": Standardizer.fit(pressure_target_array[train_ids]),
        "pressure_state": Standardizer.fit(arrays["b_next"][train_ids]),
    }
    if pressure_input_array is not None:
        scalers["pressure_input"] = Standardizer.fit(pressure_input_array[train_ids])
    alpha_rel_floor = (
        np.percentile(np.sum(arrays["a_next"][train_ids] ** 2, axis=1), 10)
        * args.relative_floor_frac
        + EPS
    )
    rhs_rel_floor = (
        np.percentile(np.sum(arrays["adot"][train_ids] ** 2, axis=1), 10)
        * args.relative_floor_frac
        + EPS
    )
    pressure_rel_floor = (
        np.percentile(np.sum(arrays["b_next"][train_ids] ** 2, axis=1), 10)
        * args.relative_floor_frac
        + EPS
    )
    pressure_energy_train = np.sum(arrays["b_next"][train_ids] ** 2, axis=1).astype(np.float32)
    pressure_energy_ref = float(np.median(pressure_energy_train) + EPS)
    i0_proto, i1_proto = parse_hopf_pair(args)
    hopf_train_ids = train_ids[arrays["attractor_id"][train_ids] == 1]
    if float(args.attractor_r_floor) > 0.0:
        args.attractor_r_floor_effective = float(args.attractor_r_floor)
    elif len(hopf_train_ids) > 0:
        hopf_radius_train = np.sqrt(
            np.sum(arrays["a_next"][hopf_train_ids][:, [i0_proto, i1_proto]] ** 2, axis=1)
            + EPS
        )
        q = float(np.clip(getattr(args, "v16_1_hopf_floor_quantile", 10.0), 1.0, 50.0))
        args.attractor_r_floor_effective = float(
            max(1.0e-5, np.percentile(hopf_radius_train, q))
        )
    else:
        args.attractor_r_floor_effective = 1.0e-5
    hopf_a_norm_train = (
        np.linalg.norm(arrays["a_next"][hopf_train_ids], axis=1)
        if len(hopf_train_ids) > 0
        else np.linalg.norm(arrays["a_next"][train_ids], axis=1)
    )
    args.v16_1_a_floor_effective = float(
        max(1.0e-5, np.percentile(hopf_a_norm_train, float(np.clip(getattr(args, "v16_1_hopf_floor_quantile", 10.0), 1.0, 50.0))))
    )

    x_train = torch.tensor(scalers["x"].transform(arrays["x"][train_ids]), device=device)
    rhs_train = torch.tensor(
        scalers["rhs_operator"].transform(rhs_target_array[train_ids]), device=device
    )
    pressure_train = torch.tensor(
        scalers["pressure_next"].transform(pressure_target_array[train_ids]),
        device=device,
    )
    pressure_input_train = None
    if pressure_input_array is not None:
        pressure_input_train = torch.tensor(
            scalers["pressure_input"].transform(pressure_input_array[train_ids]),
            device=device,
        )
    pressure_weight_train = torch.ones(len(train_ids), dtype=torch.float32, device=device)
    if args.pressure_amplitude_weight_power > 0.0:
        p_energy = torch.tensor(pressure_energy_train, dtype=torch.float32, device=device)
        pressure_weight_train = (
            torch.tensor(pressure_energy_ref, dtype=torch.float32, device=device)
            / torch.clamp(p_energy, min=EPS)
        ) ** args.pressure_amplitude_weight_power
        pressure_weight_train = torch.clamp(
            pressure_weight_train,
            min=1.0 / max(args.pressure_amplitude_weight_max, 1.0),
            max=max(args.pressure_amplitude_weight_max, 1.0),
        )
        pressure_weight_train = pressure_weight_train / torch.mean(pressure_weight_train)
    hopf_direct_weight_np = np.ones(len(train_ids), dtype=np.float32)
    train_re_for_hopf = arrays["re"][train_ids].astype(np.float32)
    hopf_direct_weight_np = np.where(
        (train_re_for_hopf >= float(args.hopf_re_min))
        & (train_re_for_hopf <= float(args.hopf_re_max)),
        hopf_direct_weight_np * float(args.hopf_sample_weight),
        hopf_direct_weight_np,
    ).astype(np.float32)
    hopf_direct_weight_np = np.where(
        np.abs(train_re_for_hopf - float(args.hopf_focus_re)) <= float(args.hopf_focus_width),
        hopf_direct_weight_np * float(args.hopf_focus_sample_weight),
        hopf_direct_weight_np,
    ).astype(np.float32)
    hopf_direct_weight_train = torch.tensor(hopf_direct_weight_np, dtype=torch.float32, device=device)
    x_all_t = torch.tensor(scalers["x"].transform(arrays["x"]), device=device)
    current_a_train = torch.tensor(arrays["a"][train_ids], device=device)
    rhs_g_train = torch.tensor(arrays["rhs_g"][train_ids], device=device)
    dt_train = torch.tensor(arrays["dt_next"][train_ids][:, None], device=device)
    train_ids_t = torch.tensor(train_ids, dtype=torch.long, device=device)
    phi_recon = torch.tensor(arrays["phi_recon"], device=device)
    prototype_np = build_attractor_prototypes_np(arrays, train_ids, args)
    prototype_t = build_attractor_prototypes_torch(prototype_np, device)
    pressure_mean_by_label_np = np.zeros((len(arrays["labels"]), args.r_p), dtype=np.float32)
    pressure_energy_mean_by_label_np = np.zeros((len(arrays["labels"]),), dtype=np.float32)
    for label_id in range(len(arrays["labels"])):
        ids = train_ids[arrays["label_id"][train_ids] == label_id]
        if len(ids) == 0:
            ids = np.where(arrays["label_id"] == label_id)[0]
        if len(ids):
            b_vals = arrays["b_next"][ids].astype(np.float32)
            pressure_mean_by_label_np[label_id] = np.mean(b_vals, axis=0)
            pressure_energy_mean_by_label_np[label_id] = np.mean(np.sum(b_vals * b_vals, axis=1))
    np.savez(
        args.output_dir / f"{args.experiment_name}_pressure_anchor_stats.npz",
        Re_labels=arrays["labels"],
        pressure_mean_by_label=pressure_mean_by_label_np,
        pressure_energy_mean_by_label=pressure_energy_mean_by_label_np,
        train_Re_labels=np.asarray(split_stats["train_re_labels"], dtype=object),
        r_floor=np.asarray([float(args.attractor_r_floor_effective)], dtype=np.float32),
        a_floor=np.asarray([float(args.v16_1_a_floor_effective)], dtype=np.float32),
    )
    train_sampling_weights_t = None
    if args.attractor_balanced_sampling:
        attractor_train = arrays["attractor_id"][train_ids].astype(np.int64)
        counts = np.bincount(
            attractor_train, minlength=len(arrays["attractor_vocab"])
        ).astype(np.float64)
        inv = np.zeros_like(counts, dtype=np.float64)
        inv[counts > 0] = 1.0 / counts[counts > 0]
        weights = inv[attractor_train]
        weights = weights / np.sum(weights)
        train_sampling_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        split_stats["attractor_balanced_sampling"] = {
            "enabled": True,
            "attractor_counts": {
                str(arrays["attractor_vocab"][i]): int(counts[i])
                for i in range(len(arrays["attractor_vocab"]))
            },
            "expected_epoch_fraction": {
                str(arrays["attractor_vocab"][i]): float(1.0 / max(1, int(np.sum(counts > 0))))
                if counts[i] > 0
                else 0.0
                for i in range(len(arrays["attractor_vocab"]))
            },
        }
        split_stats["regime_balanced_sampling"] = {"enabled": False}
    elif args.regime_balanced_sampling:
        regime_train = arrays["regime_id"][train_ids].astype(np.int64)
        counts = np.bincount(regime_train, minlength=len(arrays["regime_vocab"])).astype(np.float64)
        inv = np.zeros_like(counts, dtype=np.float64)
        inv[counts > 0] = 1.0 / counts[counts > 0]
        weights = inv[regime_train]
        re_train = arrays["re"][train_ids].astype(np.float64)
        weights = np.where(
            (re_train >= float(args.hopf_re_min)) & (re_train <= float(args.hopf_re_max)),
            weights * float(args.hopf_sample_weight),
            weights,
        )
        weights = np.where(
            np.abs(re_train - float(args.hopf_focus_re)) <= float(args.hopf_focus_width),
            weights * float(args.hopf_focus_sample_weight),
            weights,
        )
        weights = weights / np.sum(weights)
        train_sampling_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        split_stats["regime_balanced_sampling"] = {
            "enabled": True,
            "regime_counts": {
                str(arrays["regime_vocab"][i]): int(counts[i])
                for i in range(len(arrays["regime_vocab"]))
            },
            "expected_epoch_fraction": {
                str(arrays["regime_vocab"][i]): float(1.0 / max(1, int(np.sum(counts > 0))))
                if counts[i] > 0
                else 0.0
                for i in range(len(arrays["regime_vocab"]))
            },
            "hopf_sample_weight": float(args.hopf_sample_weight),
            "hopf_focus_sample_weight": float(args.hopf_focus_sample_weight),
            "hopf_focus_re": float(args.hopf_focus_re),
            "hopf_re_range": [float(args.hopf_re_min), float(args.hopf_re_max)],
        }
        split_stats["attractor_balanced_sampling"] = {"enabled": False}
    else:
        split_stats["regime_balanced_sampling"] = {"enabled": False}
        split_stats["attractor_balanced_sampling"] = {"enabled": False}

    arrays_t = {
        "a": torch.tensor(arrays["a"], dtype=torch.float32, device=device),
        "b": torch.tensor(arrays["b"], dtype=torch.float32, device=device),
        "a_next": torch.tensor(arrays["a_next"], dtype=torch.float32, device=device),
        "b_next": torch.tensor(arrays["b_next"], dtype=torch.float32, device=device),
        "pressure_base_next": torch.tensor(
            arrays["pressure_base_next"], dtype=torch.float32, device=device
        ),
        "pressure_residual": torch.tensor(
            arrays["pressure_residual"], dtype=torch.float32, device=device
        ),
        "adot": torch.tensor(arrays["adot"], dtype=torch.float32, device=device),
        "rhs_g": torch.tensor(arrays["rhs_g"], dtype=torch.float32, device=device),
        "re": torch.tensor(arrays["re"], dtype=torch.float32, device=device),
        "label_id": torch.tensor(arrays["label_id"], dtype=torch.long, device=device),
        "regime_id": torch.tensor(arrays["regime_id"], dtype=torch.long, device=device),
        "attractor_id": torch.tensor(arrays["attractor_id"], dtype=torch.long, device=device),
        "phase": torch.tensor(arrays["phase"], dtype=torch.float32, device=device),
        "time": torch.tensor(arrays["time"], dtype=torch.float32, device=device),
        "prev_idx": torch.tensor(arrays["prev_idx"], dtype=torch.long, device=device),
        "next_idx": torch.tensor(arrays["next_idx"], dtype=torch.long, device=device),
        "hist_idx": torch.tensor(arrays["hist_idx"], dtype=torch.long, device=device),
        "pressure_mean_by_label": torch.tensor(
            pressure_mean_by_label_np, dtype=torch.float32, device=device
        ),
        "pressure_energy_mean_by_label": torch.tensor(
            pressure_energy_mean_by_label_np, dtype=torch.float32, device=device
        ),
    }
    scalers_t = {
        "x_mean": torch.tensor(scalers["x"].mean, dtype=torch.float32, device=device),
        "x_scale": torch.tensor(scalers["x"].scale, dtype=torch.float32, device=device),
        "rhs_op_mean": torch.tensor(
            scalers["rhs_operator"].mean, dtype=torch.float32, device=device
        ),
        "rhs_op_scale": torch.tensor(
            scalers["rhs_operator"].scale, dtype=torch.float32, device=device
        ),
        "alpha_mean": torch.tensor(scalers["alpha_next"].mean, dtype=torch.float32, device=device),
        "alpha_scale": torch.tensor(scalers["alpha_next"].scale, dtype=torch.float32, device=device),
        "pressure_mean": torch.tensor(
            scalers["pressure_next"].mean, dtype=torch.float32, device=device
        ),
        "pressure_scale": torch.tensor(
            scalers["pressure_next"].scale, dtype=torch.float32, device=device
        ),
        "pressure_state_scale": torch.tensor(
            scalers["pressure_state"].scale, dtype=torch.float32, device=device
        ),
        "pressure_input_mean": torch.tensor(
            scalers.get(
                "pressure_input",
                Standardizer(
                    mean=np.zeros(args.r_u + args.r_p, dtype=np.float32),
                    scale=np.ones(args.r_u + args.r_p, dtype=np.float32),
                ),
            ).mean,
            dtype=torch.float32,
            device=device,
        ),
        "pressure_input_scale": torch.tensor(
            scalers.get(
                "pressure_input",
                Standardizer(
                    mean=np.zeros(args.r_u + args.r_p, dtype=np.float32),
                    scale=np.ones(args.r_u + args.r_p, dtype=np.float32),
                ),
            ).scale,
            dtype=torch.float32,
            device=device,
        ),
        "alpha_rel_floor": torch.tensor(alpha_rel_floor, dtype=torch.float32, device=device),
        "rhs_rel_floor": torch.tensor(rhs_rel_floor, dtype=torch.float32, device=device),
        "pressure_rel_floor": torch.tensor(
            pressure_rel_floor, dtype=torch.float32, device=device
        ),
    }
    gal = build_galerkin_torch(tensors, arrays["labels"], args.r_u, args.r_p, device)
    sur = build_pressure_surrogate_torch(
        pressure_tensors, arrays["labels"], args.r_u, args.r_p, device
    )
    base_bundle = build_base_bundle(args, arrays, device)

    model = OperatorSpaceMoEROM(
        in_dim=arrays["x"].shape[1],
        out_dim=args.r_u,
        pressure_dim=args.r_p,
        hidden_dim=args.hidden_dim,
        expert_hidden=args.expert_hidden,
        num_blocks=args.num_blocks,
        num_experts=args.num_experts,
        num_operator_spaces=args.num_shared_experts,
        num_regime_groups=args.num_regime_groups,
        experts_per_group=args.experts_per_group,
        top_k=args.top_k,
        group_top_k=args.group_top_k,
        dropout=args.dropout,
        temperature=args.temperature,
        gate_floor=args.gate_floor,
        group_temperature=args.group_temperature,
        group_gate_floor=args.group_gate_floor,
        shared_scale=args.shared_scale,
        routed_scale=args.routed_scale,
        expert_blocks=args.expert_blocks,
        quadratic_rank=args.quadratic_rank,
        quadratic_scale=args.quadratic_scale,
        phase_harmonics=args.phase_harmonics,
        closure_mode=args.closure_mode,
        pressure_base_mode=args.pressure_base_mode,
        film_base_hidden=args.film_base_hidden,
        film_base_scale=args.film_base_scale,
        attractor_conditioned=args.attractor_conditioned,
        attractor_adapter_dim=args.attractor_adapter_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(20, args.epochs))

    max_curriculum_steps = max(parse_curriculum_steps(args.curriculum_steps))
    train_roll_starts = sequence_start_ids(
        train_ids,
        arrays["label_id"],
        arrays["next_idx"],
        max_curriculum_steps,
        sequence_pool_ids=rollout_pool_ids,
    )
    train_roll_starts_t = torch.tensor(train_roll_starts, dtype=torch.long, device=device)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    history = []
    num_train = len(train_ids)
    epoch_iter = range(1, args.epochs + 1)
    if args.eval_only_checkpoint is not None:
        ckpt = torch.load(args.eval_only_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        best_epoch = int(ckpt.get("best_epoch", -1))
        best_val = float(ckpt.get("best_val_score", float("nan")))
        history.append(
            {
                "event": "eval_only_checkpoint_loaded",
                "checkpoint": str(args.eval_only_checkpoint),
                "best_epoch": best_epoch,
                "best_val_score": best_val,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        epoch_iter = range(0)

    for epoch in epoch_iter:
        model.train()
        args.scheduled_sampling_probability = scheduled_sampling_probability(args, epoch)
        if train_sampling_weights_t is None:
            perm = torch.randperm(num_train, device=device)
        else:
            perm = torch.multinomial(train_sampling_weights_t, num_train, replacement=True)
        batch_losses = []
        for start in range(0, num_train, args.batch_size):
            bi = perm[start : start + args.batch_size]
            x = x_train[bi]
            batch_ids = train_ids_t[bi]
            next_train_ids = arrays_t["next_idx"][batch_ids]
            pressure_state_override = (
                pressure_input_train[bi] if pressure_input_train is not None else None
            )
            rhs_std, pressure_pred_std, gates, expert_stack, closure_params = model(
                x,
                pressure_state_override=pressure_state_override,
                return_closure_params=True,
            )
            rhs_base_batch = velocity_base_from_sample_torch(
                model,
                current_a_train[bi],
                arrays_t["b"][batch_ids],
                batch_ids,
                arrays_t,
                gal,
                base_bundle,
                closure_params,
            )
            rhs_target_raw = (
                arrays_t["adot"][batch_ids] - rhs_base_batch
                if args.rhs_target == "residual"
                else arrays_t["adot"][batch_ids]
            )
            rhs_target = (rhs_target_raw - scalers_t["rhs_op_mean"]) / scalers_t["rhs_op_scale"]
            dyn_loss = F.mse_loss(rhs_std, rhs_target)

            rhs_pred = rhs_std * scalers_t["rhs_op_scale"] + scalers_t["rhs_op_mean"]
            if args.rhs_target == "residual":
                rhs_pred = rhs_base_batch + rhs_pred
            alpha_euler = current_a_train[bi] + dt_train[bi] * rhs_pred
            pressure_base_next_batch = pressure_base_from_sample_torch(
                model,
                arrays_t["a_next"][batch_ids],
                batch_ids,
                arrays_t,
                sur,
                base_bundle,
                closure_params,
            )
            if args.pressure_target == "state":
                pressure_target_raw = arrays_t["b_next"][batch_ids]
            else:
                _base_only_pred, pressure_base_component, _, _, _ = closure_components_torch(
                    args.closure_mode,
                    pressure_base_next_batch,
                    torch.zeros_like(pressure_base_next_batch),
                    closure_params,
                )
                pressure_target_raw = arrays_t["b_next"][batch_ids] - pressure_base_component
            pressure_target = (
                pressure_target_raw - scalers_t["pressure_mean"]
            ) / scalers_t["pressure_scale"]
            pressure_mse = torch.mean((pressure_pred_std - pressure_target) ** 2, dim=1)
            pressure_loss = torch.mean(pressure_weight_train[bi] * pressure_mse)
            coeff_loss = F.mse_loss(
                (alpha_euler - arrays_t["a"][next_train_ids]) / scalers_t["alpha_scale"],
                torch.zeros_like(alpha_euler),
            )
            alpha_rel_loss = relative_vector_loss_torch(
                alpha_euler - arrays_t["a"][next_train_ids],
                arrays_t["a"][next_train_ids],
                scalers_t["alpha_rel_floor"],
            )
            rhs_rel_loss = relative_vector_loss_torch(
                rhs_pred - arrays_t["adot"][batch_ids],
                arrays_t["adot"][batch_ids],
                scalers_t["rhs_rel_floor"],
            )
            pressure_op_phy = (
                pressure_pred_std * scalers_t["pressure_scale"]
                + scalers_t["pressure_mean"]
            )
            if args.pressure_target == "state":
                pressure_next_phy = pressure_op_phy
            else:
                pressure_next_phy, _, _, _, _ = closure_components_torch(
                    args.closure_mode,
                    pressure_base_next_batch,
                    pressure_op_phy,
                    closure_params,
                )
            pressure_rel_loss = relative_vector_loss_torch(
                pressure_next_phy - arrays_t["b_next"][batch_ids],
                arrays_t["b_next"][batch_ids],
                scalers_t["pressure_rel_floor"],
            )
            attr_logits = closure_params.get("attr_logits")
            attr_pi = closure_params.get("attr_pi") if args.attractor_conditioned else None
            if attr_logits is not None and float(args.lambda_attractor_ce) > 0.0:
                attractor_ce_loss = F.cross_entropy(
                    attr_logits, arrays_t["attractor_id"][batch_ids]
                )
            else:
                attractor_ce_loss = torch.tensor(0.0, device=device)
            finite_step_rhs = (
                arrays_t["a"][next_train_ids] - current_a_train[bi]
            ) / torch.clamp(dt_train[bi], min=1.0e-8)
            consistency_loss = F.mse_loss(
                (rhs_pred - finite_step_rhs) / scalers_t["rhs_op_scale"],
                torch.zeros_like(rhs_pred),
            )

            recon_delta = (alpha_euler - arrays_t["a"][next_train_ids]) @ phi_recon
            recon_true = arrays_t["a"][next_train_ids] @ phi_recon
            recon_loss = torch.mean(recon_delta**2) / (torch.mean(recon_true**2) + EPS)
            hopf_step_loss, hopf_step_info = hopf_envelope_loss_torch(
                alpha_euler,
                arrays_t["a"][next_train_ids],
                arrays_t["re"][batch_ids],
                args,
                epoch,
                hopf_direct_weight_train[bi],
            )
            attractor_loss, attractor_info = attractor_specific_loss_torch(
                alpha_euler,
                pressure_next_phy,
                arrays_t["a"][next_train_ids],
                arrays_t["b_next"][batch_ids],
                current_a_train[bi],
                arrays_t["b"][batch_ids],
                rhs_pred,
                arrays_t["attractor_id"][batch_ids],
                args,
                attr_pi,
            )
            proto_loss, proto_info = prototype_loss_torch(
                alpha_euler,
                pressure_next_phy,
                arrays_t["label_id"][batch_ids],
                prototype_t,
                args,
            )

            balance_loss, entropy_loss, router_info = router_regularization(gates)
            group_gate, group_logits = model.group_router_outputs(x)
            group_balance_loss, group_entropy_loss, group_supervision_loss = (
                group_router_regularization(group_gate, group_logits, arrays_t["re"][batch_ids])
            )
            diversity_loss = expert_diversity_loss(expert_stack)
            regime_loss = regime_router_loss(gates, arrays_t["re"][batch_ids])
            prev_ids = arrays_t["prev_idx"][batch_ids]
            smooth_mask = (prev_ids >= 0) & (
                arrays_t["label_id"][prev_ids] == arrays_t["label_id"][batch_ids]
            )
            if bool(torch.any(smooth_mask)):
                _, _, prev_gates, _ = model(
                    x_all_t[prev_ids[smooth_mask]], return_expert_stack=False
                )
                smooth_loss = gate_smoothness_loss([g[smooth_mask] for g in gates], prev_gates)
            else:
                smooth_loss = torch.tensor(0.0, device=device)

            loss = (
                args.lambda_coeff * coeff_loss
                + args.lambda_dyn * dyn_loss
                + args.lambda_pressure * pressure_loss
                + args.lambda_alpha_rel * alpha_rel_loss
                + args.lambda_rhs_rel * rhs_rel_loss
                + args.lambda_pressure_rel * pressure_rel_loss
                + args.lambda_recon * recon_loss
                + args.lambda_consistency * consistency_loss
                + args.lambda_router_balance * balance_loss
                + args.lambda_router_entropy * entropy_loss
                + args.lambda_group_balance * group_balance_loss
                + args.lambda_group_entropy * group_entropy_loss
                + args.lambda_group_supervision * group_supervision_loss
                + args.lambda_router_smooth * smooth_loss
                + args.lambda_expert_diversity * diversity_loss
                + args.lambda_regime_router * regime_loss
                + hopf_step_loss
                + attractor_loss
                + float(args.lambda_attractor_ce) * attractor_ce_loss
                + proto_loss
            )

            batch_no = start // args.batch_size
            do_rollout = (
                args.lambda_rollout > 0
                and train_roll_starts_t.numel() > 0
                and batch_no % max(1, args.rollout_every_batches) == 0
            )
            if do_rollout:
                current_rollout_steps = curriculum_step_for_epoch(args, epoch)
                count = min(args.rollout_batch, int(train_roll_starts_t.numel()))
                rb = train_roll_starts_t[torch.randint(0, train_roll_starts_t.numel(), (count,), device=device)]
                (
                    roll_loss,
                    energy_loss,
                    trajectory_loss,
                    hopf_rollout_loss,
                    v16_1_rollout_loss,
                    hopf_rollout_info,
                ) = rollout_loss_torch(
                    model,
                    rb,
                    arrays_t,
                    scalers_t,
                    gal,
                    sur,
                    base_bundle,
                    args,
                    current_rollout_steps,
                    epoch,
                )
                loss = (
                    loss
                    + args.lambda_rollout * roll_loss
                    + args.lambda_energy * energy_loss
                    + args.lambda_trajectory_consistency * trajectory_loss
                    + hopf_rollout_loss
                    + v16_1_rollout_loss
                )
            else:
                current_rollout_steps = 0
                roll_loss = torch.tensor(0.0, device=device)
                energy_loss = torch.tensor(0.0, device=device)
                trajectory_loss = torch.tensor(0.0, device=device)
                hopf_rollout_loss = torch.tensor(0.0, device=device)
                v16_1_rollout_loss = torch.tensor(0.0, device=device)
                hopf_rollout_info = {
                    "log_amp": hopf_rollout_loss,
                    "energy": hopf_rollout_loss,
                    "overshoot": hopf_rollout_loss,
                    "overshoot_ratio": hopf_rollout_loss,
                    "v16_1_hopf_growth": hopf_rollout_loss,
                    "v16_1_hopf_false_growth": hopf_rollout_loss,
                    "v16_1_hopf_floor_rel": hopf_rollout_loss,
                    "v16_1_steady_p_state": hopf_rollout_loss,
                    "v16_1_steady_p_mean": hopf_rollout_loss,
                    "v16_1_steady_p_delta": hopf_rollout_loss,
                    "v16_1_steady_residual_damp": hopf_rollout_loss,
                    "v16_1_steady_p_energy": hopf_rollout_loss,
                    "v16_1_loss_warmup": hopf_rollout_loss,
                }

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "coeff": float(coeff_loss.detach().cpu()),
                    "dyn": float(dyn_loss.detach().cpu()),
                    "pressure": float(pressure_loss.detach().cpu()),
                    "alpha_rel": float(alpha_rel_loss.detach().cpu()),
                    "rhs_rel": float(rhs_rel_loss.detach().cpu()),
                    "pressure_rel": float(pressure_rel_loss.detach().cpu()),
                    "recon": float(recon_loss.detach().cpu()),
                    "rollout": float(roll_loss.detach().cpu()),
                    "energy": float(energy_loss.detach().cpu()),
                    "trajectory_consistency": float(trajectory_loss.detach().cpu()),
                    "hopf_step_envelope": float(hopf_step_loss.detach().cpu()),
                    "hopf_step_log_amp": float(hopf_step_info["log_amp"].detach().cpu()),
                    "hopf_step_energy": float(hopf_step_info["energy"].detach().cpu()),
                    "hopf_step_overshoot": float(hopf_step_info["overshoot"].detach().cpu()),
                    "hopf_step_overshoot_ratio": float(
                        hopf_step_info["overshoot_ratio"].detach().cpu()
                    ),
                    "attractor_total": float(attractor_loss.detach().cpu()),
                    "attractor_ce": float(attractor_ce_loss.detach().cpu()),
                    "steady_rhs": float(attractor_info["steady_rhs"].detach().cpu()),
                    "steady_state": float(attractor_info["steady_state"].detach().cpu()),
                    "attractor_hopf_radius": float(
                        attractor_info["hopf_radius"].detach().cpu()
                    ),
                    "attractor_hopf_overshoot": float(
                        attractor_info["hopf_overshoot"].detach().cpu()
                    ),
                    "attractor_hopf_onset": float(
                        attractor_info["hopf_onset"].detach().cpu()
                    ),
                    "periodic_energy": float(attractor_info["periodic_energy"].detach().cpu()),
                    "periodic_radius": float(attractor_info["periodic_radius"].detach().cpu()),
                    "prototype_total": float(proto_loss.detach().cpu()),
                    "prototype_energy": float(proto_info["energy"].detach().cpu()),
                    "prototype_radius": float(proto_info["radius"].detach().cpu()),
                    "v16_1_rollout_total": float(v16_1_rollout_loss.detach().cpu()),
                    "v16_1_hopf_growth": float(
                        hopf_rollout_info["v16_1_hopf_growth"].detach().cpu()
                    ),
                    "v16_1_hopf_false_growth": float(
                        hopf_rollout_info["v16_1_hopf_false_growth"].detach().cpu()
                    ),
                    "v16_1_hopf_floor_rel": float(
                        hopf_rollout_info["v16_1_hopf_floor_rel"].detach().cpu()
                    ),
                    "v16_1_steady_p_state": float(
                        hopf_rollout_info["v16_1_steady_p_state"].detach().cpu()
                    ),
                    "v16_1_steady_p_mean": float(
                        hopf_rollout_info["v16_1_steady_p_mean"].detach().cpu()
                    ),
                    "v16_1_steady_p_delta": float(
                        hopf_rollout_info["v16_1_steady_p_delta"].detach().cpu()
                    ),
                    "v16_1_steady_residual_damp": float(
                        hopf_rollout_info["v16_1_steady_residual_damp"].detach().cpu()
                    ),
                    "v16_1_steady_p_energy": float(
                        hopf_rollout_info["v16_1_steady_p_energy"].detach().cpu()
                    ),
                    "v16_1_loss_warmup": float(
                        hopf_rollout_info["v16_1_loss_warmup"].detach().cpu()
                    ),
                    "hopf_rollout_envelope": float(hopf_rollout_loss.detach().cpu()),
                    "hopf_rollout_log_amp": float(
                        hopf_rollout_info["log_amp"].detach().cpu()
                    ),
                    "hopf_rollout_energy": float(
                        hopf_rollout_info["energy"].detach().cpu()
                    ),
                    "hopf_rollout_overshoot": float(
                        hopf_rollout_info["overshoot"].detach().cpu()
                    ),
                    "hopf_rollout_overshoot_ratio": float(
                        hopf_rollout_info["overshoot_ratio"].detach().cpu()
                    ),
                    "rollout_steps": int(current_rollout_steps),
                    "scheduled_sampling": float(args.scheduled_sampling_probability),
                    "router_smooth": float(smooth_loss.detach().cpu()),
                    "group_balance": float(group_balance_loss.detach().cpu()),
                    "group_entropy": float(group_entropy_loss.detach().cpu()),
                    "group_supervision": float(group_supervision_loss.detach().cpu()),
                    "expert_diversity": float(diversity_loss.detach().cpu()),
                    "regime_router": float(regime_loss.detach().cpu()),
                }
            )
        sched.step()

        if epoch == 1 or epoch % max(1, args.eval_every) == 0:
            val = evaluate_model(
                model, arrays, scalers, val_ids, args, device, arrays_t, gal, sur, base_bundle
            )
            val_score = (
                val["rhs_relative_l2"]
                + val["alpha_head_relative_l2"]
                + 0.35 * val["pressure_head_relative_l2"]
            )
            mean_loss = float(np.mean([b["loss"] for b in batch_losses]))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "val_score": float(val_score),
                    "val_rhs_relative_l2": float(val["rhs_relative_l2"]),
                    "val_alpha_relative_l2": float(val["alpha_head_relative_l2"]),
                    "val_pressure_relative_l2": float(val["pressure_head_relative_l2"]),
                }
            )
            event = {
                "event": "epoch_eval",
                "experiment": args.experiment_name,
                "test_label": str(arrays["labels"][test_label_id]),
                "epoch": int(epoch),
                "train_loss": mean_loss,
                "val_score": float(val_score),
                "val_rhs_relative_l2": float(val["rhs_relative_l2"]),
                "val_alpha_relative_l2": float(val["alpha_head_relative_l2"]),
                "val_pressure_relative_l2": float(val["pressure_head_relative_l2"]),
                "scheduled_sampling": float(args.scheduled_sampling_probability),
                "rollout_steps": int(curriculum_step_for_epoch(args, epoch)),
            }
            if args.eval_routing_every > 0 and epoch % args.eval_routing_every == 0:
                route_ids = np.asarray(val_ids)
                if len(route_ids) > args.eval_routing_max_samples:
                    rng = np.random.default_rng(args.seed + epoch)
                    route_ids = np.sort(
                        rng.choice(
                            route_ids,
                            size=args.eval_routing_max_samples,
                            replace=False,
                        )
                    )
                route = routing_analysis(model, arrays, scalers, route_ids, args, device)
                group_route = group_routing_analysis(model, arrays, scalers, route_ids, device)
                attr_route = attractor_router_analysis(
                    model, arrays, scalers, route_ids, device
                )
                event["eval_routing"] = {
                    "num_samples": int(len(route_ids)),
                    "group_mean_load": group_route.get("mean_load", []),
                    "group_top1_fraction": group_route.get("top1_fraction", []),
                    "active_experts_mean": route.get("active_experts_mean"),
                    "active_experts_std": route.get("active_experts_std"),
                    "dead_experts_threshold_1pct": route.get(
                        "dead_experts_threshold_1pct"
                    ),
                    "load_cv": route.get("load_cv"),
                    "attractor_accuracy": attr_route.get("accuracy"),
                    "attractor_pi_mean": attr_route.get("pi_mean", []),
                    "adapter_delta_to_h_norm_mean": attr_route.get(
                        "adapter_delta_to_h_norm_mean"
                    ),
                }
                history[-1]["eval_routing"] = event["eval_routing"]
            print(json.dumps(event), flush=True)
            swanlab_log(
                swan_run,
                swanlab_epoch_payload(
                    batch_losses,
                    val,
                    float(val_score),
                    int(epoch),
                    args,
                    float(sched.get_last_lr()[0]),
                ),
                step=int(epoch),
            )
            if val_score < best_val - args.early_stop_min_delta:
                best_val = float(val_score)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = args.output_dir / (
        f"{args.experiment_name}_{str(arrays['labels'][test_label_id])}_checkpoint.pt"
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "test_label_id": int(test_label_id),
            "holdout_label_ids": [int(v) for v in holdout_label_ids],
            "holdout_Re": split_stats["holdout_Re"],
            "test_Re_label": str(arrays["labels"][test_label_id]),
            "best_epoch": int(best_epoch),
            "best_val_score": float(best_val),
            "scalers": {
                key: {"mean": value.mean, "scale": value.scale}
                for key, value in scalers.items()
            },
        },
        ckpt_path,
    )

    train_metrics = evaluate_model(
        model, arrays, scalers, train_ids, args, device, arrays_t, gal, sur, base_bundle
    )
    val_metrics = evaluate_model(
        model, arrays, scalers, val_ids, args, device, arrays_t, gal, sur, base_bundle
    )
    route_train = routing_analysis(model, arrays, scalers, train_ids, args, device)
    route_re_groups = routing_by_re_group_analysis(model, arrays, scalers, train_ids, device)
    route_regime_train = routing_by_regime_analysis(model, arrays, scalers, train_ids, device)
    group_route_train = group_routing_analysis(model, arrays, scalers, train_ids, device)
    attr_route_train = attractor_router_analysis(model, arrays, scalers, train_ids, device)
    shared_train = shared_operator_analysis(model, arrays, scalers, train_ids, device)

    results: List[Dict[str, object]] = []
    for eval_label_id in holdout_label_ids:
        test_ids = sample_ids[arrays["label_id"][sample_ids] == int(eval_label_id)]
        if len(test_ids) == 0:
            continue
        test_metrics = evaluate_model(
            model, arrays, scalers, test_ids, args, device, arrays_t, gal, sur, base_bundle
        )
        base_rhs = arrays["rhs_g"][test_ids]
        true_rhs = arrays["adot"][test_ids]
        base_euler = arrays["a"][test_ids] + arrays["dt_next"][test_ids][:, None] * base_rhs
        base_metrics = {
            "rhs_relative_l2": relative_l2_np(true_rhs, base_rhs),
            "rhs_rmse": rmse_np(true_rhs, base_rhs),
            "rhs_centered_r2": centered_r2_np(true_rhs, base_rhs),
            "one_step_euler_relative_l2": relative_l2_np(arrays["a_next"][test_ids], base_euler),
            "one_step_euler_rmse": rmse_np(arrays["a_next"][test_ids], base_euler),
            "pressure_surrogate_relative_l2": relative_l2_np(
                arrays["b_next"][test_ids], arrays["pressure_base_next"][test_ids]
            ),
            "pressure_surrogate_rmse": rmse_np(
                arrays["b_next"][test_ids], arrays["pressure_base_next"][test_ids]
            ),
        }
        one_step_integrator = one_step_integrator_eval_np(
            model, arrays, scalers, tensors, test_ids, args, device
        )
        one_step_autonomous = one_step_autonomous_pressure_eval_np(
            model, arrays, scalers, tensors, pressure_tensors, test_ids, args, device
        )
        rollout = rollout_eval_np(model, arrays, scalers, tensors, int(eval_label_id), args, device)
        rollout_autonomous = rollout_autonomous_pressure_eval_np(
            model, arrays, scalers, tensors, pressure_tensors, int(eval_label_id), args, device
        )
        hopf_horizon_diagnostics: Dict[str, object] = {}
        if abs(float(arrays["re"][test_ids[0]]) - float(args.hopf_focus_re)) <= max(
            float(args.hopf_focus_width), 1.0e-6
        ):
            original_rollout_steps = int(args.rollout_steps)
            for horizon in parse_curriculum_steps(args.hopf_diagnostic_horizons):
                args.rollout_steps = int(horizon)
                hopf_horizon_diagnostics[str(int(horizon))] = rollout_autonomous_pressure_eval_np(
                    model,
                    arrays,
                    scalers,
                    tensors,
                    pressure_tensors,
                    int(eval_label_id),
                    args,
                    device,
                )
            args.rollout_steps = original_rollout_steps
        route_test = routing_analysis(model, arrays, scalers, test_ids, args, device)
        route_regime_test = routing_by_regime_analysis(model, arrays, scalers, test_ids, device)
        group_route_test = group_routing_analysis(model, arrays, scalers, test_ids, device)
        attr_route_test = attractor_router_analysis(model, arrays, scalers, test_ids, device)
        shared_test = shared_operator_analysis(model, arrays, scalers, test_ids, device)
        expert_diversity = expert_operator_diversity_analysis(
            model, arrays, scalers, test_ids, device
        )
        if args.skip_expert_error_analysis:
            expert_errors = {"skipped": True}
        else:
            expert_errors = expert_error_analysis_np(
                model,
                arrays,
                scalers,
                tensors,
                pressure_tensors,
                test_ids,
                int(eval_label_id),
                args,
                device,
            )
        result_item = {
            "test_Re": float(arrays["re"][test_ids[0]]),
            "test_Re_label": str(arrays["labels"][int(eval_label_id)]),
            "test_regime": str(arrays["regime"][test_ids[0]]),
            "test_attractor": str(arrays["attractor"][test_ids[0]]),
            "stage_label": "attractor",
            "train_Re_labels": split_stats["train_re_labels"],
            "num_train": int(len(train_ids)),
            "num_val": int(len(val_ids)),
            "num_test": int(len(test_ids)),
            "best_epoch": int(best_epoch),
            "best_val_score": float(best_val),
            "checkpoint_path": str(ckpt_path),
            "data_ablation_split": split_stats,
            "baseline_galerkin": base_metrics,
            "deep_moe": test_metrics,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "one_step_integrator": one_step_integrator,
            "one_step_autonomous_pressure": one_step_autonomous,
            "rollout_teacher_forced_pressure": rollout,
            "rollout_autonomous_pressure": rollout_autonomous,
            "hopf_horizon_diagnostics": hopf_horizon_diagnostics,
            "routing_analysis_test": route_test,
            "routing_analysis_train": route_train,
            "routing_by_re_group_train": route_re_groups,
            "routing_by_regime_train": route_regime_train,
            "routing_by_regime_test": route_regime_test,
            "group_routing_analysis_test": group_route_test,
            "group_routing_analysis_train": group_route_train,
            "attractor_router_analysis_test": attr_route_test,
            "attractor_router_analysis_train": attr_route_train,
            "shared_operator_analysis_test": shared_test,
            "shared_operator_analysis_train": shared_train,
            "expert_operator_diversity": expert_diversity,
            "expert_error_analysis_test": expert_errors,
            "improvement_percent_vs_galerkin_rhs": float(
                100.0
                * (
                    1.0
                    - test_metrics["rhs_relative_l2"]
                    / (base_metrics["rhs_relative_l2"] + EPS)
                )
            ),
            "improvement_percent_vs_galerkin_one_step": float(
                100.0
                * (
                    1.0
                    - test_metrics["one_step_euler_relative_l2"]
                    / (base_metrics["one_step_euler_relative_l2"] + EPS)
                )
            ),
            "history_tail": history[-12:],
        }
        results.append(result_item)
        re_tag = f"Re_{float(result_item['test_Re']):.3f}"
        swanlab_log(
            swan_run,
            {
                f"test/{re_tag}/rhs_relative_l2": test_metrics["rhs_relative_l2"],
                f"test/{re_tag}/one_step_velocity_l2": one_step_autonomous[
                    "a_relative_l2"
                ],
                f"test/{re_tag}/one_step_pressure_l2": one_step_autonomous[
                    "b_relative_l2"
                ],
                f"test/{re_tag}/rollout_velocity_l2": rollout_autonomous[
                    "a_relative_l2_mean"
                ],
                f"test/{re_tag}/rollout_pressure_l2": rollout_autonomous[
                    "b_relative_l2_mean"
                ],
                f"test/{re_tag}/rollout_pressure_energy_error": rollout_autonomous[
                    "b_energy_relative_error"
                ],
                f"test/{re_tag}/alpha_mean": test_metrics.get(
                    "closure_alpha_mean", float("nan")
                ),
                f"test/{re_tag}/beta_mean": test_metrics.get(
                    "closure_beta_mean", float("nan")
                ),
                f"test/{re_tag}/base_contribution_ratio_mean": test_metrics.get(
                    "closure_base_contribution_ratio_mean", float("nan")
                ),
                f"test/{re_tag}/residual_contribution_ratio_mean": test_metrics.get(
                    "closure_residual_contribution_ratio_mean", float("nan")
                ),
                f"test/{re_tag}/attractor_router_accuracy": attr_route_test.get(
                    "accuracy", float("nan")
                ),
                f"test/{re_tag}/attr_pi_steady": (
                    attr_route_test.get("pi_mean", [float("nan")] * 3)[0]
                    if attr_route_test.get("pi_mean")
                    else float("nan")
                ),
                f"test/{re_tag}/attr_pi_hopf": (
                    attr_route_test.get("pi_mean", [float("nan")] * 3)[1]
                    if attr_route_test.get("pi_mean")
                    else float("nan")
                ),
                f"test/{re_tag}/attr_pi_periodic": (
                    attr_route_test.get("pi_mean", [float("nan")] * 3)[2]
                    if attr_route_test.get("pi_mean")
                    else float("nan")
                ),
                f"test/{re_tag}/adapter_delta_to_h_norm": attr_route_test.get(
                    "adapter_delta_to_h_norm_mean", float("nan")
                ),
            },
            step=int(args.epochs + len(results)),
        )
    final_metrics = aggregate_result_metrics(results)
    final_payload: Dict[str, object] = {
        "final/best_epoch": int(best_epoch),
        "final/best_val_score": float(best_val),
    }
    for metric_name, stats in final_metrics.items():
        if isinstance(stats, dict):
            for stat_name, stat_value in stats.items():
                final_payload[f"final/{metric_name}_{stat_name}"] = stat_value
    swanlab_log(swan_run, final_payload, step=int(args.epochs + 100))
    swanlab_finish(swan_run, state="success")
    return results


METRIC_PATHS = {
    "rhs_l2": ("deep_moe", "rhs_relative_l2"),
    "pressure_head_l2": ("deep_moe", "pressure_head_relative_l2"),
    "pressure_base_l2": ("deep_moe", "pressure_surrogate_base_relative_l2"),
    "pressure_residual_only_l2": ("deep_moe", "pressure_residual_only_relative_l2"),
    "pressure_effective_residual_l2": (
        "deep_moe",
        "pressure_effective_residual_relative_l2",
    ),
    "one_step_a_l2": ("one_step_autonomous_pressure", "a_relative_l2"),
    "one_step_b_l2": ("one_step_autonomous_pressure", "b_relative_l2"),
    "rollout_a_l2": ("rollout_autonomous_pressure", "a_relative_l2_mean"),
    "rollout_b_l2": ("rollout_autonomous_pressure", "b_relative_l2_mean"),
    "one_step_pressure_energy_error": (
        "one_step_autonomous_pressure",
        "b_energy_relative_error",
    ),
    "rollout_pressure_energy_error": (
        "rollout_autonomous_pressure",
        "b_energy_relative_error",
    ),
    "alpha_mean": ("deep_moe", "closure_alpha_mean"),
    "alpha_std": ("deep_moe", "closure_alpha_std"),
    "beta_mean": ("deep_moe", "closure_beta_mean"),
    "beta_std": ("deep_moe", "closure_beta_std"),
    "base_scale_mean": ("deep_moe", "closure_base_scale_mean"),
    "residual_scale_mean": ("deep_moe", "closure_residual_scale_mean"),
    "base_error_mean": ("deep_moe", "closure_base_error_mean"),
    "residual_magnitude_mean": ("deep_moe", "closure_residual_magnitude_mean"),
    "effective_residual_magnitude_mean": (
        "deep_moe",
        "closure_effective_residual_magnitude_mean",
    ),
    "base_contribution_ratio_mean": (
        "deep_moe",
        "closure_base_contribution_ratio_mean",
    ),
    "residual_contribution_ratio_mean": (
        "deep_moe",
        "closure_residual_contribution_ratio_mean",
    ),
    "alpha_base_error_corr": ("deep_moe", "closure_alpha_base_error_corr"),
    "alpha_residual_magnitude_corr": (
        "deep_moe",
        "closure_alpha_residual_magnitude_corr",
    ),
    "beta_base_error_corr": ("deep_moe", "closure_beta_base_error_corr"),
    "beta_residual_magnitude_corr": (
        "deep_moe",
        "closure_beta_residual_magnitude_corr",
    ),
}


def nested_metric(item: Dict[str, object], path: Tuple[str, str]) -> float:
    group = item.get(path[0], {})
    if not isinstance(group, dict):
        return float("nan")
    return float(group.get(path[1], float("nan")))


def aggregate_result_metrics(results: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name, path in METRIC_PATHS.items():
        vals = np.asarray([nested_metric(item, path) for item in results], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[name] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        else:
            out[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    out["num_test_re"] = int(len(results))
    return out


def write_error_curve_csv(path: Path, results: List[Dict[str, object]]) -> None:
    fields = ["Re", "label", "regime", "attractor", *METRIC_PATHS.keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in sorted(results, key=lambda row: float(row["test_Re"])):
            row = {
                "Re": float(item["test_Re"]),
                "label": str(item["test_Re_label"]),
                "regime": str(item.get("test_regime", "unknown")),
                "attractor": str(item.get("test_attractor", "unknown")),
            }
            for name, metric_path in METRIC_PATHS.items():
                row[name] = nested_metric(item, metric_path)
            writer.writerow(row)


def write_error_curve_svg(path: Path, results: List[Dict[str, object]]) -> None:
    ordered = sorted(results, key=lambda row: float(row["test_Re"]))
    if not ordered:
        path.write_text("", encoding="utf-8")
        return
    width, height = 1200, 720
    left, right, top, bottom = 85, 35, 55, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    re_vals = np.asarray([float(item["test_Re"]) for item in ordered], dtype=np.float64)
    series = {
        "one-step a": [nested_metric(item, METRIC_PATHS["one_step_a_l2"]) for item in ordered],
        "one-step b": [nested_metric(item, METRIC_PATHS["one_step_b_l2"]) for item in ordered],
        "rollout a": [nested_metric(item, METRIC_PATHS["rollout_a_l2"]) for item in ordered],
        "rollout b": [nested_metric(item, METRIC_PATHS["rollout_b_l2"]) for item in ordered],
        "RHS": [nested_metric(item, METRIC_PATHS["rhs_l2"]) for item in ordered],
        "pressure head": [nested_metric(item, METRIC_PATHS["pressure_head_l2"]) for item in ordered],
    }
    all_vals = np.asarray([v for vals in series.values() for v in vals if np.isfinite(v)], dtype=np.float64)
    y_max = float(max(0.1, np.nanmax(all_vals) * 1.08)) if len(all_vals) else 1.0
    x_min, x_max = float(np.min(re_vals)), float(np.max(re_vals))
    if abs(x_max - x_min) < EPS:
        x_max = x_min + 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - min(max(y, 0.0), y_max) / y_max * plot_h

    colors = {
        "one-step a": "#1f77b4",
        "one-step b": "#ff7f0e",
        "rollout a": "#2ca02c",
        "rollout b": "#d62728",
        "RHS": "#9467bd",
        "pressure head": "#8c564b",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2d3a}.axis{stroke:#536575;stroke-width:1.4}.grid{stroke:#d8e0e8;stroke-width:1}.line{fill:none;stroke-width:2.4}.dot{stroke:#fff;stroke-width:1.2}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f7f9fb"/>',
        f'<text x="{left}" y="32" font-size="24" font-weight="700">V16 attractor error vs Reynolds number</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
    ]
    for frac in np.linspace(0, 1, 6):
        y = top + plot_h - frac * plot_h
        val = frac * y_max
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text x="18" y="{y + 4:.1f}" font-size="12">{val:.2f}</text>')
    for x in np.linspace(x_min, x_max, 6):
        px = sx(float(x))
        parts.append(f'<text x="{px - 20:.1f}" y="{top + plot_h + 28}" font-size="12">{x:.0f}</text>')
    parts.append(f'<text x="{left + plot_w / 2 - 35:.1f}" y="{height - 28}" font-size="14">Reynolds number</text>')
    parts.append(f'<text x="18" y="{top - 16}" font-size="14">relative L2</text>')
    legend_x = left + 15
    legend_y = top + 18
    for i, (name, vals) in enumerate(series.items()):
        points = []
        for re_value, value in zip(re_vals, vals):
            if np.isfinite(value):
                points.append(f"{sx(float(re_value)):.1f},{sy(float(value)):.1f}")
        if len(points) >= 2:
            parts.append(f'<polyline class="line" points="{" ".join(points)}" stroke="{colors[name]}"/>')
        for re_value, value in zip(re_vals, vals):
            if np.isfinite(value):
                parts.append(f'<circle class="dot" cx="{sx(float(re_value)):.1f}" cy="{sy(float(value)):.1f}" r="4" fill="{colors[name]}"/>')
        lx = legend_x + (i % 3) * 220
        ly = legend_y + (i // 3) * 24
        parts.append(f'<rect x="{lx}" y="{ly - 11}" width="14" height="4" fill="{colors[name]}"/>')
        parts.append(f'<text x="{lx + 22}" y="{ly - 5}" font-size="13">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary_md(path: Path, result: Dict[str, object]) -> None:
    settings = result["settings"]
    lines = [
        "# AttractorMoE-ROM V16 Physics-Generalizable Attractor Summary",
        "",
        "## Architecture",
        "",
        (
            f"Shared encoder + {settings['num_blocks']} latent refinement blocks, "
            f"hidden_dim={settings['hidden_dim']}, regime_groups={settings['num_regime_groups']}, "
            f"experts_per_group={settings['experts_per_group']}, "
            f"shared_experts_per_group={settings['shared_experts_per_group']}, "
            f"group_top_k={settings['group_top_k']}, in_group_top_k={settings['top_k']}, "
            f"expert_hidden={settings['expert_hidden']}, expert_blocks={settings['expert_blocks']}, "
            f"quadratic_rank={settings['quadratic_rank']}."
        ),
        "",
        (
            f"Shared/routed scales: {settings['shared_scale']:.3g} / "
            f"{settings['routed_scale']:.3g}; routed gate floor: "
            f"{settings['gate_floor']:.3g}."
        ),
        "",
        (
            f"A shared group router selects a physics regime, then group-local velocity/pressure "
            f"Top-2 routers mix a group-shared expert with routed physics-aware operator experts. "
            f"Experts output `{settings['rhs_target']}` velocity operator targets plus a pressure "
            f"`{settings['pressure_target']}` branch. For `residual`, the learned closure is added "
            "to the Galerkin RHS before RK4."
        ),
        "",
        (
            f"V16 pressure input mode: `{settings.get('pressure_input_mode', 'pressure_only')}`. "
            "V16 keeps the V14/V15 best `[a_t,b_t]` pressure state while changing "
            "only attractor losses or the lightweight attractor conditioning layer."
        ),
        "",
        "Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop "
        "multi-step rollout, energy consistency, trajectory consistency, pressure closure, "
        "relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, "
        "and weak Re-regime supervision.",
        "",
        "## Dense Training Split",
        "",
        (
            f"Test selection: `{settings['test_re_selection']}`, "
            f"time stride={settings['train_time_stride']}, Re stride={settings['train_re_stride']}."
        ),
        "",
    ]
    split = result["results"][0].get("data_ablation_split", {}) if result["results"] else {}
    if split:
        lines.extend(
            [
                f"- Train Re count: {split.get('train_re_count')}",
                f"- Test Re count: {split.get('test_re_count')}",
                f"- Excluded Re count from Re sparsity: {split.get('excluded_re_count')}",
                f"- Dense train samples before time sparsity: {split.get('dense_train_samples_before_time_sparsity')}",
                f"- Kept train samples: {split.get('kept_train_samples')}",
                f"- Validation samples: {split.get('val_samples')}",
                f"- Test samples: {split.get('test_samples')}",
                f"- Compression vs dense train: {split.get('compression_ratio_vs_dense_train'):.6g}",
                f"- Compression vs all non-test candidates: {split.get('compression_ratio_vs_all_non_test_candidates'):.6g}",
                "",
                "| Re | role | total | dense train | kept train | val | test |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in split.get("per_re", []):
            lines.append(
                f"| {row['Re']:.6g} | {row['role']} | {row['total_valid_samples']} | "
                f"{row['dense_train_samples']} | {row['kept_train_samples']} | "
                f"{row['val_samples']} | {row['test_samples']} |"
            )
        lines.append("")
    aggregate = result.get("aggregate_metrics", {})
    if aggregate:
        lines.extend(
            [
                "## Aggregate Held-out Metrics",
                "",
                "| Metric | mean | std | min | max |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for key in [
            "rhs_l2",
            "pressure_head_l2",
            "one_step_a_l2",
            "one_step_b_l2",
            "rollout_a_l2",
            "rollout_b_l2",
            "one_step_pressure_energy_error",
            "rollout_pressure_energy_error",
        ]:
            stats = aggregate.get(key, {})
            lines.append(
                f"| {key} | {stats.get('mean', float('nan')):.6g} | "
                f"{stats.get('std', float('nan')):.6g} | "
                f"{stats.get('min', float('nan')):.6g} | "
                f"{stats.get('max', float('nan')):.6g} |"
            )
        lines.extend(
            [
                "",
                f"Error curve CSV: `{result.get('error_curve_csv', '-')}`",
                "",
                f"Error curve SVG: `{result.get('error_curve_svg', '-')}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Metrics",
        "",
        f"Integrator: `{settings['integrator']}`.",
        "",
        "| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in result["results"]:
        base = item["baseline_galerkin"]
        deep = item["deep_moe"]
        roll = item["rollout_teacher_forced_pressure"]
        roll_auto = item["rollout_autonomous_pressure"]
        lines.append(
            f"| {item['test_Re']} | Galerkin only | {base['rhs_relative_l2']:.6g} | "
            f"{base['pressure_surrogate_relative_l2']:.6g} | - | "
            f"{base['one_step_euler_relative_l2']:.6g} | - | - | - | - | - | - | - | - | - |"
        )
        routing = item["routing_analysis_test"]
        one = item["one_step_integrator"]
        one_auto = item["one_step_autonomous_pressure"]
        lines.append(
            f"| {item['test_Re']} | HPRS-MoE | {deep['rhs_relative_l2']:.6g} | "
            f"{base['pressure_surrogate_relative_l2']:.6g} | "
            f"{deep['pressure_head_relative_l2']:.6g} | {one['relative_l2']:.6g} | "
            f"{one_auto['a_relative_l2']:.6g} | {one_auto['b_relative_l2']:.6g} | "
            f"{roll['relative_l2_mean']:.6g} | {roll_auto['a_relative_l2_mean']:.6g} | "
            f"{roll_auto['b_relative_l2_mean']:.6g} | {routing['active_experts_mean']:.3g} | "
            f"{routing['load_cv']:.6g} | "
            f"{routing['entropy_mean']:.6g} | {routing['dead_experts_threshold_1pct']} |"
        )
    lines.append("")
    lines.append("## Expert Diagnostics")
    lines.append("")
    lines.append("| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |")
    lines.append("|---:|---|---|---|---:|")
    for item in result["results"]:
        shared = item.get("shared_operator_analysis_test", {})
        weights = shared.get("group_mean_load", shared.get("mean_mixer_weight", []))
        weight_text = "[" + ", ".join(f"{float(v):.3f}" for v in weights) + "]" if weights else "-"
        top = shared.get("group_top1_fraction", [])
        top_text = "[" + ", ".join(f"{float(v):.3f}" for v in top) + "]" if top else "-"
        entropy = shared.get("mixer_entropy_mean", float("nan"))
        lines.append(
            f"| {item['test_Re']} | {shared.get('always_active_in_selected_group', False)} | "
            f"{weight_text} | {top_text} | {entropy:.6g} |"
        )
    lines.append("")
    lines.append(
        "| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |"
    )
    lines.append("|---:|---:|---|---|")
    for item in result["results"]:
        diversity = item.get("expert_operator_diversity", {})
        groups = item.get("routing_by_re_group_train", {})
        group_bits = []
        for key in ["low_Re_lt_80", "mid_Re_80_160", "high_Re_ge_160"]:
            info = groups.get(key, {})
            top = info.get("top1_fraction", [])
            if top:
                group_bits.append(f"{key}: e{int(np.argmax(top))}")
        lines.append(
            f"| {item['test_Re']} | {diversity.get('pairwise_cosine_max_abs', float('nan')):.6g} | "
            f"{diversity.get('collapse_flag_abs_cos_gt_0p95', False)} | "
            f"{'; '.join(group_bits)} |"
        )
    lines.append("")
    lines.append(f"Runtime: {result['runtime_seconds']:.2f} s.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.pressure_target != "closure":
        raise ValueError("V16_AttractorMoEROM requires --pressure-target=closure.")
    if args.pressure_input_mode != "pressure_only":
        raise ValueError(
            "V16 keeps the V14/V15 best pressure input [a_t,b_t]; "
            "use --pressure-input-mode=pressure_only."
        )
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if args.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    started = time.time()
    arrays, meta = build_arrays(args)
    args.test_re_indices = resolve_test_re_indices(args, arrays)
    tensors = np.load(args.tensor_path)
    pressure_tensors = np.load(args.pressure_surrogate_path)

    results = []
    if not args.test_re_indices:
        raise ValueError("No test Reynolds numbers selected.")
    results.extend(
        train_one_split(
            args,
            arrays,
            meta,
            tensors,
            pressure_tensors,
            int(args.test_re_indices[0]),
            device,
        )
    )
    aggregate_metrics = aggregate_result_metrics(results)
    curve_csv = args.output_dir / f"{args.experiment_name}_error_vs_re.csv"
    curve_svg = args.output_dir / f"{args.experiment_name}_error_vs_re.svg"
    write_error_curve_csv(curve_csv, results)
    write_error_curve_svg(curve_svg, results)

    settings = {
        "experiment_name": args.experiment_name,
        "experiment_tag": args.experiment_tag,
        "r_u": args.r_u,
        "r_p": args.r_p,
        "phase_harmonics": args.phase_harmonics,
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "num_experts": int(args.num_regime_groups * (args.experts_per_group + 1)),
        "legacy_num_experts_arg": args.num_experts,
        "num_operator_spaces": args.num_regime_groups,
        "num_regime_groups": args.num_regime_groups,
        "experts_per_group": args.experts_per_group,
        "shared_experts_per_group": 1,
        "top_k": args.top_k,
        "group_top_k": args.group_top_k,
        "expert_hidden": args.expert_hidden,
        "expert_blocks": args.expert_blocks,
        "quadratic_rank": args.quadratic_rank,
        "quadratic_scale": args.quadratic_scale,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "gate_floor": args.gate_floor,
        "group_temperature": args.group_temperature,
        "group_gate_floor": args.group_gate_floor,
        "shared_scale": args.shared_scale,
        "routed_scale": args.routed_scale,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "eval_every": args.eval_every,
        "early_stop_min_delta": args.early_stop_min_delta,
        "curriculum_steps": parse_curriculum_steps(args.curriculum_steps),
        "train_rollout_steps": args.train_rollout_steps,
        "rollout_steps": args.rollout_steps,
        "rollout_batch": args.rollout_batch,
        "rollout_every_batches": args.rollout_every_batches,
        "recon_dim": args.recon_dim,
        "loss_weights": {
            "coeff": args.lambda_coeff,
            "dyn": args.lambda_dyn,
            "pressure": args.lambda_pressure,
            "recon": args.lambda_recon,
            "rollout": args.lambda_rollout,
            "pressure_rollout": args.lambda_pressure_rollout,
            "consistency": args.lambda_consistency,
            "router_balance": args.lambda_router_balance,
            "router_entropy": args.lambda_router_entropy,
            "group_balance": args.lambda_group_balance,
            "group_entropy": args.lambda_group_entropy,
            "group_supervision": args.lambda_group_supervision,
            "router_smooth": args.lambda_router_smooth,
            "expert_diversity": args.lambda_expert_diversity,
            "regime_router": args.lambda_regime_router,
            "energy": args.lambda_energy,
            "trajectory_consistency": args.lambda_trajectory_consistency,
            "alpha_rel": args.lambda_alpha_rel,
            "rhs_rel": args.lambda_rhs_rel,
            "pressure_rel": args.lambda_pressure_rel,
            "hopf_log_amp": args.lambda_hopf_log_amp,
            "hopf_energy": args.lambda_hopf_energy,
            "hopf_overshoot": args.lambda_hopf_overshoot,
            "attractor_ce": args.lambda_attractor_ce,
            "steady_rhs": args.lambda_steady_rhs,
            "steady_state": args.lambda_steady_state,
            "attractor_hopf_radius": args.lambda_attractor_hopf_radius,
            "attractor_hopf_overshoot": args.lambda_attractor_hopf_overshoot,
            "attractor_hopf_onset": args.lambda_attractor_hopf_onset,
            "v16_1_hopf_growth": args.lambda_v16_1_hopf_growth,
            "v16_1_hopf_false_growth": args.lambda_v16_1_hopf_false_growth,
            "v16_1_hopf_floor_rel": args.lambda_v16_1_hopf_floor_rel,
            "v16_1_steady_p_state": args.lambda_v16_1_steady_p_state,
            "v16_1_steady_p_mean": args.lambda_v16_1_steady_p_mean,
            "v16_1_steady_p_delta": args.lambda_v16_1_steady_p_delta,
            "v16_1_steady_residual_damp": args.lambda_v16_1_steady_residual_damp,
            "v16_1_steady_p_energy": args.lambda_v16_1_steady_p_energy,
            "v16_1_loss_warmup_epochs": args.v16_1_loss_warmup_epochs,
            "periodic_energy": args.lambda_periodic_energy,
            "periodic_radius": args.lambda_periodic_radius,
            "prototype_energy": args.lambda_proto_energy,
            "prototype_radius": args.lambda_proto_radius,
            "rollout_relative_mix": args.rollout_relative_mix,
            "relative_floor_frac": args.relative_floor_frac,
        },
        "hopf_focus": {
            "pair": [int(v) for v in parse_hopf_pair(args)],
            "re_min": args.hopf_re_min,
            "re_max": args.hopf_re_max,
            "focus_re": args.hopf_focus_re,
            "focus_width": args.hopf_focus_width,
            "loss_warmup_epochs": args.hopf_loss_warmup_epochs,
            "overshoot_ratio": args.hopf_overshoot_ratio,
            "sample_weight": args.hopf_sample_weight,
            "focus_sample_weight": args.hopf_focus_sample_weight,
            "rollout_weight": args.hopf_rollout_weight,
            "diagnostic_horizons": parse_curriculum_steps(args.hopf_diagnostic_horizons),
        },
        "scheduled_sampling": {
            "start": args.scheduled_sampling_start,
            "end": args.scheduled_sampling_end,
            "warmup_frac": args.scheduled_sampling_warmup_frac,
        },
        "history_len": args.history_len,
        "database_interpretation": "Physics-Generalizable Attractor Database",
        "stage_label": "attractor",
        "attractor_balanced_sampling": bool(args.attractor_balanced_sampling),
        "attractor_conditioned": bool(args.attractor_conditioned),
        "attractor_adapter_dim": args.attractor_adapter_dim,
        "attractor_r_floor": args.attractor_r_floor,
        "attractor_r_floor_effective": float(
            getattr(args, "attractor_r_floor_effective", args.attractor_r_floor)
        ),
        "v16_1_hopf_floor_quantile": args.v16_1_hopf_floor_quantile,
        "v16_1_a_floor_effective": float(getattr(args, "v16_1_a_floor_effective", 0.0)),
        "v16_1_diagnostic_basis": {
            "baseline": "V16_AttractorMoEROM_FullRegimeLoss32",
            "hopf_near_onset": (
                "Train Re near 47.7229 shows false oscillation/growth: "
                "r_true about 1.55e-6, r_pred about 2.50e-5, standard relative "
                "velocity error about 7.21, floor-normalized error about 1.96, "
                "frequency error about 2.78e-2, predicted growth positive while true growth is negative."
            ),
            "steady_pressure": (
                "Low-Re steady pressure drift has alpha about 0.063, raw Poisson base "
                "relative error about 80-107, base contribution about 0.18-0.32, "
                "and residual contribution about 1.0; closed-loop residual drift is the target."
            ),
        },
        "attractor_onset_threshold": args.attractor_onset_threshold,
        "rhs_target": args.rhs_target,
        "pressure_target": args.pressure_target,
        "pressure_input_mode": args.pressure_input_mode,
        "closure_mode": args.closure_mode,
        "pressure_base_mode": args.pressure_base_mode,
        "regime_rom_root": str(args.regime_rom_root),
        "film_base_hidden": args.film_base_hidden,
        "film_base_scale": args.film_base_scale,
        "pressure_amplitude_weight_power": args.pressure_amplitude_weight_power,
        "pressure_amplitude_weight_max": args.pressure_amplitude_weight_max,
        "test_re_indices": args.test_re_indices,
        "test_re_values": args.test_re_values,
        "test_re_selection": args.test_re_selection,
        "num_uniform_test_re": args.num_uniform_test_re,
        "train_time_stride": args.train_time_stride,
        "train_time_offset": args.train_time_offset,
        "train_re_stride": args.train_re_stride,
        "train_re_offset": args.train_re_offset,
        "regime_balanced_sampling": bool(args.regime_balanced_sampling),
        "allow_tf32": bool(args.allow_tf32),
        "integrator": args.integrator,
        "analysis_bins": args.analysis_bins,
        "eval_routing_every": args.eval_routing_every,
        "eval_routing_max_samples": args.eval_routing_max_samples,
        "skip_expert_error_analysis": bool(args.skip_expert_error_analysis),
        "pressure_surrogate_path": str(args.pressure_surrogate_path),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    out = {
        "scheme": "v16_attractor_moe_rom_re20_200_closed_loop_rk4",
        "data_root": str(args.data_root),
        "tensor_path": str(args.tensor_path),
        "pressure_surrogate_path": str(args.pressure_surrogate_path),
        "settings": settings,
        "data_meta": meta,
        "results": results,
        "aggregate_metrics": aggregate_metrics,
        "error_curve_csv": str(curve_csv),
        "error_curve_svg": str(curve_svg),
        "runtime_seconds": float(time.time() - started),
    }
    json_path = args.output_dir / f"{args.experiment_name}_metrics.json"
    md_path = args.output_dir / f"{args.experiment_name}_summary.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_summary_md(md_path, out)
    if args.print_full_json:
        print(json.dumps(out, indent=2))
    else:
        print(
            json.dumps(
                {
                    "scheme": out["scheme"],
                    "experiment_name": args.experiment_name,
                    "metrics_json": str(json_path),
                    "summary_md": str(md_path),
                    "runtime_seconds": out["runtime_seconds"],
                    "test_Re": [item["test_Re"] for item in results],
                }
            )
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
