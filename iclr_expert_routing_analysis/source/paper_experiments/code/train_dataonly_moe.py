#!/usr/bin/env python3
"""Data-only sparse MoE ablation on frozen specialist-local POD coordinates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1.0e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_links(re_values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((times, re_values))
    if not np.array_equal(order, np.arange(len(order))):
        raise RuntimeError("coefficient asset is not ordered by (Re,time)")
    previous = np.full(len(re_values), -1, dtype=np.int64)
    following = np.full(len(re_values), -1, dtype=np.int64)
    for value in np.unique(re_values):
        ids = np.flatnonzero(np.abs(re_values - value) <= 5.0e-7)
        previous[ids[1:]] = ids[:-1]
        following[ids[:-1]] = ids[1:]
    return previous, following


def build_history(previous: np.ndarray, length: int = 3) -> np.ndarray:
    history = np.full((len(previous), length), -1, dtype=np.int64)
    history[:, 0] = np.arange(len(previous))
    for column in range(1, length):
        valid = history[:, column - 1] >= 0
        history[valid, column] = previous[history[valid, column - 1]]
    return history


def load_data(root: Path, regime: str) -> dict[str, Any]:
    if regime == "steady":
        path = root / "steady_specialist_v1/data/steady_trainval_modal_r32.npz"
        z = np.load(path, allow_pickle=False)
        a = z["a_raw"].astype(np.float32)
        b = z["b_raw"].astype(np.float32)
        re_values = z["Re"].astype(np.float64)
        times = z["time"].astype(np.float64)
        split = z["split"].astype(str)
        previous = z["prev_index"].astype(np.int64)
        following = z["next_index"].astype(np.int64)
        history = z["history_index"].astype(np.int64)
        phase = np.zeros(len(a), dtype=np.float32)
        phase_harmonics = 0
        assets = [path]
    elif regime == "hopf":
        path = (
            root
            / "Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainval_r32.npz"
        )
        z = np.load(path, allow_pickle=False)
        a = z["coeff_uv"].astype(np.float32)[:, :32]
        b = z["coeff_p"].astype(np.float32)[:, :32]
        re_values = z["Re"].astype(np.float64)
        times = z["time"].astype(np.float64)
        split = z["split"].astype(str)
        previous, following = make_links(re_values, times)
        history = build_history(previous)
        phase = np.zeros(len(a), dtype=np.float32)
        phase_harmonics = 0
        assets = [path]
    elif regime == "periodic":
        specialist_root = root / "periodic_specialist_r32"
        data_root = specialist_root / "assets/Global_POD_AreaWeighted_L2"
        velocity_path = data_root / "global_velocity_pod_area_weighted_l2.npz"
        pressure_path = data_root / "global_pressure_pod_area_weighted_l2.npz"
        index_path = data_root / "pod_snapshot_index.csv"
        trainer_path = specialist_root / "code/train_periodic_moe.py"
        spec = importlib.util.spec_from_file_location(
            "dataonly_periodic_native_builder", trainer_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(trainer_path)
        native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = native
        spec.loader.exec_module(native)
        native_arrays, _ = native.build_arrays(
            SimpleNamespace(
                data_root=data_root,
                tensor_path=specialist_root / "assets/velocity_rom_periodic.npz",
                pressure_surrogate_path=specialist_root
                / "assets/pressure_poisson_surrogate_periodic.npz",
                r_u=32,
                r_p=32,
                history_len=3,
                phase_harmonics=4,
                recon_dim=0,
                seed=1600,
            )
        )
        velocity = np.load(velocity_path, allow_pickle=False)
        pressure = np.load(pressure_path, allow_pickle=False)
        a = np.asarray(native_arrays["a"], dtype=np.float32)[:, :32]
        b = np.asarray(native_arrays["b"], dtype=np.float32)[:, :32]
        split = velocity["snapshot_splits"].astype(str)
        with index_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        re_values = np.asarray([float(row["Re"]) for row in rows], dtype=np.float64)
        times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
        period = np.asarray(
            [
                float(row["estimated_period"])
                if row.get("estimated_period", "").strip()
                else np.nan
                for row in rows
            ],
            dtype=np.float64,
        )
        if len(rows) != len(a):
            raise RuntimeError("Periodic coefficient/index length mismatch")
        previous = np.asarray(native_arrays["prev_idx"], dtype=np.int64)
        following = np.asarray(native_arrays["next_idx"], dtype=np.int64)
        history = np.asarray(native_arrays["hist_idx"], dtype=np.int64)
        phase = np.asarray(native_arrays["phase"], dtype=np.float32)
        if not np.all(np.isfinite(phase)):
            raise RuntimeError("Periodic native phase is missing")
        phase_harmonics = 4
        assets = [
            velocity_path,
            pressure_path,
            index_path,
            trainer_path,
            specialist_root / "assets/velocity_rom_periodic.npz",
            specialist_root / "assets/pressure_poisson_surrogate_periodic.npz",
        ]
    else:
        raise ValueError(regime)
    if a.shape[1:] != (32,) or b.shape[1:] != (32,):
        raise RuntimeError("DataOnly requires ru=rp=32")
    if len({len(a), len(b), len(re_values), len(times), len(split)}) != 1:
        raise RuntimeError("array length mismatch")
    split = np.char.lower(split.astype(str))
    split[split == "test"] = "heldout"
    allowed = {"train", "validation"}
    if not allowed.issubset(set(split.tolist())):
        raise RuntimeError(f"missing train/validation split: {set(split.tolist())}")
    train_re = set(np.round(re_values[split == "train"], 6).tolist())
    val_re = set(np.round(re_values[split == "validation"], 6).tolist())
    heldout_re = set(np.round(re_values[split == "heldout"], 6).tolist())
    if train_re & val_re or train_re & heldout_re or val_re & heldout_re:
        raise RuntimeError("complete-Re split leakage")
    return {
        "a": a,
        "b": b,
        "re": re_values.astype(np.float32),
        "time": times.astype(np.float32),
        "split": split,
        "prev": previous,
        "next": following,
        "history": history,
        "phase": phase,
        "phase_harmonics": phase_harmonics,
        "assets": assets,
        "split_re": {
            "train": sorted(train_re),
            "validation": sorted(val_re),
            "heldout": sorted(heldout_re),
        },
    }


class PlainExpert(nn.Module):
    def __init__(self, dim: int, hidden: int, output: int, blocks: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.GELU()]
        for _ in range(max(0, blocks - 1)):
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU()])
        final = nn.Linear(hidden, output)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalSparseMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: int,
        expert_hidden: int,
        groups: int,
        routed_per_group: int,
        blocks: int,
        dropout: float,
    ):
        super().__init__()
        self.groups = groups
        self.routed_per_group = routed_per_group
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.group_router = nn.Linear(hidden, groups)
        self.local_routers = nn.ModuleList(
            [nn.Linear(hidden, routed_per_group) for _ in range(groups)]
        )
        self.routed = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PlainExpert(hidden, expert_hidden, output_dim, blocks, dropout)
                        for _ in range(routed_per_group)
                    ]
                )
                for _ in range(groups)
            ]
        )
        self.shared = nn.ModuleList(
            [
                PlainExpert(hidden, expert_hidden, output_dim, blocks, dropout)
                for _ in range(groups)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.encoder(x)
        group = torch.softmax(self.group_router(h), dim=-1)
        group_value, group_index = torch.topk(group, 1, dim=-1)
        hard_group = torch.zeros_like(group).scatter(1, group_index, group_value)
        hard_group = hard_group / (hard_group.sum(1, keepdim=True) + EPS)
        hard_group = hard_group - group.detach() + group
        outputs = []
        local_probabilities = []
        for group_id in range(self.groups):
            local = torch.softmax(self.local_routers[group_id](h), dim=-1)
            value, index = torch.topk(local, min(2, self.routed_per_group), dim=-1)
            sparse = torch.zeros_like(local).scatter(1, index, value)
            sparse = sparse / (sparse.sum(1, keepdim=True) + EPS)
            sparse = sparse - local.detach() + local
            routed_stack = torch.stack(
                [expert(h) for expert in self.routed[group_id]], dim=1
            )
            routed_output = torch.sum(routed_stack * sparse.unsqueeze(-1), dim=1)
            outputs.append(0.75 * routed_output + self.shared[group_id](h))
            local_probabilities.append(local)
        stack = torch.stack(outputs, dim=1)
        output = torch.sum(stack * hard_group.unsqueeze(-1), dim=1)
        diagnostics = {
            "group_probs": group,
            "local_probs": torch.stack(local_probabilities, dim=1),
        }
        return output, diagnostics


def advance(data: dict[str, Any], starts: np.ndarray, horizon: int) -> np.ndarray:
    indices = np.empty((len(starts), horizon + 1), dtype=np.int64)
    indices[:, 0] = starts
    for step in range(horizon):
        indices[:, step + 1] = data["next"][indices[:, step]]
    return indices


def legal_starts(data: dict[str, Any], split: str, horizon: int) -> np.ndarray:
    candidates = np.flatnonzero(
        (data["split"] == split)
        & (data["next"] >= 0)
        & np.all(data["history"] >= 0, axis=1)
    )
    legal = []
    for start in candidates.tolist():
        current = start
        ok = True
        for _ in range(horizon):
            current = int(data["next"][current])
            if current < 0 or data["split"][current] != split:
                ok = False
                break
        if ok:
            legal.append(start)
    return np.asarray(legal, dtype=np.int64)


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.root.resolve()
        self.run = args.output_dir.resolve()
        if self.run.exists():
            raise FileExistsError(self.run)
        self.run.mkdir(parents=True)
        seed_all(args.seed)
        self.data = load_data(self.root, args.regime)
        train = self.data["split"] == "train"
        states = np.concatenate([self.data["a"], self.data["b"]], axis=1)
        self.mean = states[train].mean(0).astype(np.float32)
        self.scale = states[train].std(0).astype(np.float32)
        self.scale[self.scale < 1.0e-7] = 1.0
        re_train = self.data["re"][train]
        self.re_center = float(np.mean(re_train))
        self.re_scale = float(max(np.std(re_train), 1.0))
        self.states = torch.as_tensor(states, device="cuda")
        self.device = torch.device("cuda")
        self.mean_t = torch.as_tensor(self.mean, device=self.device)
        self.scale_t = torch.as_tensor(self.scale, device=self.device)
        self.re_t = torch.as_tensor(self.data["re"], device=self.device)
        self.phase_t = torch.as_tensor(self.data["phase"], device=self.device)
        self.history_t = torch.as_tensor(self.data["history"], device=self.device)
        phase_dim = 2 * self.data["phase_harmonics"]
        input_dim = 3 * 64 + 1 + phase_dim
        groups = 1 if args.regime == "hopf" else 3
        hidden = 256 if args.regime == "hopf" else 224
        expert_hidden = 1024 if args.regime == "hopf" else 768
        blocks = 4 if args.regime == "hopf" else 3
        self.model = HierarchicalSparseMoE(
            input_dim, 64, hidden, expert_hidden, groups, 6, blocks, 0.04
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, args.max_steps, eta_min=args.lr * 0.05
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=not torch.cuda.is_bf16_supported()
        )
        self.amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        self.train_starts = {
            horizon: legal_starts(self.data, "train", horizon)
            for horizon in sorted(set(args.curriculum_horizons))
        }
        self.val_starts = legal_starts(self.data, "validation", args.validation_horizon)
        if any(not len(value) for value in self.train_starts.values()) or not len(
            self.val_starts
        ):
            raise RuntimeError("empty legal training/validation starts")
        self.rng = np.random.default_rng(args.seed + 71)
        self.best_score = float("inf")
        self.best_step = -1
        config_path = self.run / "config.json"
        atomic_json(
            config_path,
            {
                "schema": "dataonly_specialist_ablation/v2",
                "ablation_mode": args.ablation_mode,
                "args": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
                "split_re": self.data["split_re"],
                "model_parameter_count": int(
                    sum(parameter.numel() for parameter in self.model.parameters())
                ),
                "router_contract": {
                    "groups": groups,
                    "routed_experts_per_group": 6,
                    "shared_experts_per_group": 1,
                    "group_top_k": 1,
                    "local_top_k": 2,
                },
                "allowed_changes": [
                    "direct discrete POD-state mapping",
                    "no continuous RHS/RK4/Galerkin/pressure fixed operator",
                    "data-supervised one-step and autonomous rollout losses only",
                ],
                "preserved_contract": [
                    "specialist-local POD state and ru=rp=32",
                    "complete-Re split and history length 3",
                    "hierarchical sparse MoE router and native expert counts",
                    "validation-only checkpoint selection",
                ],
                "test_truth_read": False,
            },
        )
        (self.run / "config.sha256").write_text(
            f"{sha256(config_path)}  config.json\n", encoding="utf-8"
        )

    def features(
        self, current: torch.Tensor, history_state: torch.Tensor
    ) -> torch.Tensor:
        normalized = (history_state - self.mean_t) / self.scale_t
        re_feature = (
            (self.re_t[current] - self.re_center) / self.re_scale
        ).unsqueeze(1)
        columns = [normalized.flatten(1), re_feature]
        for harmonic in range(1, self.data["phase_harmonics"] + 1):
            angle = 2.0 * math.pi * harmonic * self.phase_t[current]
            columns.extend([torch.sin(angle).unsqueeze(1), torch.cos(angle).unsqueeze(1)])
        return torch.cat(columns, dim=1)

    def rollout(
        self, starts: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        history_indices = self.history_t[starts]
        history_state = self.states[history_indices].clone()
        current = starts
        predicted = []
        target = []
        diagnostics: dict[str, torch.Tensor] = {}
        for _ in range(horizon):
            x = self.features(current, history_state)
            delta_std, diagnostics = self.model(x)
            next_state = history_state[:, 0] + delta_std * self.scale_t
            current = torch.as_tensor(
                self.data["next"][current.detach().cpu().numpy()],
                device=self.device,
                dtype=torch.long,
            )
            predicted.append(next_state)
            target.append(self.states[current])
            history_state = torch.cat(
                [next_state.unsqueeze(1), history_state[:, :-1]], dim=1
            )
        return torch.stack(predicted, 1), torch.stack(target, 1), diagnostics

    def loss(
        self, starts: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prediction, truth, diagnostics = self.rollout(starts, horizon)
        normalized_error = (prediction - truth) / self.scale_t
        one = normalized_error[:, 0].square().mean()
        rollout = normalized_error.square().mean()
        group = diagnostics["group_probs"].mean(0)
        local = diagnostics["local_probs"].mean((0, 1))
        balance = group.square().sum() + local.square().sum()
        entropy = -(
            diagnostics["group_probs"]
            * torch.log(diagnostics["group_probs"] + EPS)
        ).sum(1).mean()
        total = one + (0.45 if horizon > 1 else 0.0) * rollout + 0.01 * balance - 0.001 * entropy
        return total, {
            "loss": float(total.detach()),
            "one": float(one.detach()),
            "rollout": float(rollout.detach()),
        }

    @torch.inference_mode()
    def validate(self, step: int) -> dict[str, float]:
        self.model.eval()
        starts = self.val_starts
        if len(starts) > 128:
            starts = starts[np.linspace(0, len(starts) - 1, 128, dtype=np.int64)]
        prediction, truth, _ = self.rollout(
            torch.as_tensor(starts, device=self.device), self.args.validation_horizon
        )
        finite = torch.isfinite(prediction).all(2)
        velocity = torch.linalg.vector_norm(
            prediction[:, :, :32] - truth[:, :, :32]
        ) / (torch.linalg.vector_norm(truth[:, :, :32]) + EPS)
        pressure = torch.linalg.vector_norm(
            prediction[:, :, 32:] - truth[:, :, 32:]
        ) / (torch.linalg.vector_norm(truth[:, :, 32:]) + EPS)
        score = float(velocity + pressure)
        result = {
            "step": step,
            "score": score,
            "velocity_modal_relative_l2": float(velocity),
            "pressure_modal_relative_l2": float(pressure),
            "finite_fraction": float(finite.float().mean()),
            "divergent_windows": int(
                (
                    torch.linalg.vector_norm(prediction, dim=2)
                    > 20.0
                    * torch.clamp(
                        torch.linalg.vector_norm(truth, dim=2), min=1.0e-6
                    )
                )
                .any(1)
                .sum()
            ),
        }
        self.model.train()
        return result

    def checkpoint(self, step: int, validation: dict[str, float], name: str) -> None:
        atomic_torch(
            self.run / name,
            {
                "schema": "dataonly_specialist_ablation/v1",
                "regime": self.args.regime,
                "seed": self.args.seed,
                "step": step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "normalization": {"mean": self.mean, "scale": self.scale},
                "validation": validation,
                "args": vars(self.args),
                "split_re": self.data["split_re"],
            },
        )

    def initialize_swanlab(self) -> None:
        import swanlab

        self.swan = swanlab.init(
            project=self.args.swanlab_project,
            group=f"DataOnly-{self.args.regime}",
            name=f"DataOnly-{self.args.regime}-seed{self.args.seed}",
            mode="online",
            config={
                **vars(self.args),
                "split_re": self.data["split_re"],
                "method": "direct discrete POD-state mapping; no RK4/Galerkin/pressure physics",
            },
            log_dir=str(self.run / "swanlog"),
            reinit=True,
            parallel="shared",
        )

    def run_smoke(self) -> None:
        reports = []
        torch.cuda.reset_peak_memory_stats()
        for horizon in (1, 4, 8, 16):
            candidates = legal_starts(self.data, "train", horizon)
            if not len(candidates):
                raise RuntimeError(f"DataOnly has no legal K{horizon} smoke starts")
            batch = torch.as_tensor(
                candidates[: min(self.args.micro_batch, len(candidates))],
                device=self.device,
            )
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=self.amp_dtype):
                loss, metrics = self.loss(batch, horizon)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise FloatingPointError(
                    f"DataOnly K{horizon} smoke produced non-finite loss/grad"
                )
            reports.append(
                {
                    "horizon": horizon,
                    "loss": float(loss.detach()),
                    "grad_norm": float(grad),
                    "metrics": metrics,
                }
            )
        validation = self.validate(0)
        self.checkpoint(0, validation, "smoke.pt")
        atomic_json(
            self.run / "SMOKE_PASS.json",
            {
                "status": "PASS",
                "regime": self.args.regime,
                "seed": self.args.seed,
                "horizons": [1, 4, 8, 16],
                "reports": reports,
                "validation": validation,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "split_re": self.data["split_re"],
            },
        )

    def run_training(self) -> None:
        self.initialize_swanlab()
        import swanlab

        atomic_json(
            self.run / "STARTED.json",
            {
                "status": "RUNNING",
                "pid": os.getpid(),
                "regime": self.args.regime,
                "seed": self.args.seed,
                "assets": [
                    {"path": str(path), "sha256": sha256(path)}
                    for path in self.data["assets"]
                ],
                "split_re": self.data["split_re"],
            },
        )
        validation_history = []
        started = time.time()
        for step in range(1, self.args.max_steps + 1):
            horizon = self.args.curriculum_horizons[
                min(
                    len(self.args.curriculum_horizons) - 1,
                    (step - 1)
                    * len(self.args.curriculum_horizons)
                    // self.args.max_steps,
                )
            ]
            candidates = self.train_starts[horizon]
            sampled = self.rng.choice(
                candidates, self.args.effective_batch, replace=True
            )
            self.optimizer.zero_grad(set_to_none=True)
            aggregate = {"loss": 0.0, "one": 0.0, "rollout": 0.0}
            for offset in range(0, len(sampled), self.args.micro_batch):
                batch = torch.as_tensor(
                    sampled[offset : offset + self.args.micro_batch],
                    device=self.device,
                )
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    loss, metrics = self.loss(batch, horizon)
                    scaled = loss / (len(sampled) / self.args.micro_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                self.scaler.scale(scaled).backward()
                for key in aggregate:
                    aggregate[key] += metrics[key] / (
                        len(sampled) / self.args.micro_batch
                    )
            self.scaler.unscale_(self.optimizer)
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise FloatingPointError(f"non-finite grad at step {step}")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            if step == 1 or step % self.args.log_every == 0:
                swanlab.log(
                    {
                        **{f"train/{key}": value for key, value in aggregate.items()},
                        "train/horizon": horizon,
                        "train/grad_norm": float(grad),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )
                print(
                    json.dumps(
                        {
                            "event": "optimizer_step",
                            "step": step,
                            "regime": self.args.regime,
                            **aggregate,
                        }
                    ),
                    flush=True,
                )
            if step % self.args.validation_every == 0:
                validation = self.validate(step)
                validation_history.append(validation)
                swanlab.log(
                    {f"validation/{key}": value for key, value in validation.items()},
                    step=step,
                )
                self.checkpoint(step, validation, "last.pt")
                if validation["score"] < self.best_score:
                    self.best_score = validation["score"]
                    self.best_step = step
                    self.checkpoint(step, validation, "best_validation.pt")
                atomic_json(self.run / "validation_history.json", validation_history)
        final = self.validate(self.args.max_steps)
        self.checkpoint(self.args.max_steps, final, "final.pt")
        atomic_json(
            self.run / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "best_step": self.best_step,
                "best_score": self.best_score,
                "runtime_seconds": time.time() - started,
            },
        )
        swanlab.finish(state="success")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-mode", choices=("data-only",), required=True
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--regime", choices=("steady", "hopf", "periodic"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=5.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1.5e-4)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--effective-batch", type=int, default=24)
    parser.add_argument("--validation-every", type=int, default=200)
    parser.add_argument("--validation-horizon", type=int, default=16)
    parser.add_argument("--curriculum-horizons", default="1,4,8,16")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--swanlab-project", default="PaperSpecialistAblations20260723")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.curriculum_horizons = tuple(
        int(value) for value in args.curriculum_horizons.split(",")
    )
    if args.effective_batch % args.micro_batch:
        parser.error("effective batch must be divisible by micro batch")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    trainer = Trainer(parsed)
    trainer.run_smoke() if parsed.smoke_only else trainer.run_training()
