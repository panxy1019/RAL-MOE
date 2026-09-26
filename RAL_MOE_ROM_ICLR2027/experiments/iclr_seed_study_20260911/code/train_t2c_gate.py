#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    ConvexGate,
    atomic_json,
    blend_relative,
    clean_numeric,
    router_probabilities,
    sha256,
)


def summarize(alpha: np.ndarray, quad_u: np.ndarray, quad_p: np.ndarray, reynolds: np.ndarray) -> dict:
    weight = alpha[:, None]
    eu = np.maximum(
        (
            weight * weight * quad_u[:, :, 0]
            + (1 - weight) ** 2 * quad_u[:, :, 1]
            + 2 * weight * (1 - weight) * quad_u[:, :, 2]
        )
        / np.maximum(quad_u[:, :, 3], 1.0e-12),
        0,
    )
    ep = np.maximum(
        (
            weight * weight * quad_p[:, :, 0]
            + (1 - weight) ** 2 * quad_p[:, :, 1]
            + 2 * weight * (1 - weight) * quad_p[:, :, 2]
        )
        / np.maximum(quad_p[:, :, 3], 1.0e-12),
        0,
    )
    joint = np.sqrt(eu) + np.sqrt(ep)
    per_re = {
        str(float(value)): {
            "joint_mean": float(np.mean(joint[reynolds == value])),
            "joint_worst": float(np.max(joint[reynolds == value])),
            "alpha_mean": float(np.mean(alpha[reynolds == value])),
            "alpha_min": float(np.min(alpha[reynolds == value])),
            "alpha_max": float(np.max(alpha[reynolds == value])),
        }
        for value in np.unique(reynolds)
    }
    return {
        "velocity_mean": float(np.mean(np.sqrt(eu))),
        "pressure_mean": float(np.mean(np.sqrt(ep))),
        "joint_mean": float(np.mean(joint)),
        "joint_worst": float(np.max(joint)),
        "worst_Re_mean": float(max(row["joint_mean"] for row in per_re.values())),
        "per_Re": per_re,
        "alpha_mean": float(np.mean(alpha)),
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-mode", choices=("full", "mu_only"), default="full")
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    with np.load(args.cache, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if set(np.unique(data["split"])) != {"train", "validation"}:
        raise RuntimeError("development cache must contain train and validation only")
    if not bool(np.all(data["candidate_1_finite"]) and np.all(data["candidate_2_finite"])):
        raise RuntimeError("preflight failed: non-finite candidate rollout in cache")
    train = data["split"] == "train"
    validation = data["split"] == "validation"
    mean = data["features"][train].mean(0)
    std = data["features"][train].std(0)
    std[std < 1.0e-8] = 1.0
    features = ((data["features"] - mean) / std).astype(np.float32)
    if args.input_mode == "mu_only":
        features[:, 1:] = 0.0
    probabilities = router_probabilities(args.router_checkpoint, data["re"])
    pair_indices = data["pair_indices"][0].astype(int)
    pair = probabilities[:, pair_indices]
    pair /= pair.sum(axis=1, keepdims=True)
    base = np.clip(pair[:, 0], 1.0e-6, 1 - 1.0e-6)
    base_logit = np.log(base / (1 - base)).astype(np.float32)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(features, device=device)
    logits = torch.as_tensor(base_logit, device=device)
    quad_u = torch.as_tensor(data["quad_u"], dtype=torch.float32, device=device)
    quad_p = torch.as_tensor(data["quad_p"], dtype=torch.float32, device=device)
    train_ids = torch.as_tensor(np.flatnonzero(train), device=device)
    validation_ids = torch.as_tensor(np.flatnonzero(validation), device=device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = ConvexGate(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    atomic_json(args.output_dir / "config.json", {
        "seed": args.seed, "steps": args.steps, "eval_every": args.eval_every,
        "input_mode": args.input_mode, "input_dim": int(features.shape[1]),
        "mask_after_standardization": args.input_mode == "mu_only",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer": "AdamW", "lr": 3.0e-4, "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0, "batch_size": min(256, len(train_ids)),
        "batch_sampling": "with_replacement",
        "selection": ["validation worst_Re_mean", "validation joint_mean"],
        "cache_sha256": sha256(args.cache),
        "router_checkpoint_sha256": sha256(args.router_checkpoint),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "device": str(device), "started_unix": time.time(),
        "swanlab_mode": args.swanlab_mode,
    })
    run = None
    boundary = str(data["boundary"].item())
    if args.swanlab_mode != "disabled":
        import swanlab
        run = swanlab.init(
            project="CenteredSquare_Fusion_E2_T2C",
            group=f"T2C_{boundary}",
            name=f"T2C_{boundary}_K{int(data['horizon'].item())}_seed{args.seed}",
            mode=args.swanlab_mode,
            logdir=str(args.output_dir / "swanlog"),
            config={
                "method": "T2-C_LearnedConvexCorrection_FieldBlend",
                "boundary": boundary,
                "steps": args.steps,
                "seed": args.seed,
                "cache_sha256": sha256(args.cache),
                "specialists_frozen": True,
                "fusion_feedback": False,
            },
            reinit=True,
        )
    best = None
    history = []
    started = time.time()
    try:
        for step in range(1, args.steps + 1):
            batch = train_ids[torch.randint(len(train_ids), (min(256, len(train_ids)),), device=device)]
            alpha = model(x[batch], logits[batch])
            loss = blend_relative(alpha, quad_u[batch]).mean() + blend_relative(alpha, quad_p[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                with torch.inference_mode():
                    val_alpha = model(x[validation_ids], logits[validation_ids]).cpu().numpy()
                metrics = summarize(
                    val_alpha,
                    data["quad_u"][validation],
                    data["quad_p"][validation],
                    data["re"][validation],
                )
                selection = (metrics["worst_Re_mean"], metrics["joint_mean"])
                row = {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "grad_norm": float(grad_norm.detach()),
                    **metrics,
                }
                history.append(row)
                if best is None or selection < best[0]:
                    best = (
                        selection,
                        step,
                        {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                        metrics,
                    )
                atomic_json(
                    args.output_dir / "runtime_status.json",
                    {"status": "running", "boundary": boundary, "updated_unix": time.time(), **row},
                )
                if run is not None:
                    import swanlab
                    swanlab.log(clean_numeric({f"train/{key}": value for key, value in row.items()}), step=step)
        assert best is not None
        model.load_state_dict(best[2])
        with torch.inference_mode():
            all_alpha = model(x, logits).cpu().numpy()
        validation_metrics = summarize(
            all_alpha[validation],
            data["quad_u"][validation],
            data["quad_p"][validation],
            data["re"][validation],
        )
        checkpoint = {
            "schema_version": 1,
            "method": "T2-C_LearnedConvexCorrection_FieldBlend",
            "boundary": boundary,
            "gate_state": best[2],
            "feature_mean": mean,
            "feature_std": std,
            "best_step": best[1],
            "validation_metrics": validation_metrics,
            "router_checkpoint_sha256": sha256(args.router_checkpoint),
            "cache_sha256": sha256(args.cache),
            "pair_indices": pair_indices,
            "seed": args.seed,
            "input_mode": args.input_mode,
            "fusion_feedback": False,
        }
        torch.save(checkpoint, args.output_dir / "best.pt")
        atomic_json(args.output_dir / "history.json", history)
        atomic_json(args.output_dir / "validation_metrics.json", validation_metrics)
        atomic_json(
            args.output_dir / "runtime_status.json",
            {
                "status": "complete",
                "boundary": boundary,
                "best_step": best[1],
                "elapsed_seconds": time.time() - started,
                **validation_metrics,
            },
        )
    finally:
        if run is not None:
            import swanlab
            swanlab.finish()


if __name__ == "__main__":
    main()
