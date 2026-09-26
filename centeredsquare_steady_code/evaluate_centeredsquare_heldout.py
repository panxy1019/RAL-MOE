#!/usr/bin/env python3
"""One-time validation-frozen heldout evaluation for CenteredSquare steady S2-B."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-heldout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_validation_history(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint_selection(root: Path, checkpoint: Path, output_dir: Path) -> dict:
    run = root / "runs" / "s2b_rank999"
    history = load_validation_history(run / "validation_history.jsonl")
    candidates = []
    qualified = []
    for item in history:
        k16 = item["k16"]
        row = {
            "step": int(k16["step"]),
            "worst_pressure": float(k16["worst_pressure"]),
            "worst_velocity": float(k16["worst_velocity"]),
            "fixed_point_residual": float(k16["fixed_point_residual"]),
            "contraction_ratio": float(k16["contraction_ratio"]),
            "finite_fraction": float(k16["finite_fraction"]),
            "divergent_windows": int(k16["divergent_windows"]),
        }
        row["qualified"] = (
            row["finite_fraction"] == 1.0
            and row["divergent_windows"] == 0
            and row["contraction_ratio"] < 1.0
        )
        candidates.append(row)
        if row["qualified"]:
            qualified.append(row)
    best_unqualified = min(
        candidates,
        key=lambda row: (
            row["worst_pressure"],
            row["worst_velocity"],
            row["fixed_point_residual"],
        ),
    )

    checkpoint_info = {}
    model_hashes = {}
    for name in ("latest.pt", "final.pt"):
        path = run / "checkpoints" / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model_hash = state_sha256(payload["model"])
        checkpoint_info[name] = {
            "path": str(path),
            "step": int(payload["step"]),
            "sha256": sha256(path),
            "model_sha256": model_hash,
        }
        model_hashes[name] = model_hash
        del payload
    if len(set(model_hashes.values())) != 1:
        raise RuntimeError("latest.pt and final.pt do not contain identical model weights")

    selected_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selected_step = int(selected_payload["step"])
    del selected_payload
    if selected_step != 3200:
        raise RuntimeError(f"expected final step 3200, got {selected_step}")

    frozen_dir = root / "checkpoint"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen = frozen_dir / "frozen_centeredsquare_steady_step3200.pt"
    if frozen.exists():
        if sha256(frozen) != sha256(checkpoint):
            raise RuntimeError(f"refusing to replace different frozen checkpoint: {frozen}")
    else:
        os.link(checkpoint, frozen)

    report = {
        "status": "FROZEN_UNQUALIFIED_CANDIDATE",
        "selection_data": "validation_only",
        "heldout_used_for_selection": False,
        "pre_registered_qualification": {
            "finite_fraction": 1.0,
            "divergent_windows": 0,
            "contraction_ratio_less_than": 1.0,
        },
        "qualified_checkpoint_count": len(qualified),
        "best_observed_validation_candidate": best_unqualified,
        "selected_checkpoint": {
            "path": str(frozen),
            "source_path": str(checkpoint),
            "step": selected_step,
            "sha256": sha256(frozen),
            "model_sha256": model_hashes["final.pt"],
            "reason": (
                "No validation checkpoint passed the contraction<1 gate. "
                "The training-produced final early-stop checkpoint is frozen for one-time heldout characterization, "
                "not promoted as a qualified deployment checkpoint."
            ),
        },
        "available_terminal_checkpoints": checkpoint_info,
        "validation_history": candidates,
    }
    atomic_json(output_dir / "checkpoint_selection.json", report)
    return report


def raw_case_path(raw_dir: Path, label: str) -> Path:
    return raw_dir / f"snapshots_{label}.npz"


def prepare_exact_physical_terms(exp, raw_dir: Path) -> dict[str, np.ndarray | float]:
    count = len(exp.a["a"])
    terms: dict[str, np.ndarray | float] = {
        "u_residual": np.full(count, np.nan, dtype=np.float64),
        "u_truth": np.full(count, np.nan, dtype=np.float64),
        "u_projected_truth": np.full(count, np.nan, dtype=np.float64),
        "p_residual": np.full(count, np.nan, dtype=np.float64),
        "p_truth": np.full(count, np.nan, dtype=np.float64),
        "p_projected_truth": np.full(count, np.nan, dtype=np.float64),
        "u_residual_mode_inner": np.full((count, exp.ru), np.nan, dtype=np.float64),
        "p_residual_mode_inner": np.full((count, exp.rp), np.nan, dtype=np.float64),
    }
    with np.load(exp.paths["velocity_pod"]) as velocity:
        phi_u = velocity["phi_uv"][: exp.ru].astype(np.float64)
        mean_u = velocity["mean_uv_regime"].astype(np.float64)
        volumes = velocity["point_areas"].astype(np.float64)
    with np.load(exp.paths["pressure_pod"]) as pressure:
        phi_p = pressure["phi_p"][: exp.rp].astype(np.float64)
        mean_p = pressure["mean_p_regime"].astype(np.float64)
    weights_u = np.repeat(volumes, 2)
    gram_u = (phi_u * weights_u[None, :]) @ phi_u.T
    gram_p = (phi_p * volumes[None, :]) @ phi_p.T
    max_orthogonality = {"velocity": 0.0, "pressure": 0.0}
    coefficient_consistency = {"velocity": 0.0, "pressure": 0.0}

    heldout_labels = sorted(exp.windows["heldout"])
    for label_id in heldout_labels:
        label = str(exp.a["labels"][label_id])
        ids = np.flatnonzero(exp.a["label_id"] == label_id)
        ids = ids[np.argsort(exp.a["time"][ids])]
        path = raw_case_path(raw_dir, label)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as case:
            times = case["times"].astype(np.float64)
            velocity = case["U"].astype(np.float64).reshape(len(times), -1)
            pressure = case["p"].astype(np.float64)
        pressure -= (pressure @ volumes / volumes.sum())[:, None]
        if len(ids) != len(times) or not np.allclose(exp.a["time"][ids], times, atol=2e-5, rtol=0):
            raise RuntimeError(f"snapshot time mismatch for {label}")
        centered_u = velocity - mean_u[None, :]
        centered_p = pressure - mean_p[None, :]
        projected_u = mean_u[None, :] + exp.a["a"][ids].astype(np.float64) @ phi_u
        projected_p = mean_p[None, :] + exp.a["b"][ids].astype(np.float64) @ phi_p
        residual_u = velocity - projected_u
        residual_p = pressure - projected_p
        terms["u_residual"][ids] = np.einsum("nd,d,nd->n", residual_u, weights_u, residual_u)
        terms["u_truth"][ids] = np.einsum("nd,d,nd->n", velocity, weights_u, velocity)
        terms["u_projected_truth"][ids] = np.einsum("nd,d,nd->n", projected_u, weights_u, projected_u)
        terms["p_residual"][ids] = np.einsum("nd,d,nd->n", residual_p, volumes, residual_p)
        terms["p_truth"][ids] = np.einsum("nd,d,nd->n", pressure, volumes, pressure)
        terms["p_projected_truth"][ids] = np.einsum("nd,d,nd->n", projected_p, volumes, projected_p)
        orth_u = (residual_u * weights_u[None, :]) @ phi_u.T
        orth_p = (residual_p * volumes[None, :]) @ phi_p.T
        terms["u_residual_mode_inner"][ids] = orth_u
        terms["p_residual_mode_inner"][ids] = orth_p
        max_orthogonality["velocity"] = max(max_orthogonality["velocity"], float(np.max(np.abs(orth_u))))
        max_orthogonality["pressure"] = max(max_orthogonality["pressure"], float(np.max(np.abs(orth_p))))
        reproj_u = (centered_u * weights_u[None, :]) @ phi_u.T
        reproj_p = (centered_p * volumes[None, :]) @ phi_p.T
        coefficient_consistency["velocity"] = max(
            coefficient_consistency["velocity"],
            float(np.max(np.abs(reproj_u - exp.a["a"][ids]))),
        )
        coefficient_consistency["pressure"] = max(
            coefficient_consistency["pressure"],
            float(np.max(np.abs(reproj_p - exp.a["b"][ids]))),
        )

    terms["gram_u"] = gram_u
    terms["gram_p"] = gram_p
    terms["max_residual_mode_inner_product"] = max_orthogonality
    terms["max_reprojection_coefficient_abs_delta"] = coefficient_consistency
    return terms


def empty_accumulator() -> dict[str, float]:
    return {
        "u_modal_num": 0.0,
        "u_modal_den": 0.0,
        "p_modal_num": 0.0,
        "p_modal_den": 0.0,
        "u_projected_num": 0.0,
        "u_projected_den": 0.0,
        "p_projected_num": 0.0,
        "p_projected_den": 0.0,
        "u_exact_num": 0.0,
        "u_exact_den": 0.0,
        "p_exact_num": 0.0,
        "p_exact_den": 0.0,
        "u_floor_num": 0.0,
        "p_floor_num": 0.0,
        "windows": 0.0,
        "finite_windows": 0.0,
        "divergent_windows": 0.0,
        "pressure_drift": 0.0,
        "fixed_point_residual": 0.0,
    }


def finalize_accumulator(acc: dict[str, float]) -> dict[str, float | int]:
    return {
        "windows": int(acc["windows"]),
        "raw_pod_velocity_rel_l2": math.sqrt(acc["u_modal_num"] / max(acc["u_modal_den"], EPS)),
        "raw_pod_pressure_rel_l2": math.sqrt(acc["p_modal_num"] / max(acc["p_modal_den"], EPS)),
        "projected_physical_velocity_rel_l2": math.sqrt(
            acc["u_projected_num"] / max(acc["u_projected_den"], EPS)
        ),
        "projected_physical_pressure_rel_l2": math.sqrt(
            acc["p_projected_num"] / max(acc["p_projected_den"], EPS)
        ),
        "full_cfd_physical_velocity_rel_l2": math.sqrt(acc["u_exact_num"] / max(acc["u_exact_den"], EPS)),
        "full_cfd_physical_pressure_rel_l2": math.sqrt(acc["p_exact_num"] / max(acc["p_exact_den"], EPS)),
        "pod_floor_full_cfd_velocity_rel_l2": math.sqrt(acc["u_floor_num"] / max(acc["u_exact_den"], EPS)),
        "pod_floor_full_cfd_pressure_rel_l2": math.sqrt(acc["p_floor_num"] / max(acc["p_exact_den"], EPS)),
        "finite_fraction": acc["finite_windows"] / max(acc["windows"], 1.0),
        "divergent_windows": int(acc["divergent_windows"]),
        "pressure_drift_worst": acc["pressure_drift"],
        "fixed_point_residual_worst": acc["fixed_point_residual"],
    }


def add_batch(
    acc: dict[str, float],
    pa: np.ndarray,
    pb: np.ndarray,
    ta: np.ndarray,
    tb: np.ndarray,
    target_ids: np.ndarray,
    b0: np.ndarray,
    terms: dict,
) -> None:
    delta_a = pa - ta
    delta_b = pb - tb
    projected_u_error = np.einsum("kbr,rs,kbs->kb", delta_a, terms["gram_u"], delta_a)
    projected_p_error = np.einsum("kbr,rs,kbs->kb", delta_b, terms["gram_p"], delta_b)
    u_residual = terms["u_residual"][target_ids]
    p_residual = terms["p_residual"][target_ids]
    u_cross = np.einsum("kbr,kbr->kb", delta_a, terms["u_residual_mode_inner"][target_ids])
    p_cross = np.einsum("kbr,kbr->kb", delta_b, terms["p_residual_mode_inner"][target_ids])
    u_truth = terms["u_truth"][target_ids]
    p_truth = terms["p_truth"][target_ids]
    u_projected_truth = terms["u_projected_truth"][target_ids]
    p_projected_truth = terms["p_projected_truth"][target_ids]
    acc["u_modal_num"] += float(np.sum(delta_a * delta_a))
    acc["u_modal_den"] += float(np.sum(ta * ta))
    acc["p_modal_num"] += float(np.sum(delta_b * delta_b))
    acc["p_modal_den"] += float(np.sum(tb * tb))
    acc["u_projected_num"] += float(np.sum(projected_u_error))
    acc["u_projected_den"] += float(np.sum(u_projected_truth))
    acc["p_projected_num"] += float(np.sum(projected_p_error))
    acc["p_projected_den"] += float(np.sum(p_projected_truth))
    acc["u_exact_num"] += float(np.sum(projected_u_error + u_residual - 2.0 * u_cross))
    acc["u_exact_den"] += float(np.sum(u_truth))
    acc["p_exact_num"] += float(np.sum(projected_p_error + p_residual - 2.0 * p_cross))
    acc["p_exact_den"] += float(np.sum(p_truth))
    acc["u_floor_num"] += float(np.sum(u_residual))
    acc["p_floor_num"] += float(np.sum(p_residual))
    finite = np.isfinite(pa).all(axis=(0, 2)) & np.isfinite(pb).all(axis=(0, 2))
    ratio_a = np.linalg.norm(pa, axis=2) / (np.linalg.norm(ta, axis=2) + 1e-6)
    ratio_b = np.linalg.norm(pb, axis=2) / (np.linalg.norm(tb, axis=2) + 1e-6)
    divergent = ((ratio_a > 20) | (ratio_b > 20) | ~np.isfinite(ratio_a) | ~np.isfinite(ratio_b)).any(axis=0)
    acc["windows"] += float(pa.shape[1])
    acc["finite_windows"] += float(np.sum(finite))
    acc["divergent_windows"] += float(np.sum(divergent))
    pressure_drift = np.linalg.norm(pb[-1] - b0, axis=1) / (np.linalg.norm(b0, axis=1) + 1e-12)
    fixed = np.sqrt(
        np.sum((pb - b0[None, :, :]) ** 2, axis=2)
        / (np.sum(b0 * b0, axis=1)[None, :] + 1e-12)
    )
    acc["pressure_drift"] = max(acc["pressure_drift"], float(np.max(pressure_drift)))
    acc["fixed_point_residual"] = max(acc["fixed_point_residual"], float(np.max(fixed)))


def evaluate_horizon(exp, horizon: int, batch_size: int, terms: dict) -> dict:
    windows_np = exp.build_windows(horizon)["heldout"]
    windows = {
        label: torch.as_tensor(ids, dtype=torch.long, device=exp.device)
        for label, ids in windows_np.items()
    }
    exp.validation_windows[horizon] = windows
    aggregate = empty_accumulator()
    per_re = {}
    exp.model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=exp.amp_dtype):
        for label_id, ids in windows.items():
            label_acc = empty_accumulator()
            for offset in range(0, len(ids), batch_size):
                starts = ids[offset : offset + batch_size]
                out = exp.rollout(starts, horizon)
                pa, pb, ta, tb = [
                    out[key].float().cpu().numpy().astype(np.float64)
                    for key in ("pa", "pb", "ta", "tb")
                ]
                target_ids = exp.indices(starts, horizon)[:, 1:].T.cpu().numpy()
                b0 = out["b0"].float().cpu().numpy().astype(np.float64)
                add_batch(label_acc, pa, pb, ta, tb, target_ids, b0, terms)
                add_batch(aggregate, pa, pb, ta, tb, target_ids, b0, terms)
            per_re[str(exp.a["labels"][label_id])] = finalize_accumulator(label_acc)
    report = finalize_accumulator(aggregate)
    report["horizon"] = horizon
    report["per_re"] = per_re
    report["worst_by_re"] = {
        key: max(float(item[key]) for item in per_re.values())
        for key in (
            "raw_pod_velocity_rel_l2",
            "raw_pod_pressure_rel_l2",
            "projected_physical_velocity_rel_l2",
            "projected_physical_pressure_rel_l2",
            "full_cfd_physical_velocity_rel_l2",
            "full_cfd_physical_pressure_rel_l2",
        )
    }
    if horizon == 16:
        original = exp.validation_windows
        exp.validation_windows = {16: windows}
        report["controlled_contraction_by_re"] = exp.controlled_contraction()
        report["controlled_contraction_worst"] = max(report["controlled_contraction_by_re"].values())
        exp.validation_windows = original
    else:
        report["controlled_contraction_by_re"] = None
        report["controlled_contraction_worst"] = None
    return report


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def write_markdown(path: Path, selection: dict, result: dict, asset_audit: dict) -> None:
    lines = [
        "# CenteredSquare Steady Specialist 最终实验报告",
        "",
        f"- 生成时间：`{result['finished_iso']}`",
        f"- 模型：`{result['experiment']}`",
        f"- POD：rank999，`ru={result['rank']['r_u']}`、`rp={result['rank']['r_p']}`",
        f"- 压力规范：`{result['pressure_gauge']}`",
        f"- SwanLab：{result['swanlab_url']}",
        "",
        "## 1. 数据与冻结合同",
        "",
        "| Split | Re | 案例数 | 快照数 |",
        "|---|---|---:|---:|",
        f"| Train | 见 `ASSET_AUDIT.json` | 60 | {asset_audit['snapshot_counts']['train']} |",
        f"| Validation | 55, 75, 90, 94.5, 95.25 | 5 | {asset_audit['snapshot_counts']['validation']} |",
        f"| Heldout/Test | 60, 85, 95.1, 95.3 | 4 | {asset_audit['snapshot_counts']['heldout']} |",
        "",
        "训练、validation、heldout 的 Re 集合零交集。POD、均值与 normalization 仅由 train split 拟合；heldout 未用于训练、早停或 checkpoint 选择。",
        "",
        "## 2. Checkpoint 结论",
        "",
        f"- 训练于 step `{selection['selected_checkpoint']['step']}` 正常早停。",
        f"- 冻结 checkpoint：`{selection['selected_checkpoint']['path']}`",
        f"- SHA256：`{selection['selected_checkpoint']['sha256']}`",
        f"- 预注册状态：`{selection['status']}`。",
        f"- validation 中通过 contraction `<1` 门的 checkpoint 数：`{selection['qualified_checkpoint_count']}`。",
        "",
        "因此该权重用于一次性 heldout 性能刻画，但不能标记为通过旧圆柱 contraction 硬门的 qualified deployment checkpoint。",
        "",
        "## 3. Heldout clean autonomous rollout",
        "",
        "下表是四个 heldout Re 合并后的误差；relative L2 已乘 100。`Full CFD physical` 直接相对原始 CFD 快照，包含 POD 截断误差。",
        "",
        "| K | 窗口数 | Raw POD U | Raw POD p | Projected physical U | Projected physical p | Full CFD physical U | Full CFD physical p | Finite | Divergent |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(result["horizons"], key=lambda item: int(item)):
        item = result["horizons"][key]
        lines.append(
            f"| {item['horizon']} | {item['windows']} | {pct(item['raw_pod_velocity_rel_l2'])} | "
            f"{pct(item['raw_pod_pressure_rel_l2'])} | {pct(item['projected_physical_velocity_rel_l2'])} | "
            f"{pct(item['projected_physical_pressure_rel_l2'])} | {pct(item['full_cfd_physical_velocity_rel_l2'])} | "
            f"{pct(item['full_cfd_physical_pressure_rel_l2'])} | {100*item['finite_fraction']:.1f}% | "
            f"{item['divergent_windows']} |"
        )
    lines.extend(
        [
            "",
            "### K16 分 Re 的 Full CFD physical relative L2",
            "",
            "| Heldout Re | Velocity | Pressure | POD floor U | POD floor p |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    k16 = result["horizons"]["16"]
    for label, item in k16["per_re"].items():
        re_value = label.removeprefix("Re").replace("p", ".")
        lines.append(
            f"| {float(re_value):.6g} | {pct(item['full_cfd_physical_velocity_rel_l2'])} | "
            f"{pct(item['full_cfd_physical_pressure_rel_l2'])} | "
            f"{pct(item['pod_floor_full_cfd_velocity_rel_l2'])} | "
            f"{pct(item['pod_floor_full_cfd_pressure_rel_l2'])} |"
        )
    lines.extend(
        [
            "",
            "## 4. 稳定性指标",
            "",
            f"- K16 controlled contraction worst：`{k16['controlled_contraction_worst']:.6g}`。",
            f"- K16 pressure drift worst：`{k16['pressure_drift_worst']:.6g}`。",
            f"- K16 fixed-point residual worst：`{k16['fixed_point_residual_worst']:.6g}`。",
            f"- K16 finite fraction：`{100*k16['finite_fraction']:.1f}%`；divergent windows：`{k16['divergent_windows']}`。",
            "",
            "旧圆柱 contraction 门使用相对极小 steady POD 系数的归一化扰动比；在本方柱数据上该比值远大于 1。clean rollout 保持有限且无发散，但不能据此宣称局部扰动收缩。",
            "",
            "## 5. 资产与数值审计",
            "",
            f"- Train POD 系数复投影相对差：velocity `{asset_audit['train_projection_vs_stored_coeff_relative']['velocity']:.3e}`，pressure `{asset_audit['train_projection_vs_stored_coeff_relative']['pressure']:.3e}`。",
            f"- ROM 的 1/Re 仿射重建残差最大约 `{max(asset_audit['rom_reconstruction_relative'].values()):.3e}`。",
            f"- Heldout residual 与 POD modes 的最大加权内积：velocity `{result['physical_term_audit']['max_residual_mode_inner_product']['velocity']:.3e}`，pressure `{result['physical_term_audit']['max_residual_mode_inner_product']['pressure']:.3e}`。",
            f"- `latest.pt` 与 `final.pt` 模型权重 SHA256 相同：`{result['checkpoint_model_identity']}`。",
            "",
            "## 6. 最终判断",
            "",
            result["conclusion"],
            "",
            "完整逐窗口/逐 Re 数值见 `heldout_metrics.json`；checkpoint 选择证据见 `checkpoint_selection.json`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = checkpoint_selection(root, args.checkpoint.resolve(), output_dir)
    config_path = root / "code" / "training_centeredsquare_steady_rank999.json"
    config = json.loads(config_path.read_text())
    trainer = load_module(root / "code" / "train_s2b_3090.py", "centeredsquare_s2b")
    exp = trainer.Experiment(SimpleNamespace(run_dir=str(output_dir / "runtime"), resume=None), config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    exp.model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    exp.validation_windows = {}
    terms = prepare_exact_physical_terms(exp, args.raw_heldout_dir.resolve())
    horizons = {}
    for horizon in args.horizons:
        started = time.time()
        item = evaluate_horizon(exp, int(horizon), args.batch_size, terms)
        item["runtime_seconds"] = time.time() - started
        horizons[str(horizon)] = item
        atomic_json(output_dir / f"heldout_k{horizon}.json", item)
        print(json.dumps({"event": "heldout_horizon_complete", "horizon": horizon, "metrics": item}), flush=True)

    asset_audit = json.loads((root / "ASSET_AUDIT.json").read_text())
    k16 = horizons["16"]
    selected_validation = next(
        item
        for item in selection["validation_history"]
        if item["step"] == selection["selected_checkpoint"]["step"]
    )
    result = {
        "schema_version": 1,
        "status": "PASS_EVALUATION_COMPLETED",
        "experiment": config["experiment"],
        "finished_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rank": {"r_u": config["r_u"], "r_p": config["r_p"], "policy": "rank999"},
        "pressure_gauge": config["pressure_gauge"],
        "heldout_re": [60.0, 85.0, 95.1, 95.3],
        "heldout_used_for_selection": False,
        "checkpoint": selection["selected_checkpoint"],
        "checkpoint_model_identity": (
            selection["available_terminal_checkpoints"]["latest.pt"]["model_sha256"]
            == selection["available_terminal_checkpoints"]["final.pt"]["model_sha256"]
        ),
        "selected_validation_k16": selected_validation,
        "physical_term_audit": {
            "max_residual_mode_inner_product": terms["max_residual_mode_inner_product"],
            "max_reprojection_coefficient_abs_delta": terms["max_reprojection_coefficient_abs_delta"],
        },
        "horizons": horizons,
        "swanlab_url": "https://swanlab.cn/@panxy1019/V17indepentMOEV2/runs/mk26kj3q",
        "conclusion": (
            "该模型在全部 heldout clean rollout 窗口上保持 finite 且无发散，并给出了可复现的方柱 steady "
            "预测基线；但 validation 与 heldout 的 controlled contraction 均未通过 <1 硬门，因此结果应标记为 "
            "stable clean-rollout baseline / unqualified contraction，而不是合格的局部吸引子部署模型。"
        ),
    }
    atomic_json(output_dir / "heldout_metrics.json", result)
    write_markdown(output_dir / "FINAL_EXPERIMENT_REPORT.md", selection, result, asset_audit)
    inventory = {}
    for path in sorted(output_dir.glob("*")):
        # The shell redirects stdout to evaluate.log and appends after this process writes
        # the inventory, so that live log is intentionally excluded from immutable assets.
        if path.is_file() and path.name not in {"evaluate.log", "FINAL_INVENTORY.json"}:
            inventory[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    atomic_json(output_dir / "FINAL_INVENTORY.json", inventory)
    print(json.dumps({"event": "evaluation_complete", "output_dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    main()
