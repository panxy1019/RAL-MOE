from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


HORIZONS = (1, 4, 8, 16, 56, 128)
GAIN_HORIZONS = (1, 4, 8, 16, 56)
SELECTION_SCALES = (0.005, 0.01, 0.02, 0.05)
GAIN_METRICS = (
    "normalized_state",
    "raw_velocity",
    "raw_pressure",
    "physical_velocity",
    "physical_pressure_gauged",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "worst": float(np.max(array)),
    }


def is_selection_scale(value: float) -> bool:
    return any(bool(np.isclose(value, scale, rtol=0.0, atol=1e-7)) for scale in SELECTION_SCALES)


def selected_gain_summary(records: list[dict], model: str) -> dict:
    result = {}
    for horizon in GAIN_HORIZONS:
        result[f"k{horizon}"] = {}
        rows = [
            row for row in records
            if row["model"] == model and row["horizon"] == horizon
            and is_selection_scale(row["scale"])
        ]
        for metric in GAIN_METRICS:
            valid = [row for row in rows if row["metrics"][metric]["gain_terminal"] is not None]
            values = [row["metrics"][metric]["gain_terminal"] for row in valid]
            stats = quantiles(values)
            if valid:
                worst = max(valid, key=lambda row: row["metrics"][metric]["gain_terminal"])
                stats["worst_case"] = {
                    key: worst[key]
                    for key in ("re", "scale", "direction_id", "direction_category", "window_id")
                }
            result[f"k{horizon}"][metric] = stats
    return result


def selected_gain_by_re(records: list[dict], model: str) -> dict:
    result = {}
    re_names = sorted({row["re"] for row in records if row["model"] == model})
    for re_name in re_names:
        result[re_name] = {}
        for horizon in GAIN_HORIZONS:
            result[re_name][f"k{horizon}"] = {}
            rows = [
                row for row in records
                if row["model"] == model and row["re"] == re_name and row["horizon"] == horizon
                and is_selection_scale(row["scale"])
            ]
            for metric in GAIN_METRICS:
                values = [
                    row["metrics"][metric]["gain_terminal"] for row in rows
                    if row["metrics"][metric]["gain_terminal"] is not None
                ]
                result[re_name][f"k{horizon}"][metric] = quantiles(values)
    return result


def fixed_point_metrics(s3, evaluator, bank: dict[str, np.ndarray]) -> dict:
    exp = s3.exp
    names = sorted(set(bank["re_name"].astype(str).tolist()))
    by_re = {}
    for name in names:
        mask = bank["re_name"].astype(str) == name
        fixed_id = int(bank["fixed_id"][np.flatnonzero(mask)[0]])
        fixed = torch.tensor([fixed_id], dtype=torch.long, device=exp.device)
        a_star, b_star = exp.tensor("a", fixed), exp.tensor("b", fixed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=exp.amp_dtype):
            pa, pb = s3.free_rollout(fixed, a_star, b_star, 56)
        item = {}
        for horizon in (1, 16, 56):
            pred_a, pred_b = pa[horizon - 1 : horizon].float(), pb[horizon - 1 : horizon].float()
            raw_u = float((torch.linalg.vector_norm(pred_a - a_star) / (torch.linalg.vector_norm(a_star) + 1e-8)).cpu())
            raw_p = float((torch.linalg.vector_norm(pred_b - b_star) / (torch.linalg.vector_norm(b_star) + 1e-8)).cpu())
            phy_u = evaluator.physical_relative(
                s3.finalizer, pred_a, a_star[None], exp.cache["velocity_pod_basis"],
                exp.cache["velocity_mean"], s3.velocity_weight,
            )
            phy_p = evaluator.physical_relative(
                s3.finalizer, pred_b, b_star[None], exp.cache["pressure_pod_basis"],
                exp.cache["pressure_mean"], s3.pressure_weight,
            )
            item[f"k{horizon}"] = {
                "coefficient_space": {"velocity_relative_l2": raw_u, "pressure_relative_l2": raw_p},
                "physical_reconstruction_area_weighted": {
                    "velocity_relative_l2": phy_u, "pressure_relative_l2": phy_p,
                },
            }
        by_re[name] = item
    overall = {}
    for horizon in (1, 16, 56):
        overall[f"k{horizon}"] = {}
        for space in ("coefficient_space", "physical_reconstruction_area_weighted"):
            overall[f"k{horizon}"][space] = {
                field: max(item[f"k{horizon}"][space][field] for item in by_re.values())
                for field in ("velocity_relative_l2", "pressure_relative_l2")
            }
    return {"by_re": by_re, "overall_worst": overall}


def table(headers, rows) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]


def pct(value) -> str:
    return "—" if value is None else f"{100 * float(value):.4f}%"


def num(value, digits=3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def make_report(raw: dict, path: Path) -> None:
    selection = raw["selection"]
    heldout = raw["heldout"]
    decision = raw["decision"]
    lines = [
        "# V17 S4-Steady Anchor-Only 最终客观实验报告",
        "",
        "## 执行与选择",
        "",
        f"训练按用户指令停止于最后记录 optimizer step {selection['training_stop']['final_logged_optimizer_step']}；最后完整 validation 为 step {selection['training_stop']['last_completed_validation_step']}。Heldout 在 validation 选择冻结后执行，成功指标产物仅生成一次，且未参与选择。",
        "",
        f"冻结 checkpoint：step {selection['selected']['step']}，SHA256 `{selection['selected']['sha256']}`。该 checkpoint 通过预注册保护门；选择规则为在所有保护门通过点中最小化 validation K16 physical pressure fixed-point residual。",
        "",
        "## Validation 历史",
        "",
    ]
    if raw["fairness"]["execution_attempt_count"] > 1:
        lines += [
            f"执行审计：heldout evaluator 共启动 {raw['fairness']['execution_attempt_count']} 次，首次在全部 rollout 完成后的纯汇总阶段因浮点尺度精确匹配失败，未产生指标或选择；修正为容差匹配后，成功产物仅生成一次。两次均使用同一冻结 checkpoint 与同一 perturbation bank。",
            "",
        ]
    rows = []
    for item in selection["all_validation_candidates"]:
        rows.append([
            item["step"], "PASS" if item["protection"]["pass"] else "FAIL",
            pct(item["k16_pressure_fixed_point_worst"]),
            pct(item["k56_pressure_fixed_point_worst"]),
            num(item["k16_pressure_gain_p95"]),
            ", ".join(item["protection"]["reasons"]) or "—",
        ])
    lines += table(["step", "保护门", "K16 压力 FP worst", "K56 压力 FP worst", "K16 压力 gain P95", "失败原因"], rows)
    lines += ["", "## Heldout clean autonomous rollout：全部测试集", "", "所有 relative L2 均乘 100，以百分数表示。", ""]
    rows = []
    clean = heldout["clean_horizons"]
    for re_name in sorted(clean["k1"]["by_re"]):
        for horizon in GAIN_HORIZONS:
            item = clean[f"k{horizon}"]["by_re"][re_name]
            physical = item["physical_reconstruction_area_weighted"]
            modal = item["coefficient_space"]
            rows.append([
                re_name, f"K{horizon}", item["window_count"], pct(physical["velocity_relative_l2"]),
                pct(physical["pressure_relative_l2"]), pct(modal["velocity_relative_l2"]),
                pct(modal["pressure_relative_l2"]), pct(item["finite_fraction"]),
                item["divergent_windows"], item["first_divergent_step"] or "—",
                num(item["state_norm_max_multiple"]), pct(item["pressure_drift"]), pct(item["fixed_point_residual"]),
            ])
    lines += table([
        "Heldout Re", "Horizon", "窗口", "物理速度", "物理压力", "Raw POD 速度", "Raw POD 压力",
        "finite", "发散窗口", "首次异常", "状态范数倍数", "压力 drift", "轨迹 FP residual",
    ], rows)
    lines += ["", "K128 按原数据连续窗口合同如实报告为不可用，不进行外推填补。", "", "## Heldout 固定点残差：全部测试集", ""]
    rows = []
    for re_name, item in heldout["fixed_point"]["by_re"].items():
        for horizon in (1, 16, 56):
            p = item[f"k{horizon}"]["physical_reconstruction_area_weighted"]
            c = item[f"k{horizon}"]["coefficient_space"]
            rows.append([re_name, f"K{horizon}", pct(p["velocity_relative_l2"]), pct(p["pressure_relative_l2"]), pct(c["velocity_relative_l2"]), pct(c["pressure_relative_l2"])])
    lines += table(["Heldout Re", "Horizon", "物理速度 FP", "物理压力 FP", "Raw POD 速度 FP", "Raw POD 压力 FP"], rows)
    lines += ["", "## 四空间严格 paired gain", "", "以下统计使用 0.5%、1%、2%、5% 预注册选择尺度；分子为 perturbed prediction 与 clean prediction 的同空间距离，分母为同空间实际初始扰动范数。", ""]
    gain = heldout["paired_gain"]["selection_summary"]
    rows = []
    labels = {
        "normalized_state": "Normalized POD state", "raw_velocity": "Raw velocity POD",
        "raw_pressure": "Raw pressure POD", "physical_velocity": "Area-weighted velocity field",
        "physical_pressure_gauged": "Gauged area-weighted pressure field",
    }
    for horizon in GAIN_HORIZONS:
        for metric in GAIN_METRICS:
            s = gain[f"k{horizon}"][metric]
            rows.append([f"K{horizon}", labels[metric], s.get("count", 0), num(s.get("median")), num(s.get("p90")), num(s.get("p95")), num(s.get("p99")), num(s.get("worst"))])
    lines += table(["Horizon", "度量空间", "N", "median", "P90", "P95", "P99", "worst"], rows)
    lines += ["", "### 每个 Heldout Re 的 physical pressure paired gain", ""]
    rows = []
    for re_name, item in heldout["paired_gain"]["selection_by_re"].items():
        for horizon in (16, 56):
            s = item[f"k{horizon}"]["physical_pressure_gauged"]
            rows.append([re_name, f"K{horizon}", s["count"], num(s["median"]), num(s["p90"]), num(s["p95"]), num(s["worst"])])
    lines += table(["Heldout Re", "Horizon", "N", "median", "P90", "P95", "worst"], rows)
    lines += ["", "## 与 S3-A、S3-B 的同口径对照", ""]
    rows = []
    for model, summary in raw["comparators"]["selection_summary"].items():
        for horizon in (16, 56):
            s = summary[f"k{horizon}"]["physical_pressure_gauged"]
            rows.append([model, f"K{horizon}", num(s["median"]), num(s["p90"]), num(s["p95"]), num(s["worst"])])
    lines += table(["模型", "Horizon", "pressure gain median", "P90", "P95", "worst"], rows)
    lines += ["", "### Clean physical worst 与 fixed-point 同口径对照", ""]
    rows = []
    for model in ("S3-A", "S3-B", "S4-Steady"):
        clean_source = heldout["clean_horizons"] if model == "S4-Steady" else raw["comparators"]["clean_horizons"][model]
        fixed_source = heldout["fixed_point"] if model == "S4-Steady" else raw["comparators"]["fixed_point"][model]
        for horizon in (1, 16, 56):
            p = clean_source[f"k{horizon}"]["overall_worst"]["physical_reconstruction_area_weighted"]
            if model == "S4-Steady":
                fp = fixed_source["overall_worst"][f"k{horizon}"]["physical_reconstruction_area_weighted"]["pressure_relative_l2"]
            else:
                fp = fixed_source["overall_worst"][f"k{horizon}"]["fixed_point_physical_pressure_worst"]
            rows.append([model, f"K{horizon}", pct(p["velocity_relative_l2"]), pct(p["pressure_relative_l2"]), pct(fp)])
    lines += table(["模型", "Horizon", "clean 物理速度 worst", "clean 物理压力 worst", "压力 FP worst"], rows)
    lines += ["", "## 预注册成功判定", ""]
    checks = decision["checks"]
    lines += table(["检查项", "结果", "测量值"], [[name, "PASS" if value["pass"] else "FAIL", value["detail"]] for name, value in checks.items()])
    lines += [
        "",
        f"最终判定：**{decision['status']}**。{decision['consequence']}",
        "",
        f"非强制目标：{decision['non_required_observations']['heldout_k16_pressure_fp_below_5pct']}。该观察不覆盖预注册强制门槛。",
        "",
        "## 三个概念的客观区分",
        "",
        "1. 低误差有限时域自主预测：由 clean K1/K16/K56 physical relative L2 单独衡量。",
        "2. 数值不发散：由 finite fraction、divergent windows 和首次异常 step 单独衡量；它不等价于局部扰动收缩。",
        "3. 局部扰动收缩/固定点吸引性：由同空间 paired gain 与真实 Steady fixed-point residual 衡量；median<1 不代表 P95/worst 也收缩。",
        "",
        "异常方向、低方差 pressure modes 的贡献、逐方向/尺度原始记录、分母分布与 raw/physical 一致性统计见 `paired_gain_four_space.json`。Heldout 未参与 checkpoint 选择，canonical best 未修改，也未启动其他训练。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--s3-trainer", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--heldout-evaluator", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--training-bank", type=Path, required=True)
    parser.add_argument("--heldout-bank", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--s3-heldout-metrics", type=Path, required=True)
    parser.add_argument("--s3-audit-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    args = parser.parse_args()

    done = args.output_dir / "HELDOUT_EVALUATION_DONE.json"
    started = args.output_dir / "HELDOUT_EVALUATION_STARTED.json"
    if done.exists() or started.exists():
        raise RuntimeError("heldout evaluation marker exists; refusing a second evaluation")
    selection = json.loads(args.selection_manifest.read_text())
    if not selection["selection_completed_before_heldout"] or selection["heldout_used_for_selection"]:
        raise AssertionError("validation selection was not frozen before heldout")
    selected_sha = sha256(args.selected_checkpoint)
    if selected_sha != selection["selected"]["sha256"]:
        raise AssertionError("selected checkpoint SHA256 mismatch")
    bank_sha = sha256(args.heldout_bank)
    atomic_json(started, {
        "started_unix": time.time(), "heldout_used_for_selection": False,
        "selected_checkpoint_step": selection["selected"]["step"],
        "selected_checkpoint_sha256": selected_sha,
        "heldout_bank_sha256": bank_sha,
    })

    audit = load_module("s4_metric_audit", args.audit_script)
    evaluator = load_module("s4_heldout_helpers", args.heldout_evaluator)
    s3_module = load_module("s4_s3_runtime", args.s3_trainer)
    init_args = SimpleNamespace(
        experiment="S3-B", run_dir=str(args.output_dir / "evaluation_runtime"), config=str(args.config),
        trainer=str(args.trainer), finalizer=str(args.finalizer), checkpoint=str(args.source_checkpoint),
        bank=str(args.training_bank), validation_lock=str(args.output_dir / "heldout.lock"),
        learning_rate=args.learning_rate,
    )
    s3 = s3_module.S3(init_args)
    s3.init_eval_assets()
    checkpoint = torch.load(args.selected_checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != int(selection["selected"]["step"]):
        raise AssertionError("selected checkpoint step mismatch")
    s3.exp.model.load_state_dict(checkpoint["model"], strict=True)
    s3.exp.model.eval()
    with np.load(args.heldout_bank) as loaded:
        bank = {key: loaded[key] for key in loaded.files}
    fixed_ids = np.unique(bank["fixed_id"].astype(np.int64))
    audit.repair_terminal_dt(s3, fixed_ids)

    clean = {f"k{k}": s3.finalizer.evaluate_horizon(s3.exp, "heldout", k) for k in HORIZONS}
    fixed = fixed_point_metrics(s3, evaluator, bank)
    with np.load(s3.exp.paths["velocity_pod"]) as velocity:
        sqrt_area = torch.as_tensor(velocity["sqrt_point_areas"], dtype=torch.float32, device=s3.exp.device)
    area = sqrt_area.square()
    velocity_weight = torch.cat((sqrt_area, sqrt_area))
    records = []
    groups = []
    re_values = bank["re_name"].astype(str)
    for re_name in sorted(set(re_values.tolist())):
        for scale in sorted(set(bank["scale"][re_values == re_name].astype(float).tolist())):
            groups.append((re_name, scale))
    for re_name, scale in groups:
        mask = (re_values == re_name) & np.isclose(bank["scale"].astype(float), scale)
        fixed_values = bank["fixed_id"][mask].astype(np.int64)
        if len(set(fixed_values.tolist())) != 1:
            raise AssertionError("a Re/scale group must share one fixed window")
        fixed_id = int(fixed_values[0])
        count = int(mask.sum())
        ids = torch.full((count,), fixed_id, dtype=torch.long, device=s3.exp.device)
        a0, b0 = s3.exp.tensor("a", ids), s3.exp.tensor("b", ids)
        du = torch.as_tensor(bank["du"][mask], dtype=torch.float32, device=s3.exp.device)
        dp = torch.as_tensor(bank["dp"][mask], dtype=torch.float32, device=s3.exp.device)
        da, db = float(scale) * s3.exp.avt * du, float(scale) * s3.exp.bvt * dp
        with torch.inference_mode(), torch.autocast("cuda", dtype=s3.exp.amp_dtype):
            clean_a, clean_b = s3.free_rollout(ids, a0, b0, 56)
            pert_a, pert_b = s3.free_rollout(ids, a0 + da, b0 + db, 56)
        calculated = audit.calculate_metrics(
            s3, clean_a, clean_b, pert_a, pert_b, da, db, GAIN_HORIZONS, area, velocity_weight,
        )
        categories = bank["category"][mask].astype(str)
        bank_indices = np.flatnonzero(mask)
        for local_id, sample in enumerate(calculated):
            metadata = {
                "re": re_name, "scale": float(scale), "direction_id": int(local_id),
                "bank_entry_id": int(bank_indices[local_id]), "direction_category": str(categories[local_id]),
                "window_id": fixed_id,
            }
            for horizon in GAIN_HORIZONS:
                records.append(audit.flatten_sample(sample, metadata, "S4-Steady", horizon))
        print(json.dumps({"event": "heldout_group_complete", "re": re_name, "scale": scale}), flush=True)

    audit.MODELS = ("S4-Steady",)
    summary, grouped, _ = audit.summarize_records(records)
    denominator_stats = {}
    for metric in audit.METRICS:
        values = [
            row["metrics"][metric]["initial_actual_norm"] for row in records
            if row["horizon"] == 1 and not row["metrics"][metric]["denominator_zero"]
        ]
        denominator_stats[metric] = quantiles(values)
    paired = {
        "definition": {
            "primary": "terminal paired distance at K divided by same-space actual initial perturbation norm",
            "secondary": "maximum paired gain over steps 1..K",
            "truth_not_used_in_gain": True,
            "epsilon_floor": "none; zero same-space denominator is reported undefined",
            "selection_scales": list(SELECTION_SCALES),
        },
        "records": records,
        "all_scale_summary": summary["S4-Steady"],
        "all_scale_grouped_summary": grouped["S4-Steady"],
        "selection_summary": selected_gain_summary(records, "S4-Steady"),
        "selection_by_re": selected_gain_by_re(records, "S4-Steady"),
        "denominator_distribution": denominator_stats,
        "agreement": {
            "raw_velocity_vs_physical_velocity": audit.agreement(records, "raw_velocity", "physical_velocity"),
            "raw_pressure_vs_gauged_physical_pressure": audit.agreement(records, "raw_pressure", "physical_pressure_gauged"),
            "normalized_pressure_vs_raw_pressure": audit.agreement(records, "normalized_pressure", "raw_pressure"),
        },
        "worst_k16_diagnostics": audit.top_mode_diagnostics(s3, records)["S4-Steady"],
    }

    existing_heldout = json.loads(args.s3_heldout_metrics.read_text())
    existing_audit = json.loads(args.s3_audit_metrics.read_text())
    comparator_records = [row for row in existing_audit["records"] if row["model"] in ("S3-A", "S3-B")]
    comparator_summary = {
        model: selected_gain_summary(comparator_records, model) for model in ("S3-A", "S3-B")
    }
    comparators = {
        "selection_summary": comparator_summary,
        "clean_horizons": {model: existing_heldout["experiments"][model]["horizons"] for model in ("S3-A", "S3-B")},
        "fixed_point": {model: existing_heldout["experiments"][model]["attractivity"] for model in ("S3-A", "S3-B")},
        "checkpoints": {model: existing_audit["checkpoint_audit"][model] for model in ("S3-A", "S3-B")},
    }

    s3b_clean = comparators["clean_horizons"]["S3-B"]
    clean_reasons = []
    for horizon in (1, 16, 56):
        current = clean[f"k{horizon}"]["overall_worst"]
        baseline = s3b_clean[f"k{horizon}"]["overall_worst"]
        for field in ("velocity_relative_l2", "pressure_relative_l2"):
            c = current["physical_reconstruction_area_weighted"][field]
            b = baseline["physical_reconstruction_area_weighted"][field]
            if c > 1.05 * b:
                clean_reasons.append(f"K{horizon} {field}: {100*c:.4f}% > 105% of {100*b:.4f}%")
        if current["finite_fraction"] != 1 or current["divergent_windows"] != 0:
            clean_reasons.append(f"K{horizon} finite/divergence")
    s4_fp = fixed["overall_worst"]["k16"]["physical_reconstruction_area_weighted"]["pressure_relative_l2"]
    s3b_fp = comparators["fixed_point"]["S3-B"]["overall_worst"]["k16"]["fixed_point_physical_pressure_worst"]
    heldout_fp_improvement = (s3b_fp - s4_fp) / s3b_fp
    val_rows = json.loads((args.output_dir / "validation.json").read_text())["history"]
    val0 = next(row for row in val_rows if row["step"] == 0)["fixed_point"]["k16"]["worst_physical_pressure"]
    vals = next(row for row in val_rows if row["step"] == selection["selected"]["step"])["fixed_point"]["k16"]["worst_physical_pressure"]
    val_fp_improvement = (val0 - vals) / val0
    contraction_reasons = []
    for horizon in (16, 56):
        cur = paired["selection_summary"][f"k{horizon}"]["physical_pressure_gauged"]
        base = comparator_summary["S3-B"][f"k{horizon}"]["physical_pressure_gauged"]
        if cur["median"] >= 1:
            contraction_reasons.append(f"K{horizon} median={cur['median']:.4g} >= 1")
        if cur["p95"] > 1.05 * base["p95"]:
            contraction_reasons.append(f"K{horizon} P95={cur['p95']:.4g} > 105% of S3-B {base['p95']:.4g}")
    checks = {
        "validation_fixed_point_improvement_ge_10pct": {"pass": val_fp_improvement >= 0.10, "detail": f"{100*val_fp_improvement:.2f}%"},
        "heldout_fixed_point_improvement_ge_10pct": {"pass": heldout_fp_improvement >= 0.10, "detail": f"{100*heldout_fp_improvement:.2f}% (S4 {100*s4_fp:.4f}% vs S3-B {100*s3b_fp:.4f}%)"},
        "clean_rollout_protection": {"pass": not clean_reasons, "detail": "; ".join(clean_reasons) or "all K1/K16/K56 physical errors within +5%, finite=100%, divergence=0"},
        "pressure_contraction_protection": {"pass": not contraction_reasons, "detail": "; ".join(contraction_reasons) or "K16/K56 median<1 and P95 within +5% of S3-B"},
    }
    success = all(item["pass"] for item in checks.values())
    decision = {
        "status": "S4_SUCCESS_CANDIDATE" if success else "S4_FAILED",
        "checks": checks,
        "consequence": (
            "S4 may be frozen only as the new Steady sub-MoE candidate; canonical best remains unchanged."
            if success else
            "S4 must not replace S3-B; the existing S3-B remains the final Steady sub-MoE."
        ),
        "canonical_best_modified": False,
    }
    below_5 = sum(
        item["k16"]["physical_reconstruction_area_weighted"]["pressure_relative_l2"] < 0.05
        for item in fixed["by_re"].values()
    )
    decision["non_required_observations"] = {
        "heldout_k16_pressure_fp_below_5pct": f"{below_5}/4 heldout Re below 5% (aspirational, not a mandatory gate)"
    }

    configuration = {
        "schema_version": 1, "selected_checkpoint": {"path": str(args.selected_checkpoint), "sha256": selected_sha, "step": int(checkpoint["step"])},
        "runtime_base_checkpoint": {"path": str(args.source_checkpoint), "sha256": sha256(args.source_checkpoint)},
        "training_bank": {"path": str(args.training_bank), "sha256": sha256(args.training_bank)},
        "heldout_bank": {"path": str(args.heldout_bank), "sha256": bank_sha},
        "config": {"path": str(args.config), "sha256": sha256(args.config)},
        "ru": s3.exp.ru, "rp": s3.exp.rp, "amp_dtype": str(s3.exp.amp_dtype),
        "scalers": {"velocity": audit.scaler_stats(s3.exp.avt), "pressure": audit.scaler_stats(s3.exp.bvt)},
        "basis_and_pressure_gauge": audit.basis_audit(s3),
        "heldout_re": sorted(set(re_values.tolist())), "horizons": list(HORIZONS),
        "perturbation_scales": sorted(set(float(x) for x in bank["scale"].tolist())),
        "directions_per_re_scale": 20,
    }
    heldout = {"clean_horizons": clean, "fixed_point": fixed, "paired_gain": paired}
    failed_attempt = args.output_dir / "HELDOUT_EVALUATION_ATTEMPT1_FAILED.json"
    fairness = {
        "schema_version": 1, "heldout_used_for_selection": False, "heldout_evaluation_count": 1,
        "execution_attempt_count": 2 if failed_attempt.exists() else 1,
        "successful_artifact_count": 1,
        "failed_attempt_before_metrics_or_decision": failed_attempt.exists(),
        "validation_selection_frozen_before_heldout": True, "canonical_best_modified": False,
        "relative_errors_stored_as_fraction": True, "report_display_multiplier": 100,
        "four_space_gain": True, "same_space_numerator_denominator": True,
        "perturbed_to_truth_not_used_as_paired_gain": True, "pressure_gauge": "subtract_area_mean_per_snapshot",
    }
    raw = {
        "schema_version": 1, "selection": selection, "configuration": configuration,
        "heldout": heldout, "comparators": comparators, "decision": decision, "fairness": fairness,
    }
    atomic_json(args.output_dir / "configuration.json", configuration)
    atomic_json(args.output_dir / "heldout_metrics.json", {"clean_horizons": clean, "fixed_point": fixed})
    atomic_json(args.output_dir / "paired_gain_four_space.json", paired)
    atomic_json(args.output_dir / "fairness_manifest.json", fairness)
    atomic_json(args.output_dir / "raw_metrics.json", raw)
    make_report(raw, args.output_dir / "experiment_report.md")
    atomic_json(done, {
        "finished_unix": time.time(), "heldout_evaluation_count": 1,
        "selected_checkpoint_step": int(checkpoint["step"]), "selected_checkpoint_sha256": selected_sha,
        "decision": decision["status"], "raw_metrics_sha256": sha256(args.output_dir / "raw_metrics.json"),
    })
    inventory = {"schema_version": 1, "artifacts": {}}
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_inventory.json":
            inventory["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    atomic_json(args.output_dir / "artifact_inventory.json", inventory)
    del s3
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"event": "s4_finalization_complete", "decision": decision["status"], "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
