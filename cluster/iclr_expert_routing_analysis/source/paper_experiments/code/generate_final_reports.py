#!/usr/bin/env python3
"""Generate traceable final ablation reports from frozen project artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(
    "/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE"
)
REPORTS = ROOT / "paper_experiments/reports"
MANUSCRIPT = ROOT / "paper_experiments/manuscript"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figure_data"


P = {
    "steady_vanilla": ROOT
    / "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/steady/vanilla-fnn/frozen_user_stop_step6200_20260724/evaluation/test/metrics.json",
    "steady_vanilla_freeze": ROOT
    / "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/steady/vanilla-fnn/FROZEN_SELECTION.json",
    "hopf_vanilla_validation": ROOT
    / "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/hopf_v6/hopf/vanilla-fnn/evaluation/validation/metrics.json",
    "hopf_vanilla_freeze": ROOT
    / "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/hopf_v6/hopf/vanilla-fnn/FROZEN_SELECTION.json",
    "periodic_vanilla": ROOT
    / "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/periodic_v4/periodic/vanilla-fnn/evaluation/test/native/periodic_r32_multihorizon_evaluation.json",
    "periodic_vanilla_freeze": ROOT
    / "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/periodic_v4/periodic/vanilla-fnn/FROZEN_SELECTION.json",
    "steady_data": ROOT
    / "paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/steady/data-only/evaluation/test/metrics.json",
    "hopf_data": ROOT
    / "paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/hopf/data-only/evaluation/test/metrics.json",
    "periodic_data": ROOT
    / "paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/periodic/data-only/evaluation/test/metrics.json",
    "steady_proposed": ROOT
    / "steady_specialist_v1/portability_runs/best_checkpoint_evaluation_retry/heldout_metrics.json",
    "hopf_proposed": ROOT
    / "Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/heldout_metrics.json",
    "hopf_attractor": ROOT
    / "Hopf/migrated_h4_expanded/native_attractor_audit_20260723_v1/results/HOPF_NATIVE_ATTRACTOR_AUDIT.json",
    "periodic_proposed": ROOT
    / "periodic_specialist_r32/reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json",
    "global": ROOT
    / "V16_1_SteadyPressureAnchor32/reproduction/heldout_eval_frozen_checkpoint_20260723/V16_1_SteadyPressureAnchor32_ru32_rp32_reproduced_metrics.json",
    "e2_summary": ROOT
    / "trajectory_router_e1_e2_e3_20260722/E1_E2_E3_TOP1_SUMMARY.json",
    "e2_report": ROOT
    / "trajectory_router_e1_e2_e3_20260722/E1_E2_E3_TOP1_REPORT.md",
    "t2c_eval": ROOT
    / "top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/final_test_evaluation/EVALUATION_REPORT.json",
    "t2c_oracle": ROOT
    / "top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/supplements/per_re_matched_oracle_v4/PER_RE_MATCHED_ORACLE_REPORT.json",
    "t2c_attractor": ROOT
    / "top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/supplements/t2c_attractor_v2/T2C_ATTRACTOR_ANALYSIS.json",
    "hp_report": ROOT
    / "top2_boundary_experiments_20260722/one_sided_recovery_hp_20260723_v1/H_P_RECOVERY_REPORT_20260723.md",
}


def load(key: str):
    return json.loads(P[key].read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def f(value) -> str:
    if value is None:
        return "NA"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.3f}"
    if abs(value) >= 1:
        return f"{value:.5f}"
    return f"{value:.6g}"


def source(key: str) -> str:
    return f"`{P[key]}`"


def steady_metrics(key: str) -> dict:
    payload = load(key)
    horizons = payload.get("clean_horizons", payload.get("horizons"))
    result = {}
    for horizon in ("k1", "k4", "k8", "k16", "k56"):
        rows = list(horizons[horizon]["by_re"].values())
        pairs = [
            row["physical_reconstruction_area_weighted"] for row in rows
        ]
        result[horizon] = {
            "u": mean(row["velocity_relative_l2"] for row in pairs),
            "p": mean(row["pressure_relative_l2"] for row in pairs),
            "joint": mean(
                0.5 * (row["velocity_relative_l2"] + row["pressure_relative_l2"])
                for row in pairs
            ),
            "worst": max(
                0.5 * (row["velocity_relative_l2"] + row["pressure_relative_l2"])
                for row in pairs
            ),
            "finite": min(row["finite_fraction"] for row in rows),
            "divergent": sum(row["divergent_windows"] for row in rows),
        }
    return result


def data_metrics(key: str) -> dict:
    payload = load(key)
    return {
        horizon: {
            "u": row["velocity_mean_over_windows"],
            "p": row["pressure_mean_over_windows"],
            "joint": row["joint_mean_over_windows"],
            "worst": row["worst_window_joint"],
            "finite": row["finite_fraction"],
            "divergent": row["divergent_windows"],
        }
        for horizon, row in payload["overall"].items()
    }


def cycle_metrics(key: str, hopf: bool = False) -> dict:
    payload = load(key)
    if hopf:
        rows = list(next(iter(payload["experiments"].values()))["by_re"].values())
        uk = "velocity_area_weighted_physical_relative_l2"
        pk = "pressure_area_weighted_physical_relative_l2"
    else:
        rows = payload["results"]
        uk = "velocity_area_weighted_l2"
        pk = "pressure_area_weighted_l2"
    result = {}
    for horizon in ("1", "4", "8", "16", "48", "56"):
        entries = [row["horizons"][horizon] for row in rows if horizon in row["horizons"]]
        if not entries:
            continue
        result[f"k{horizon}"] = {
            "u": mean(row[uk] for row in entries),
            "p": mean(row[pk] for row in entries),
            "joint": mean(0.5 * (row[uk] + row[pk]) for row in entries),
            "worst": max(0.5 * (row[uk] + row[pk]) for row in entries),
            "finite": min(row["finite_fraction"] for row in entries),
            "divergent": sum(row["divergent_windows"] for row in entries),
        }
    return result


def hopf_k56_from_audit() -> dict:
    payload = load("hopf_attractor")
    rows = [row for row in payload["per_Re"] if row["split"] == "heldout"]
    entries = [row["horizons"]["56"] for row in rows]
    return {
        "u": mean(row["velocity_area_weighted_physical_relative_l2"] for row in entries),
        "p": mean(row["pressure_area_weighted_physical_relative_l2"] for row in entries),
        "joint": mean(
            0.5
            * (
                row["velocity_area_weighted_physical_relative_l2"]
                + row["pressure_area_weighted_physical_relative_l2"]
            )
            for row in entries
        ),
        "worst": max(
            0.5
            * (
                row["velocity_area_weighted_physical_relative_l2"]
                + row["pressure_area_weighted_physical_relative_l2"]
            )
            for row in entries
        ),
        "finite": min(row["finite_fraction"] for row in entries),
        "divergent": sum(row["divergent_windows"] for row in entries),
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    text = ["| " + " | ".join(headers) + " |"]
    text.append("|" + "|".join("---" for _ in headers) + "|")
    text.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(text)


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def build_records() -> list[dict]:
    records = []
    configurations = [
        ("Steady", "Vanilla-FNN-MoE", steady_metrics("steady_vanilla"), "steady_vanilla", "EARLY_STOPPED_STEP_6200"),
        ("Steady", "DataOnly-MoE", data_metrics("steady_data"), "steady_data", "DONE"),
        ("Steady", "Proposed Specialist MoE", steady_metrics("steady_proposed"), "steady_proposed", "DONE"),
        ("Hopf", "Vanilla-FNN-MoE", {}, "hopf_vanilla_validation", "BLOCKED_VALIDATION_NONFINITE"),
        ("Hopf", "DataOnly-MoE", data_metrics("hopf_data"), "hopf_data", "DONE_WITH_DIVERGENT_WINDOWS"),
        ("Hopf", "Proposed Specialist MoE", cycle_metrics("hopf_proposed", hopf=True), "hopf_proposed", "DONE"),
        ("Periodic", "Vanilla-FNN-MoE", cycle_metrics("periodic_vanilla"), "periodic_vanilla", "DONE"),
        ("Periodic", "DataOnly-MoE", data_metrics("periodic_data"), "periodic_data", "DONE"),
        ("Periodic", "Proposed Specialist MoE", cycle_metrics("periodic_proposed"), "periodic_proposed", "DONE"),
    ]
    configurations[5][2]["k56"] = hopf_k56_from_audit()
    for regime, method, metrics, key, status in configurations:
        long_key = "k56" if "k56" in metrics else ("k48" if "k48" in metrics else None)
        records.append(
            {
                "regime": regime,
                "method": method,
                "metrics": metrics,
                "long_key": long_key,
                "source_key": key,
                "status": status,
            }
        )
    return records


def table_a1(records: list[dict]) -> tuple[str, list[list[object]]]:
    headers = [
        "Regime", "Method", "u K1", "p K1", "u K16", "p K16",
        "Long", "u long", "p long", "joint long", "worst long",
        "finite", "divergent", "status",
    ]
    csv_rows = []
    md_rows = []
    for record in records:
        m = record["metrics"]
        long_key = record["long_key"]
        k1, k16 = m.get("k1"), m.get("k16")
        long = m.get(long_key) if long_key else None
        row = [
            record["regime"], record["method"],
            None if not k1 else k1["u"], None if not k1 else k1["p"],
            None if not k16 else k16["u"], None if not k16 else k16["p"],
            long_key, None if not long else long["u"], None if not long else long["p"],
            None if not long else long["joint"], None if not long else long["worst"],
            None if not long else long["finite"], None if not long else long["divergent"],
            record["status"],
        ]
        csv_rows.append(row)
        md_rows.append([f(x) if isinstance(x, (int, float)) else ("NA" if x is None else str(x)) for x in row])
    return markdown_table(headers, md_rows), csv_rows


def build_b2() -> tuple[str, list[list[object]]]:
    attractor = load("t2c_attractor")
    oracle = load("t2c_oracle")["per_Re"]
    entries = attractor.get("per_Re", attractor.get("by_Re", attractor))
    rows = []
    if isinstance(entries, list):
        iterator = [(str(row["Re"]), row) for row in entries]
    else:
        iterator = list(entries.items())
    for re_key, payload in iterator:
        re_value = float(re_key)
        for method_key, method_name in (("E2_Top1", "E2 Top-1"), ("T2_C", "T2-C")):
            values = payload[method_key]
            weights = oracle[f"{re_value:g}" if f"{re_value:g}" in oracle else f"{re_value:.6f}"]["T2_C_weights"]
            rows.append(
                [
                    re_value,
                    method_name,
                    values["tail_center_field_error"]["mean"],
                    values["tail_amplitude_absolute_error"]["mean"],
                    values["tail_orbit_geometry_error"]["mean"],
                    values["tail_increment_field_error"]["mean"],
                    values["terminal_increment_field_error"]["mean"],
                    weights["alpha_S_mean"] if method_name == "T2-C" else None,
                    weights["alpha_H_mean"] if method_name == "T2-C" else None,
                ]
            )
    headers = ["Re", "Method", "Center", "Amplitude abs.", "Orbit", "Tail inc.", "Terminal inc.", "alpha_S", "alpha_H"]
    md = markdown_table(
        headers,
        [[f(x) if isinstance(x, (int, float)) else ("NA" if x is None else x) for x in row] for row in rows],
    )
    return md, rows


def main() -> None:
    for path in P.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    records = build_records()
    a1_md, a1_rows = table_a1(records)
    a1_headers = [
        "regime", "method", "u_k1", "p_k1", "u_k16", "p_k16",
        "long_horizon", "u_long", "p_long", "joint_long", "worst_long",
        "finite_fraction", "divergent_windows", "status",
    ]
    write_csv(TABLES / "TABLE_A1_SPECIALIST_ABLATIONS.csv", a1_headers, a1_rows)

    b2_md, b2_rows = build_b2()
    write_csv(
        TABLES / "TABLE_B2_SH_ATTRACTOR.csv",
        ["Re", "method", "center", "amplitude_abs", "orbit", "tail_increment", "terminal_increment", "alpha_S", "alpha_H"],
        b2_rows,
    )

    t2c = load("t2c_eval")
    t2c_method = t2c["routes"]["T2-C_LearnedConvexCorrection_FieldBlend"]
    e2_pair = t2c["baselines"]["E2_pair_Top1"]
    global_metrics = load("global")["aggregate_metrics"]
    a2_rows = [
        [
            "Global MoE", "all available heldout Re", "modal, not unified physical K56",
            global_metrics["one_step_a_l2"]["mean"], global_metrics["one_step_b_l2"]["mean"],
            global_metrics["rollout_a_l2"]["mean"], global_metrics["rollout_b_l2"]["mean"],
            "Not numerically commensurate with S-H cache",
        ],
        [
            "E2 Top-1", "S-H boundary", "area-weighted physical joint",
            None, None, e2_pair["horizons"]["K56"]["joint_mean"], e2_pair["joint_all_worst"],
            "E2 selects S on this S-native cache",
        ],
        [
            "Multi-chart + T2-C", "S-H boundary", "area-weighted physical joint",
            None, None, t2c_method["horizons"]["K56"]["joint_mean"], t2c_method["joint_all_worst"],
            "window-conditioned convex field blend",
        ],
        [
            "Multi-chart admissibility", "H-P boundary", "validation admissibility",
            None, None, None, None, "H inadmissible; forced P-only / hard routing",
        ],
    ]
    a2_headers = ["System", "Region", "Metric contract", "one-step u", "one-step p", "long/rollout", "worst", "Interpretation"]
    write_csv(TABLES / "TABLE_A2_FULL_SYSTEM.csv", a2_headers, a2_rows)
    a2_md = markdown_table(
        a2_headers,
        [[f(x) if isinstance(x, (int, float)) else ("NA" if x is None else str(x)) for x in row] for row in a2_rows],
    )

    hopf_audit = load("hopf_attractor")
    periodic_proposed = load("periodic_proposed")
    periodic_vanilla = load("periodic_vanilla")
    data_attr = {
        regime: load(f"{regime.lower()}_data")["attractor"]
        for regime in ("Steady", "Hopf", "Periodic")
    }
    b1_rows = [
        ["Steady", "Vanilla-FNN-MoE", "No preregistered strict binary count", "K56 finite; early-stopped checkpoint"],
        ["Steady", "DataOnly-MoE", "Generic tail diagnostics only", f"center={f(data_attr['Steady']['tail_center_modal_relative_l2'])}, amp={f(data_attr['Steady']['tail_amplitude_relative_error'])}"],
        ["Steady", "Proposed Specialist MoE", "No preregistered strict binary count", "K56 finite; fixed-point metrics available"],
        ["Hopf", "Vanilla-FNN-MoE", "Not evaluated on test", "validation K16 non-finite; fail closed"],
        ["Hopf", "DataOnly-MoE", "No certified Hopf strict conjunction", f"center={f(data_attr['Hopf']['tail_center_modal_relative_l2'])}, amp={f(data_attr['Hopf']['tail_amplitude_relative_error'])}; divergent windows present"],
        ["Hopf", "Proposed Specialist MoE", "3/29 train; 0/2 validation; 0/3 heldout", "strict train passes: Re=49.3, 49.6, 50.0"],
        ["Periodic", "Vanilla-FNN-MoE", f"{periodic_vanilla['heldout_preserved_count']}/4 heldout", "native K48 cycle contract"],
        ["Periodic", "DataOnly-MoE", "No certified phase/frequency conjunction", f"center={f(data_attr['Periodic']['tail_center_modal_relative_l2'])}, amp={f(data_attr['Periodic']['tail_amplitude_relative_error'])}"],
        ["Periodic", "Proposed Specialist MoE", f"{periodic_proposed['heldout_preserved_count']}/4 heldout", "native K48 cycle contract"],
    ]
    b1_headers = ["Regime", "Method", "Strict attractor result", "Notes"]
    write_csv(TABLES / "TABLE_B1_SPECIALIST_ATTRACTORS.csv", b1_headers, b1_rows)
    b1_md = markdown_table(b1_headers, b1_rows)

    per_re = load("t2c_oracle")["per_Re"]
    figure_rows = []
    for key, row in per_re.items():
        figure_rows.append(
            [
                row["Re"],
                row["S_only"]["overall_mean"],
                row["H_only"]["overall_mean"],
                row["E2_Top1"]["overall_mean"],
                row["T2_C"]["overall_mean"],
                row["T2_C"]["K56_mean"],
                row["T2_C_weights"]["alpha_S_mean"],
                row["T2_C_weights"]["alpha_H_mean"],
                row["per_Re_shared_constant_oracle"]["overall_mean"],
            ]
        )
    write_csv(
        FIGURES / "SH_BOUNDARY_PER_RE.csv",
        ["Re", "S_only", "H_only", "E2_Top1", "T2_C", "T2_C_K56", "alpha_S_mean", "alpha_H_mean", "matched_oracle"],
        figure_rows,
    )
    write_csv(
        FIGURES / "SPECIALIST_LONG_HORIZON.csv",
        ["regime", "method", "horizon", "velocity", "pressure", "joint", "worst", "finite", "divergent"],
        [
            [
                record["regime"], record["method"], record["long_key"],
                *(record["metrics"][record["long_key"]][key] for key in ("u", "p", "joint", "worst", "finite", "divergent")),
            ]
            for record in records
            if record["long_key"]
        ],
    )
    write_csv(
        FIGURES / "HOPF_STRICT_ATTRACTOR_PER_RE.csv",
        ["split", "Re", "strict_pass", "u_K56", "p_K56", "rms_amp_error", "p2p_amp_error", "frequency_error", "phase_drift_cycles", "orbit_distance"],
        [
            [
                row["split"], row["Re"], row["attractor_preserved_K56"],
                row["horizons"]["56"]["velocity_area_weighted_physical_relative_l2"],
                row["horizons"]["56"]["pressure_area_weighted_physical_relative_l2"],
                row["attractor_K56"]["rms_amplitude_error"],
                row["attractor_K56"]["peak_to_peak_amplitude_error"],
                row["attractor_K56"]["frequency_relative_error"],
                row["attractor_K56"]["terminal_phase_drift_cycles"],
                row["attractor_K56"]["normalized_orbit_distance"],
            ]
            for row in hopf_audit["per_Re"]
        ],
    )

    status_rows = [
        [r["regime"], r["method"], r["status"], r["source_key"], str(P[r["source_key"]])]
        for r in records
    ]
    status_md = markdown_table(
        ["Regime", "Method", "Status", "Source key", "Result path"], status_rows
    )
    status_text = f"""# Final experiment status

Generated mechanically from frozen artifacts. No model selection or threshold was changed.

{status_md}

Important execution facts:

- Exactly one fixed seed was used per new ablation; no mean±std across seeds exists.
- Steady Vanilla is a user-requested early stop at optimizer step 6200 and is not a strict qualified checkpoint.
- Hopf Vanilla failed the validation long-rollout contract; test access was therefore blocked.
- Periodic Vanilla and all three DataOnly jobs reached terminal evaluation states.
- Proposed specialist results are the previously frozen native evaluations.

Source: {source('steady_vanilla_freeze')}; {source('hopf_vanilla_validation')}; {source('periodic_vanilla_freeze')}.
"""
    (REPORTS / "EXPERIMENT_STATUS.md").write_text(status_text)

    ablation_text = f"""# Specialist ablation objective report

## Table A1 — transient physical-field prediction

All errors are relative physical-field errors. Values are means over the Re-level summaries exposed by each frozen native evaluator. `long` is K56 except Periodic Proposed, whose frozen native result ends at K48. A finite fraction of 1 does not imply non-divergence: the DataOnly evaluator separately flags trajectories exceeding its 10× norm rule.

{a1_md}

## Objective interpretation

- Steady: the early-stopped Vanilla checkpoint has lower K56 mean joint error than the frozen Proposed Steady checkpoint ({f(steady_metrics('steady_vanilla')['k56']['joint'])} vs {f(steady_metrics('steady_proposed')['k56']['joint'])}). This ablation therefore does not support a blanket claim that the proposed expert architecture is superior in Steady long-horizon field error. DataOnly has finite arithmetic outputs but severe pressure error and three divergent K56 windows.
- Hopf: Proposed has low physical-field error and zero divergence on the native held-out split. DataOnly is finite in the NaN/Inf sense but has 39 divergent K56 windows and large pressure error. Vanilla cannot be compared on test because validation K16 contains non-finite trajectories.
- Periodic: Proposed improves over Vanilla at K1/K16/K48 and preserves 3/4 held-out attractors versus 1/4 for Vanilla. DataOnly has competitive short-horizon field error but lacks the certified phase/frequency/orbit contract.

These statements are single-seed observations, not uncertainty estimates.

Primary sources: {source('steady_vanilla')}; {source('steady_data')}; {source('steady_proposed')}; {source('hopf_data')}; {source('hopf_proposed')}; {source('periodic_vanilla')}; {source('periodic_data')}; {source('periodic_proposed')}.
"""
    (REPORTS / "SPECIALIST_ABLATION_REPORT.md").write_text(ablation_text)
    (REPORTS / "SPECIALIST_ABLATION_OBJECTIVE_REPORT.md").write_text(ablation_text)

    full_system_text = f"""# Full-system comparison

## Table A2

{a2_md}

The table is deliberately partial. The available Global MoE artifact reports modal one-step and rollout errors, whereas the S–H sealed cache reports area-weighted physical-field errors at fixed horizons. These numbers are not placed in a single ranking. A fresh common-population physical-field cache would be required for a defensible all-region numerical ranking.

Within the sealed S–H cache, T2-C reduces K56 mean joint error from {f(e2_pair['horizons']['K56']['joint_mean'])} (E2/S-only) to {f(t2c_method['horizons']['K56']['joint_mean'])}, but the worst-window statistic is {f(t2c_method['joint_all_worst'])} and T2-C is not better at every Re. At H–P, the Hopf expert failed the development K56 admissibility gate (0/6 validation trajectories finite), so the unified system correctly collapses to P-only/native-time hard routing.

Sources: {source('global')}; {source('t2c_eval')}; {source('hp_report')}.
"""
    (REPORTS / "FULL_SYSTEM_COMPARISON.md").write_text(full_system_text)

    attractor_text = f"""# Attractor and asymptotic-dynamics analysis

## Table B1 — specialist attractor results

{b1_md}

The three strict Hopf train cases are Re=49.3, 49.6, and 50.0. They are in-sample diagnostics because these Re values participated in training; they must not be described as held-out generalization. The frozen Hopf specialist is finite at every audited native K56 trajectory, but strict attractor preservation is 3/29 train, 0/2 validation, and 0/3 heldout.

## Table B2 — S–H boundary

{b2_md}

At Re=43.50, T2-C has worse full-window physical-field error than E2 ({f(per_re['43.5']['T2_C']['overall_mean'])} vs {f(per_re['43.5']['E2_Top1']['overall_mean'])}) while improving all reported tail-attractor diagnostics. This is a genuine transient–attractor discrepancy and is retained as a counterexample. At Re=43.90, T2-C strongly improves both the full-window error and the tail diagnostics. Phase/frequency are not reported for this S-native cache because no certified complete-cycle observable exists.

Sources: {source('hopf_attractor')}; {source('periodic_proposed')}; {source('periodic_vanilla')}; {source('t2c_attractor')}; {source('t2c_oracle')}.
"""
    (REPORTS / "ATTRACTOR_ANALYSIS.md").write_text(attractor_text)
    (REPORTS / "SPECIALIST_ATTRACTOR_RESULTS.md").write_text(attractor_text)

    routing_rows = []
    for key, row in per_re.items():
        weights = row["T2_C_weights"]
        routing_rows.append(
            [
                key,
                f(weights["alpha_S_mean"]),
                f(weights["alpha_S_min"]),
                f(weights["alpha_S_max"]),
                f(weights["alpha_H_mean"]),
                f(row["T2_C"]["overall_mean"]),
                f(row["E2_Top1"]["overall_mean"]),
                f(row["differences"]["T2_C_minus_shared_constant_oracle_overall_mean"]),
            ]
        )
    routing_text = f"""# Routing diagnostics

E2 is a trajectory-level Re-only `temporal-consistent regime router`; it is not evidence of learned physical S→H→P transitions. Its frozen test classification has balanced accuracy=1.0, macro-F1=1.0, NLL=0.183816, Brier=0.093280, ECE-10=0.150093, and minimum Top-1 margin=0.004595.

{markdown_table(['Re','alpha_S mean','min','max','alpha_H mean','T2-C','E2','gap to matched oracle'], routing_rows)}

T2-C uses Top-2 on all S–H cache windows. It beats E2 on 2/3 test Re values, not all three. RiskPrediction and LookAhead produced the same final predictions as their shared convex gate; removing their critic/lookahead modules did not change output error, so they provide no demonstrated incremental value here.

H–P Top-2 is disabled by the admissibility mask. This is a method outcome, not a missing favorable experiment.

Sources: {source('e2_report')}; {source('t2c_eval')}; {source('t2c_oracle')}; {source('hp_report')}.
"""
    (REPORTS / "ROUTING_DIAGNOSTICS.md").write_text(routing_text)

    failures_text = f"""# Failures, limitations, and unsupported claims

1. **Steady Vanilla was early-stopped.** The final ablation checkpoint is step 6200, before the preregistered 11000-step maximum, and it did not satisfy the trainer's strict contraction qualification. Its test result is valid for that frozen checkpoint but not evidence of a completed curriculum. Source: {source('steady_vanilla_freeze')}.
2. **Hopf Vanilla failed closed.** Its validation rollout becomes non-finite at long horizons; no test result is reported. Source: {source('hopf_vanilla_validation')}.
3. **Single seed only.** The revised execution contract explicitly removed repeated seeds. No mean±std across random seeds can be claimed.
4. **DataOnly divergence and pressure failure.** Steady and Hopf DataOnly remain arithmetically finite but violate the norm-based divergence rule and have very large pressure errors. Source: {source('steady_data')}; {source('hopf_data')}.
5. **Hopf attractor claim is narrow.** Strict K56 preservation occurs only at train Re=49.3, 49.6, 50.0; none of the validation or held-out Re passes the full conjunction. Source: {source('hopf_attractor')}.
6. **Periodic long-horizon mismatch.** Proposed Periodic is frozen through K48, while the new Vanilla evaluator also reports K56. Table A1 therefore uses K48 for Proposed and does not invent K56.
7. **Global-vs-multichart metric mismatch.** The archived Global MoE artifact reports modal errors and does not share the sealed S–H physical-field cache. It cannot support a numeric overall ranking against E2/T2-C without a new common evaluator. Source: {source('global')}.
8. **H–P Top-2 unavailable.** Hopf is finite on only 5/53 P-native train trajectories and 0/6 validation trajectories at K56. The correct frozen output is P-only, not a trained fusion checkpoint. Source: {source('hp_report')}.
9. **Not blind test.** Historical test information had been viewed before the final S–H rerun. The rerun did not use test for tuning, but it must not be described as fully blind.
10. **No true transition trajectories.** Router data contain fixed-Re trajectories, not certified startup or slowly varying Re(t) transitions. The system is a temporal-consistent regime router.

Unsupported statements include: “Proposed beats every ablation in every regime,” “all Re improve under T2-C,” “Hopf attractors are uniformly preserved,” “H–P fusion was validated,” and “the final comparison is a blind test.”
"""
    (REPORTS / "FAILURES_AND_LIMITATIONS.md").write_text(failures_text)

    transient_text = f"""# Specialist transient results

{a1_md}

See `{REPORTS / 'SPECIALIST_ABLATION_REPORT.md'}` for interpretation and source paths.
"""
    (REPORTS / "SPECIALIST_TRANSIENT_RESULTS.md").write_text(transient_text)

    manuscript = f"""# Numerical Experiments Draft

## 4.1 Flow Configurations, Datasets, and Multi-chart POD Spaces

The parameter domain is represented by three independently trained reduced-order specialists associated with steady, Hopf, and periodic wake dynamics. Each specialist retains its native velocity and pressure POD bases, regime mean fields, scalers, history builder, pressure gauge, physics operators, and time-advancement contract. Reynolds-number trajectories are kept intact within train, validation, and test populations. The newly trained ablations use one fixed seed per regime and method; consequently, the reported differences are deterministic single-run comparisons rather than estimates of seed-to-seed uncertainty.

The final router is hierarchical and sparse. Steady-core trajectories use the Steady specialist, the certified S–H overlap uses window-conditioned T2-C field fusion, Hopf-core trajectories use the Hopf specialist, and the H–P/Periodic region uses P-only native-time hard routing. Soft fusion is enabled only when both adjacent specialists pass a validation-supported K56 admissibility gate. The router data contain fixed-Re trajectories and therefore support the term *temporal-consistent regime router*, not a claim of learned physical regime-transition dynamics.

## 4.2 Transient Physical-field Prediction

### 4.2.1 Regime-specific specialist ablations

Table A1 compares Vanilla-FNN-MoE, DataOnly-MoE, and the proposed physics–data specialist using frozen physical-field evaluators.

{a1_md}

The results do not support a uniform architectural advantage across all regimes. In the Steady regime, the user-frozen Vanilla checkpoint at step 6200 yields a lower K56 mean joint error than the proposed specialist ({f(steady_metrics('steady_vanilla')['k56']['joint'])} versus {f(steady_metrics('steady_proposed')['k56']['joint'])}). This checkpoint was stopped before the full curriculum and did not meet the strict contraction qualification, so the result is reported as an early-stop ablation rather than a completed training optimum. Steady DataOnly predicts finite coefficients but exhibits extreme pressure error and three divergent K56 windows.

In the Hopf regime, the proposed specialist remains finite and non-divergent on all native held-out K56 evaluations, whereas DataOnly produces 39 divergent K56 windows and a mean pressure error of {f(data_metrics('hopf_data')['k56']['p'])}. Vanilla-FNN-MoE fails the validation long-rollout gate and is not evaluated on test. In the Periodic regime, the proposed specialist improves the K48 mean joint field error from {f(cycle_metrics('periodic_vanilla')['k48']['joint'])} for Vanilla to {f(cycle_metrics('periodic_proposed')['k48']['joint'])}, while DataOnly is competitive at short horizons but lacks a certified cycle diagnostic.

### 4.2.2 Full-domain comparison

{a2_md}

The current artifacts do not permit a single commensurate all-domain ranking: the Global MoE archive reports modal one-step/rollout errors, while the S–H experiment uses area-weighted physical-field errors on a sealed boundary cache. We therefore restrict the numeric full-system comparison to the certified S–H population. There, T2-C reduces K56 mean joint error from {f(e2_pair['horizons']['K56']['joint_mean'])} to {f(t2c_method['horizons']['K56']['joint_mean'])}. At H–P, the Hopf specialist fails the development admissibility gate and the system degenerates to P-only hard routing.

### 4.2.3 Embedded routing diagnostics

The frozen E2 router classifies all 11 fixed-Re test trajectories correctly (balanced accuracy and macro-F1 both 1.0), but its minimum margin is only 0.004595 near the S–H boundary. T2-C is active on the S–H cache and beats E2 on two of three test Reynolds numbers. Its mean Hopf weight changes from 0.00125 at Re=42.359071 to 0.91389 at Re=43.50 and 0.81090 at Re=43.90, with substantial window-to-window variation. RiskPrediction and LookAhead reduce exactly to their common convex gate in this split; their extra scoring mechanisms do not improve the final output.

## 4.3 Attractor and Asymptotic-dynamics Preservation

### 4.3.1 Specialist attractor preservation

{b1_md}

The frozen Hopf specialist is numerically stable but not uniformly attractor preserving. The strict K56 conjunction passes only three training Reynolds numbers (49.3, 49.6, and 50.0), zero validation Reynolds numbers, and zero held-out Reynolds numbers. Those three cases are in-sample diagnostics and are not presented as generalization. The Periodic proposed specialist preserves three of four held-out cycles under its native K48 amplitude/frequency/phase/orbit conjunction, compared with one of four for Vanilla.

### 4.3.2 S–H boundary comparison

{b2_md}

### 4.3.3 Transient–attractor discrepancy

Re=43.50 is retained as a counterexample. T2-C increases the full-window mean field error from {f(per_re['43.5']['E2_Top1']['overall_mean'])} to {f(per_re['43.5']['T2_C']['overall_mean'])}, yet it decreases the tail-center error, amplitude error, orbit distance, tail increment, and terminal increment. Thus short-to-medium transient fidelity and local asymptotic geometry need not rank methods identically. At Re=43.90, T2-C improves both families of metrics.

## 4.4 Generalization and Computational Cost

All reported test evaluations were performed after freezing validation-selected checkpoints or, for Steady Vanilla, after the explicit user-requested step-6200 freeze. The S–H rerun did not use test data for optimization or selection, but earlier test disclosure means the study is not described as fully blind. T2-C requires two native specialist rollouts on S–H boundary windows; the archived routing overhead is approximately {f(t2c['inference']['T2-C_LearnedConvexCorrection_FieldBlend']['routing_seconds_per_trajectory'])} s per trajectory, excluding the two native rollouts. Away from the certified overlap, the admissibility mask recovers single-specialist cost.

### Remaining TODOs

- Run Global MoE, E2, and the final multi-chart system through one common physical-field cache covering all five regions before claiming an overall numerical winner.
- Add K56 to the frozen Proposed Periodic evaluator if exact horizon parity is required.
- Obtain independent Hopf validation/held-out evidence near Re=49.3–50.0; current strict passes are training diagnostics.
- Add certified startup or slowly varying Re(t) trajectories before making dynamic regime-transition claims.
"""
    (MANUSCRIPT / "NUMERICAL_EXPERIMENTS_DRAFT.md").write_text(manuscript)

    manifest = {
        "status": "REPORTS_GENERATED",
        "generated_files": sorted(
            str(path)
            for directory in (REPORTS, MANUSCRIPT)
            for path in directory.rglob("*")
            if path.is_file()
        ),
        "source_files": {key: str(path) for key, path in P.items()},
        "source_sha256": {key: sha256(path) for key, path in P.items()},
        "claim": {
            "paper_main_conclusion_supported": "PARTIALLY",
            "supported": [
                "admissibility-constrained sparse multi-chart routing",
                "S-H T2-C improvement on 2/3 test Re and mean K56",
                "Periodic proposed specialist improvement over Vanilla",
                "physics-data advantage over DataOnly for Steady/Hopf pressure stability",
            ],
            "not_supported": [
                "uniform superiority across every regime/method/Re",
                "uniform Hopf attractor preservation",
                "validated H-P soft fusion",
                "fully blind all-domain comparison",
            ],
        },
    }
    (REPORTS / "FINAL_REPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
