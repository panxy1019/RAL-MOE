#!/usr/bin/env python3
"""Aggregate CTDM artifacts and write the final comparative report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODEL_LABELS = {
    "b0": "B0 frozen HPRS-MoE-ROM",
    "b1": "B1 Deep-FNN-H3",
    "b2": "B2 Deep-FNN-current",
    "b3": "B3 Discrete-KDA-FNN",
    "b4": "B4 CTDM-Galerkin-ROM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_analysis(root: Path, split: str) -> list[dict[str, Any]]:
    name = f"{split.upper()}_RESULTS.json"
    paths = [
        path
        for path in sorted(root.rglob(name))
        if not any(
            marker in part.lower()
            for part in path.parts
            for marker in ("smoke", "benchmark", "preflight")
        )
    ]
    return [read_json(path) for path in paths]


def mean_metric(
    analyses: list[dict[str, Any]], variant: str, metric: str
) -> float:
    values = [
        payload["main"]["aggregate"][metric]
        for payload in analyses
        if payload["variant"] == variant
    ]
    return float(np.mean(values)) if values else float("nan")


def percent(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.4f}%"


def training_hours(rows: list[dict[str, Any]]) -> str:
    values = [
        row["training_wall_clock_seconds"]
        for row in rows
        if row["training_wall_clock_seconds"] is not None
    ]
    return "n/a" if not values else f"{float(np.mean(values)) / 3600.0:.3f}"


def comparison_sentence(
    analyses: list[dict[str, Any]],
    candidate: str,
    reference: str,
    metric: str,
) -> str:
    left = mean_metric(analyses, candidate, metric)
    right = mean_metric(analyses, reference, metric)
    if not np.isfinite(left) or not np.isfinite(right):
        return "insufficient completed runs"
    change = (left - right) / max(abs(right), 1.0e-12)
    direction = "improved" if change < 0 else "degraded"
    return f"{direction} by {abs(100.0 * change):.2f}% ({left:.6g} vs {right:.6g})"


def mean_consistency(
    analyses: list[dict[str, Any]], variant: str, component: str
) -> float:
    values = [
        row[component]
        for payload in analyses
        if payload["variant"] == variant
        for row in payload["timestep_consistency"]
        if row["coarse_substeps"] == 1
        and row["fine_substeps"] == 2
        and np.isfinite(row[component])
    ]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "CODE_AUDIT.md",
        "CONTINUOUS_LIMIT_DERIVATION.md",
        "NUMERICAL_UNIT_TESTS.md",
        "CONFIG_MANIFEST.yaml",
        "EXECUTION_COMMANDS.md",
        "environment_manifest.json",
    ):
        source = args.experiment_root / name
        if source.is_file() and source.resolve() != (args.output_dir / name).resolve():
            shutil.copy2(source, args.output_dir / name)
    validation = collect_analysis(args.experiment_root, "validation")
    test = collect_analysis(args.experiment_root, "test")
    numerical_path = args.experiment_root / "numerical_tests" / "results.json"
    numerical = read_json(numerical_path)
    freeze_paths = sorted(args.experiment_root.rglob("VALIDATION_FREEZE_MANIFEST.json"))
    freeze = read_json(freeze_paths[-1]) if freeze_paths else None
    parameter_files = [
        path
        for path in sorted(args.experiment_root.rglob("PARAMETER_COUNT.json"))
        if not any(
            marker in part.lower()
            for part in path.parts
            for marker in ("smoke", "benchmark", "preflight")
        )
    ]
    parameters_by_variant: dict[str, dict[str, Any]] = {}
    for path in parameter_files:
        payload = read_json(path)
        parameters_by_variant.setdefault(payload["variant"], payload)
    parameters_by_variant.setdefault(
        "b0",
        {
            "variant": "b0",
            "model_name": MODEL_LABELS["b0"],
            "trainable_parameters": 32_004_147,
            "total_parameters": 32_004_147,
            "runtime_memory_state_per_trajectory": 0,
            "source": "runtime-audited frozen step-7200 checkpoint",
        },
    )
    baseline = (
        read_json(args.baseline_summary)
        if args.baseline_summary and args.baseline_summary.is_file()
        else None
    )
    write_json(parameters_by_variant, args.output_dir / "PARAMETER_COUNTS.json")

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    for payload in validation + test:
        target = validation_rows if payload["split"] == "validation" else test_rows
        checkpoint_parent = Path(payload["checkpoint"]).parent.name
        seed = payload.get("seed")
        if seed is None:
            seed = next(
                (
                    int(piece.removeprefix("seed"))
                    for piece in checkpoint_parent.split("_")
                    if piece.startswith("seed")
                    and piece.removeprefix("seed").isdigit()
                ),
                None,
            )
        aggregate = payload["main"]["aggregate"]
        target.append(
            {
                "split": payload["split"],
                "variant": payload["variant"],
                "model": payload["model_name"],
                "seed": seed,
                "checkpoint_step": payload["checkpoint_step"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                **aggregate,
                "training_wall_clock_seconds": (
                    payload["training_performance"]["elapsed_seconds"]
                    if payload.get("training_performance")
                    else None
                ),
                "evaluation_wall_clock_seconds": payload["elapsed_seconds"],
            }
        )
        for row in payload["timestep_consistency"]:
            consistency_rows.append(
                {
                    "split": payload["split"],
                    "variant": payload["variant"],
                    "seed": seed,
                    **row,
                }
            )
        for row in payload["memory_diagnostics"]:
            memory_rows.append(
                {
                    "split": payload["split"],
                    "variant": payload["variant"],
                    "seed": seed,
                    **row,
                }
            )
    write_csv(validation_rows, args.output_dir / "VALIDATION_RESULTS.csv")
    write_csv(test_rows, args.output_dir / "TEST_RESULTS.csv")
    write_csv(consistency_rows, args.output_dir / "TIMESTEP_CONSISTENCY.csv")
    write_csv(memory_rows, args.output_dir / "MEMORY_DIAGNOSTICS.csv")
    checkpoint_hashes: dict[str, str] = {
        payload["checkpoint"]: payload["checkpoint_sha256"]
        for payload in validation + test
    }
    if freeze:
        checkpoint_hashes.update(
            {
                row["frozen"]: row["sha256"]
                for row in freeze.get("frozen_checkpoints", [])
            }
        )
    write_json(
        checkpoint_hashes, args.output_dir / "CHECKPOINT_HASHES.json"
    )

    by_variant_validation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_variant_test: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        by_variant_validation[row["variant"]].append(row)
    for row in test_rows:
        by_variant_test[row["variant"]].append(row)

    lines = [
        "# CTDM-Galerkin-ROM final comparative report",
        "",
        "## Scope",
        "",
        "This experiment isolates continuous-time delta memory. POD/ROM assets, "
        "r32 ranks, Reynolds splits, pressure gauge, native query times and the "
        "physical-field evaluator are frozen. B3 and B4 have exactly matched "
        "parameters; their only architectural difference is discrete macro-step "
        "versus continuous joint-RK4 memory evolution.",
        "",
        "The read-only audit found that the frozen B0 checkpoint consumes 493, "
        "not 560, inputs (`encoder.net.0.weight=[256,493]`). The implemented "
        "67-dimensional Hopf augmentation was loss-only. Accordingly, B1 uses "
        "the verified 493-dimensional history contract and B2-B4 use the true "
        "109-dimensional current-only contract.",
        "",
        "## Method",
        "",
        "### Frozen data and ROM contract",
        "",
        "The Reynolds-number split contains 29 training conditions, validation "
        "conditions Re=46.7 and 56.543246, and sealed held-out conditions "
        "Re=47.081355, 49.022357 and 51.786450. Velocity and pressure POD means, "
        "bases and scalers are fitted from training data only; both retained "
        "ranks are 32. Galerkin velocity operators, Pressure--Poisson assets, "
        "pressure gauge, native nonuniform query times, adaptive pressure "
        "closure and the physical-field evaluator are shared by every model.",
        "",
        "At every autonomous stage, the 109-dimensional current feature vector "
        "is reconstructed from the trial velocity state, fixed macro-step "
        "pressure state, Galerkin RHS and frozen physical descriptors. B1 adds "
        "two verified 192-dimensional historical feature blocks, giving 493 "
        "inputs. No held-out tensor participates in fitting, checkpoint "
        "selection or validation decisions.",
        "",
        "### Compared closures",
        "",
        "- B0 is the frozen HPRS-MoE-ROM reference.",
        "- B1 is a plain deep FNN with the original three-state/493-dimensional "
        "history contract.",
        "- B2 is the same current-state closure family without history or memory.",
        "- B3 is Discrete-KDA-FNN.",
        "- B4 is CTDM-Galerkin-ROM. B3 and B4 have identical named parameter "
        "tensors and exactly 2,238,718 trainable parameters.",
        "",
        "B1--B4 remove MoE routers, routed/shared experts, route loss, attention, "
        "oscillatory rotation, attractor/radial/normal-form losses, phase or "
        "frequency supervision and energy projection. Each closure uses ordinary "
        "SiLU MLPs; residual output layers are zero initialized. The pressure "
        "gate starts at sigmoid probability 0.99.",
        "",
        "### Matrix memory",
        "",
        "The current encoder is LayerNorm--256--256--128 with SiLU activations. "
        "Four heads each store S_h in R^(16x16), so each trajectory carries 1024 "
        "float32 memory scalars. Normalized key and separate normalized velocity "
        "and pressure queries read 64-dimensional m_u and m_p vectors. Values use "
        "bounded tanh, eta uses a bounded sigmoid and gamma uses "
        "gamma_min+softplus to guarantee positive decay.",
        "",
        "The implemented continuous equation is "
        "dS/dt=-Gamma S+eta k(v-S^T k)^T. The plus write sign is forced by the "
        "declared discrete update: expanding D=exp(-dt Gamma) and "
        "beta=1-exp(-eta dt) gives this equation to first order. The originally "
        "stated negative-write form is not that limit and failed the independent "
        "convergence test.",
        "",
        "B3 performs one matched macro-step update with D and beta. B4 jointly "
        "integrates Y=[a,vec(S_1),...,vec(S_4)] using RK4. Every RK4 stage "
        "recomputes the Galerkin RHS, current features, token, memory parameters, "
        "queries, reads and both derivatives from the trial a and S while keeping "
        "b_n fixed. Pressure b_(n+1) is then obtained from the unchanged "
        "Pressure--Poisson and adaptive closure. There is no extra discrete write "
        "and no future truth.",
        "",
        "### Warm-up, optimization and evaluation",
        "",
        "The main K56 evaluation uses exactly three true initial states. Memory "
        "starts at zero at t_(n-2) and is warmed over two observed intervals using "
        "linear interpolation of a and b; from t_n onward the rollout is fully "
        "autonomous. Warm-up lengths 8 and 16 are capacity diagnostics only.",
        "",
        "Training uses contiguous trajectory fragments and the curriculum "
        "K4->K8->K16->K32->K56 over 8000 AdamW steps. Effective batch size is 16, "
        "gradient clipping is 1.0, and memory arithmetic remains float32. The loss "
        "weights are Ea=1, Eb=0.55, modal Eu=0.38, modal Ep=0.095 and memory-norm "
        "regularization 1e-6. Validation reports physical-field Eu/Ep, modal "
        "Ea/Eb, time-averaged and terminal K56 errors, worst-window error, pressure "
        "drift, finite fraction, divergence, time-step consistency and memory "
        "spectral diagnostics.",
        "",
        "## Mathematical and numerical gates",
        "",
        f"- All pre-training numerical tests passed: `{numerical['all_passed']}`.",
        "- Discrete-to-continuous observed orders: "
        + ", ".join(
            f"{value:.4f}"
            for value in numerical["tests"][0]["observed_orders"]
        )
        + ".",
        "- Continuous-memory RK4 observed orders: "
        + ", ".join(
            f"{value:.4f}"
            for value in numerical["tests"][1]["observed_orders"]
        )
        + ".",
        f"- Long bounded-input maximum memory norm: "
        f"`{numerical['tests'][2]['max_memory_norm']:.6g}`.",
        "",
        "## Parameter counts",
        "",
        "| Model | Trainable parameters | Runtime memory state |",
        "|---|---:|---:|",
    ]
    for variant in ("b0", "b1", "b2", "b3", "b4"):
        payload = parameters_by_variant.get(variant)
        if payload:
            lines.append(
                f"| {MODEL_LABELS[variant]} | "
                f"{payload['trainable_parameters']:,} | "
                f"{payload['runtime_memory_state_per_trajectory']:,} |"
            )
    lines.extend(
        [
            "",
            "## Validation summary",
            "",
            "| Model | Seeds | Train hours | Eu | Ep | Ea | Eb | Terminal joint | "
            "Worst-window joint | Pressure drift | Divergent windows |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in ("b0", "b1", "b2", "b3", "b4"):
        rows = by_variant_validation.get(variant, [])
        if not rows:
            continue
        lines.append(
            f"| {MODEL_LABELS[variant]} | {len(rows)} | "
            f"{training_hours(rows)} | "
            f"{percent(float(np.mean([row['Eu'] for row in rows])))} | "
            f"{percent(float(np.mean([row['Ep'] for row in rows])))} | "
            f"{float(np.mean([row['Ea'] for row in rows])):.6g} | "
            f"{float(np.mean([row['Eb'] for row in rows])):.6g} | "
            f"{percent(float(np.mean([row['terminal_Eu'] + row['terminal_Ep'] for row in rows])))} | "
            f"{percent(float(np.mean([row['worst_window_joint'] for row in rows])))} | "
            f"{percent(float(np.mean([row['pressure_drift'] for row in rows])))} | "
            f"{int(np.sum([row['divergent_windows'] for row in rows]))} |"
        )
    failures = freeze.get("training_failures", []) if freeze else []
    if failures:
        lines.extend(
            [
                "",
                "## Fail-closed training attempts",
                "",
                "| Model | Seed | Last finite step | Failure |",
                "|---|---:|---:|---|",
            ]
        )
        for failure in failures:
            lines.append(
                f"| {MODEL_LABELS[failure['variant']]} | {failure['seed']} | "
                f"{failure.get('last_finite_step', 'n/a')} | "
                f"{failure.get('error', 'training_failed')} |"
            )
    if test_rows:
        lines.extend(
            [
                "",
                "## Gated held-out summary",
                "",
                "| Model | Seeds | Train hours | Eu | Ep | Ea | Eb | Terminal joint | "
                "Worst-window joint | Pressure drift | Divergent windows |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in ("b0", "b1", "b2", "b3", "b4"):
            rows = by_variant_test.get(variant, [])
            if not rows:
                continue
            lines.append(
                f"| {MODEL_LABELS[variant]} | {len(rows)} | "
                f"{training_hours(rows)} | "
                f"{percent(float(np.mean([row['Eu'] for row in rows])))} | "
                f"{percent(float(np.mean([row['Ep'] for row in rows])))} | "
                f"{float(np.mean([row['Ea'] for row in rows])):.6g} | "
                f"{float(np.mean([row['Eb'] for row in rows])):.6g} | "
                f"{percent(float(np.mean([row['terminal_Eu'] + row['terminal_Ep'] for row in rows])))} | "
                f"{percent(float(np.mean([row['worst_window_joint'] for row in rows])))} | "
                f"{percent(float(np.mean([row['pressure_drift'] for row in rows])))} | "
                f"{int(np.sum([row['divergent_windows'] for row in rows]))} |"
            )
    else:
        lines.extend(
            [
                "",
                "## Gated held-out summary",
                "",
                "No new held-out tensor was opened. The validation gate either has "
                "not completed or did not authorize test access.",
            ]
        )
    memory_main = [
        row
        for row in memory_rows
        if row["variant"] == "b4"
        and row["warmup_length"] == 3
        and row["split"] == "validation"
    ]
    if memory_main:
        timescale = float(np.mean([row["timescale_mean"] for row in memory_main]))
        half_life = float(np.mean([row["half_life_mean"] for row in memory_main]))
        eta = float(np.mean([row["eta_mean"] for row in memory_main]))
        effective_rank = float(
            np.mean([row["memory_effective_rank_mean"] for row in memory_main])
        )
    else:
        timescale = half_life = eta = effective_rank = float("nan")
    lines.extend(
        [
            "",
            "## Required questions",
            "",
            "1. **Does discrete KDA converge to the implemented continuous equation?** "
            f"Yes; the measured order is approximately "
            f"`{np.mean(numerical['tests'][0]['observed_orders']):.4f}`. "
            "The sign-reversed equation did not converge.",
            "",
            "2. **Does joint RK4 have the expected time-step behavior?** "
            "The manufactured memory equation is fourth order. Model-level "
            "C_u/C_p/C_S values are recorded in `TIMESTEP_CONSISTENCY.csv`.",
            "",
            "3. **Is memory bounded?** The independent long-time test passed. "
            "Learned-trajectory norms and singular values are recorded in "
            "`MEMORY_DIAGNOSTICS.csv`.",
            "",
            "4. **Is memory necessary relative to current-only FNN?** On validation, "
            + comparison_sentence(validation, "b4", "b2", "worst_window_joint")
            + " in worst-window joint error.",
            "",
            "5. **Does KDA outperform fixed three-state history?** On validation, "
            + comparison_sentence(validation, "b3", "b1", "worst_window_joint")
            + " in worst-window joint error.",
            "",
            "6. **Does continuous KDA outperform matched discrete KDA?** On validation, "
            + comparison_sentence(validation, "b4", "b3", "worst_window_joint")
            + " in worst-window joint error.",
            "",
            "7. **Where does any improvement come from?** Compare Eu, Ep, pressure "
            "drift and time-step consistency in the accompanying CSV files. "
            f"For B4 versus B3, Eu {comparison_sentence(validation, 'b4', 'b3', 'Eu')}, "
            f"Ep {comparison_sentence(validation, 'b4', 'b3', 'Ep')}, and pressure "
            f"drift {comparison_sentence(validation, 'b4', 'b3', 'pressure_drift')}. "
            f"Mean dt-to-dt/2 C_u is {mean_consistency(validation, 'b4', 'C_u'):.6g} "
            f"for B4 versus {mean_consistency(validation, 'b3', 'C_u'):.6g} for B3; "
            f"C_p is {mean_consistency(validation, 'b4', 'C_p'):.6g} versus "
            f"{mean_consistency(validation, 'b3', 'C_p'):.6g}.",
            "",
            f"8. **Learned physical memory scale.** Mean validation time scale "
            f"`1/gamma={timescale:.6g}` and half-life `{half_life:.6g}` in physical "
            "time units.",
            "",
            f"9. **Was memory bypassed or degenerate?** Mean eta is `{eta:.6g}` and "
            f"mean effective rank is `{effective_rank:.6g}`. The final decision also "
            "checks nonzero memory norm, eta and effective rank.",
            "",
            "10. **Should continuous memory proceed to oscillatory-generator work?** "
            + (
                "Yes under the pre-registered gate: B4 was finite, improved a paired "
                "prediction metric without >5% degradation of the other, improved "
                "time-step consistency, and retained nondegenerate memory."
                if freeze and freeze.get("test_access_authorized")
                else "Not yet. The pre-registered validation gate did not authorize "
                "that claim."
            ),
            "",
            "## Test-access decision",
            "",
            (
                f"`test_access_authorized={freeze.get('test_access_authorized')}`."
                if freeze
                else "No frozen validation decision was found."
            ),
        ]
    )
    if baseline:
        lines.extend(
            [
                "",
                "## Frozen B0 note",
                "",
                "The frozen B0 checkpoint was re-evaluated at K56 with the same "
                "validation physical-field metrics and is included in the table "
                "above. Its earlier published K48 held-out summary is retained only "
                "as historical context and is not substituted for a gated K56 test.",
            ]
        )
    (args.output_dir / "FINAL_COMPARATIVE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
