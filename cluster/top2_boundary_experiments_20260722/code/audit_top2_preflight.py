"""Fail-closed audit for adjacent dual-native output-field fusion."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch


ROOT = Path("/root/panxy/particalMOE")
ASSETS = {
    "Steady": {
        "index": ROOT / "steady_specialist_v1/source_artifacts/steady/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv",
        "velocity": ROOT / "steady_specialist_v1/source_artifacts/steady/velocity_pod_steady.npz",
        "pressure": ROOT / "steady_specialist_v1/source_artifacts/steady/pressure_pod_steady.npz",
        "checkpoint": ROOT / "steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt",
        "config": ROOT / "steady_specialist_v1/code/training_s2b_3090.json",
        "runtime_source": ROOT / "steady_specialist_v1/code/train_s2b_3090.py",
    },
    "Hopf": {
        "index": ROOT / "Hopf/artifacts/hopf/projection_snapshots_velocity_hopf.csv",
        "velocity": ROOT / "Hopf/artifacts/hopf/velocity_pod_hopf.npz",
        "pressure": ROOT / "Hopf/artifacts/hopf/pressure_pod_hopf.npz",
        "checkpoint": ROOT / "Hopf/migrated_h4_expanded/runs/HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt",
        "runtime_source": ROOT / "Hopf/migrated_h4_expanded/code/train_hopf_moe_expanded.py",
    },
    "Periodic": {
        "index": ROOT / "periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv",
        "velocity": ROOT / "periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz",
        "pressure": ROOT / "periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz",
        "checkpoint": ROOT / "periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt",
        "runtime_source": ROOT / "periodic_specialist_r32/code/train_periodic_moe.py",
    },
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def normalize_split(value: str) -> str:
    value = value.strip().lower()
    return "validation" if value in {"validation", "val"} else "test" if value in {"heldout", "test"} else value


def index_audit(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if normalize_split(row["split"]) in {"train", "validation"}]
    re_key = "Re" if "Re" in fields else "re"
    groups: dict[float, list[float]] = {}
    for row in rows:
        groups.setdefault(float(row[re_key]), []).append(float(row["time"]))
    dt = []
    for times in groups.values():
        ordered = sorted(times)
        dt.extend(right-left for left, right in zip(ordered[:-1], ordered[1:]) if right > left)
    values = np.asarray(dt, dtype=np.float64)
    return {
        "columns": fields,
        "phase_column": "phase" in fields,
        "period_column": "period" in fields or "estimated_period" in fields,
        "train_validation_Re_count": len(groups),
        "dt": {
            "count": int(values.size), "min": float(values.min()), "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max()),
        },
    }


def phase_harmonics(name: str, spec: dict[str, Path]) -> int:
    if name == "Steady":
        config = json.loads(spec["config"].read_text(encoding="utf-8"))
        return int(config["model"]["phase_harmonics"])
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    if hasattr(args, "__dict__"):
        args = vars(args)
    return int(args.get("phase_harmonics", 0))


def mesh_audit() -> dict[str, Any]:
    reference = None
    rows = {}
    for name, spec in ASSETS.items():
        with np.load(spec["velocity"]) as velocity, np.load(spec["pressure"]) as pressure:
            points = np.asarray(velocity["points"]); areas = np.asarray(velocity["point_areas"])
            gauge = str(np.asarray(pressure["pressure_gauge"]).item())
            current = (points, areas, gauge)
            if reference is None:
                reference = current
            rows[name] = {
                "velocity_pressure_points_equal": bool(np.array_equal(points, np.asarray(pressure["points"]))),
                "velocity_pressure_areas_equal": bool(np.array_equal(areas, np.asarray(pressure["point_areas"]))),
                "common_points_equal": bool(np.array_equal(reference[0], points)),
                "common_areas_equal": bool(np.array_equal(reference[1], areas)),
                "pressure_gauge": gauge,
                "common_pressure_gauge": reference[2] == gauge,
            }
    return {"charts": rows, "passed": all(all(v for k, v in row.items() if k != "pressure_gauge") for row in rows.values())}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty audit directory: {args.output_dir}")

    charts = {}
    failures = []
    for name, spec in ASSETS.items():
        idx = index_audit(spec["index"])
        harmonics = phase_harmonics(name, spec)
        source = spec["runtime_source"].read_text(encoding="utf-8")
        reads_indexed_phase = ("arrays[\"phase\"][" in source or "self.tensor(\"phase\", cur)" in source)
        endpoint_fallback = name in {"Steady", "Periodic"} and not idx["phase_column"] and not idx["period_column"]
        # A database phase column would still not define phase for an arbitrary
        # physical history from the neighboring chart.  Only a phase-free
        # checkpoint or a separately certified estimator can pass this gate.
        autonomous_phase = harmonics == 0
        if endpoint_fallback:
            autonomous_phase = False
            failures.append(f"{name}: phase_harmonics={harmonics}, index has no phase/period, and native loader derives phase from full trajectory endpoint")
        if harmonics > 0 and reads_indexed_phase:
            failures.append(f"{name}: native forward is coupled to arrays['phase'][current] and exposes no certified source-independent phase argument")
        charts[name] = {
            "checkpoint": str(spec["checkpoint"]), "checkpoint_sha256": sha256(spec["checkpoint"]),
            "index": str(spec["index"]), "index_audit_train_validation_only": idx,
            "phase_harmonics": harmonics, "native_reads_indexed_phase": reads_indexed_phase,
            "phase_uses_full_trajectory_endpoint_fallback": endpoint_fallback,
            "future_truth_free_cross_source_phase_contract": autonomous_phase,
        }

    mesh = mesh_audit()
    if not mesh["passed"]:
        failures.append("common mesh/cell-area/pressure-gauge contract failed")
    pair_contract = {
        "Steady-Hopf": {
            "adjacent": True,
            "future_truth_free_dual_native_start": charts["Steady"]["future_truth_free_cross_source_phase_contract"] and charts["Hopf"]["future_truth_free_cross_source_phase_contract"],
        },
        "Hopf-Periodic": {
            "adjacent": True,
            "future_truth_free_dual_native_start": charts["Hopf"]["future_truth_free_cross_source_phase_contract"] and charts["Periodic"]["future_truth_free_cross_source_phase_contract"],
        },
        "Steady-Periodic": {"adjacent": False, "forbidden": True},
    }
    for pair in ("Steady-Hopf", "Hopf-Periodic"):
        if not pair_contract[pair]["future_truth_free_dual_native_start"]:
            failures.append(f"{pair}: cannot run both frozen specialists from the same arbitrary physical history without database phase/endpoint metadata")

    audit = {
        "schema": "TOP2_OUTPUT_FIELD_PREFLIGHT_V1",
        "test_physical_modal_or_metric_data_loaded": False,
        "test_index_split_labels_scanned_for_exclusion": True,
        "charts": charts,
        "mesh_pressure_contract": mesh,
        "pair_contract": pair_contract,
        "time_alignment": {
            "status": "NOT_CERTIFIED",
            "reason": "dt ranges overlap, but no frozen cross-source wrapper has been certified to drive both indexed native feature builders on one query clock; phase failure prevents a valid dual-rollout alignment smoke",
        },
        "failures": sorted(set(failures)),
        "decision": "FAIL_CLOSED" if failures else "PREFLIGHT_PASS",
        "authorized_next_action": "no cache construction, training, validation selection, or test access" if failures else "construct train/validation dual-native cache",
    }
    atomic_json(args.output_dir / "TOP2_PREFLIGHT_AUDIT.json", audit)
    atomic_json(args.output_dir / ("FAIL_CLOSED.json" if failures else "PASS.json"), {
        "decision": audit["decision"], "failure_count": len(audit["failures"]),
        "test_physical_modal_or_metric_data_loaded": False,
        "test_index_split_labels_scanned_for_exclusion": True,
    })
    print(json.dumps({"decision": audit["decision"], "failures": audit["failures"]}, indent=2))


if __name__ == "__main__":
    main()
