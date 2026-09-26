"""Build a balanced S-native S/H cache for an explicit complete-Re split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


RANK = 32
HORIZON = 56
WINDOWS_PER_RE = 5
SPLIT = {
    "train": [40.711525, 43.093929, 43.20, 43.40, 43.60, 43.797394, 44.478355],
    "validation": [41.576575, 43.30, 43.70],
    "heldout": [42.359071, 43.50, 43.90],
}
EXTERNAL_RE = {43.20, 43.30, 43.40, 43.50, 43.60, 43.70, 43.90}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evenly(ids: np.ndarray, count: int) -> np.ndarray:
    if len(ids) < count:
        raise RuntimeError(f"only {len(ids)} valid windows, need {count}")
    if len(ids) == count:
        return ids
    return ids[np.linspace(0, len(ids) - 1, count, dtype=np.int64)]


def quad_stats(s, h, y, source, target, vector: bool):
    areas = np.asarray(source["point_areas"], dtype=np.float64)
    weights = np.concatenate((areas, areas)) if vector else areas
    phi_key, mean_key = ("phi_uv", "mean_uv_regime") if vector else ("phi_p", "mean_p_regime")
    phi_s = np.asarray(source[phi_key][:RANK], dtype=np.float64)
    phi_h = np.asarray(target[phi_key][:RANK], dtype=np.float64)
    mean_s = np.asarray(source[mean_key], dtype=np.float64)
    mean_h = np.asarray(target[mean_key], dtype=np.float64)
    gss = (phi_s * weights) @ phi_s.T
    ghh = (phi_h * weights) @ phi_h.T
    cross = (phi_s * weights) @ phi_h.T
    mean_delta = mean_h - mean_s
    psmd = (phi_s * weights) @ mean_delta
    phmd = (phi_h * weights) @ mean_delta
    mean_delta_sq = float(np.dot(mean_delta * weights, mean_delta))
    psms = (phi_s * weights) @ mean_s
    mean_s_sq = float(np.dot(mean_s * weights, mean_s))
    delta_s = s.astype(np.float64) - y.astype(np.float64)
    h = h.astype(np.float64)
    y = y.astype(np.float64)
    qss = np.einsum("ti,ij,tj->t", delta_s, gss, delta_s)
    qhh = (
        mean_delta_sq
        + np.einsum("ti,ij,tj->t", h, ghh, h)
        + np.einsum("ti,ij,tj->t", y, gss, y)
        + 2 * h @ phmd
        - 2 * y @ psmd
        - 2 * np.einsum("ti,ij,tj->t", y, cross, h)
    )
    qsh = np.einsum("ti,ti->t", delta_s, psmd[None] + h @ cross.T - y @ gss.T)
    truth_norm = mean_s_sq + 2 * y @ psms + np.einsum("ti,ij,tj->t", y, gss, y)
    return np.stack((qss, qhh, qsh, truth_norm), axis=-1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("development", "final_test"), required=True)
    parser.add_argument("--frozen-marker", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    if args.mode == "final_test" and (
        args.frozen_marker is None or not args.frozen_marker.is_file()
    ):
        raise RuntimeError("final-test cache requires an existing validation-frozen marker")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_splits = ("train", "validation") if args.mode == "development" else ("heldout",)
    atomic_json(
        args.output_dir / "STARTED.json",
        {
            "mode": args.mode,
            "selected_complete_Re": {name: SPLIT[name] for name in selected_splits},
            "windows_per_Re": WINDOWS_PER_RE,
            "test_metrics_used_for_training_or_selection": False,
        },
    )
    scan = load_module(
        "resplit_external_scan",
        Path(__file__).with_name("scan_expanded_s_native_capability.py"),
    )
    preflight = load_module(
        "resplit_preflight",
        Path(__file__).with_name("one_sided_preflight_sh.py"),
    )
    preflight.ROOT = args.root
    top1 = load_module(
        "resplit_top1",
        args.root / "trajectory_router_e1_e2_e3_20260722/code/run_top1_experiment.py",
    )
    device = torch.device("cuda")
    exp, steady_checkpoint = scan.build_steady_runtime(args.root, args.output_dir)
    hopf = preflight.HopfRuntime(device)
    modal_path = args.dataset / "processed/steady_expanded_validation_modal_r32.npz"
    with np.load(modal_path, allow_pickle=True) as archive:
        external = {key: np.asarray(archive[key]) for key in archive.files}
    steady_root = args.root / "steady_specialist_v1/source_artifacts/steady"
    hopf_root = args.root / "Hopf/artifacts/hopf"
    records = []
    source_audit = []
    with (
        np.load(steady_root / "velocity_pod_steady.npz") as su,
        np.load(steady_root / "pressure_pod_steady.npz") as sp,
        np.load(hopf_root / "velocity_pod_hopf.npz") as hu,
        np.load(hopf_root / "pressure_pod_hopf.npz") as hp,
    ):
        if not (
            np.array_equal(su["points"], hu["points"])
            and np.array_equal(su["point_areas"], hu["point_areas"])
        ):
            raise RuntimeError("S/H mesh or area mismatch")
        if str(np.asarray(sp["pressure_gauge"]).item()) != str(np.asarray(hp["pressure_gauge"]).item()):
            raise RuntimeError("S/H pressure gauge mismatch")
        map_u, off_u = preflight.affine_map(su, hu, True)
        map_p, off_p = preflight.affine_map(sp, hp, False)
        areas = np.asarray(su["point_areas"], dtype=np.float64)
        velocity_geometry = top1.weighted_geometry(
            su["phi_uv"], su["mean_uv_regime"], areas, True
        )
        pressure_geometry = top1.weighted_geometry(
            sp["phi_p"], sp["mean_p_regime"], areas, False
        )
        windows = exp.build_windows(HORIZON)
        old_candidates = []
        for native_split, label_map in windows.items():
            for label, starts in label_map.items():
                re_value = float(np.mean(exp.a["re"][exp.a["label_id"] == label]))
                old_candidates.append(
                    (re_value, native_split, int(label), np.asarray(starts, dtype=np.int64))
                )
        for assigned_split in selected_splits:
            for requested_re in SPLIT[assigned_split]:
                is_external = round(float(requested_re), 2) in EXTERNAL_RE
                if is_external:
                    ids = np.flatnonzero(
                        np.isclose(external["Re"], requested_re, rtol=0.0, atol=5e-6)
                    )
                    if len(ids) != 64:
                        raise RuntimeError(f"external Re={requested_re}: expected 64 snapshots")
                    a_all = external["a_raw"][ids].astype(np.float32)
                    b_all = external["b_raw"][ids].astype(np.float32)
                    times_all = external["time"][ids].astype(np.float64)
                    starts = evenly(np.arange(2, len(ids) - HORIZON), WINDOWS_PER_RE)
                    gal, sur, rom_re, rom_label = scan.native_rom_for_re(
                        exp, float(requested_re)
                    )
                    source_audit.append(
                        {
                            "assigned_split": assigned_split,
                            "requested_Re": requested_re,
                            "source": "expanded_external_S_native",
                            "source_split_metadata": str(external["split"][ids[0]]),
                            "valid_K56_windows": int(len(np.arange(2, len(ids) - HORIZON))),
                            "selected_windows": starts.tolist(),
                            "steady_native_ROM_Re": rom_re,
                            "steady_native_ROM_label": rom_label,
                        }
                    )
                    for start in starts.tolist():
                        sa, sb, ya, yb = scan.steady_rollout(
                            exp,
                            a_all,
                            b_all,
                            times_all,
                            int(start),
                            float(requested_re),
                            gal,
                            sur,
                        )
                        history_ids = np.asarray([start, start - 1, start - 2], dtype=np.int64)
                        history_times = times_all[history_ids]
                        order = np.argsort(history_times)
                        descriptor = top1.physical_descriptors(
                            a_all[history_ids][order][0],
                            a_all[history_ids][order][1],
                            a_all[history_ids][order][2],
                            b_all[history_ids][order][0],
                            b_all[history_ids][order][1],
                            b_all[history_ids][order][2],
                            history_times[order][0],
                            history_times[order][1],
                            history_times[order][2],
                            velocity_geometry,
                            pressure_geometry,
                            float(areas.sum()),
                        )
                        ha0 = map_u @ a_all[start] + off_u
                        hb0 = map_p @ b_all[start] + off_p
                        hah = a_all[history_ids] @ map_u.T + off_u
                        hbh = b_all[history_ids] @ map_p.T + off_p
                        query_times = times_all[start : start + HORIZON + 1]
                        hpa, hpb = hopf.rollout(
                            ha0,
                            hb0,
                            hah,
                            hbh,
                            float(requested_re),
                            np.diff(query_times).astype(np.float32),
                        )
                        ha = np.asarray(hpa, dtype=np.float32)
                        hb = np.asarray(hpb, dtype=np.float32)
                        records.append(
                            (
                                assigned_split,
                                float(requested_re),
                                -1,
                                int(start),
                                query_times,
                                np.r_[requested_re, descriptor].astype(np.float32),
                                sa,
                                sb,
                                ha,
                                hb,
                                ya,
                                yb,
                                quad_stats(sa, ha, ya, su, hu, True),
                                quad_stats(sb, hb, yb, sp, hp, False),
                            )
                        )
                else:
                    match = min(old_candidates, key=lambda item: abs(item[0] - requested_re))
                    re_value, native_split, label, valid_starts = match
                    if abs(re_value - requested_re) > 2e-5:
                        raise RuntimeError(
                            f"no old native match for {requested_re}; nearest={re_value}"
                        )
                    starts = evenly(valid_starts, WINDOWS_PER_RE)
                    source_audit.append(
                        {
                            "assigned_split": assigned_split,
                            "requested_Re": requested_re,
                            "actual_Re": re_value,
                            "source": "original_S_native",
                            "source_split_metadata": native_split,
                            "label_id": label,
                            "valid_K56_windows": int(len(valid_starts)),
                            "selected_windows": starts.tolist(),
                        }
                    )
                    for start in starts.tolist():
                        start_t = torch.tensor([int(start)], dtype=torch.long, device=device)
                        indices = exp.indices(start_t, HORIZON)[0]
                        query_times = np.asarray(exp.a["time"])[indices.cpu().numpy()]
                        current = indices[0:1]
                        history_ids_t = exp.tensor("hist_idx", current, torch.long)
                        a0 = exp.tensor("a", current)[0].float().cpu().numpy()
                        b0 = exp.tensor("b", current)[0].float().cpu().numpy()
                        ah = exp.tensor("a", history_ids_t)[0].float().cpu().numpy()
                        bh = exp.tensor("b", history_ids_t)[0].float().cpu().numpy()
                        history_times = np.asarray(exp.a["time"])[
                            history_ids_t[0].cpu().numpy()
                        ]
                        order = np.argsort(history_times)
                        descriptor = top1.physical_descriptors(
                            ah[order][0],
                            ah[order][1],
                            ah[order][2],
                            bh[order][0],
                            bh[order][1],
                            bh[order][2],
                            history_times[order][0],
                            history_times[order][1],
                            history_times[order][2],
                            velocity_geometry,
                            pressure_geometry,
                            float(areas.sum()),
                        )
                        ha0 = map_u @ a0 + off_u
                        hb0 = map_p @ b0 + off_p
                        hah = ah @ map_u.T + off_u
                        hbh = bh @ map_p.T + off_p
                        with torch.inference_mode():
                            steady = exp.rollout(start_t, HORIZON)
                        hpa, hpb = hopf.rollout(
                            ha0,
                            hb0,
                            hah,
                            hbh,
                            re_value,
                            np.diff(query_times).astype(np.float32),
                        )
                        sa = steady["pa"][:, 0].float().cpu().numpy()
                        sb = steady["pb"][:, 0].float().cpu().numpy()
                        ya = steady["ta"][:, 0].float().cpu().numpy()
                        yb = steady["tb"][:, 0].float().cpu().numpy()
                        ha = np.asarray(hpa, dtype=np.float32)
                        hb = np.asarray(hpb, dtype=np.float32)
                        records.append(
                            (
                                assigned_split,
                                float(requested_re),
                                int(label),
                                int(start),
                                query_times.astype(np.float64),
                                np.r_[requested_re, descriptor].astype(np.float32),
                                sa,
                                sb,
                                ha,
                                hb,
                                ya,
                                yb,
                                quad_stats(sa, ha, ya, su, hu, True),
                                quad_stats(sb, hb, yb, sp, hp, False),
                            )
                        )
    names = (
        "split",
        "re",
        "label",
        "start",
        "times",
        "features",
        "s_a",
        "s_b",
        "h_a",
        "h_b",
        "true_a",
        "true_b",
        "quad_u",
        "quad_p",
    )
    payload = {
        name: np.asarray([record[index] for record in records])
        for index, name in enumerate(names)
    }
    expected_counts = {
        split: len(SPLIT[split]) * WINDOWS_PER_RE for split in selected_splits
    }
    actual_counts = {
        split: int(np.sum(payload["split"] == split)) for split in selected_splits
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"cache count mismatch: {actual_counts} != {expected_counts}")
    if not all(
        np.isfinite(payload[key]).all()
        for key in names
        if payload[key].dtype.kind in "fc"
    ):
        raise RuntimeError("non-finite cache values")
    filename = (
        "G_SH_resplit_final_test_cache.npz"
        if args.mode == "final_test"
        else "G_SH_resplit_train_validation_cache.npz"
    )
    cache_path = args.output_dir / filename
    np.savez_compressed(cache_path, **payload)
    manifest = {
        "status": "CACHE_COMPLETE",
        "mode": args.mode,
        "cache": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "selected_complete_Re": {name: SPLIT[name] for name in selected_splits},
        "windows_per_Re": WINDOWS_PER_RE,
        "split_counts": actual_counts,
        "horizon": HORIZON,
        "source_audit": source_audit,
        "fusion_feedback": False,
        "timestamps_shared": True,
        "steady_checkpoint_sha256": sha256(steady_checkpoint),
        "hopf_checkpoint_sha256": hopf.checkpoint_sha256,
        "expanded_modal_archive_sha256": sha256(modal_path),
        "test_metrics_used_for_training_or_selection": False,
        "known_prior_test_disclosure": True,
    }
    atomic_json(args.output_dir / "CACHE_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
