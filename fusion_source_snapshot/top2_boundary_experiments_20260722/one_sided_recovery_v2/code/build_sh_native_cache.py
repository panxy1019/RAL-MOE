"""Build the shared S-native/S+H train-validation cache for one-sided Top-2."""

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


ROOT = Path("/root/panxy/particalMOE")
RANK = 32
HORIZON = 56
WINDOWS_PER_RE = 32


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def affine_map(source, target, vector: bool):
    areas = np.asarray(source["point_areas"], dtype=np.float64)
    w = np.concatenate((areas, areas)) if vector else areas
    pk, mk = ("phi_uv", "mean_uv_regime") if vector else ("phi_p", "mean_p_regime")
    ps = np.asarray(source[pk][:RANK], dtype=np.float64)
    pt = np.asarray(target[pk][:RANK], dtype=np.float64)
    ms = np.asarray(source[mk], dtype=np.float64)
    mt = np.asarray(target[mk], dtype=np.float64)
    return ((pt * w) @ ps.T).astype(np.float32), ((pt * w) @ (ms - mt)).astype(np.float32)


def quad_stats(s, h, y, source, target, vector: bool):
    areas = np.asarray(source["point_areas"], dtype=np.float64)
    w = np.concatenate((areas, areas)) if vector else areas
    pk, mk = ("phi_uv", "mean_uv_regime") if vector else ("phi_p", "mean_p_regime")
    ps = np.asarray(source[pk][:RANK], dtype=np.float64)
    ph = np.asarray(target[pk][:RANK], dtype=np.float64)
    ms = np.asarray(source[mk], dtype=np.float64)
    mh = np.asarray(target[mk], dtype=np.float64)
    gss = (ps * w) @ ps.T
    ghh = (ph * w) @ ph.T
    csh = (ps * w) @ ph.T
    md = mh - ms
    psmd = (ps * w) @ md
    phmd = (ph * w) @ md
    md2 = float(np.dot(md * w, md))
    psms = (ps * w) @ ms
    ms2 = float(np.dot(ms * w, ms))
    ds = s - y
    qss = np.einsum("ti,ij,tj->t", ds, gss, ds)
    qhh = (md2 + np.einsum("ti,ij,tj->t", h, ghh, h)
           + np.einsum("ti,ij,tj->t", y, gss, y) + 2 * h @ phmd
           - 2 * y @ psmd - 2 * np.einsum("ti,ij,tj->t", y, csh, h))
    qsh = np.einsum("ti,ti->t", ds, psmd[None] + h @ csh.T - y @ gss.T)
    yn = ms2 + 2 * y @ psms + np.einsum("ti,ij,tj->t", y, gss, y)
    return np.stack((qss, qhh, qsh, yn), axis=-1).astype(np.float32)


def evenly(ids: np.ndarray, count: int) -> np.ndarray:
    if len(ids) <= count:
        return ids
    return ids[np.linspace(0, len(ids) - 1, count, dtype=np.int64)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("development", "final_test"), default="development")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_mode = args.mode == "final_test"
    if test_mode and not (args.output_dir.parent.parent / "training/S_H_routes_attempt3_test_sealed/ALL_ROUTES_VALIDATION_FROZEN.json").exists():
        raise RuntimeError("final test cache forbidden before all routes are validation-frozen")
    atomic_json(args.output_dir / "STARTED.json", {"test_loaded": test_mode, "source": "Steady heldout only" if test_mode else "Steady train/validation only", "horizon": HORIZON})
    pre = load_module("sh_cache_preflight", Path(__file__).with_name("one_sided_preflight_sh.py"))
    top1 = load_module("sh_cache_top1", ROOT / "trajectory_router_e1_e2_e3_20260722/code/run_top1_experiment.py")
    device = torch.device("cuda")
    exp, steady_ckpt = pre.build_steady_runtime(args.output_dir)
    hopf = pre.HopfRuntime(device)
    sroot = ROOT / "steady_specialist_v1/source_artifacts/steady"
    hroot = ROOT / "Hopf/artifacts/hopf"
    with np.load(sroot / "velocity_pod_steady.npz") as su, np.load(sroot / "pressure_pod_steady.npz") as sp, \
         np.load(hroot / "velocity_pod_hopf.npz") as hu, np.load(hroot / "pressure_pod_hopf.npz") as hp:
        if not np.array_equal(su["points"], hu["points"]):
            raise RuntimeError("mesh mismatch")
        map_u, off_u = affine_map(su, hu, True)
        map_p, off_p = affine_map(sp, hp, False)
        areas = np.asarray(su["point_areas"], dtype=np.float64)
        ug = top1.weighted_geometry(su["phi_uv"], su["mean_uv_regime"], areas, True)
        pg = top1.weighted_geometry(sp["phi_p"], sp["mean_p_regime"], areas, False)
        windows = exp.build_windows(HORIZON)
        selected_labels = {}
        split_plan = (("heldout", 1),) if test_mode else (("train", 3), ("validation", 1))
        for split, count in split_plan:
            ranked = []
            for label, ids in windows[split].items():
                re_value = float(np.mean(exp.a["re"][exp.a["label_id"] == label]))
                ranked.append((re_value, label, np.asarray(ids, dtype=np.int64)))
            selected_labels[split] = sorted(ranked, reverse=True)[:count]
        records = []
        for split, _count in split_plan:
            for re_value, label, ids in selected_labels[split]:
                for start in evenly(ids, WINDOWS_PER_RE):
                    start_t = torch.tensor([int(start)], dtype=torch.long, device=device)
                    indices = exp.indices(start_t, HORIZON)[0]
                    times = np.asarray(exp.a["time"])[indices.cpu().numpy()]
                    dts = np.diff(times).astype(np.float32)
                    current = indices[0:1]
                    hist_ids = exp.tensor("hist_idx", current, torch.long)
                    a0 = exp.tensor("a", current)[0].float().cpu().numpy(); b0 = exp.tensor("b", current)[0].float().cpu().numpy()
                    ah = exp.tensor("a", hist_ids)[0].float().cpu().numpy(); bh = exp.tensor("b", hist_ids)[0].float().cpu().numpy()
                    ht = np.asarray(exp.a["time"])[hist_ids[0].cpu().numpy()]
                    order = np.argsort(ht)
                    desc = top1.physical_descriptors(ah[order][0], ah[order][1], ah[order][2], bh[order][0], bh[order][1], bh[order][2], ht[order][0], ht[order][1], ht[order][2], ug, pg, float(areas.sum()))
                    ha0, hb0 = map_u @ a0 + off_u, map_p @ b0 + off_p
                    hah, hbh = ah @ map_u.T + off_u, bh @ map_p.T + off_p
                    with torch.inference_mode():
                        sr = exp.rollout(start_t, HORIZON)
                    hpa, hpb = hopf.rollout(ha0, hb0, hah, hbh, re_value, dts)
                    sa = sr["pa"][:, 0].float().cpu().numpy(); sb = sr["pb"][:, 0].float().cpu().numpy()
                    ya = sr["ta"][:, 0].float().cpu().numpy(); yb = sr["tb"][:, 0].float().cpu().numpy()
                    ha = np.asarray(hpa, dtype=np.float32); hb = np.asarray(hpb, dtype=np.float32)
                    records.append((split, re_value, int(label), int(start), times.astype(np.float64), np.r_[re_value, desc].astype(np.float32), sa, sb, ha, hb, ya, yb, quad_stats(sa, ha, ya, su, hu, True), quad_stats(sb, hb, yb, sp, hp, False)))
        payload = {}
        names = ("split", "re", "label", "start", "times", "features", "s_a", "s_b", "h_a", "h_b", "true_a", "true_b", "quad_u", "quad_p")
        for i, name in enumerate(names):
            values = [row[i] for row in records]
            payload[name] = np.asarray(values)
        cache_path = args.output_dir / ("G_SH_native_final_test_cache.npz" if test_mode else "G_SH_native_train_validation_cache.npz")
        np.savez_compressed(cache_path, **payload)
    split_counts = {split: int(np.sum(payload["split"] == split)) for split, _count in split_plan}
    manifest = {
        "cache": str(cache_path), "cache_sha256": sha256(cache_path), "test_loaded": test_mode,
        "source": "Steady database-native heldout trajectories" if test_mode else "Steady database-native train/validation trajectories", "candidates": ["Steady", "Hopf"],
        "horizon": HORIZON, "windows_per_Re": WINDOWS_PER_RE, "split_counts": split_counts,
        "selected_complete_Re": {split: sorted(set(payload["re"][payload["split"] == split].tolist())) for split in split_counts},
        "steady_checkpoint_sha256": sha256(steady_ckpt), "hopf_checkpoint_sha256": hopf.checkpoint_sha256,
        "fusion_feedback": False, "timestamps_shared": True,
    }
    atomic_json(args.output_dir / "CACHE_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
