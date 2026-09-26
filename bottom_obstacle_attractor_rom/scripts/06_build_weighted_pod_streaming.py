#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build area-weighted velocity and pressure POD bases.")
    parser.add_argument("--npz-dir", type=Path, default=ROOT / "data" / "npz")
    parser.add_argument("--area-file", type=Path, default=ROOT / "data" / "npz" / "point_area.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "pod")
    parser.add_argument("--ru", type=int, default=32)
    parser.add_argument("--rp", type=int, default=32)
    parser.add_argument(
        "--pressure-gauge",
        choices=["none", "subtract_area_mean_per_snapshot"],
        default="subtract_area_mean_per_snapshot",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--blas-threads", type=int)
    return parser.parse_args()


def configure_threads(args: argparse.Namespace) -> None:
    if args.blas_threads:
        for name in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            os.environ[name] = str(args.blas_threads)


def load_case_index(files):
    import numpy as np

    entries = []
    offset = 0
    coords_ref = None
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            coords = data["coords"]
            times = data["times"]
            re_value = float(data["Re"])
            if coords_ref is None:
                coords_ref = coords.copy()
            elif coords.shape != coords_ref.shape or not np.allclose(coords, coords_ref, atol=1e-10):
                raise RuntimeError(f"Coordinate mismatch in {path}")
            entries.append(
                {
                    "path": path,
                    "Re": re_value,
                    "times": times.copy(),
                    "start": offset,
                    "stop": offset + len(times),
                }
            )
            offset += len(times)
    if coords_ref is None:
        raise RuntimeError("No npz files found")
    return entries, coords_ref, offset


def compute_means(entries, area, pressure_gauge):
    import numpy as np

    u_sum = None
    p_sum = None
    count = 0
    area_sum = float(np.sum(area))
    for entry in entries:
        with np.load(entry["path"], allow_pickle=True) as data:
            u = data["U"].astype(np.float64)
            p = data["p"].astype(np.float64)
            if pressure_gauge == "subtract_area_mean_per_snapshot":
                p = p - (p @ area / area_sum)[:, None]
            u_sum = np.sum(u, axis=0) if u_sum is None else u_sum + np.sum(u, axis=0)
            p_sum = np.sum(p, axis=0) if p_sum is None else p_sum + np.sum(p, axis=0)
            count += u.shape[0]
    return u_sum / count, p_sum / count


def load_centered(entry, u_mean, p_mean, area, pressure_gauge):
    import numpy as np

    area_sum = float(np.sum(area))
    with np.load(entry["path"], allow_pickle=True) as data:
        u = data["U"].astype(np.float64) - u_mean[None, :, :]
        p = data["p"].astype(np.float64)
        if pressure_gauge == "subtract_area_mean_per_snapshot":
            p = p - (p @ area / area_sum)[:, None]
        p = p - p_mean[None, :]
    return u, p


def weighted_velocity_matrix(u, sqrt_area):
    return (u * sqrt_area[None, :, None]).reshape(u.shape[0], -1)


def weighted_pressure_matrix(p, sqrt_area):
    return p * sqrt_area[None, :]


def build_grams(entries, u_mean, p_mean, area, pressure_gauge):
    import numpy as np

    total_snapshots = entries[-1]["stop"]
    gram_u = np.zeros((total_snapshots, total_snapshots), dtype=np.float64)
    gram_p = np.zeros_like(gram_u)
    sqrt_area = np.sqrt(area)

    for i, entry_i in enumerate(entries):
        u_i, p_i = load_centered(entry_i, u_mean, p_mean, area, pressure_gauge)
        wu_i = weighted_velocity_matrix(u_i, sqrt_area)
        wp_i = weighted_pressure_matrix(p_i, sqrt_area)
        si = slice(entry_i["start"], entry_i["stop"])
        for entry_j in entries[i:]:
            u_j, p_j = load_centered(entry_j, u_mean, p_mean, area, pressure_gauge)
            wu_j = weighted_velocity_matrix(u_j, sqrt_area)
            wp_j = weighted_pressure_matrix(p_j, sqrt_area)
            sj = slice(entry_j["start"], entry_j["stop"])
            block_u = wu_i @ wu_j.T
            block_p = wp_i @ wp_j.T
            gram_u[si, sj] = block_u
            gram_p[si, sj] = block_p
            if entry_i is not entry_j:
                gram_u[sj, si] = block_u.T
                gram_p[sj, si] = block_p.T
    return gram_u, gram_p


def eig_from_gram(gram, rank):
    import numpy as np

    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    keep = min(rank, int(np.sum(values > 0)))
    values = values[:keep]
    vectors = vectors[:, :keep]
    singular = np.sqrt(values)
    energy = values / np.sum(values) if np.sum(values) > 0 else values
    return singular, energy, vectors


def build_modes(entries, u_mean, p_mean, area, pressure_gauge, vec_u, sig_u, vec_p, sig_p):
    import numpy as np

    ru = sig_u.shape[0]
    rp = sig_p.shape[0]
    n = area.shape[0]
    phi_u = np.zeros((ru, n, 2), dtype=np.float64)
    phi_p = np.zeros((rp, n), dtype=np.float64)

    for entry in entries:
        u, p = load_centered(entry, u_mean, p_mean, area, pressure_gauge)
        idx = slice(entry["start"], entry["stop"])
        if ru:
            coeff_u = vec_u[idx, :] / sig_u[None, :]
            phi_u += np.einsum("tk,tni->kni", coeff_u, u, optimize=True)
        if rp:
            coeff_p = vec_p[idx, :] / sig_p[None, :]
            phi_p += np.einsum("tk,tn->kn", coeff_p, p, optimize=True)

    # Normalize defensively in the requested weighted inner products.
    for k in range(ru):
        norm = np.sqrt(np.sum(area * np.sum(phi_u[k] ** 2, axis=1)))
        if norm > 0:
            phi_u[k] /= norm
    for k in range(rp):
        norm = np.sqrt(np.sum(area * phi_p[k] ** 2))
        if norm > 0:
            phi_p[k] /= norm
    return phi_u, phi_p


def coefficients_by_re(entries, vec_u, sig_u, vec_p, sig_p):
    import numpy as np

    re_list = []
    times = []
    a = []
    b = []
    for entry in entries:
        idx = slice(entry["start"], entry["stop"])
        re_list.append(entry["Re"])
        times.append(entry["times"])
        a.append(vec_u[idx, :] * sig_u[None, :])
        b.append(vec_p[idx, :] * sig_p[None, :])
    return (
        np.asarray(re_list, dtype=np.float64),
        np.asarray(times, dtype=object),
        np.asarray(a, dtype=object),
        np.asarray(b, dtype=object),
    )


def main() -> None:
    args = parse_args()
    configure_threads(args)

    import numpy as np

    files = sorted(p for p in args.npz_dir.glob("Re_*.npz") if p.name != "point_area.npz")
    if not files:
        raise SystemExit(f"No Re_*.npz files found in {args.npz_dir}")
    with np.load(args.area_file, allow_pickle=True) as area_data:
        coords_area = area_data["coords"]
        area = area_data["area"].astype(np.float64)

    entries, coords, total_snapshots = load_case_index(files)
    if coords.shape != coords_area.shape or not np.allclose(coords, coords_area, atol=1e-10):
        raise RuntimeError("Area weights do not match npz coordinates")
    if args.num_workers != 1:
        print("Note: --num-workers is accepted for workflow compatibility; dense BLAS handles the heavy dot products.")

    print(f"cases={len(entries)} snapshots={total_snapshots} points={coords.shape[0]}")
    u_mean, p_mean = compute_means(entries, area, args.pressure_gauge)
    gram_u, gram_p = build_grams(entries, u_mean, p_mean, area, args.pressure_gauge)
    sig_u, energy_u, vec_u = eig_from_gram(gram_u, args.ru)
    sig_p, energy_p, vec_p = eig_from_gram(gram_p, args.rp)
    phi_u, phi_p = build_modes(
        entries, u_mean, p_mean, area, args.pressure_gauge, vec_u, sig_u, vec_p, sig_p
    )
    re_list, times_by_re, a_by_re, b_by_re = coefficients_by_re(entries, vec_u, sig_u, vec_p, sig_p)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "velocity_pod.npz",
        U_mean=u_mean.astype(np.float32),
        Phi_u=phi_u.astype(np.float32),
        singular_values_u=sig_u,
        energy_u=energy_u,
        area=area,
        coords=coords,
    )
    np.savez_compressed(
        args.output_dir / "pressure_pod.npz",
        p_mean=p_mean.astype(np.float32),
        Phi_p=phi_p.astype(np.float32),
        singular_values_p=sig_p,
        energy_p=energy_p,
        area=area,
        coords=coords,
        pressure_gauge=np.asarray(args.pressure_gauge),
    )
    np.savez_compressed(
        args.output_dir / "modal_coefficients.npz",
        Re_list=re_list,
        times_by_Re=times_by_re,
        a_by_Re=a_by_re,
        b_by_Re=b_by_re,
    )
    print(f"Wrote POD files to {args.output_dir}")


if __name__ == "__main__":
    main()
