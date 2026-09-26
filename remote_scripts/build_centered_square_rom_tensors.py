#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_DATASET = Path("/home/ray/Desktop/centeredSquare/dataset_Re50_150_N100_npz")


@dataclass(frozen=True)
class RankSpec:
    label: str
    ru: int
    rp: int


@dataclass
class LSQOperator:
    indices: np.ndarray
    coeffs: np.ndarray
    neighbors: int
    power: float
    ridge: float
    coordinate_dims: int


@dataclass
class VTKDerivativeOperator:
    grid: object
    vtk_path: Path
    n_cells: int
    n_points: int
    max_center_delta: float
    center_atol: float
    preference: str = "cell"


def parse_rank_spec(text: str) -> RankSpec:
    try:
        label, ru, rp = text.split(":")
        return RankSpec(label=label, ru=int(ru), rp=int(rp))
    except Exception as exc:
        raise argparse.ArgumentTypeError("rank spec must look like rank99:12:6") from exc


def re_tag(re_value: float) -> str:
    return ("Re" + f"{float(re_value):010.6f}").replace(".", "p")


def configure_threads(threads: int | None) -> None:
    if not threads:
        return
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = str(threads)


def find_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of these files exist: " + ", ".join(str(p) for p in paths))


def load_re_list(dataset: Path) -> tuple[np.ndarray, list[str]]:
    manifest = dataset / "manifest" / "re_points_100.csv"
    if manifest.exists():
        rows = list(csv.DictReader(manifest.open(newline="")))
        re_values = np.asarray([float(r["Re"]) for r in rows], dtype=np.float64)
        tags = [r.get("case_tag") or re_tag(float(r["Re"])) for r in rows]
        return re_values, tags
    left = np.linspace(50.0, 80.0, 30, endpoint=False)
    mid = np.linspace(80.0, 100.0, 40, endpoint=False)
    right = np.linspace(100.0, 150.0, 30, endpoint=True)
    re_values = np.concatenate([left, mid, right]).astype(np.float64)
    return re_values, [re_tag(x) for x in re_values]


def load_mesh_and_pod(dataset: Path, max_ru: int, max_rp: int) -> dict[str, np.ndarray | Path]:
    pod_dir = dataset / "pod"
    vel_path = find_existing([pod_dir / "weighted_pod_velocity.npz", pod_dir / "velocity_pod.npz"])
    prs_path = find_existing([pod_dir / "weighted_pod_pressure.npz", pod_dir / "pressure_pod.npz"])
    mesh_path = find_existing([dataset / "mesh" / "mesh_metadata.npz"])

    vel = np.load(vel_path, allow_pickle=True)
    prs = np.load(prs_path, allow_pickle=True)
    mesh = np.load(mesh_path, allow_pickle=True)

    coords = np.asarray(mesh["cellCenters"] if "cellCenters" in mesh.files else mesh["coords"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise RuntimeError(f"invalid coordinates shape: {coords.shape}")
    volumes = np.asarray(mesh["cellVolumes"] if "cellVolumes" in mesh.files else mesh["volumes"], dtype=np.float64)
    if volumes.ndim != 1 or volumes.shape[0] != coords.shape[0]:
        raise RuntimeError(f"invalid volume shape {volumes.shape} for coords {coords.shape}")
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise RuntimeError("mesh volumes contain non-positive or non-finite values")

    u_mean = np.asarray(vel["mean"] if "mean" in vel.files else vel["U_mean"], dtype=np.float64)
    p_mean = np.asarray(prs["mean"] if "mean" in prs.files else prs["p_mean"], dtype=np.float64)
    if u_mean.shape != (coords.shape[0], 2):
        raise RuntimeError(f"velocity mean shape {u_mean.shape} does not match coords")
    if p_mean.shape != (coords.shape[0],):
        raise RuntimeError(f"pressure mean shape {p_mean.shape} does not match coords")

    phi_u = np.asarray(vel["modes"] if "modes" in vel.files else vel["Phi_u"], dtype=np.float64)
    phi_p = np.asarray(prs["modes"] if "modes" in prs.files else prs["Phi_p"], dtype=np.float64)
    n = coords.shape[0]
    if phi_u.ndim == 2:
        phi_u = phi_u.reshape(phi_u.shape[0], n, 2)
    if phi_p.ndim == 3 and phi_p.shape[-1] == 1:
        phi_p = phi_p[..., 0]
    if phi_u.shape[1:] != (n, 2):
        raise RuntimeError(f"velocity modes shape {phi_u.shape} does not match (modes,{n},2)")
    if phi_p.shape[1:] != (n,):
        raise RuntimeError(f"pressure modes shape {phi_p.shape} does not match (modes,{n})")
    if phi_u.shape[0] < max_ru or phi_p.shape[0] < max_rp:
        raise RuntimeError(f"POD files do not contain requested ranks: velocity={phi_u.shape[0]}, pressure={phi_p.shape[0]}")

    out: dict[str, np.ndarray | Path] = {
        "coords": coords,
        "volumes": volumes,
        "u_mean": u_mean,
        "p_mean": p_mean,
        "phi_u": phi_u[:max_ru],
        "phi_p": phi_p[:max_rp],
        "velocity_pod_path": vel_path,
        "pressure_pod_path": prs_path,
        "mesh_path": mesh_path,
    }
    for key in ["singular_values", "cumulative_energy", "coefficients", "snapshot_times", "snapshot_case_tags"]:
        if key in vel.files:
            out[f"u_{key}"] = np.asarray(vel[key])
        if key in prs.files:
            out[f"p_{key}"] = np.asarray(prs[key])
    if "energy_u" in vel.files:
        out["u_cumulative_energy"] = np.cumsum(np.asarray(vel["energy_u"], dtype=np.float64))
    if "energy_p" in prs.files:
        out["p_cumulative_energy"] = np.cumsum(np.asarray(prs["energy_p"], dtype=np.float64))
    if "singular_values_u" in vel.files:
        out["u_singular_values"] = np.asarray(vel["singular_values_u"])
    if "singular_values_p" in prs.files:
        out["p_singular_values"] = np.asarray(prs["singular_values_p"])
    return out


def build_lsq_operator(
    coords: np.ndarray,
    neighbors: int,
    power: float,
    ridge: float,
    coordinate_dims: int = 2,
) -> LSQOperator:
    xy = np.asarray(coords[:, :coordinate_dims], dtype=np.float64)
    if coordinate_dims != 2:
        raise RuntimeError("this ROM builder currently expects a 2D flow field")
    distances, indices = nearest_neighbors(xy, neighbors)

    n = xy.shape[0]
    out_idx = np.empty((n, neighbors), dtype=np.int64)
    coeffs = np.empty((n, 5, neighbors), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for i in range(n):
        cand = indices[i]
        dist = distances[i]
        if cand.size != neighbors:
            raise RuntimeError(f"could not find {neighbors} neighbors for cell {i}")
        out_idx[i] = cand

        dxy = xy[cand] - xy[i]
        a = np.column_stack(
            [
                dxy[:, 0],
                dxy[:, 1],
                0.5 * dxy[:, 0] ** 2,
                dxy[:, 0] * dxy[:, 1],
                0.5 * dxy[:, 1] ** 2,
            ]
        )
        w = 1.0 / np.maximum(dist, eps) ** power
        w /= np.max(w)
        lhs = a.T @ (w[:, None] * a)
        scale = float(np.trace(lhs) / max(lhs.shape[0], 1))
        lhs = lhs + (ridge * max(scale, 1.0)) * np.eye(5)
        rhs = a.T * w[None, :]
        try:
            coeffs[i] = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coeffs[i] = np.linalg.pinv(lhs) @ rhs

    return LSQOperator(
        indices=out_idx,
        coeffs=coeffs,
        neighbors=neighbors,
        power=power,
        ridge=ridge,
        coordinate_dims=coordinate_dims,
    )


def discover_reference_vtk(dataset: Path) -> Path | None:
    ref_dir = dataset / "reference_vtk"
    if not ref_dir.exists():
        return None
    candidates = sorted(
        p
        for p in ref_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".vtk", ".vtu", ".vtp", ".vti"}
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise RuntimeError(f"expected one reference VTK in {ref_dir}, found {len(candidates)}")
    return candidates[0]


def build_vtk_operator(vtk_path: Path, coords: np.ndarray, center_atol: float) -> VTKDerivativeOperator:
    try:
        import pyvista as pv
    except Exception as exc:
        raise RuntimeError("pyvista/vtk is required for the VTK derivative backend") from exc

    grid = pv.read(vtk_path)
    if grid.n_cells != coords.shape[0]:
        raise RuntimeError(f"VTK cell count {grid.n_cells} does not match POD cells {coords.shape[0]}")
    centers = np.asarray(grid.cell_centers().points, dtype=np.float64)
    if centers.shape != coords.shape:
        raise RuntimeError(f"VTK cell center shape {centers.shape} does not match POD coords {coords.shape}")
    max_delta = float(np.max(np.abs(centers - coords)))
    if max_delta > center_atol:
        raise RuntimeError(f"VTK/POD cell centers are not aligned: max_abs_delta={max_delta:.6e} > {center_atol:.6e}")
    if "cellID" in grid.cell_data:
        cell_id = np.asarray(grid.cell_data["cellID"])
        if cell_id.shape == (grid.n_cells,) and not np.array_equal(cell_id, np.arange(grid.n_cells)):
            raise RuntimeError("VTK cellID is not the identity ordering expected by the POD arrays")
    return VTKDerivativeOperator(
        grid=grid,
        vtk_path=vtk_path,
        n_cells=int(grid.n_cells),
        n_points=int(grid.n_points),
        max_center_delta=max_delta,
        center_atol=center_atol,
    )


def nearest_neighbors(xy: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    if neighbors <= 5:
        raise RuntimeError("at least 6 neighbors are required for the quadratic LSQ derivative")
    if xy.shape[0] <= neighbors:
        raise RuntimeError(f"neighbors={neighbors} is too large for {xy.shape[0]} cells")
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xy)
        distances, indices = tree.query(xy, k=neighbors + 1)
        distances = np.asarray(distances)
        indices = np.asarray(indices)
        if indices.ndim == 1:
            indices = indices[:, None]
            distances = distances[:, None]
        return distances[:, 1:], indices[:, 1:]
    except Exception:
        return nearest_neighbors_numpy(xy, neighbors)


def nearest_neighbors_numpy(xy: np.ndarray, neighbors: int, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    n = xy.shape[0]
    all_idx = np.empty((n, neighbors), dtype=np.int64)
    all_dist = np.empty((n, neighbors), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        diff = xy[start:stop, None, :] - xy[None, :, :]
        dist2 = np.einsum("bnd,bnd->bn", diff, diff, optimize=True)
        rows = np.arange(stop - start)
        dist2[rows, np.arange(start, stop)] = np.inf
        part = np.argpartition(dist2, kth=neighbors - 1, axis=1)[:, :neighbors]
        part_dist2 = np.take_along_axis(dist2, part, axis=1)
        order = np.argsort(part_dist2, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        d2 = np.take_along_axis(part_dist2, order, axis=1)
        all_idx[start:stop] = idx
        all_dist[start:stop] = np.sqrt(d2)
    return all_dist, all_idx


def derivative_scalars(
    op: LSQOperator | VTKDerivativeOperator,
    fields: np.ndarray,
    chunk_fields: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(op, VTKDerivativeOperator):
        return derivative_scalars_vtk(op, fields)
    return derivative_scalars_lsq(op, fields, chunk_fields=chunk_fields)


def derivative_scalars_lsq(op: LSQOperator, fields: np.ndarray, chunk_fields: int = 32) -> tuple[np.ndarray, np.ndarray]:
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim == 1:
        fields = fields[None, :]
    f_count, n = fields.shape
    if n != op.indices.shape[0]:
        raise RuntimeError(f"field length {n} does not match derivative operator")
    deriv = np.empty((f_count, n, 5), dtype=np.float64)
    for start in range(0, f_count, chunk_fields):
        stop = min(start + chunk_fields, f_count)
        block = fields[start:stop]
        neighbor_values = block[:, op.indices]
        diffs = neighbor_values - block[:, :, None]
        deriv[start:stop] = np.einsum("ndk,fnk->fnd", op.coeffs, diffs, optimize=True)
    grad = deriv[:, :, :2]
    lap = deriv[:, :, 2] + deriv[:, :, 4]
    return grad, lap


def derivative_scalars_vtk(op: VTKDerivativeOperator, fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim == 1:
        fields = fields[None, :]
    f_count, n = fields.shape
    if n != op.n_cells:
        raise RuntimeError(f"field length {n} does not match VTK cell count {op.n_cells}")
    grad = np.empty((f_count, n, 2), dtype=np.float64)
    lap = np.empty((f_count, n), dtype=np.float64)
    grid = op.grid.copy(deep=True)
    scalar_name = "__rom_scalar"
    grad_name = "__rom_grad"
    for i in range(f_count):
        grid.cell_data[scalar_name] = fields[i]
        out = grid.compute_derivative(scalars=scalar_name, gradient=True, preference=op.preference)
        full_grad = np.asarray(out.cell_data["gradient"], dtype=np.float64)
        if full_grad.shape != (n, 3):
            raise RuntimeError(f"unexpected VTK gradient shape {full_grad.shape}")
        grad[i] = full_grad[:, :2]
        out.cell_data[grad_name] = full_grad
        out_lap = out.compute_derivative(scalars=grad_name, divergence=True, preference=op.preference)
        div = np.asarray(out_lap.cell_data["divergence"], dtype=np.float64)
        if div.shape != (n,):
            raise RuntimeError(f"unexpected VTK divergence shape {div.shape}")
        lap[i] = div
    return grad, lap


def derivative_vectors(op: LSQOperator | VTKDerivativeOperator, fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim == 2:
        fields = fields[None, :, :]
    f_count, n, comp = fields.shape
    if comp != 2:
        raise RuntimeError("velocity fields must have two components")
    flat = fields.transpose(0, 2, 1).reshape(f_count * comp, n)
    grad_flat, lap_flat = derivative_scalars(op, flat)
    grad = grad_flat.reshape(f_count, comp, n, 2).transpose(0, 2, 1, 3)
    lap = lap_flat.reshape(f_count, comp, n).transpose(0, 2, 1)
    return grad, lap


def mass_inner_vectors(weights: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("n,inc,jnc->ij", weights, left, right, optimize=True)


def solve_left_mass(g: np.ndarray, arr: np.ndarray) -> np.ndarray:
    original_shape = arr.shape
    flat = arr.reshape(original_shape[0], -1)
    solved = np.linalg.solve(g, flat)
    return solved.reshape(original_shape)


def build_rank_tensors(
    spec: RankSpec,
    dataset: Path,
    pod: dict[str, np.ndarray | Path],
    op: LSQOperator | VTKDerivativeOperator,
    re_values: np.ndarray,
    re_tags: list[str],
    output_root: Path,
) -> dict[str, object]:
    weights = np.asarray(pod["volumes"], dtype=np.float64)
    coords = np.asarray(pod["coords"], dtype=np.float64)
    ubar = np.asarray(pod["u_mean"], dtype=np.float64)
    pbar = np.asarray(pod["p_mean"], dtype=np.float64)
    phi_u = np.asarray(pod["phi_u"], dtype=np.float64)[: spec.ru]
    phi_p = np.asarray(pod["phi_p"], dtype=np.float64)[: spec.rp]

    grad_u_all, lap_u_all = derivative_vectors(op, np.concatenate([ubar[None, :, :], phi_u], axis=0))
    grad_p_all, _ = derivative_scalars(op, np.concatenate([pbar[None, :], phi_p], axis=0))
    grad_ubar, lap_ubar = grad_u_all[0], lap_u_all[0]
    grad_phi, lap_phi = grad_u_all[1:], lap_u_all[1:]
    grad_pbar = grad_p_all[0]
    grad_psi = grad_p_all[1:]

    g_u = mass_inner_vectors(weights, phi_u, phi_u)
    g_cond = float(np.linalg.cond(g_u))
    g_max_i = float(np.max(np.abs(g_u - np.eye(spec.ru))))

    conv_ubar = np.einsum("nd,ncd->nc", ubar, grad_ubar, optimize=True)
    ubar_dot_grad_phi = np.einsum("nd,jncd->jnc", ubar, grad_phi, optimize=True)
    phi_dot_grad_ubar = np.einsum("jnd,ncd->jnc", phi_u, grad_ubar, optimize=True)
    cross_conv = ubar_dot_grad_phi + phi_dot_grad_ubar
    mode_conv = np.einsum("jnd,kncd->jknc", phi_u, grad_phi, optimize=True)

    c_no_nu = np.einsum("n,inc,nc->i", weights, phi_u, -conv_ubar - grad_pbar, optimize=True)
    c_nu = np.einsum("n,inc,nc->i", weights, phi_u, lap_ubar, optimize=True)
    a_no_nu = np.einsum("n,inc,jnc->ij", weights, phi_u, -cross_conv, optimize=True)
    a_nu = np.einsum("n,inc,jnc->ij", weights, phi_u, lap_phi, optimize=True)
    h_raw = -np.einsum("n,inc,jknc->ijk", weights, phi_u, mode_conv, optimize=True)
    p_raw = -np.einsum("n,ind,mnd->im", weights, phi_u, grad_psi, optimize=True)

    h = solve_left_mass(g_u, h_raw)
    p_tensor = solve_left_mass(g_u, p_raw)

    c_all = np.empty((len(re_values), spec.ru), dtype=np.float64)
    a_all = np.empty((len(re_values), spec.ru, spec.ru), dtype=np.float64)
    for i, re_value in enumerate(re_values):
        nu = 1.0 / float(re_value)
        c_all[i] = solve_left_mass(g_u, c_no_nu + nu * c_nu)
        a_all[i] = solve_left_mass(g_u, a_no_nu + nu * a_nu)

    l_mat = -np.einsum("n,mnd,knd->mk", weights, grad_psi, grad_psi, optimize=True)
    l_rank = int(np.linalg.matrix_rank(l_mat))
    eig_l = np.linalg.eigvalsh(0.5 * (l_mat + l_mat.T))
    l_pinv = np.linalg.pinv(l_mat)
    l_pbar = -np.einsum("n,mnd,nd->m", weights, grad_psi, grad_pbar, optimize=True)
    cp_conv = np.einsum("n,mnd,nd->m", weights, grad_psi, conv_ubar, optimize=True)
    cp_lap = np.einsum("n,mnd,nd->m", weights, grad_psi, lap_ubar, optimize=True)
    ap_conv = np.einsum("n,mnd,jnd->mj", weights, grad_psi, cross_conv, optimize=True)
    ap_lap = np.einsum("n,mnd,jnd->mj", weights, grad_psi, lap_phi, optimize=True)
    hp_raw = np.einsum("n,mnd,jknd->mjk", weights, grad_psi, mode_conv, optimize=True)

    cp_all = np.empty((len(re_values), spec.rp), dtype=np.float64)
    ap_all = np.empty((len(re_values), spec.rp, spec.ru), dtype=np.float64)
    for i, re_value in enumerate(re_values):
        nu = 1.0 / float(re_value)
        cp_all[i] = cp_conv - nu * cp_lap - l_pbar
        ap_all[i] = ap_conv - nu * ap_lap
    c_tilde_all = np.einsum("mk,sk->sm", l_pinv, cp_all, optimize=True)
    a_tilde_all = np.einsum("mk,skj->smj", l_pinv, ap_all, optimize=True)
    h_tilde = np.einsum("mk,kij->mij", l_pinv, hp_raw, optimize=True)

    out_dir = output_root / f"{spec.label}_ru{spec.ru}_rp{spec.rp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    velocity_payload: dict[str, np.ndarray] = {
        "Re_list": re_values,
        "G_u": g_u,
        "c_all": c_all,
        "A_all": a_all,
        "H": h,
        "P": p_tensor,
        "c_raw_no_nu": c_no_nu,
        "c_raw_nu": c_nu,
        "A_raw_no_nu": a_no_nu,
        "A_raw_nu": a_nu,
        "H_raw": h_raw,
        "P_raw": p_raw,
        "ru": np.asarray(spec.ru, dtype=np.int64),
        "rp": np.asarray(spec.rp, dtype=np.int64),
    }
    pressure_payload: dict[str, np.ndarray] = {
        "Re_list": re_values,
        "L": l_mat,
        "L_pinv": l_pinv,
        "c_p_all": cp_all,
        "A_p_all": ap_all,
        "H_p": hp_raw,
        "c_tilde_all": c_tilde_all,
        "A_tilde_all": a_tilde_all,
        "H_tilde": h_tilde,
        "L_pbar": l_pbar,
        "ru": np.asarray(spec.ru, dtype=np.int64),
        "rp": np.asarray(spec.rp, dtype=np.int64),
    }
    for idx, tag in enumerate(re_tags):
        velocity_payload[f"{tag}_c"] = c_all[idx]
        velocity_payload[f"{tag}_A"] = a_all[idx]
        pressure_payload[f"{tag}_c_tilde"] = c_tilde_all[idx]
        pressure_payload[f"{tag}_A_tilde"] = a_tilde_all[idx]

    vel_out = out_dir / f"semi_intrusive_galerkin_tensors_Re50_150_N100_{spec.label}_ru{spec.ru}_rp{spec.rp}_compact.npz"
    prs_out = out_dir / f"pressure_poisson_surrogate_tensors_Re50_150_N100_{spec.label}_ru{spec.ru}_rp{spec.rp}.npz"
    np.savez_compressed(vel_out, **velocity_payload)
    np.savez_compressed(prs_out, **pressure_payload)

    pod_pack = out_dir / f"pod_rank_pack_Re50_150_N100_{spec.label}_ru{spec.ru}_rp{spec.rp}.npz"
    pack_payload: dict[str, np.ndarray] = {
        "coords": coords,
        "cellVolumes": weights,
        "U_mean": ubar.astype(np.float32),
        "p_mean": pbar.astype(np.float32),
        "Phi_u": phi_u.astype(np.float32),
        "Phi_p": phi_p.astype(np.float32),
        "Re_list": re_values,
        "ru": np.asarray(spec.ru, dtype=np.int64),
        "rp": np.asarray(spec.rp, dtype=np.int64),
    }
    optional_map = {
        "u_singular_values": "singular_values_u",
        "p_singular_values": "singular_values_p",
        "u_cumulative_energy": "cumulative_energy_u",
        "p_cumulative_energy": "cumulative_energy_p",
        "u_coefficients": "coefficients_u",
        "p_coefficients": "coefficients_p",
        "u_snapshot_times": "snapshot_times",
        "u_snapshot_case_tags": "snapshot_case_tags",
    }
    for src, dst in optional_map.items():
        if src in pod:
            arr = np.asarray(pod[src])
            if src.endswith("singular_values") or src.endswith("cumulative_energy"):
                limit = spec.ru if src.startswith("u_") else spec.rp
                arr = arr[:limit]
            elif src.endswith("coefficients"):
                limit = spec.ru if src.startswith("u_") else spec.rp
                arr = arr[:, :limit]
            pack_payload[dst] = arr
    np.savez_compressed(pod_pack, **pack_payload)

    pressure_consistency = float(
        np.linalg.norm(np.einsum("mk,kij->mij", l_mat, h_tilde, optimize=True) - hp_raw)
        / max(np.linalg.norm(hp_raw), np.finfo(np.float64).eps)
    )
    finite_ok = all(
        np.all(np.isfinite(x))
        for x in [c_all, a_all, h, p_tensor, l_mat, cp_all, ap_all, hp_raw, c_tilde_all, a_tilde_all, h_tilde]
    )
    summary = {
        "label": spec.label,
        "dataset": str(dataset),
        "output_dir": str(out_dir),
        "velocity_tensor_file": str(vel_out),
        "pressure_tensor_file": str(prs_out),
        "pod_rank_pack": str(pod_pack),
        "ru": spec.ru,
        "rp": spec.rp,
        "num_re": int(len(re_values)),
        "num_cells": int(coords.shape[0]),
        "G_u_shape": list(g_u.shape),
        "c_all_shape": list(c_all.shape),
        "A_all_shape": list(a_all.shape),
        "H_shape": list(h.shape),
        "P_shape": list(p_tensor.shape),
        "L_shape": list(l_mat.shape),
        "H_p_shape": list(hp_raw.shape),
        "H_tilde_shape": list(h_tilde.shape),
        "cond_G_u": g_cond,
        "max_abs_G_u_minus_I": g_max_i,
        "norm_H_fro": float(np.linalg.norm(h)),
        "norm_P_fro": float(np.linalg.norm(p_tensor)),
        "rank_L": l_rank,
        "eig_L_min": float(eig_l.min()),
        "eig_L_max": float(eig_l.max()),
        "pressure_H_tilde_relative_residual": pressure_consistency,
        "finite": bool(finite_ok),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown_report(out_dir / f"ROM_BUILD_REPORT_Re50_150_N100_{spec.label}_ru{spec.ru}_rp{spec.rp}.md", summary, op)
    return summary


def derivative_metadata(op: LSQOperator | VTKDerivativeOperator) -> dict[str, object]:
    if isinstance(op, VTKDerivativeOperator):
        return {
            "backend": "pyvista_vtk_compute_derivative_cell_data",
            "reference_vtk": str(op.vtk_path),
            "preference": op.preference,
            "n_cells": op.n_cells,
            "n_points": op.n_points,
            "max_center_delta": op.max_center_delta,
            "center_atol": op.center_atol,
        }
    return {
        "backend": "meshfree_quadratic_lsq_cell_centers",
        "neighbors": op.neighbors,
        "power": op.power,
        "ridge": op.ridge,
        "coordinate_dims": op.coordinate_dims,
    }


def write_markdown_report(path: Path, summary: dict[str, object], op: LSQOperator | VTKDerivativeOperator) -> None:
    if isinstance(op, VTKDerivativeOperator):
        derivative_lines = [
            "- Derivative backend: PyVista `compute_derivative()` on reference VTK cell data",
            f"- Reference VTK: `{op.vtk_path}`",
            f"- VTK cells/points: `{op.n_cells}` / `{op.n_points}`",
            f"- Max VTK/POD cell-center delta: `{op.max_center_delta:.6e}`",
        ]
    else:
        derivative_lines = [
            "- Derivative backend: meshfree quadratic least-squares over cell centers",
            f"- LSQ neighbors: `{op.neighbors}`, weight power: `{op.power}`, ridge: `{op.ridge}`",
        ]
    lines = [
        f"# CenteredSquare ROM Build Report - {summary['label']}",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Output dir: `{summary['output_dir']}`",
        f"- Ranks: `ru={summary['ru']}`, `rp={summary['rp']}`",
        f"- Re points: `{summary['num_re']}`",
        f"- Cells: `{summary['num_cells']}`",
        *derivative_lines,
        "",
        "## Velocity ROM",
        "",
        "```text",
        "da/dt = c(Re) + A(Re) a + H(a,a) + P b",
        "```",
        "",
        f"- `G_u` shape: `{summary['G_u_shape']}`",
        f"- `c_all` shape: `{summary['c_all_shape']}`",
        f"- `A_all` shape: `{summary['A_all_shape']}`",
        f"- `H` shape: `{summary['H_shape']}`",
        f"- `P` shape: `{summary['P_shape']}`",
        f"- `cond(G_u)`: `{summary['cond_G_u']:.6e}`",
        f"- `max|G_u-I|`: `{summary['max_abs_G_u_minus_I']:.6e}`",
        f"- `||H||_F`: `{summary['norm_H_fro']:.6e}`",
        f"- `||P||_F`: `{summary['norm_P_fro']:.6e}`",
        "",
        "## Pressure Poisson Surrogate",
        "",
        "```text",
        "L b(t) = c^p(Re) + A^p(Re) a(t) + H^p(a(t),a(t))",
        "b(t) = c_tilde(Re) + A_tilde(Re) a(t) + H_tilde(a(t),a(t))",
        "```",
        "",
        f"- `L` shape: `{summary['L_shape']}`",
        f"- `H_p` shape: `{summary['H_p_shape']}`",
        f"- `H_tilde` shape: `{summary['H_tilde_shape']}`",
        f"- `rank(L)`: `{summary['rank_L']}/{summary['rp']}`",
        f"- `eig(L) min/max`: `{summary['eig_L_min']:.6e}` / `{summary['eig_L_max']:.6e}`",
        f"- `rel ||L H_tilde-H_p||`: `{summary['pressure_H_tilde_relative_residual']:.6e}`",
        "",
        "## Files",
        "",
        f"- `{Path(str(summary['velocity_tensor_file'])).name}`",
        f"- `{Path(str(summary['pressure_tensor_file'])).name}`",
        f"- `{Path(str(summary['pod_rank_pack'])).name}`",
        f"- `manifest.json`",
        "",
        f"- Finite arrays: `{summary['finite']}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def make_bundle(dataset: Path, output_root: Path, summaries: list[dict[str, object]], pod: dict[str, np.ndarray | Path]) -> Path:
    bundle = output_root / "centeredSquare_Re50_150_N100_rank99_rank999_ROM_and_POD.tar.gz"
    include_paths = [
        Path(str(pod["velocity_pod_path"])),
        Path(str(pod["pressure_pod_path"])),
        Path(str(pod["mesh_path"])),
        dataset / "pod" / "pod_energy_report.csv",
        dataset / "RUN_SUMMARY.md",
    ]
    with tarfile.open(bundle, "w:gz") as tar:
        for path in include_paths:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(dataset)))
        ref_dir = dataset / "reference_vtk"
        if ref_dir.exists():
            for path in sorted(ref_dir.iterdir()):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(dataset)))
        for summary in summaries:
            out_dir = Path(str(summary["output_dir"]))
            for path in sorted(out_dir.glob("*")):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(dataset)))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rank99/rank99.9 semi-intrusive ROM tensors for the centeredSquare NPZ POD dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--rank", dest="ranks", type=parse_rank_spec, action="append", default=None)
    parser.add_argument("--derivative-backend", choices=["auto", "vtk", "lsq"], default="auto")
    parser.add_argument("--reference-vtk", type=Path, default=None)
    parser.add_argument("--vtk-center-atol", type=float, default=5e-6)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--lsq-power", type=float, default=1.0)
    parser.add_argument("--lsq-ridge", type=float, default=1e-12)
    parser.add_argument("--blas-threads", type=int, default=4)
    parser.add_argument("--skip-bundle", action="store_true")
    args = parser.parse_args()

    configure_threads(args.blas_threads)
    ranks = args.ranks or [RankSpec("rank99", 12, 6), RankSpec("rank999", 26, 13)]
    output_root = args.output_root or (args.dataset / "pod" / "rom")
    output_root.mkdir(parents=True, exist_ok=True)

    max_ru = max(r.ru for r in ranks)
    max_rp = max(r.rp for r in ranks)
    re_values, tags = load_re_list(args.dataset)
    pod = load_mesh_and_pod(args.dataset, max_ru=max_ru, max_rp=max_rp)
    reference_vtk = args.reference_vtk or discover_reference_vtk(args.dataset)
    if args.derivative_backend == "vtk" and reference_vtk is None:
        raise RuntimeError(f"--derivative-backend vtk requested, but no reference VTK was found in {args.dataset / 'reference_vtk'}")
    if args.derivative_backend == "vtk" or (args.derivative_backend == "auto" and reference_vtk is not None):
        op: LSQOperator | VTKDerivativeOperator = build_vtk_operator(
            reference_vtk,
            np.asarray(pod["coords"]),
            center_atol=args.vtk_center_atol,
        )
    else:
        op = build_lsq_operator(
            np.asarray(pod["coords"]),
            neighbors=args.neighbors,
            power=args.lsq_power,
            ridge=args.lsq_ridge,
        )
    summaries = [build_rank_tensors(spec, args.dataset, pod, op, re_values, tags, output_root) for spec in ranks]
    manifest = {
        "dataset": str(args.dataset),
        "output_root": str(output_root),
        "rank_specs": [asdict(r) for r in ranks],
        "derivative_operator": derivative_metadata(op),
        "summaries": summaries,
    }
    if not args.skip_bundle:
        manifest["bundle"] = str(make_bundle(args.dataset, output_root, summaries, pod))
    manifest_path = output_root / "ROM_PhysicsGeneralizable_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
