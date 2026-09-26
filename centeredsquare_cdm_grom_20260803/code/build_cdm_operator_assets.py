#!/usr/bin/env python3
"""Construct deterministic CDM-GROM auxiliary operators from a full Galerkin ROM.

The first 11 velocity coordinates are resolved. Coordinates 12--64 define a
53-state unresolved linear realization. Each parameter node is linearized at
an operator equilibrium when a trustworthy root is found; otherwise the
train-trajectory tail mean is used and the affine unresolved forcing is kept.
No validation or held-out field is loaded by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolved-rank", type=int, default=11)
    parser.add_argument("--stability-margin", type=float, default=0.02)
    parser.add_argument("--tail-snapshots", type=int, default=32)
    parser.add_argument("--root-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--root-max-nfev", type=int, default=3000)
    parser.add_argument(
        "--reference-policy",
        choices=["train_tail_mean", "operator_equilibrium"],
        default="train_tail_mean",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def single_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, found {matches}")
    return matches[0]


def effective_tensors(
    c: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    pressure_coupling: np.ndarray,
    pressure_c: np.ndarray,
    pressure_linear: np.ndarray,
    pressure_quadratic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    effective_c = c + pressure_coupling @ pressure_c
    effective_linear = linear + pressure_coupling @ pressure_linear
    effective_quadratic = quadratic + np.einsum(
        "im,mjk->ijk", pressure_coupling, pressure_quadratic, optimize=True
    )
    return effective_c, effective_linear, effective_quadratic


def make_field(
    constant: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    def field(state: np.ndarray) -> np.ndarray:
        return (
            constant
            + linear @ state
            + np.einsum("ijk,j,k->i", quadratic, state, state, optimize=True)
        )

    def jacobian(state: np.ndarray) -> np.ndarray:
        return (
            linear
            + np.einsum("ilk,k->il", quadratic, state, optimize=True)
            + np.einsum("ijl,j->il", quadratic, state, optimize=True)
        )

    return field, jacobian


def relative_residual(
    state: np.ndarray,
    field_value: np.ndarray,
    constant: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
) -> float:
    scale = max(
        1.0,
        float(np.linalg.norm(constant)),
        float(np.linalg.norm(linear @ state)),
        float(np.linalg.norm(np.einsum("ijk,j,k->i", quadratic, state, state))),
    )
    return float(np.linalg.norm(field_value) / scale)


def equilibrium_candidate(
    initial: np.ndarray,
    field: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = least_squares(
        field,
        initial,
        jac=jacobian,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=max_nfev,
        x_scale="jac",
    )
    return np.asarray(result.x, dtype=np.float64), {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "njev": int(result.njev) if result.njev is not None else None,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
    }


def finite_difference_jacobian_error(
    state: np.ndarray,
    field: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=len(state))
    direction /= np.linalg.norm(direction)
    epsilon = 2.0e-6 * max(1.0, float(np.linalg.norm(state)))
    finite_difference = (
        field(state + epsilon * direction) - field(state - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = jacobian(state) @ direction
    return float(
        np.linalg.norm(finite_difference - analytic)
        / max(np.linalg.norm(analytic), 1.0e-12)
    )


def reduced_polynomial(
    effective_c: np.ndarray,
    effective_linear: np.ndarray,
    effective_quadratic: np.ndarray,
    reference_unresolved: np.ndarray,
    resolved_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = resolved_rank
    u = slice(r, None)
    constant = (
        effective_c[:r]
        + effective_linear[:r, u] @ reference_unresolved
        + np.einsum(
            "ijk,j,k->i",
            effective_quadratic[:r, u, u],
            reference_unresolved,
            reference_unresolved,
            optimize=True,
        )
    )
    linear = (
        effective_linear[:r, :r]
        + np.einsum(
            "ijk,k->ij",
            effective_quadratic[:r, :r, u],
            reference_unresolved,
            optimize=True,
        )
        + np.einsum(
            "ikj,k->ij",
            effective_quadratic[:r, u, :r],
            reference_unresolved,
            optimize=True,
        )
    )
    quadratic = effective_quadratic[:r, :r, :r]
    return constant, linear, quadratic


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.stability_margin <= 0:
        raise ValueError("stability margin must be positive")

    velocity_path = single_match(args.tensor_dir, "semi_intrusive_*compact.npz")
    pressure_path = single_match(args.tensor_dir, "pressure_poisson_*.npz")
    velocity_pod_path = args.dataset_root / "pod" / "weighted_pod_velocity.npz"
    pressure_pod_path = args.dataset_root / "pod" / "weighted_pod_pressure.npz"

    with np.load(velocity_path, allow_pickle=False) as velocity:
        reynolds = np.asarray(velocity["Re_list"], dtype=np.float64)
        c_all = np.asarray(velocity["c_all"], dtype=np.float64)
        a_all = np.asarray(velocity["A_all"], dtype=np.float64)
        h_tensor = np.asarray(velocity["H"], dtype=np.float64)
        pressure_coupling = np.asarray(velocity["P"], dtype=np.float64)
    with np.load(pressure_path, allow_pickle=False) as pressure:
        pressure_re = np.asarray(pressure["Re_list"], dtype=np.float64)
        pressure_c_all = np.asarray(pressure["c_tilde_all"], dtype=np.float64)
        pressure_a_all = np.asarray(pressure["A_tilde_all"], dtype=np.float64)
        pressure_h = np.asarray(pressure["H_tilde"], dtype=np.float64)
    if not np.allclose(reynolds, pressure_re, atol=1.0e-10, rtol=0):
        raise RuntimeError("velocity and pressure Re nodes differ")

    with np.load(velocity_pod_path, allow_pickle=False) as velocity_pod:
        coefficients = np.asarray(velocity_pod["coefficients"], dtype=np.float64)
        tags = np.asarray(velocity_pod["snapshot_case_tags"]).astype(str)
        stored_rank = int(velocity_pod["stored_modes"])
    with np.load(pressure_pod_path, allow_pickle=False) as pressure_pod:
        pressure_stored_rank = int(pressure_pod["stored_modes"])
    full_rank = c_all.shape[1]
    if stored_rank != full_rank or pressure_stored_rank != pressure_c_all.shape[1]:
        raise RuntimeError(
            f"POD/tensor rank mismatch: velocity {stored_rank}/{full_rank}, "
            f"pressure {pressure_stored_rank}/{pressure_c_all.shape[1]}"
        )
    if not 0 < args.resolved_rank < full_rank:
        raise ValueError("resolved rank must be smaller than full rank")
    if h_tensor.shape != (full_rank, full_rank, full_rank):
        raise RuntimeError(f"unexpected velocity H shape {h_tensor.shape}")
    if pressure_h.shape[1:] != (full_rank, full_rank):
        raise RuntimeError(f"unexpected pressure H shape {pressure_h.shape}")

    references = []
    reference_kinds = []
    f_references = []
    base_constants = []
    base_linears = []
    base_quadratics = []
    trunc_constants = []
    trunc_linears = []
    trunc_quadratics = []
    a_ru_all = []
    a_ur_all = []
    a_uu_raw_all = []
    a_uu_stable_all = []
    shifts = []
    raw_abscissae = []
    stable_abscissae = []
    jacobian_errors = []
    node_reports: list[dict[str, Any]] = []
    previous_equilibrium: np.ndarray | None = None

    for node, re_value in enumerate(reynolds):
        effective_c, effective_a, effective_h = effective_tensors(
            c_all[node],
            a_all[node],
            h_tensor,
            pressure_coupling,
            pressure_c_all[node],
            pressure_a_all[node],
            pressure_h,
        )
        field, jacobian = make_field(effective_c, effective_a, effective_h)
        tag = ("Re" + f"{float(re_value):010.6f}").replace(".", "p")
        ids = np.flatnonzero(tags == tag)
        if len(ids) < args.tail_snapshots:
            raise RuntimeError(f"insufficient train coefficients for {tag}: {len(ids)}")
        tail_mean = coefficients[ids[-args.tail_snapshots :]].mean(axis=0)
        candidate_reports = []
        best_state = tail_mean
        best_kind = "train_tail_mean"
        best_residual = relative_residual(
            best_state, field(best_state), effective_c, effective_a, effective_h
        )
        if args.reference_policy == "operator_equilibrium":
            candidates = [("tail_mean_root", tail_mean)]
            if previous_equilibrium is not None:
                candidates.insert(0, ("continuation_root", previous_equilibrium))
            for kind, initial in candidates:
                state, report = equilibrium_candidate(
                    initial, field, jacobian, args.root_max_nfev
                )
                residual = relative_residual(
                    state, field(state), effective_c, effective_a, effective_h
                )
                norm_guard = float(np.linalg.norm(state)) <= 10.0 * max(
                    1.0, float(np.linalg.norm(initial))
                )
                accepted = (
                    report["success"]
                    and residual <= args.root_relative_tolerance
                    and norm_guard
                )
                candidate_reports.append(
                    {
                        "kind": kind,
                        **report,
                        "relative_residual": residual,
                        "state_norm": float(np.linalg.norm(state)),
                        "norm_guard": norm_guard,
                        "accepted": accepted,
                    }
                )
                if accepted and residual < best_residual:
                    best_state = state
                    best_kind = kind
                    best_residual = residual
        if best_kind.endswith("root"):
            previous_equilibrium = best_state.copy()

        f_reference = field(best_state)
        full_jacobian = jacobian(best_state)
        jacobian_error = finite_difference_jacobian_error(
            best_state, field, jacobian, seed=20260803 + node
        )
        if jacobian_error > 2.0e-5:
            raise RuntimeError(
                f"Jacobian finite-difference check failed at Re={re_value}: {jacobian_error}"
            )

        r = args.resolved_rank
        a_ru = full_jacobian[:r, r:]
        a_ur = full_jacobian[r:, :r]
        a_uu_raw = full_jacobian[r:, r:]
        raw_eigenvalues = np.linalg.eigvals(a_uu_raw)
        raw_abscissa = float(np.max(raw_eigenvalues.real))
        shift = max(0.0, raw_abscissa + args.stability_margin)
        a_uu_stable = a_uu_raw - shift * np.eye(full_rank - r)
        stable_eigenvalues = np.linalg.eigvals(a_uu_stable)
        stable_abscissa = float(np.max(stable_eigenvalues.real))
        if stable_abscissa > -args.stability_margin + 2.0e-10:
            raise RuntimeError(
                f"stability correction failed at Re={re_value}: {stable_abscissa}"
            )
        base_c, base_a, base_h = reduced_polynomial(
            effective_c, effective_a, effective_h, best_state[r:], r
        )

        references.append(best_state)
        reference_kinds.append(best_kind)
        f_references.append(f_reference)
        base_constants.append(base_c)
        base_linears.append(base_a)
        base_quadratics.append(base_h)
        trunc_constants.append(effective_c[:r])
        trunc_linears.append(effective_a[:r, :r])
        trunc_quadratics.append(effective_h[:r, :r, :r])
        a_ru_all.append(a_ru)
        a_ur_all.append(a_ur)
        a_uu_raw_all.append(a_uu_raw)
        a_uu_stable_all.append(a_uu_stable)
        shifts.append(shift)
        raw_abscissae.append(raw_abscissa)
        stable_abscissae.append(stable_abscissa)
        jacobian_errors.append(jacobian_error)
        node_reports.append(
            {
                "Re": float(re_value),
                "reference_kind": best_kind,
                "reference_relative_residual": best_residual,
                "reference_norm": float(np.linalg.norm(best_state)),
                "f_reference_norm": float(np.linalg.norm(f_reference)),
                "jacobian_directional_relative_error": jacobian_error,
                "raw_unresolved_spectral_abscissa": raw_abscissa,
                "stability_shift": shift,
                "stable_unresolved_spectral_abscissa": stable_abscissa,
                "raw_unstable_eigenvalue_count": int(np.sum(raw_eigenvalues.real >= 0)),
                "candidate_roots": candidate_reports,
            }
        )

    args.output_dir.mkdir(parents=True)
    asset_path = args.output_dir / "centeredsquare_hopf_cdm_operator_r11_u53.npz"
    np.savez_compressed(
        asset_path,
        Re_nodes=reynolds,
        resolved_rank=np.asarray(args.resolved_rank, dtype=np.int64),
        full_velocity_rank=np.asarray(full_rank, dtype=np.int64),
        full_pressure_rank=np.asarray(pressure_c_all.shape[1], dtype=np.int64),
        reference=np.asarray(references),
        reference_kind=np.asarray(reference_kinds),
        f_reference=np.asarray(f_references),
        base_c=np.asarray(base_constants),
        base_A=np.asarray(base_linears),
        base_H=np.asarray(base_quadratics),
        trunc_c=np.asarray(trunc_constants),
        trunc_A=np.asarray(trunc_linears),
        trunc_H=np.asarray(trunc_quadratics),
        A_ru=np.asarray(a_ru_all),
        A_ur=np.asarray(a_ur_all),
        A_uu_raw=np.asarray(a_uu_raw_all),
        A_uu_stable=np.asarray(a_uu_stable_all),
        stability_shift=np.asarray(shifts),
        raw_spectral_abscissa=np.asarray(raw_abscissae),
        stable_spectral_abscissa=np.asarray(stable_abscissae),
        pressure_c=pressure_c_all,
        pressure_A=pressure_a_all,
        pressure_H=pressure_h,
        stability_margin=np.asarray(args.stability_margin),
    )
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "method": "operator-based CDM-GROM",
        "dataset_root": str(args.dataset_root),
        "resolved_rank": args.resolved_rank,
        "unresolved_rank": full_rank - args.resolved_rank,
        "pressure_rank": int(pressure_c_all.shape[1]),
        "stability_policy": (
            "use exact A_uu if spectral abscissa <= -margin; otherwise subtract "
            "(spectral_abscissa + margin) I"
        ),
        "stability_margin": args.stability_margin,
        "validation_loaded": False,
        "heldout_loaded": False,
        "reference_policy": args.reference_policy,
        "reference_policy_detail": (
            "train-tail temporal mean with affine f_reference retained"
            if args.reference_policy == "train_tail_mean"
            else "accepted operator equilibrium with train-tail fallback"
        ),
        "root_relative_tolerance": args.root_relative_tolerance,
        "max_jacobian_directional_relative_error": float(max(jacobian_errors)),
        "nodes_requiring_stability_shift": int(np.sum(np.asarray(shifts) > 0)),
        "maximum_stability_shift": float(max(shifts)),
        "maximum_raw_spectral_abscissa": float(max(raw_abscissae)),
        "maximum_stable_spectral_abscissa": float(max(stable_abscissae)),
        "nodes": node_reports,
        "sha256": {
            "operator_asset": sha256(asset_path),
            "velocity_tensor": sha256(velocity_path),
            "pressure_tensor": sha256(pressure_path),
            "velocity_pod": sha256(velocity_pod_path),
            "pressure_pod": sha256(pressure_pod_path),
        },
    }
    atomic_json(manifest, args.output_dir / "CDM_OPERATOR_MANIFEST.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
