#!/usr/bin/env python3
"""Reproducible best-case flow-field selection and paper plotting.

The VTK file supplies topology only.  Predictions remain in their native POD
charts until the selected snapshot is reconstructed on the common CFD mesh.
No checkpoint is trained or modified by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista as pv
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter


EPS = 1.0e-12
METHOD_LABELS = {
    "truth": "Truth (CFD)",
    "proposed": "Proposed Specialist MoE",
    "vanilla": "FNN-MoE",
    "dataonly": "DataOnly-MoE",
    "s_only": "Steady Specialist",
    "p_only": "Periodic Specialist",
    "h_only": "Hopf Specialist",
    "t2c": "RAL-MoE-ROM",
}
METHOD_ROW_LABELS = {
    **METHOD_LABELS,
    "proposed": "Proposed\nSpecialist MoE",
    "t2c": "RAL-MoE-ROM",
}


@dataclass(frozen=True)
class Basis:
    phi_u: np.ndarray
    phi_p: np.ndarray
    mean_u: np.ndarray
    mean_p: np.ndarray
    areas: np.ndarray
    velocity_layout: str
    velocity_path: Path
    pressure_path: Path
    gram_u: np.ndarray
    cross_u: np.ndarray
    mean_u_energy: float
    gram_p: np.ndarray
    cross_p: np.ndarray
    mean_p_energy: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("specialist", "sh_fusion", "boundary_fusion"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--template-vtk", type=Path, required=True)
    parser.add_argument("--candidate-scan", action="store_true")
    parser.add_argument("--re", type=float)
    parser.add_argument("--time-index", type=int)
    parser.add_argument("--split", choices=("train", "validation", "heldout", "test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-vtk", action="store_true")
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Write PNG figures only; omit PDF outputs.",
    )
    parser.add_argument("--plot-truth-pred", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-last-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-best-frame-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--xlim", type=float, nargs=2, default=(2.0, 17.0))
    parser.add_argument("--ylim", type=float, nargs=2, default=(0.0, 4.0))
    parser.add_argument(
        "--error-cmap",
        type=str,
        default="magma",
        choices=("magma", "Reds", "thermal_r", "haline_r"),
        help="Colormap for pointwise error panels. thermal_r=lajolla-like warm, haline_r=batlowW-like scientific (default: magma).",
    )
    return parser.parse_args()


def _resolve_error_cmap(name):
    """Return a valid matplotlib colormap from the requested key.

    Maps user-facing names to the closest available colormap:
    haline_r  (cmocean) -- batlowW-like scientific
    thermal_r (cmocean) -- lajolla-like warm error
    Reds      (built-in)
    """
    if name in ("magma", "Reds"):
        return name
    try:
        from matplotlib import colormaps
        _ = colormaps[name]
        return name
    except Exception:
        pass
    try:
        import cmocean
        base = name.replace("_r", "")
        if hasattr(cmocean.cm, base):
            cmap_obj = getattr(cmocean.cm, base)
            return cmap_obj.reversed() if name.endswith("_r") else cmap_obj
    except ImportError:
        pass
    print(f"[WARN] colormap '{name}' not found; falling back to 'magma'")
    return "magma"


def load_basis(spec: dict[str, Any]) -> Basis:
    velocity_path = Path(spec["velocity"])
    pressure_path = Path(spec["pressure"])
    with np.load(velocity_path, allow_pickle=False) as velocity:
        phi_u = np.asarray(velocity[spec.get("phi_u_key", "phi_uv")], np.float64)[
            : int(spec.get("r_u", 32))
        ]
        mean_u = np.asarray(
            velocity[spec.get("mean_u_key", "mean_uv_regime")], np.float64
        )
        areas = np.asarray(
            velocity[spec.get("area_key", "point_areas")], np.float64
        )
    with np.load(pressure_path, allow_pickle=False) as pressure:
        phi_p = np.asarray(pressure[spec.get("phi_p_key", "phi_p")], np.float64)[
            : int(spec.get("r_p", 32))
        ]
        mean_p = np.asarray(
            pressure[spec.get("mean_p_key", "mean_p_regime")], np.float64
        )
    if phi_u.shape[1] != 2 * areas.size:
        raise ValueError(f"velocity basis/grid mismatch: {phi_u.shape}, N={areas.size}")
    if phi_p.shape[1] != areas.size:
        raise ValueError(f"pressure basis/grid mismatch: {phi_p.shape}, N={areas.size}")
    velocity_layout = spec.get("velocity_layout", "blocked")
    if velocity_layout not in ("blocked", "interleaved"):
        raise ValueError(f"unsupported velocity_layout={velocity_layout!r}")
    area_sum = float(np.sum(areas))
    mean_p = mean_p - np.sum(areas * mean_p) / area_sum
    phi_p = phi_p - (
        np.sum(phi_p * areas[None, :], axis=1) / area_sum
    )[:, None]
    weights_u = (
        np.repeat(areas, 2)
        if velocity_layout == "interleaved"
        else np.concatenate((areas, areas))
    )
    gram_u = (phi_u * weights_u[None, :]) @ phi_u.T
    cross_u = (phi_u * weights_u[None, :]) @ mean_u
    gram_p = (phi_p * areas[None, :]) @ phi_p.T
    cross_p = (phi_p * areas[None, :]) @ mean_p
    return Basis(
        phi_u,
        phi_p,
        mean_u,
        mean_p,
        areas,
        velocity_layout,
        velocity_path,
        pressure_path,
        gram_u,
        cross_u,
        float(np.sum(weights_u * mean_u * mean_u)),
        gram_p,
        cross_p,
        float(np.sum(areas * mean_p * mean_p)),
    )


def reconstruct(a: np.ndarray, b: np.ndarray, basis: Basis) -> tuple[np.ndarray, ...]:
    uv = basis.mean_u + np.asarray(a, np.float64) @ basis.phi_u
    pressure = basis.mean_p + np.asarray(b, np.float64) @ basis.phi_p
    n_points = basis.areas.size
    pressure -= np.sum(basis.areas * pressure) / np.sum(basis.areas)
    if basis.velocity_layout == "interleaved":
        paired = uv.reshape(n_points, 2)
        return paired[:, 0], paired[:, 1], pressure
    return uv[:n_points], uv[n_points:], pressure


def field_errors(
    truth: tuple[np.ndarray, ...],
    prediction: tuple[np.ndarray, ...],
    areas: np.ndarray,
) -> dict[str, float]:
    u_true, v_true, p_true = truth
    u_pred, v_pred, p_pred = prediction
    velocity_num = np.sum(areas * ((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2))
    velocity_den = np.sum(areas * (u_true**2 + v_true**2))
    pressure_num = np.sum(areas * (p_pred - p_true) ** 2)
    pressure_den = np.sum(areas * p_true**2)
    velocity = math.sqrt(max(float(velocity_num), 0.0) / max(float(velocity_den), EPS))
    pressure = math.sqrt(max(float(pressure_num), 0.0) / max(float(pressure_den), EPS))
    return {
        "velocity_relative_l2": velocity,
        "pressure_relative_l2": pressure,
        "joint_field_error": 0.5 * (velocity + pressure),
    }


def coefficient_field_errors(
    true_a: np.ndarray,
    true_b: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    basis: Basis,
) -> dict[str, float]:
    da = np.asarray(pred_a, np.float64) - np.asarray(true_a, np.float64)
    db = np.asarray(pred_b, np.float64) - np.asarray(true_b, np.float64)
    true_a = np.asarray(true_a, np.float64)
    true_b = np.asarray(true_b, np.float64)
    u_num = float(da @ basis.gram_u @ da)
    p_num = float(db @ basis.gram_p @ db)
    u_den = float(
        basis.mean_u_energy
        + 2.0 * true_a @ basis.cross_u
        + true_a @ basis.gram_u @ true_a
    )
    p_den = float(
        basis.mean_p_energy
        + 2.0 * true_b @ basis.cross_p
        + true_b @ basis.gram_p @ true_b
    )
    velocity = math.sqrt(max(u_num, 0.0) / max(u_den, EPS))
    pressure = math.sqrt(max(p_num, 0.0) / max(p_den, EPS))
    return {
        "velocity_relative_l2": velocity,
        "pressure_relative_l2": pressure,
        "joint_field_error": 0.5 * (velocity + pressure),
    }


def mesh_triangulation(
    template: Path,
) -> tuple[pv.DataSet, mtri.Triangulation, np.ndarray, np.ndarray]:
    mesh = pv.read(template)
    # CenteredSquare POD coefficients live on the OpenFOAM finite-volume cells.
    # Preserve the VTK cell order and use those centers as the plotting degrees
    # of freedom.  The reference VTK is binary float32, so coordinates can
    # differ from the float64 cache by roughly 1e-6 without implying a reorder.
    if mesh.n_cells > 0:
        centers = np.asarray(mesh.cell_centers().points, dtype=np.float64)
        mesh.cell_data["_plot_original_cell_id"] = np.arange(
            mesh.n_cells, dtype=np.int64
        )
        cut = mesh.slice(normal="z", origin=(0.0, 0.0, float(np.median(centers[:, 2]))))
        cut = cut.triangulate()
        faces = np.asarray(cut.faces, dtype=np.int64).reshape(-1, 4)
        if not np.all(faces[:, 0] == 3):
            raise RuntimeError("mid-plane VTK slice contains non-triangle cells")
        points = np.asarray(cut.points, dtype=np.float64)
        triangulation = mtri.Triangulation(
            points[:, 0], points[:, 1], faces[:, 1:]
        )
        cell_value_ids = np.asarray(
            cut.cell_data["_plot_original_cell_id"], dtype=np.int64
        )
        if (
            cell_value_ids.size != len(faces)
            or cell_value_ids.min() != 0
            or cell_value_ids.max() != mesh.n_cells - 1
            or np.unique(cell_value_ids).size != mesh.n_cells
        ):
            raise RuntimeError("VTK slice did not preserve all original cell ids")
        triangulation.cell_value_ids = cell_value_ids
        ids = np.arange(mesh.n_cells, dtype=np.int64)
        return mesh, triangulation, ids, centers

    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    faces = np.asarray(surface.faces).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise RuntimeError("triangulated VTK surface contains non-triangle cells")
    points = np.asarray(surface.points)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], faces[:, 1:])
    if "vtkOriginalPointIds" in surface.point_data:
        original_point_ids = np.asarray(
            surface.point_data["vtkOriginalPointIds"], dtype=np.int64
        )
    else:
        original_point_ids = np.arange(surface.n_points, dtype=np.int64)
        if surface.n_points != mesh.n_points or not np.allclose(surface.points, mesh.points):
            raise RuntimeError("surface extraction changed point ordering without provenance")
    return surface, triangulation, original_point_ids, np.asarray(mesh.points)


def read_weight_table(
    path: Path | None, config: dict[str, Any]
) -> dict[tuple[float, int], float]:
    if path is None:
        return {}
    result: dict[tuple[float, int], float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            re_value = float(row.get("Re", row.get("re")))
            key_column = config.get("weight_key_column", "window_index")
            alpha_column = config.get("weight_alpha_column", "alpha_S")
            key = int(row[key_column])
            result[(round(re_value, 6), key)] = float(row[alpha_column])
    return result


def method_fields(
    bundle: Any,
    config: dict[str, Any],
    bases: dict[str, Basis],
    window: int,
    time_index: int,
    weights: dict[tuple[float, int], float],
) -> dict[str, tuple[np.ndarray, ...]]:
    arrays = config["arrays"]
    re_value = float(bundle[arrays["re"]][window])
    truth_basis = bases[config["truth_basis"]]
    result = {
        "truth": reconstruct(
            bundle[arrays["truth_a"]][window, time_index],
            bundle[arrays["truth_b"]][window, time_index],
            truth_basis,
        )
    }
    for method, mapping in config["methods"].items():
        if method == "t2c":
            key_array = arrays.get("weight_key")
            weight_key = int(bundle[key_array][window]) if key_array else window
            alpha_1 = weights[(round(re_value, 6), weight_key)]
            candidate_1 = reconstruct(
                bundle[mapping["candidate_1_a"]][window, time_index],
                bundle[mapping["candidate_1_b"]][window, time_index],
                bases[mapping["candidate_1_basis"]],
            )
            candidate_2 = reconstruct(
                bundle[mapping["candidate_2_a"]][window, time_index],
                bundle[mapping["candidate_2_b"]][window, time_index],
                bases[mapping["candidate_2_basis"]],
            )
            result[method] = tuple(
                alpha_1 * x + (1.0 - alpha_1) * y
                for x, y in zip(candidate_1, candidate_2)
            )
        else:
            result[method] = reconstruct(
                bundle[mapping["a"]][window, time_index],
                bundle[mapping["b"]][window, time_index],
                bases[mapping["basis"]],
            )
    return result


def eligible_tier(
    mode: str,
    metrics: dict[str, dict[str, float]],
    target: str,
    comparators: tuple[str, ...],
) -> int | None:
    if any(name not in metrics for name in (target, *comparators)):
        return None
    t = metrics[target]
    both = all(
        t["velocity_relative_l2"] < metrics[name]["velocity_relative_l2"]
        and t["pressure_relative_l2"] < metrics[name]["pressure_relative_l2"]
        for name in comparators
    )
    joint = all(
        t["joint_field_error"] < metrics[name]["joint_field_error"]
        for name in comparators
    )
    velocity = all(
        t["velocity_relative_l2"] < metrics[name]["velocity_relative_l2"]
        for name in comparators
    )
    if both:
        return 0
    if joint and velocity:
        return 1
    if joint:
        return 2
    return None


def structural_score(truth: tuple[np.ndarray, ...], points: np.ndarray) -> float:
    u, v, pressure = truth
    wake = (
        (points[:, 0] >= 6.0)
        & (points[:, 0] <= 17.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= 4.0)
    )
    if not np.any(wake):
        return 0.0
    speed = np.hypot(u, v)
    return float(np.std(speed[wake]) + 0.25 * np.std(pressure[wake]))


def candidate_rows(
    args: argparse.Namespace,
    bundle: Any,
    config: dict[str, Any],
    bases: dict[str, Basis],
    points: np.ndarray,
    weights: dict[tuple[float, int], float],
) -> list[dict[str, Any]]:
    arrays = config["arrays"]
    re_values = np.asarray(bundle[arrays["re"]], np.float64)
    splits = np.asarray(bundle[arrays["split"]]).astype(str)
    times = np.asarray(bundle[arrays["times"]], np.float64)
    selected_windows = np.arange(re_values.size)
    if args.re is not None:
        selected_windows = selected_windows[
            np.isclose(re_values[selected_windows], args.re, atol=5.0e-6, rtol=0)
        ]
    if args.split is not None:
        wanted = "heldout" if args.split == "test" else args.split
        selected_windows = selected_windows[splits[selected_windows] == wanted]
    rows: list[dict[str, Any]] = []
    for window in selected_windows.tolist():
        count = times.shape[1] - int(config.get("time_offset", 1))
        if args.time_index is not None:
            indices = [args.time_index % count]
        elif args.prefer_last_frame:
            indices = [count - 1]
            if args.allow_best_frame_search:
                indices.extend(range(count - 2, -1, -1))
        else:
            indices = list(range(count - 1, -1, -1))
        for rank, time_index in enumerate(indices):
            if args.mode in ("sh_fusion", "boundary_fusion") and "quad_u" in arrays and "quad_p" in arrays:
                qu = np.asarray(bundle[arrays["quad_u"]][window, time_index], np.float64)
                qp = np.asarray(bundle[arrays["quad_p"]][window, time_index], np.float64)
                key_array = arrays.get("weight_key")
                weight_key = int(bundle[key_array][window]) if key_array else window
                alpha_1 = weights[(round(float(re_values[window]), 6), weight_key)]

                def relative(quad: np.ndarray, alpha: float) -> float:
                    numerator = (
                        alpha * alpha * quad[0]
                        + (1.0 - alpha) ** 2 * quad[1]
                        + 2.0 * alpha * (1.0 - alpha) * quad[2]
                    )
                    return math.sqrt(max(float(numerator), 0.0) / max(float(quad[3]), EPS))

                metrics = {}
                comparator_names = tuple(config["comparator_methods"])
                for name, alpha in (
                    (comparator_names[0], 1.0),
                    (comparator_names[1], 0.0),
                    (config.get("target_method", "t2c"), alpha_1),
                ):
                    velocity = relative(qu, alpha)
                    pressure = relative(qp, alpha)
                    metrics[name] = {
                        "velocity_relative_l2": velocity,
                        "pressure_relative_l2": pressure,
                        "joint_field_error": 0.5 * (velocity + pressure),
                    }
                structure = float(
                    np.linalg.norm(bundle[arrays["truth_a"]][window, time_index, :4])
                )
            else:
                truth_a = bundle[arrays["truth_a"]][window, time_index]
                truth_b = bundle[arrays["truth_b"]][window, time_index]
                if all(
                    mapping.get("basis") == config["truth_basis"]
                    for mapping in config["methods"].values()
                ):
                    basis = bases[config["truth_basis"]]
                    metrics = {
                        method: coefficient_field_errors(
                            truth_a,
                            truth_b,
                            bundle[mapping["a"]][window, time_index],
                            bundle[mapping["b"]][window, time_index],
                            basis,
                        )
                        for method, mapping in config["methods"].items()
                    }
                    structure = float(np.linalg.norm(truth_a[:4]))
                else:
                    fields = method_fields(
                        bundle, config, bases, window, time_index, weights
                    )
                    truth = fields["truth"]
                    metrics = {
                        method: field_errors(
                            truth, value, bases[config["truth_basis"]].areas
                        )
                        for method, value in fields.items()
                        if method != "truth"
                    }
                    structure = structural_score(truth, points)
            target = config.get(
                "target_method", "proposed" if args.mode == "specialist" else "t2c"
            )
            comparators = tuple(
                config.get(
                    "comparator_methods",
                    ("vanilla", "dataonly")
                    if args.mode == "specialist"
                    else ("s_only", "h_only"),
                )
            )
            tier = eligible_tier(args.mode, metrics, target, comparators)
            advantage = min(
                metrics[name]["joint_field_error"] - metrics[target]["joint_field_error"]
                for name in comparators
            )
            rows.append(
                {
                    "window_index": window,
                    "Re": float(re_values[window]),
                    "split": splits[window],
                    "time_index": time_index,
                    "physical_time": float(
                        times[window, time_index + int(config.get("time_offset", 1))]
                    ),
                    "last_frame": rank == 0,
                    "eligibility_tier": tier,
                    "joint_advantage": float(advantage),
                    "structure_score": structure,
                    "metrics": metrics,
                }
            )
            if rank == 0 and tier is not None:
                break
    return rows


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligibility_tier"] is not None]
    if not eligible:
        raise RuntimeError("NO_ELIGIBLE_BEST_CASE_SNAPSHOT")
    last = [row for row in eligible if row["last_frame"]]
    pool = last if last else eligible
    return min(
        pool,
        key=lambda row: (
            row["eligibility_tier"],
            -row["joint_advantage"],
            -row["structure_score"],
            row["window_index"],
        ),
    )


def style(error_cmap_name: str = "magma") -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def robust_limits(values: list[np.ndarray], divergent: bool = False) -> tuple[float, float]:
    merged = np.concatenate([np.asarray(value)[np.isfinite(value)] for value in values])
    if divergent:
        bound = max(abs(float(np.quantile(merged, 0.005))), abs(float(np.quantile(merged, 0.995))))
        return -bound, bound
    return float(np.quantile(merged, 0.005)), float(np.quantile(merged, 0.995))


def add_bottom_colorbar(
    figure: plt.Figure, axes: list[plt.Axes], mappable: Any, label: str
) -> None:
    box0, box1 = axes[0].get_position(), axes[-1].get_position()
    cax = figure.add_axes([box0.x0, 0.035, box1.x1 - box0.x0, 0.026])
    bar = figure.colorbar(mappable, cax=cax, orientation="horizontal")
    bar.locator = MaxNLocator(nbins=5)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    bar.formatter = formatter
    bar.update_ticks()
    bar.set_label(label, labelpad=3, fontsize=11)
    bar.ax.tick_params(labelsize=9, length=3)
    bar.ax.xaxis.get_offset_text().set_size(9)


def latex_scientific(value: float, precision: int = 3) -> str:
    if not np.isfinite(value):
        return r"\mathrm{NaN}"
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{precision - 1}f}\times10^{{{exponent}}}"


def draw_panel(
    axis: plt.Axes,
    triangulation: mtri.Triangulation,
    values: np.ndarray,
    cmap: str,
    norm: Normalize,
    title: str,
    panel: str,
    args: argparse.Namespace,
    box_aspect: float | None = None,
) -> Any:
    cell_value_ids = getattr(triangulation, "cell_value_ids", None)
    if cell_value_ids is None:
        image = axis.tripcolor(
            triangulation, values, shading="gouraud", cmap=cmap, norm=norm
        )
    else:
        image = axis.tripcolor(
            triangulation,
            facecolors=np.asarray(values)[cell_value_ids],
            shading="flat",
            cmap=cmap,
            norm=norm,
        )
    if box_aspect is None:
        axis.set_aspect("equal")
    else:
        axis.set_box_aspect(box_aspect)
    axis.set_xlim(*args.xlim)
    axis.set_ylim(*args.ylim)
    axis.set_xticks([])
    axis.set_yticks([])
    if title:
        axis.set_title(title, pad=3.0, linespacing=0.88)
    axis.text(
        0.012,
        0.96,
        panel,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
    )
    return image


def plot_figures(
    args: argparse.Namespace,
    triangulation: mtri.Triangulation,
    fields: dict[str, tuple[np.ndarray, ...]],
    order: list[str],
    stem: str,
    metrics: dict[str, dict[str, float]],
    method_labels: dict[str, str],
) -> list[Path]:
    error_cmap = _resolve_error_cmap(getattr(args, "error_cmap", "magma"))
    style(error_cmap)
    outputs: list[Path] = []
    speed = {name: np.hypot(value[0], value[1]) for name, value in fields.items()}
    pressure = {name: value[2] for name, value in fields.items()}
    # Physical-variable limits are anchored to the CFD reference.  This keeps
    # the common scale scientifically interpretable when an ablation is grossly
    # divergent; the divergent panel then saturates instead of flattening all
    # otherwise valid fields into a nearly white image.
    speed_lim = robust_limits([speed["truth"]])
    p_lim = robust_limits([pressure["truth"]], divergent=True)
    field_norms = (Normalize(*speed_lim), TwoSlopeNorm(0.0, *p_lim))
    output_suffixes = ("png",) if args.png_only else ("png", "pdf")
    if args.plot_truth_pred:
        names = ["truth", *order]
        # This is deliberately a wide, compact physical-field layout.  Method
        # labels are moved outside the first column, so they no longer consume
        # a title line above every row.
        fig, axes = plt.subplots(len(names), 2, figsize=(14.6, 7.0))
        fig.subplots_adjust(left=0.17, right=0.975, top=0.92, bottom=0.11, wspace=0.12, hspace=0.018)
        left, right = [], []
        letters = iter("abcdefghijklmnopqrstuvwxyz")
        for row, name in enumerate(names):
            left.append(draw_panel(axes[row, 0], triangulation, speed[name], "cividis", field_norms[0], "", f"({next(letters)})", args, box_aspect=0.32))
            right.append(draw_panel(axes[row, 1], triangulation, pressure[name], "RdBu_r", field_norms[1], "", f"({next(letters)})", args, box_aspect=0.32))
            axes[row, 0].annotate(
                method_labels.get(name, METHOD_ROW_LABELS.get(name, name)),
                xy=(-0.035, 0.5),
                xycoords="axes fraction",
                ha="right",
                va="center",
                rotation=0,
                fontsize=11,
                fontweight="semibold",
                linespacing=0.9,
            )
            axes[row, 0].set_anchor("E")
            axes[row, 1].set_anchor("W")
        axes[0, 0].set_title(r"Velocity magnitude $|\mathbf{u}|$", pad=6.0)
        axes[0, 1].set_title(r"Pressure $p$", pad=6.0)
        add_bottom_colorbar(fig, list(axes[:, 0]), left[-1], r"Velocity magnitude $|\mathbf{u}|$")
        add_bottom_colorbar(fig, list(axes[:, 1]), right[-1], r"Pressure $p$")
        for suffix in output_suffixes:
            path = args.output_dir / f"{stem}_physical_fields_ultra_compact_wide.{suffix}"
            fig.savefig(
                path,
                dpi=args.dpi if suffix == "png" else None,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            outputs.append(path)
        plt.close(fig)
    if args.plot_error:
        truth = fields["truth"]
        u_errors = {
            name: np.hypot(fields[name][0] - truth[0], fields[name][1] - truth[1])
            for name in order
        }
        p_errors = {name: np.abs(fields[name][2] - truth[2]) for name in order}
        u_lim = (0.0, max(robust_limits(list(u_errors.values()))[1], EPS))
        pe_lim = (0.0, max(robust_limits(list(p_errors.values()))[1], EPS))
        fig, axes = plt.subplots(len(order), 2, figsize=(14.6, 1.65 * len(order) + 0.75))
        if len(order) == 1:
            axes = axes[None, :]
        fig.subplots_adjust(left=0.17, right=0.975, top=0.90, bottom=0.14, wspace=0.12, hspace=0.02)
        left, right = [], []
        letters = iter("abcdefghijklmnopqrstuvwxyz")
        for row, name in enumerate(order):
            epsilon_u = latex_scientific(metrics[name]["velocity_relative_l2"])
            epsilon_p = latex_scientific(metrics[name]["pressure_relative_l2"])
            left.append(draw_panel(axes[row, 0], triangulation, u_errors[name], error_cmap, Normalize(*u_lim), "", f"({next(letters)})", args, box_aspect=0.32))
            right.append(draw_panel(axes[row, 1], triangulation, p_errors[name], error_cmap, Normalize(*pe_lim), "", f"({next(letters)})", args, box_aspect=0.32))
            axes[row, 0].annotate(
                method_labels.get(name, METHOD_ROW_LABELS.get(name, name)),
                xy=(-0.035, 0.5),
                xycoords="axes fraction",
                ha="right",
                va="center",
                rotation=0,
                fontsize=11,
                fontweight="semibold",
                linespacing=0.9,
            )
            axes[row, 0].text(
                0.985, 0.96, rf"$\epsilon_{{u}}={epsilon_u}$",
                transform=axes[row, 0].transAxes,
                ha="right", va="top", fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
            )
            axes[row, 1].text(
                0.985, 0.96, rf"$\epsilon_{{p}}={epsilon_p}$",
                transform=axes[row, 1].transAxes,
                ha="right", va="top", fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
            )
            axes[row, 0].set_anchor("E")
            axes[row, 1].set_anchor("W")
        axes[0, 0].set_title(r"Velocity error $|\Delta\mathbf{u}|$", pad=6.0)
        axes[0, 1].set_title(r"Pressure error $|\Delta p|$", pad=6.0)
        add_bottom_colorbar(fig, list(axes[:, 0]), left[-1], r"Pointwise error $|\Delta\mathbf{u}|$")
        add_bottom_colorbar(fig, list(axes[:, 1]), right[-1], r"Pointwise error $|\Delta p|$")
        for suffix in output_suffixes:
            path = args.output_dir / f"{stem}_pointwise_errors_ultra_compact_wide.{suffix}"
            fig.savefig(
                path,
                dpi=args.dpi if suffix == "png" else None,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            outputs.append(path)
        plt.close(fig)
    return outputs


def save_selected_vtk(
    surface: pv.PolyData,
    fields: dict[str, tuple[np.ndarray, ...]],
    output: Path,
) -> None:
    mesh = surface.copy(deep=True)
    for name, (u, v, pressure) in fields.items():
        mesh.point_data[f"{name}_U"] = np.column_stack((u, v, np.zeros_like(u)))
        mesh.point_data[f"{name}_speed"] = np.hypot(u, v)
        mesh.point_data[f"{name}_p"] = pressure
    mesh.save(output)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["mode"] != args.mode:
        raise ValueError(f"mode mismatch: CLI={args.mode}, config={config['mode']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(config["bundle"])
    bases = {name: load_basis(spec) for name, spec in config["bases"].items()}
    surface, triangulation, original_point_ids, aligned_points = mesh_triangulation(
        args.template_vtk
    )
    truth_basis = bases[config["truth_basis"]]
    if np.max(original_point_ids) >= truth_basis.areas.size:
        raise ValueError(
            f"VTK/POD point mismatch: max original id {np.max(original_point_ids)} "
            f"vs N={truth_basis.areas.size}"
        )
    weights_path = Path(config["weights_csv"]) if config.get("weights_csv") else None
    weights = read_weight_table(weights_path, config)
    if aligned_points.shape[0] != truth_basis.areas.size:
        raise ValueError(
            f"VTK/POD cell mismatch: VTK={aligned_points.shape[0]}, "
            f"POD={truth_basis.areas.size}"
        )
    with np.load(bundle_path, allow_pickle=False) as bundle:
        rows = candidate_rows(
            args, bundle, config, bases, aligned_points, weights
        )
        scan_path = args.output_dir / "candidate_scan.json"
        scan_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        selected = select_candidate(rows)
        fields = method_fields(
            bundle,
            config,
            bases,
            int(selected["window_index"]),
            int(selected["time_index"]),
            weights,
        )
    order = list(config["plot_order"])
    stem = config.get("figure_stem", args.mode)
    surface_fields = {
        name: tuple(component[original_point_ids] for component in value)
        for name, value in fields.items()
    }
    outputs = plot_figures(
        args,
        triangulation,
        surface_fields,
        order,
        stem,
        selected["metrics"],
        config.get("method_labels", {}),
    )
    if args.save_vtk:
        raise RuntimeError("--save-vtk is not supported for cell-centered figures")
    manifest = {
        "schema": "field_figure_manifest/v1",
        "qualitative_scope": (
            "best-case qualitative snapshot; quantitative conclusions remain those "
            "of the frozen aggregate evaluator"
        ),
        "mode": args.mode,
        "selected": selected,
        "style": {
            "target_journals": ["Journal of Computational Physics", "CMAME"],
            "velocity_colormap": "cividis",
            "pressure_colormap": "RdBu_r centered at zero",
            "error_colormap": getattr(args, "error_cmap", "magma"),
            "colorbars": "shared horizontal bottom colorbar per physical variable",
            "dpi": args.dpi,
        },
        "inputs": {
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "bundle": str(bundle_path),
            "bundle_sha256": sha256(bundle_path),
            "template_vtk": str(args.template_vtk),
            "template_vtk_sha256": sha256(args.template_vtk),
            "weights_csv": str(weights_path) if weights_path else None,
            "weights_csv_sha256": sha256(weights_path) if weights_path else None,
            "bases": {
                name: {
                    "velocity": str(basis.velocity_path),
                    "velocity_sha256": sha256(basis.velocity_path),
                    "pressure": str(basis.pressure_path),
                    "pressure_sha256": sha256(basis.pressure_path),
                }
                for name, basis in bases.items()
            },
        },
        "outputs": [
            {"path": str(path), "sha256": sha256(path)} for path in outputs
        ],
    }
    manifest_path = args.output_dir / "FIELD_FIGURE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DONE", "manifest": str(manifest_path), "selected": selected}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
