#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import ROOT


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def plot_case_npz(npz_dir: Path, fig_dir: Path) -> None:
    plt = setup_matplotlib()
    files = sorted(npz_dir.glob("Re_*.npz"))
    if not files:
        return
    re_values = []
    counts = []
    mean_ke = []
    p_rms = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            re_values.append(float(data["Re"]))
            counts.append(int(data["times"].shape[0]))
            u = data["U"].astype(float)
            p = data["p"].astype(float)
            mean_ke.append(float(np.mean(0.5 * np.sum(u**2, axis=2))))
            p_rms.append(float(np.sqrt(np.mean((p - np.mean(p)) ** 2))))

    for values, ylabel, name in [
        (counts, "snapshot count", "snapshot_count_per_Re.png"),
        (mean_ke, "mean kinetic energy", "mean_kinetic_energy_per_Re.png"),
        (p_rms, "pressure RMS", "pressure_rms_per_Re.png"),
    ]:
        plt.figure(figsize=(7, 4))
        plt.plot(re_values, values, marker="o")
        plt.xlabel("Re")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=160)
        plt.close()


def plot_pilot(fig_dir: Path) -> None:
    plt = setup_matplotlib()
    rows = read_csv(ROOT / "data" / "manifest" / "pilot_diagnostics.csv")
    if not rows:
        return
    re_values = np.asarray([float(r["Re"]) for r in rows])
    for columns, ylabel, name in [
        (["Cd_std", "Cl_std"], "force coefficient std", "Cd_Cl_std_per_Re.png"),
        (["probe_v_std"], "probe v velocity std", "probe_v_std_per_Re.png"),
    ]:
        plt.figure(figsize=(7, 4))
        for col in columns:
            vals = np.asarray([float(r[col]) if r.get(col) not in {"", "nan"} else np.nan for r in rows])
            plt.plot(re_values, vals, marker="o", label=col)
        plt.xlabel("Re")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=160)
        plt.close()


def plot_pod(fig_dir: Path) -> None:
    plt = setup_matplotlib()
    vel = ROOT / "data" / "pod" / "velocity_pod.npz"
    pre = ROOT / "data" / "pod" / "pressure_pod.npz"
    if vel.exists() or pre.exists():
        plt.figure(figsize=(7, 4))
        if vel.exists():
            with np.load(vel, allow_pickle=True) as data:
                plt.plot(np.cumsum(data["energy_u"]), marker="o", label="velocity")
        if pre.exists():
            with np.load(pre, allow_pickle=True) as data:
                plt.plot(np.cumsum(data["energy_p"]), marker="s", label="pressure")
        plt.xlabel("mode index")
        plt.ylabel("cumulative energy")
        plt.ylim(0, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "POD_cumulative_energy.png", dpi=160)
        plt.close()

    coeff = ROOT / "data" / "pod" / "modal_coefficients.npz"
    if not coeff.exists():
        return
    with np.load(coeff, allow_pickle=True) as data:
        re_list = data["Re_list"]
        times_by_re = data["times_by_Re"]
        a_by_re = data["a_by_Re"]
        if len(re_list) == 0:
            return
        idx = 0
        times = np.asarray(times_by_re[idx], dtype=float)
        a = np.asarray(a_by_re[idx], dtype=float)
        plt.figure(figsize=(8, 4))
        for k in range(min(4, a.shape[1])):
            plt.plot(times, a[:, k], label=f"a{k + 1}")
        plt.xlabel(f"time, Re={re_list[idx]:g}")
        plt.ylabel("velocity modal coefficient")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "selected_Re_velocity_modal_coefficients.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create quick static diagnostic figures.")
    parser.add_argument("--npz-dir", type=Path, default=ROOT / "data" / "npz")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "data" / "logs" / "figures")
    args = parser.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_case_npz(args.npz_dir, args.fig_dir)
    plot_pilot(args.fig_dir)
    plot_pod(args.fig_dir)
    print(f"Wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
