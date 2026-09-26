"""Deterministic phase-aware key/query/value maps for KDA-CDM-ROM."""

from __future__ import annotations

import numpy as np


def phase_coordinates(state: np.ndarray, pair: tuple[int, int], epsilon: float = 1e-12):
    """Return rho, theta for one state; indices are zero based."""
    ap = float(state[pair[0]])
    aq = float(state[pair[1]])
    rho = float(np.hypot(ap, aq))
    theta = float(np.arctan2(aq, ap))
    return max(rho, epsilon), theta


def phase_kinematics(state: np.ndarray, rhs: np.ndarray, pair: tuple[int, int], epsilon: float = 1e-12):
    """Return instantaneous angular and radial speeds of a phase pair."""
    p, q = pair
    ap, aq = float(state[p]), float(state[q])
    fp, fq = float(rhs[p]), float(rhs[q])
    rho2 = ap * ap + aq * aq
    omega = (ap * fq - aq * fp) / (rho2 + epsilon)
    rho_dot = (ap * fp + aq * fq) / (np.sqrt(rho2) + epsilon)
    return float(omega), float(rho_dot)


def phase_key_query(theta: float, rho_scaled: float, re_scaled: float, epsilon: float = 1e-12):
    feature = np.asarray([
        1.0, np.cos(theta), np.sin(theta), np.cos(2.0 * theta),
        np.sin(2.0 * theta), rho_scaled, re_scaled,
    ], dtype=np.float64)
    return feature / (np.linalg.norm(feature) + epsilon)


def phase_value(omega_scaled: float, rho_dot_scaled: float, rho_scaled: float):
    return np.asarray([omega_scaled, rho_dot_scaled, rho_scaled, 1.0], dtype=np.float64)


def phase_tangent(state: np.ndarray, pair: tuple[int, int]):
    result = np.zeros_like(state, dtype=np.float64)
    p, q = pair
    result[p] = -state[q]
    result[q] = state[p]
    return result
