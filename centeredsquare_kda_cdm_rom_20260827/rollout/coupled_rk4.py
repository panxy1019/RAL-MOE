"""Stage-consistent RK4 for the jointly coupled resolved/memory ODE."""

from __future__ import annotations

import numpy as np


def rk4_step(resolved: np.ndarray, memory: np.ndarray, step: float, joint_rhs):
    """Advance (a,S) while recomputing all features at every RK4 stage."""
    a0 = np.asarray(resolved, dtype=np.float64)
    s0 = np.asarray(memory, dtype=np.float64)
    k1a, k1s = joint_rhs(a0, s0)
    k2a, k2s = joint_rhs(a0 + 0.5 * step * k1a, s0 + 0.5 * step * k1s)
    k3a, k3s = joint_rhs(a0 + 0.5 * step * k2a, s0 + 0.5 * step * k2s)
    k4a, k4s = joint_rhs(a0 + step * k3a, s0 + step * k3s)
    a1 = a0 + step * (k1a + 2 * k2a + 2 * k3a + k4a) / 6.0
    s1 = s0 + step * (k1s + 2 * k2s + 2 * k3s + k4s) / 6.0
    return a1, s1
