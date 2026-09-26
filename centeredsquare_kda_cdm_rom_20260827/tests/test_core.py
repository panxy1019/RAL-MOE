"""Deterministic numerical checks for the KDA-CDM-ROM core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.continuous_delta_memory import memory_readout, memory_rhs
from memory.features import phase_key_query, phase_kinematics, phase_tangent
from rollout.coupled_rk4 import rk4_step
from defect.project_instantaneous_defect import make_rhs, rk4_advance


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    key = phase_key_query(theta=0.7, rho_scaled=0.2, re_scaled=-0.4)
    check(abs(np.linalg.norm(key) - 1.0) < 1e-12, "key normalization")

    state = np.asarray([2.0, -3.0, 0.0])
    pair = (0, 1)
    tangent = phase_tangent(state, pair)
    omega, rho_dot = phase_kinematics(state, tangent, pair)
    check(abs(omega - 1.0) < 1e-12, "phase tangent must add unit angular speed")
    check(abs(rho_dot) < 1e-12, "phase tangent must not add radial speed")

    memory = np.zeros((7, 4))
    value = np.asarray([0.3, -0.2, 0.1, 1.0])
    derivative = memory_rhs(memory, key, value, gamma=0.2, eta=0.5)
    check(np.allclose(derivative, 0.5 * np.outer(key, value)), "zero-state delta write")

    equilibrium = np.linalg.solve(0.2 * np.eye(7) + 0.5 * np.outer(key, key), 0.5 * np.outer(key, value))
    check(np.linalg.norm(memory_rhs(equilibrium, key, value, 0.2, 0.5)) < 1e-12, "fixed-input equilibrium")
    check(np.all(np.isfinite(memory_readout(equilibrium, key))), "finite equilibrium readout")

    # The discrete KDA recurrence must approach the continuous memory ODE.
    errors = []
    for step in [0.2, 0.1, 0.05]:
        discrete = memory.copy()
        count = round(1.0 / step)
        decay = np.exp(-step * 0.2)
        for _ in range(count):
            forgotten = decay * discrete
            discrete = forgotten + step * 0.5 * np.outer(key, value - forgotten.T @ key)
        generator = 0.2 * np.eye(7) + 0.5 * np.outer(key, key)
        eigval, eigvec = np.linalg.eigh(generator)
        continuous = equilibrium - (eigvec * np.exp(-eigval)[None]) @ eigvec.T @ equilibrium
        errors.append(np.linalg.norm(discrete - continuous))
    check(errors[1] < errors[0] and errors[2] < errors[1], "discrete KDA must converge to continuous memory")

    # Frozen baseline assembly and substepped RK4 must preserve a constant RHS.
    rank = 3
    tensor = {
        "A_conv": np.zeros((rank, rank)), "A_diff": np.zeros((rank, rank)),
        "H_conv": np.zeros((rank, rank, rank)), "P": np.zeros((rank, 1)),
        "c_conv": np.asarray([1.0, -2.0, 0.5]), "c_diff": np.zeros(rank),
        "c_pressure": np.zeros(rank),
    }
    baseline_rhs = make_rhs(tensor, np.ones(2 + 2 * rank), np.zeros((2 + 2 * rank, 1)), 0.01)
    initial = np.asarray([0.2, -0.1, 0.4])
    advanced = rk4_advance(initial, 0.37, baseline_rhs, max_step=0.05)
    check(np.allclose(advanced, initial + 0.37 * tensor["c_conv"]), "frozen baseline RK4")

    calls = {"count": 0}
    def joint_rhs(a, s):
        calls["count"] += 1
        return -a, -2.0 * s
    a1, s1 = rk4_step(np.ones(2), np.ones((2, 2)), 0.01, joint_rhs)
    check(calls["count"] == 4, "joint RHS must be evaluated at four RK4 stages")
    check(np.max(abs(a1 - np.exp(-0.01))) < 1e-10, "resolved RK4 accuracy")
    check(np.max(abs(s1 - np.exp(-0.02))) < 1e-9, "memory RK4 accuracy")
    print("PASS: key, phase tangent, delta equilibrium, and coupled RK4 checks")


if __name__ == "__main__":
    main()
