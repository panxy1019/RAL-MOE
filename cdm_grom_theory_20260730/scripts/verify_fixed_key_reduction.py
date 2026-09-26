#!/usr/bin/env python3
"""Check reduction of fixed-key matrix memory to an observable vector state."""

from __future__ import annotations

import numpy as np


def matrix_rhs(
    state: np.ndarray,
    address: np.ndarray,
    target: np.ndarray,
    gamma: float,
    eta: float,
) -> np.ndarray:
    return -gamma * state + eta * np.outer(
        address, target - state.T @ address
    )


def rk4_step(function, state: np.ndarray, time: float, step: float) -> np.ndarray:
    k1 = function(time, state)
    k2 = function(time + step / 2.0, state + step * k1 / 2.0)
    k3 = function(time + step / 2.0, state + step * k2 / 2.0)
    k4 = function(time + step, state + step * k3)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def main() -> None:
    rng = np.random.default_rng(271828)
    d_key, d_value, r = 5, 3, 4
    address = rng.normal(size=d_key)
    address /= np.linalg.norm(address)
    projector_perp = np.eye(d_key) - np.outer(address, address)
    state0 = rng.normal(scale=0.3, size=(d_key, d_value))
    coupling = rng.normal(size=(d_value, r))
    resolved0 = rng.normal(size=r)
    gamma, eta = 0.55, 0.75

    target0 = coupling @ resolved0
    derivative = matrix_rhs(state0, address, target0, gamma, eta)
    reduced_derivative = (
        -(gamma + eta) * (state0.T @ address) + eta * target0
    )
    read_derivative_error = np.linalg.norm(
        derivative.T @ address - reduced_derivative
    )
    orthogonal_derivative_error = np.linalg.norm(
        projector_perp @ derivative + gamma * projector_perp @ state0
    )

    def resolved(time: float) -> np.ndarray:
        frequencies = np.arange(1, r + 1, dtype=float)
        return resolved0 * np.cos(0.3 * frequencies * time)

    def full_function(time: float, state: np.ndarray) -> np.ndarray:
        return matrix_rhs(
            state, address, coupling @ resolved(time), gamma, eta
        )

    def reduced_function(time: float, read: np.ndarray) -> np.ndarray:
        return -(gamma + eta) * read + eta * coupling @ resolved(time)

    state = state0.copy()
    read = state0.T @ address
    time, step, final_time = 0.0, 0.002, 2.0
    for _ in range(round(final_time / step)):
        state = rk4_step(full_function, state, time, step)
        read = rk4_step(reduced_function, read, time, step)
        time += step

    trajectory_read_error = np.linalg.norm(state.T @ address - read)
    orthogonal_exact = np.exp(-gamma * final_time) * projector_perp @ state0
    orthogonal_trajectory_error = np.linalg.norm(
        projector_perp @ state - orthogonal_exact
    )

    assert read_derivative_error < 1.0e-12, read_derivative_error
    assert orthogonal_derivative_error < 1.0e-12
    assert trajectory_read_error < 1.0e-10, trajectory_read_error
    assert orthogonal_trajectory_error < 1.0e-10

    print("PASS verify_fixed_key_reduction")
    print(f"read_derivative_error={read_derivative_error:.6e}")
    print(
        f"orthogonal_derivative_error={orthogonal_derivative_error:.6e}"
    )
    print(f"trajectory_read_error={trajectory_read_error:.6e}")
    print(
        f"orthogonal_trajectory_error={orthogonal_trajectory_error:.6e}"
    )


if __name__ == "__main__":
    main()

