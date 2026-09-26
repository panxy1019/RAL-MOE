#!/usr/bin/env python3
"""Check the regularized reconstruction gradient-flow identity."""

from __future__ import annotations

import numpy as np


def objective(
    state: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gamma_matrix: np.ndarray,
    eta: float,
) -> float:
    residual = value - state.T @ key
    regularizer = np.trace(state.T @ gamma_matrix @ state) / (2.0 * eta)
    return 0.5 * float(residual @ residual) + float(regularizer)


def analytic_gradient(
    state: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gamma_matrix: np.ndarray,
    eta: float,
) -> np.ndarray:
    return -np.outer(key, value - state.T @ key) + gamma_matrix @ state / eta


def finite_difference_gradient(
    state: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gamma_matrix: np.ndarray,
    eta: float,
    epsilon: float = 1.0e-6,
) -> np.ndarray:
    gradient = np.zeros_like(state)
    for row in range(state.shape[0]):
        for col in range(state.shape[1]):
            perturbation = np.zeros_like(state)
            perturbation[row, col] = epsilon
            plus = objective(
                state + perturbation, key, value, gamma_matrix, eta
            )
            minus = objective(
                state - perturbation, key, value, gamma_matrix, eta
            )
            gradient[row, col] = (plus - minus) / (2.0 * epsilon)
    return gradient


def main() -> None:
    rng = np.random.default_rng(481516)
    d_key, d_value = 4, 3
    state = rng.normal(scale=0.3, size=(d_key, d_value))
    key = rng.normal(size=d_key)
    key /= np.linalg.norm(key)
    value = rng.normal(size=d_value)
    gamma_matrix = np.diag(0.25 + rng.random(d_key))
    eta = 0.8

    analytic = analytic_gradient(state, key, value, gamma_matrix, eta)
    numerical = finite_difference_gradient(
        state, key, value, gamma_matrix, eta
    )
    relative_gradient_error = np.linalg.norm(analytic - numerical) / max(
        np.linalg.norm(analytic), 1.0e-15
    )

    memory_rhs = -gamma_matrix @ state + eta * np.outer(
        key, value - state.T @ key
    )
    flow_identity_error = np.linalg.norm(memory_rhs + eta * analytic)

    operator = gamma_matrix + eta * np.outer(key, key)
    equilibrium = np.linalg.solve(operator, eta * np.outer(key, value))
    equilibrium_residual = np.linalg.norm(
        -gamma_matrix @ equilibrium
        + eta * np.outer(key, value - equilibrium.T @ key)
    )

    assert relative_gradient_error < 2.0e-9, relative_gradient_error
    assert flow_identity_error < 1.0e-12, flow_identity_error
    assert equilibrium_residual < 1.0e-12, equilibrium_residual

    print("PASS verify_gradient_flow")
    print(f"relative_gradient_error={relative_gradient_error:.6e}")
    print(f"flow_identity_error={flow_identity_error:.6e}")
    print(f"equilibrium_residual={equilibrium_residual:.6e}")


if __name__ == "__main__":
    main()

