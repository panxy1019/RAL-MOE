#!/usr/bin/env python3
"""Check finite-state, convolution, and block-elimination equivalence."""

from __future__ import annotations

import numpy as np


def symmetric_exponential(
    matrix: np.ndarray, time: float
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.exp(eigenvalues * time)) @ eigenvectors.T


def rk4_step(function, state: np.ndarray, step: float) -> np.ndarray:
    k1 = function(state)
    k2 = function(state + step * k1 / 2.0)
    k3 = function(state + step * k2 / 2.0)
    k4 = function(state + step * k3)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def main() -> None:
    rng = np.random.default_rng(161803)
    r, unresolved_dim = 2, 3
    rotation, _ = np.linalg.qr(rng.normal(size=(unresolved_dim, unresolved_dim)))
    decay_rates = np.array([0.45, 0.9, 1.4])
    a_uu = rotation @ np.diag(-decay_rates) @ rotation.T
    a_rr = rng.normal(scale=0.08, size=(r, r))
    a_ru = rng.normal(scale=0.15, size=(r, unresolved_dim))
    a_ur = rng.normal(scale=0.15, size=(unresolved_dim, r))
    full_matrix = np.block([[a_rr, a_ru], [a_ur, a_uu]])

    resolved0 = rng.normal(scale=0.3, size=r)
    unresolved0 = rng.normal(scale=0.2, size=unresolved_dim)
    full = np.concatenate([resolved0, unresolved0])
    auxiliary = full.copy()

    def full_rhs(state: np.ndarray) -> np.ndarray:
        return full_matrix @ state

    def auxiliary_rhs(state: np.ndarray) -> np.ndarray:
        resolved = state[:r]
        unresolved = state[r:]
        return np.concatenate(
            [
                a_rr @ resolved + a_ru @ unresolved,
                a_ur @ resolved + a_uu @ unresolved,
            ]
        )

    step, final_time = 0.001, 1.0
    count = round(final_time / step)
    resolved_history = np.empty((count + 1, r))
    unresolved_history = np.empty((count + 1, unresolved_dim))
    resolved_history[0] = resolved0
    unresolved_history[0] = unresolved0
    for index in range(1, count + 1):
        full = rk4_step(full_rhs, full, step)
        auxiliary = rk4_step(auxiliary_rhs, auxiliary, step)
        resolved_history[index] = full[:r]
        unresolved_history[index] = full[r:]

    block_auxiliary_error = np.linalg.norm(full - auxiliary)

    sample_indices = [count // 4, count // 2, 3 * count // 4, count]
    convolution_errors: list[float] = []
    for index in sample_indices:
        time = index * step
        integral = np.zeros(unresolved_dim)
        for source_index in range(index + 1):
            source_time = source_index * step
            weight = 0.5 if source_index in (0, index) else 1.0
            propagator = symmetric_exponential(a_uu, time - source_time)
            integral += (
                weight
                * propagator
                @ a_ur
                @ resolved_history[source_index]
            )
        integral *= step
        reconstructed = (
            symmetric_exponential(a_uu, time) @ unresolved0 + integral
        )
        convolution_errors.append(
            float(np.linalg.norm(reconstructed - unresolved_history[index]))
        )

    test_laplace_point = 2.0
    transfer_from_block = a_ru @ np.linalg.solve(
        test_laplace_point * np.eye(unresolved_dim) - a_uu, a_ur
    )
    lambda_matrix = -a_uu
    transfer_from_auxiliary = a_ru @ np.linalg.solve(
        test_laplace_point * np.eye(unresolved_dim) + lambda_matrix, a_ur
    )
    transfer_error = np.linalg.norm(
        transfer_from_block - transfer_from_auxiliary
    )

    test_tau = 0.37
    exact_kernel = (
        a_ru @ symmetric_exponential(a_uu, test_tau) @ a_ur
    )
    auxiliary_kernel = (
        a_ru @ symmetric_exponential(-lambda_matrix, test_tau) @ a_ur
    )
    kernel_error = np.linalg.norm(exact_kernel - auxiliary_kernel)

    assert block_auxiliary_error < 1.0e-12, block_auxiliary_error
    assert max(convolution_errors) < 2.0e-7, convolution_errors
    assert transfer_error < 1.0e-12, transfer_error
    assert kernel_error < 1.0e-12, kernel_error

    print("PASS verify_linear_memory_kernel")
    print(f"block_auxiliary_error={block_auxiliary_error:.6e}")
    print(f"convolution_errors={convolution_errors}")
    print(f"transfer_error={transfer_error:.6e}")
    print(f"kernel_error={kernel_error:.6e}")


if __name__ == "__main__":
    main()

