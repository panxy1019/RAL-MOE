#!/usr/bin/env python3
"""Check memory energy and reciprocal-coupling identities."""

from __future__ import annotations

import numpy as np


def main() -> None:
    rng = np.random.default_rng(314159)
    d_key, d_value = 6, 4
    state = rng.normal(scale=0.25, size=(d_key, d_value))
    key = rng.normal(size=d_key)
    key /= np.linalg.norm(key)
    value = rng.normal(scale=0.8, size=d_value)
    gamma = 0.4 + rng.random(d_key)
    gamma_matrix = np.diag(gamma)
    eta = 0.65

    read = state.T @ key
    state_rhs = -gamma_matrix @ state + eta * np.outer(
        key, value - read
    )
    direct_derivative = float(np.sum(state * state_rhs))
    identity_derivative = float(
        -np.sum(state * (gamma_matrix @ state))
        + eta * read @ (value - read)
    )
    identity_error = abs(direct_derivative - identity_derivative)

    gamma_min = float(np.min(gamma))
    completing_square_bound = (
        -gamma_min * np.linalg.norm(state) ** 2
        + eta * np.linalg.norm(value) ** 2 / 4.0
    )
    passivity_bound = (
        -gamma_min * np.linalg.norm(state) ** 2
        - eta * np.linalg.norm(read) ** 2 / 2.0
        + eta * np.linalg.norm(value) ** 2 / 2.0
    )

    r, d_aux = 5, 3
    resolved = rng.normal(size=r)
    auxiliary = rng.normal(size=d_aux)
    coupling = rng.normal(size=(d_aux, r))
    eta_aux = 0.9
    lambda_aux = 0.7
    galerkin_rhs = rng.normal(size=r)
    resolved_rhs = galerkin_rhs - coupling.T @ auxiliary
    auxiliary_rhs = eta_aux * coupling @ resolved - lambda_aux * auxiliary
    augmented_derivative = float(
        resolved @ resolved_rhs
        + auxiliary @ auxiliary_rhs / eta_aux
    )
    predicted_derivative = float(
        resolved @ galerkin_rhs
        - lambda_aux * (auxiliary @ auxiliary) / eta_aux
    )
    reciprocal_error = abs(augmented_derivative - predicted_derivative)

    assert identity_error < 1.0e-12, identity_error
    assert direct_derivative <= completing_square_bound + 1.0e-12
    assert direct_derivative <= passivity_bound + 1.0e-12
    assert reciprocal_error < 1.0e-12, reciprocal_error

    print("PASS verify_memory_energy_identity")
    print(f"identity_error={identity_error:.6e}")
    print(
        "completing_square_margin="
        f"{completing_square_bound - direct_derivative:.6e}"
    )
    print(f"passivity_margin={passivity_bound - direct_derivative:.6e}")
    print(f"reciprocal_cross_term_error={reciprocal_error:.6e}")


if __name__ == "__main__":
    main()

