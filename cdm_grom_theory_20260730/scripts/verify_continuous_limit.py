#!/usr/bin/env python3
"""Sanity checks for the discrete-to-continuous delta-memory limit."""

from __future__ import annotations

import math

import numpy as np


def recurrence_step(
    state: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gamma: np.ndarray,
    eta: float,
    step: float,
) -> np.ndarray:
    decay = np.diag(np.exp(-step * gamma))
    beta = 1.0 - math.exp(-step * eta)
    forgotten = decay @ state
    return forgotten + beta * np.outer(key, value - forgotten.T @ key)


def memory_rhs(
    state: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gamma: np.ndarray,
    eta: float,
) -> np.ndarray:
    return -gamma[:, None] * state + eta * np.outer(
        key, value - state.T @ key
    )


def exact_constant_solution(
    initial: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gamma: np.ndarray,
    eta: float,
    final_time: float,
) -> np.ndarray:
    operator = np.diag(gamma) + eta * np.outer(key, key)
    forcing = eta * np.outer(key, value)
    equilibrium = np.linalg.solve(operator, forcing)
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    propagator = (
        eigenvectors * np.exp(-eigenvalues * final_time)
    ) @ eigenvectors.T
    return equilibrium + propagator @ (initial - equilibrium)


def observed_orders(errors: list[float]) -> list[float]:
    return [
        math.log(errors[i] / errors[i + 1], 2.0)
        for i in range(len(errors) - 1)
    ]


def main() -> None:
    rng = np.random.default_rng(20260731)
    d_key, d_value = 5, 3
    state0 = rng.normal(scale=0.4, size=(d_key, d_value))
    key = rng.normal(size=d_key)
    key /= np.linalg.norm(key)
    value = rng.normal(scale=0.6, size=d_value)
    gamma = 0.3 + rng.random(d_key)
    eta = 0.7

    local_steps = [0.2 / (2**i) for i in range(7)]
    rhs0 = memory_rhs(state0, key, value, gamma, eta)
    expansion_errors = [
        np.linalg.norm(
            recurrence_step(state0, key, value, gamma, eta, h)
            - state0
            - h * rhs0
        )
        for h in local_steps
    ]
    expansion_orders = observed_orders(expansion_errors)
    one_step_errors = [
        np.linalg.norm(
            recurrence_step(state0, key, value, gamma, eta, h)
            - exact_constant_solution(
                state0, key, value, gamma, eta, h
            )
        )
        for h in local_steps
    ]
    one_step_orders = observed_orders(one_step_errors)

    final_time = 1.0
    exact = exact_constant_solution(
        state0, key, value, gamma, eta, final_time
    )
    global_steps = [1.0 / (8 * 2**i) for i in range(7)]
    global_errors: list[float] = []
    for h in global_steps:
        state = state0.copy()
        for _ in range(round(final_time / h)):
            state = recurrence_step(state, key, value, gamma, eta, h)
        global_errors.append(np.linalg.norm(state - exact))
    global_orders = observed_orders(global_errors)

    wrong_rhs = -gamma[:, None] * state0 - eta * np.outer(
        key, value - state0.T @ key
    )
    wrong_sign_errors = [
        np.linalg.norm(
            recurrence_step(state0, key, value, gamma, eta, h)
            - state0
            - h * wrong_rhs
        )
        for h in local_steps
    ]
    wrong_sign_orders = observed_orders(wrong_sign_errors)

    assert min(expansion_orders[-3:]) > 1.95, expansion_orders
    assert min(one_step_orders[-3:]) > 1.95, one_step_orders
    assert 0.90 < float(np.mean(global_orders[-3:])) < 1.10, global_orders
    assert max(wrong_sign_orders[-3:]) < 1.10, wrong_sign_orders

    print("PASS verify_continuous_limit")
    print(f"expansion_errors={expansion_errors}")
    print(f"expansion_orders={expansion_orders}")
    print(f"one_step_errors={one_step_errors}")
    print(f"one_step_orders={one_step_orders}")
    print(f"global_errors={global_errors}")
    print(f"global_orders={global_orders}")
    print(f"wrong_sign_orders={wrong_sign_orders}")


if __name__ == "__main__":
    main()
