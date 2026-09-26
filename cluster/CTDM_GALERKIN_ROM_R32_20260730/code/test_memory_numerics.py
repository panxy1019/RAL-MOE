#!/usr/bin/env python3
"""Pre-training mathematical tests for the CTDM matrix-memory equation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch


Tensor = torch.Tensor


def smooth_inputs(t: float, d_k: int = 4, d_v: int = 3):
    tt = torch.tensor(t, dtype=torch.float64)
    key = torch.stack(
        (
            1.0 + 0.2 * torch.sin(0.7 * tt),
            0.8 * torch.cos(0.4 * tt),
            0.5 * torch.sin(1.1 * tt + 0.2),
            0.3 + 0.1 * torch.cos(0.9 * tt),
        )
    )[:d_k]
    key = key / torch.linalg.vector_norm(key)
    value = torch.stack(
        (
            0.7 * torch.sin(0.6 * tt),
            0.5 * torch.cos(0.8 * tt + 0.1),
            0.4 * torch.sin(0.3 * tt + 0.7),
        )
    )[:d_v]
    gamma = 0.08 + torch.stack(
        (
            0.03 * (1.0 + torch.sin(0.2 * tt)),
            0.02 * (1.0 + torch.cos(0.5 * tt)),
            0.025 * (1.0 + torch.sin(0.4 * tt + 0.3)),
            0.015 * (1.0 + torch.cos(0.7 * tt)),
        )
    )[:d_k]
    eta = 0.18 + 0.04 * torch.sin(0.35 * tt)
    return key, value, gamma, eta


def rhs(t: float, state: Tensor, write_sign: float = 1.0) -> Tensor:
    key, value, gamma, eta = smooth_inputs(t, state.shape[0], state.shape[1])
    reconstruction = state.T @ key
    return (
        -gamma[:, None] * state
        + write_sign * eta * key[:, None] * (value - reconstruction)[None, :]
    )


def rk4_step(
    function: Callable[[float, Tensor], Tensor], t: float, state: Tensor, dt: float
) -> Tensor:
    k1 = function(t, state)
    k2 = function(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = function(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = function(t + dt, state + dt * k3)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_rk4(dt: float, final_time: float, write_sign: float = 1.0) -> Tensor:
    state = torch.zeros(4, 3, dtype=torch.float64)
    count = int(round(final_time / dt))
    for index in range(count):
        state = rk4_step(
            lambda time, value: rhs(time, value, write_sign),
            index * dt,
            state,
            dt,
        )
    return state


def integrate_discrete(dt: float, final_time: float) -> Tensor:
    state = torch.zeros(4, 3, dtype=torch.float64)
    count = int(round(final_time / dt))
    identity = torch.eye(4, dtype=torch.float64)
    for index in range(count):
        key, value, gamma, eta = smooth_inputs(index * dt)
        decay = torch.diag(torch.exp(-dt * gamma))
        beta = 1.0 - torch.exp(-eta * dt)
        state = (
            (identity - beta * key[:, None] @ key[None, :]) @ decay @ state
            + beta * key[:, None] @ value[None, :]
        )
    return state


def relative_error(value: Tensor, reference: Tensor) -> float:
    return float(
        torch.linalg.matrix_norm(value - reference)
        / (torch.linalg.matrix_norm(reference) + 1.0e-14)
    )


def observed_orders(steps: list[float], errors: list[float]) -> list[float]:
    return [
        math.log(errors[index] / errors[index + 1])
        / math.log(steps[index] / steps[index + 1])
        for index in range(len(errors) - 1)
    ]


def discrete_to_continuous_test() -> dict:
    final_time = 2.0
    reference = integrate_rk4(2.0 ** -12, final_time)
    steps = [2.0 ** -5, 2.0 ** -6, 2.0 ** -7, 2.0 ** -8]
    errors = [relative_error(integrate_discrete(step, final_time), reference) for step in steps]
    orders = observed_orders(steps, errors)
    # The sign-reversed equation in the prompt cannot be the limit of the
    # plus-write discrete KDA.  Its error remains O(1) as dt tends to zero.
    wrong_reference = integrate_rk4(2.0 ** -12, final_time, write_sign=-1.0)
    wrong_errors = [
        relative_error(integrate_discrete(step, final_time), wrong_reference)
        for step in steps
    ]
    passed = (
        all(errors[index + 1] < errors[index] for index in range(len(errors) - 1))
        and float(np.median(orders[-2:])) > 0.9
        and wrong_errors[-1] > 20.0 * errors[-1]
    )
    return {
        "name": "DISCRETE_TO_CONTINUOUS_CONVERGENCE",
        "steps": steps,
        "errors": errors,
        "observed_orders": orders,
        "wrong_sign_errors": wrong_errors,
        "passed": passed,
    }


def rk4_order_test() -> dict:
    final_time = 2.0
    reference = integrate_rk4(2.0 ** -12, final_time)
    steps = [2.0 ** -3, 2.0 ** -4, 2.0 ** -5, 2.0 ** -6]
    errors = [relative_error(integrate_rk4(step, final_time), reference) for step in steps]
    orders = observed_orders(steps, errors)
    passed = (
        all(errors[index + 1] < errors[index] for index in range(len(errors) - 1))
        and float(np.median(orders)) > 3.7
    )
    return {
        "name": "RK4_ORDER_TEST",
        "steps": steps,
        "errors": errors,
        "observed_orders": orders,
        "passed": passed,
    }


def bounded_inputs(t: float):
    key = torch.tensor(
        [1.0, math.sin(0.17 * t), math.cos(0.11 * t), 0.4],
        dtype=torch.float64,
    )
    key = key / torch.linalg.vector_norm(key)
    value = 0.75 * torch.tensor(
        [math.sin(0.13 * t), math.cos(0.19 * t), math.sin(0.07 * t + 0.4)],
        dtype=torch.float64,
    )
    gamma = torch.full((4,), 0.05, dtype=torch.float64)
    eta = torch.tensor(0.2, dtype=torch.float64)
    return key, value, gamma, eta


def bounded_rhs(t: float, state: Tensor) -> Tensor:
    key, value, gamma, eta = bounded_inputs(t)
    error = value - state.T @ key
    return -gamma[:, None] * state + eta * key[:, None] * error[None, :]


def memory_boundedness_test() -> dict:
    dt, final_time = 0.05, 200.0
    state = torch.zeros(4, 3, dtype=torch.float64)
    norms: list[float] = []
    for index in range(int(round(final_time / dt))):
        state = rk4_step(bounded_rhs, index * dt, state, dt)
        if index % 40 == 0:
            norms.append(float(torch.linalg.matrix_norm(state)))
    gamma_min, eta_max = 0.05, 0.2
    value_norm_bound = math.sqrt(3.0) * 0.75
    theoretical_asymptotic_norm = math.sqrt(
        eta_max * value_norm_bound**2 / (4.0 * gamma_min)
    )
    tail = np.asarray(norms[len(norms) // 2 :])
    passed = (
        np.isfinite(norms).all()
        and max(norms) < 1.25 * theoretical_asymptotic_norm
        and tail[-1] < 1.05 * max(tail)
    )
    return {
        "name": "MEMORY_BOUNDEDNESS_TEST",
        "dt": dt,
        "final_time": final_time,
        "max_memory_norm": max(norms),
        "final_memory_norm": norms[-1],
        "theoretical_asymptotic_norm": theoretical_asymptotic_norm,
        "passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    tests = [
        discrete_to_continuous_test(),
        rk4_order_test(),
        memory_boundedness_test(),
    ]
    payload = {
        "schema_version": 1,
        "dtype": "torch.float64",
        "equation": "dS/dt=-Gamma S + eta k (v-S^T k)^T",
        "all_passed": all(test["passed"] for test in tests),
        "tests": tests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
