"""Continuous state-addressed delta memory without neural parameters."""

from __future__ import annotations

import numpy as np


def memory_rhs(state: np.ndarray, key: np.ndarray, value: np.ndarray, gamma, eta: float):
    """Evaluate Sdot=-Gamma*S+eta*k*(v-S.T*k)^T."""
    state = np.asarray(state, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    if gamma.ndim == 0:
        decay = float(gamma) * state
    elif gamma.shape == (state.shape[0],):
        decay = gamma[:, None] * state
    else:
        raise ValueError("gamma must be scalar or have one value per key channel")
    prediction_error = value - state.T @ key
    return -decay + float(eta) * np.outer(key, prediction_error)


def memory_readout(state: np.ndarray, query: np.ndarray):
    return np.asarray(state, dtype=np.float64).T @ np.asarray(query, dtype=np.float64)
