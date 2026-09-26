"""Continuous-time delta memory and its matched discrete KDA update.

This module is intentionally independent of the Hopf training stack.  It contains
only the matrix-memory equations that are exercised by the pre-training numerical
tests and later imported by the formal trainer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MemoryConfig:
    current_dim: int = 109
    token_dim: int = 128
    num_heads: int = 4
    d_k: int = 16
    d_v: int = 16
    gamma_min: float = 1.0e-3
    eta_max: float = 0.25
    v_max: float = 1.0
    norm_eps: float = 1.0e-6


class CurrentEncoder(nn.Module):
    def __init__(self, config: MemoryConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.current_dim),
            nn.Linear(config.current_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, config.token_dim),
            nn.SiLU(),
        )

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.net(current)


class DeltaMemoryParameters(nn.Module):
    """Shared parameterization used by both discrete KDA and CTDM."""

    def __init__(self, config: MemoryConfig) -> None:
        super().__init__()
        self.config = config
        h, dk, dv, d = (
            config.num_heads,
            config.d_k,
            config.d_v,
            config.token_dim,
        )
        self.key = nn.Linear(d, h * dk)
        self.value = nn.Linear(d, h * dv)
        self.gamma = nn.Linear(d, h * dk)
        self.eta = nn.Linear(d, h)
        self.query_u = nn.Linear(d, h * dk)
        self.query_p = nn.Linear(d, h * dk)
        # Native Hopf query intervals can exceed twenty physical time units.
        # Start in a non-stiff regime while retaining the requested positive,
        # unbounded gamma parameterization during learning.
        initial_gamma = max(2.0e-2 - config.gamma_min, 1.0e-6)
        initial_gamma_bias = math.log(math.expm1(initial_gamma))
        initial_eta_ratio = min(max(2.0e-2 / config.eta_max, 1.0e-6), 1.0 - 1.0e-6)
        initial_eta_bias = math.log(initial_eta_ratio / (1.0 - initial_eta_ratio))
        nn.init.zeros_(self.gamma.weight)
        nn.init.constant_(self.gamma.bias, initial_gamma_bias)
        nn.init.zeros_(self.eta.weight)
        nn.init.constant_(self.eta.bias, initial_eta_bias)

    def _unit(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value.float(), p=2.0, dim=-1, eps=self.config.norm_eps)

    def write_parameters(
        self, token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, dk, dv = self.config.num_heads, self.config.d_k, self.config.d_v
        with torch.autocast(device_type=token.device.type, enabled=False):
            z = token.float()
            key = self._unit(self.key(z).reshape(-1, h, dk))
            value = self.config.v_max * torch.tanh(
                self.value(z).reshape(-1, h, dv)
            )
            # gamma_min + softplus is required for Gamma >= gamma_min I.
            gamma = self.config.gamma_min + F.softplus(
                self.gamma(z).reshape(-1, h, dk)
            )
            eta = self.config.eta_max * torch.sigmoid(
                self.eta(z).reshape(-1, h)
            )
        return key, value, gamma, eta

    def queries(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, dk = self.config.num_heads, self.config.d_k
        with torch.autocast(device_type=token.device.type, enabled=False):
            z = token.float()
            query_u = self._unit(self.query_u(z).reshape(-1, h, dk))
            query_p = self._unit(self.query_p(z).reshape(-1, h, dk))
        return query_u, query_p

    def read(
        self, memory: torch.Tensor, token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_u, query_p = self.queries(token)
        state = memory.float()
        memory_u = torch.einsum("bhkv,bhk->bhv", state, query_u).flatten(1)
        memory_p = torch.einsum("bhkv,bhk->bhv", state, query_p).flatten(1)
        return memory_u, memory_p

    def derivative_from_parameters(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gamma: torch.Tensor,
        eta: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = memory.float()
        reconstruction = torch.einsum("bhkv,bhk->bhv", state, key)
        error = value - reconstruction
        decay = -gamma.unsqueeze(-1) * state
        write = eta[..., None, None] * key.unsqueeze(-1) * error.unsqueeze(-2)
        derivative = decay + write
        diagnostics = {
            "gamma": gamma,
            "eta": eta,
            "delta_error": torch.linalg.vector_norm(error, dim=-1),
            "memory_norm": torch.linalg.matrix_norm(state, ord="fro", dim=(-2, -1)),
        }
        return derivative, diagnostics

    def derivative(
        self, memory: torch.Tensor, token: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parameters = self.write_parameters(token)
        return self.derivative_from_parameters(memory, *parameters)

    def discrete_update(
        self, memory: torch.Tensor, token: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Matched KDA: (I-beta kk^T) D S + beta k v^T."""

        key, value, gamma, eta = self.write_parameters(token)
        step = dt.float().reshape(-1, 1, 1)
        decay = torch.exp(-step * gamma)
        forgotten = decay.unsqueeze(-1) * memory.float()
        beta = 1.0 - torch.exp(-step[..., 0] * eta)
        reconstruction = torch.einsum("bhkv,bhk->bhv", forgotten, key)
        error = value - reconstruction
        updated = (
            forgotten
            + beta[..., None, None] * key.unsqueeze(-1) * error.unsqueeze(-2)
        )
        diagnostics = {
            "gamma": gamma,
            "eta": eta,
            "beta": beta,
            "delta_error": torch.linalg.vector_norm(error, dim=-1),
            "memory_norm": torch.linalg.matrix_norm(updated, ord="fro", dim=(-2, -1)),
        }
        return updated, diagnostics

    def zero_state(
        self, batch: int, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.zeros(
            batch,
            self.config.num_heads,
            self.config.d_k,
            self.config.d_v,
            device=device,
            dtype=dtype,
        )


class DeltaMemory(nn.Module):
    """Encoder plus matched matrix-memory parameterization."""

    def __init__(self, config: MemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CurrentEncoder(config)
        self.parameters_net = DeltaMemoryParameters(config)

    def encode(self, current: torch.Tensor) -> torch.Tensor:
        return self.encoder(current)
