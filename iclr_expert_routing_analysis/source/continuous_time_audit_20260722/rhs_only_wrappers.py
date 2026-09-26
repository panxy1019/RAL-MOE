"""Audit-only interfaces for the frozen specialist operator contracts.

The specialists expose a continuous velocity derivative and an algebraic pressure
closure.  This module deliberately refuses to pretend that the closure is a
pressure derivative.  It does not own or mutate any checkpoint parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch


class FullStateRHSUnavailable(RuntimeError):
    """Raised when a caller asks the frozen model for an untrained pressure RHS."""


@dataclass(frozen=True)
class RHSOnlyResult:
    velocity_rhs: torch.Tensor
    pressure_closure_output: torch.Tensor
    galerkin_rhs: torch.Tensor
    diagnostics: Mapping[str, Any]

    @property
    def pressure_rhs(self) -> torch.Tensor:
        raise FullStateRHSUnavailable(
            "The frozen specialist predicts an algebraic next-pressure closure, "
            "not db/dt.  Finite-differencing this value would change the contract."
        )

    def full_state_rhs(self) -> torch.Tensor:
        _ = self.pressure_rhs
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class SpecialistOperatorContract:
    name: str
    velocity_operator: str = "continuous_time_derivative"
    pressure_operator: str = "algebraic_next_state_closure"
    dt_is_network_input: bool = False
    supports_full_state_rhs: bool = False

    def require_full_state_rhs(self) -> None:
        if not self.supports_full_state_rhs:
            raise FullStateRHSUnavailable(
                f"{self.name} has no certified 64-dimensional continuous RHS: "
                "velocity is differential, pressure is algebraic/discrete."
            )


class ReadOnlyRHSOnlyWrapper:
    """Thin callable guard around a specialist's native pre-integrator path.

    ``native_rhs`` must perform the specialist's own scaler, feature builder,
    Galerkin, residual-MoE, and pressure-closure forward.  This class adds no
    transformations; it merely verifies that the supplied modules are frozen
    and prevents accidental gradient recording during the audit call.
    """

    def __init__(
        self,
        contract: SpecialistOperatorContract,
        native_rhs: Callable[..., RHSOnlyResult],
        frozen_modules: Sequence[torch.nn.Module],
    ) -> None:
        self.contract = contract
        self.native_rhs = native_rhs
        self.frozen_modules = tuple(frozen_modules)
        trainable = [name for module in self.frozen_modules for name, p in module.named_parameters() if p.requires_grad]
        if trainable:
            raise ValueError(f"rhs_only audit wrapper received trainable parameters: {trainable[:8]}")

    def __call__(self, *args: Any, **kwargs: Any) -> RHSOnlyResult:
        with torch.no_grad():
            result = self.native_rhs(*args, **kwargs)
        if not isinstance(result, RHSOnlyResult):
            raise TypeError("native_rhs must return RHSOnlyResult")
        return result
