#!/usr/bin/env python3
"""Dependency-free static contract checks for the strict Hopf trainer."""
from pathlib import Path
import ast

HERE = Path(__file__).resolve().parent
SOURCE = HERE.joinpath("train_hopf_moe.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

assert "HELDOUT HARD-GATE" in SOURCE
assert "phase-free by construction" in SOURCE
assert "arrays_t[\"phase\"]" not in SOURCE
assert "global_velocity_pod" not in SOURCE
assert "num_regime_groups=1" in SOURCE
assert "experts_per_group=args.experts" in SOURCE
assert "top_k=args.top_k" in SOURCE
assert "model.group_router.requires_grad_(False)" in SOURCE
assert "PENDING_USER_FREEZE_AND_HELDOUT" in SOURCE
assert "heldout_evaluation_performed\":False" in SOURCE
assert "(0, 1200, 1), (1200, 2800, 2), (2800, 4800, 4), (4800, 7500, 8)" in SOURCE

functions = {n.name for n in ast.walk(TREE) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
for required in ("audit_assets", "load_train_rom", "autonomous_step", "scale_loss",
                 "audit_scale_gradient", "validate", "validation_lock"):
    assert required in functions, required

print("static Hopf trainer contract: PASS")
