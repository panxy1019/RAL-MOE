"""Path-configurable entry point for the frozen one-sided route trainer."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    source = Path(__file__).with_name("train_sh_routes.py")
    spec = importlib.util.spec_from_file_location("resplit_route_trainer", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["resplit_route_trainer"] = module
    spec.loader.exec_module(module)
    module.ROOT = known.root
    module.METHODS = {
        "T2-C_LearnedConvexCorrection_FieldBlend": 42001,
        "RiskPredictionRouter": 42001,
        "LookAheadShortRolloutRouter": 42001,
    }
    sys.argv = [sys.argv[0], *remaining]
    module.main()


if __name__ == "__main__":
    main()
