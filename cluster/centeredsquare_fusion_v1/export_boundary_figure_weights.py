#!/usr/bin/env python3
"""Export frozen per-window T2-C weights for auditable field figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from common import ConvexGate, router_probabilities, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--boundary", choices=("SH", "HP"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    gate_path = args.run_root / f"gate_{args.boundary.lower()}" / "best.pt"
    router_path = args.run_root / "e2_router" / "best.pt"
    checkpoint = torch.load(gate_path, map_location="cpu", weights_only=False)
    if sha256(router_path) != checkpoint["router_checkpoint_sha256"]:
        raise RuntimeError("gate/router checkpoint hash mismatch")

    with np.load(args.cache, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if set(np.asarray(data["split"]).astype(str)) != {"heldout"}:
        raise RuntimeError("figure weights must be exported from heldout-only cache")

    pair_indices = np.asarray(checkpoint["pair_indices"], dtype=np.int64)
    observed_pair = np.asarray(data["pair_indices"][0], dtype=np.int64)
    if not np.array_equal(pair_indices, observed_pair):
        raise RuntimeError(
            f"pair order mismatch: gate={pair_indices}, cache={observed_pair}"
        )

    features = (
        (data["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    ).astype(np.float32)
    probabilities = router_probabilities(router_path, data["re"])
    pair = probabilities[:, pair_indices]
    pair /= pair.sum(axis=1, keepdims=True)
    base = np.clip(pair[:, 0], 1.0e-6, 1.0 - 1.0e-6)
    base_logit = np.log(base / (1.0 - base)).astype(np.float32)

    gate = ConvexGate(features.shape[1])
    gate.load_state_dict(checkpoint["gate_state"], strict=True)
    gate.eval()
    with torch.inference_mode():
        alpha = gate(
            torch.as_tensor(features), torch.as_tensor(base_logit)
        ).numpy()
    if not np.all(np.isfinite(alpha)) or np.any((alpha < 0.0) | (alpha > 1.0)):
        raise RuntimeError("invalid T2-C weights")

    candidate_1 = str(data["source_name"].item())
    candidate_2 = "Hopf"
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_csv}")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "boundary",
                "Re",
                "start",
                "candidate_1",
                "candidate_2",
                "T2_C_alpha_candidate_1",
            ),
        )
        writer.writeheader()
        for reynolds, start, weight in zip(data["re"], data["starts"], alpha):
            writer.writerow(
                {
                    "boundary": args.boundary,
                    "Re": f"{float(reynolds):.6f}",
                    "start": int(start),
                    "candidate_1": candidate_1,
                    "candidate_2": candidate_2,
                    "T2_C_alpha_candidate_1": f"{float(weight):.10f}",
                }
            )

    manifest = {
        "schema": "boundary_figure_weights/v1",
        "boundary": args.boundary,
        "candidate_order": [candidate_1, candidate_2],
        "windows": int(len(alpha)),
        "alpha_min": float(np.min(alpha)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_max": float(np.max(alpha)),
        "cache": str(args.cache),
        "cache_sha256": sha256(args.cache),
        "router_checkpoint": str(router_path),
        "router_checkpoint_sha256": sha256(router_path),
        "gate_checkpoint": str(gate_path),
        "gate_checkpoint_sha256": sha256(gate_path),
        "output_csv": str(args.output_csv),
        "output_csv_sha256": sha256(args.output_csv),
    }
    manifest_path = args.output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
