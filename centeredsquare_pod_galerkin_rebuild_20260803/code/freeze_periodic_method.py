#!/usr/bin/env python3
"""Freeze the periodic FVM-GROM before held-out field access."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""): digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-metrics",type=Path,required=True)
    parser.add_argument("--fvm-tensors",type=Path,required=True)
    parser.add_argument("--pressure-closure",type=Path,required=True)
    parser.add_argument("--evaluator",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    validation=json.loads(args.validation_metrics.read_text())
    if validation.get("heldout_loaded") is not False: raise RuntimeError("validation contamination")
    payload={
        "schema_version":1,"status":"FROZEN",
        "method":"periodic OpenFOAM-discrete FVM-Galerkin r28 with ungauged rp24 linear pressure closure",
        "fvm_tensors_sha256":sha256(args.fvm_tensors),
        "pressure_closure_sha256":sha256(args.pressure_closure),
        "evaluator_sha256":sha256(args.evaluator),
        "validation_metrics_sha256":sha256(args.validation_metrics),
        "validation_aggregate":validation["aggregate"],
        "choices":{"velocity_rank":28,"pressure_rank":24,"probe_epsilon":0.05,"pressure_kind":"linear","pressure_ridge":0.01,"warmup_intervals":2,"max_internal_step":0.05,"eddy_damping":0.0},
        "heldout_loaded_at_freeze":False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
