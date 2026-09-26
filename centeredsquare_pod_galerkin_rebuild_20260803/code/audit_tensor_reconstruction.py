#!/usr/bin/env python3
"""Compare assembled FVM-GROM tensors against a direct OpenFOAM RHS snapshot."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ray/Desktop/centeredSquare/scripts")
from dataset_tools import read_internal_field  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--pressure-pod", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--foam-time", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.tensors) as source:
        tensor = {key: np.asarray(source[key]) for key in source.files}
    ru = int(tensor["velocity_rank"]); rp = int(tensor["pressure_rank"])
    with np.load(args.mesh) as source: volumes = np.asarray(source["cellVolumes"], dtype=np.float64)
    with np.load(args.velocity_pod) as source:
        u_mean=np.asarray(source["mean"],dtype=np.float64); u_weights=np.asarray(source["weights"],dtype=np.float64); u_weighted_modes=np.asarray(source["weighted_modes"],dtype=np.float64)[:ru]; u_modes=np.asarray(source["modes"],dtype=np.float64)[:ru].reshape(ru,len(volumes),2)
    with np.load(args.pressure_pod) as source:
        p_mean=np.asarray(source["mean"],dtype=np.float64); p_weights=np.asarray(source["weights"],dtype=np.float64); p_weighted_modes=np.asarray(source["weighted_modes"],dtype=np.float64)[:rp]
    with np.load(args.snapshot) as source:
        times=np.asarray(source["times"]); index=int(np.flatnonzero(np.isclose(times,100))[0]); u=np.asarray(source["U"],dtype=np.float64)[index]; p=np.asarray(source["p"],dtype=np.float64)[index]; re_value=float(source["Re"])
    a=((u-u_mean).reshape(-1)*u_weights)@u_weighted_modes.T
    b=((p-p_mean)*p_weights)@p_weighted_modes.T
    direct_fields={name:read_internal_field(args.foam_time/name,"vector",len(volumes))[:,:2] for name in ["romConvection","romDiffusion","romPressure","romRhs"]}
    direct={name:np.einsum("n,inc,nc->i",volumes,u_modes,value,optimize=True) for name,value in direct_fields.items()}
    predicted=(tensor["c_conv"]+(1/re_value)*tensor["c_diff"]+tensor["c_pressure"]
        +(tensor["A_conv"]+(1/re_value)*tensor["A_diff"])@a
        +np.einsum("ijk,j,k->i",tensor["H_conv"],a,a,optimize=True)+tensor["P"]@b)
    payload={
        "Re":re_value,"time":100.0,"velocity_rank":ru,"pressure_rank":rp,
        "direct_rhs_norm":float(np.linalg.norm(direct["romRhs"])),
        "tensor_rhs_norm":float(np.linalg.norm(predicted)),
        "tensor_vs_direct_relative_l2":float(np.linalg.norm(predicted-direct["romRhs"])/np.linalg.norm(direct["romRhs"])),
        "state_norm":float(np.linalg.norm(a)),"pressure_coefficient_norm":float(np.linalg.norm(b)),
        "validation_loaded":False,"heldout_loaded":False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
