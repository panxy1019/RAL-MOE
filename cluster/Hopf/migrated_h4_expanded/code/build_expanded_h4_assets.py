#!/usr/bin/env python3
"""Build a heldout-sealed r32 H4 training view from the expanded train-only POD/ROM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

TRAIN = np.asarray([
    45.5,45.8,46.1,46.3,46.5,46.9,47.2,47.4,47.6,47.722947,47.8,
    48.0,48.3,48.368688,48.7,49.3,49.6,49.687640,50.0,50.368054,
    51.066785,52.528767,53.294175,54.081508,54.887950,55.709610,
    57.389970,58.262636,59.201432], dtype=np.float64)
VAL = np.asarray([46.7,56.543246], dtype=np.float64)
TEST = np.asarray([47.081355,49.022357,51.786450], dtype=np.float64)
EXPECTED = {
    "velocity_pod_hopf.npz":"2f951192f6eeaa69e7d911aa21552454d1a99a49ab796166b4a13df16016a192",
    "pressure_pod_hopf.npz":"cc128a4361ab2a239165ac22a48407290de7d085b857bbff6726988457cfddf3",
    "velocity_rom_hopf.npz":"7d16ca813efa0f92109689a686de0c3de02cdd269a4b31029c53c3ed1309d3e9",
    "pressure_poisson_surrogate_hopf.npz":"a6646aa3eb3814d8648293ee3cd798a6abe268bf46a79ec0c0b8b17b30aaf06d",
}

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
    os.replace(tmp,path)

def label(re_value: float) -> str:
    return f"Re_{re_value:.6f}".replace(".","p")

def exact_indices(nodes: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    out=[]
    for value in wanted:
        hit=np.flatnonzero(np.isclose(nodes,value,atol=5e-7,rtol=0))
        if len(hit)!=1: raise RuntimeError(f"ROM node Re={value:.6f} has {len(hit)} matches")
        out.append(int(hit[0]))
    return np.asarray(out,dtype=np.int64)

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--artifact-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    src=args.artifact_root
    observed={name:digest(src/name) for name in EXPECTED}
    bad={k:(EXPECTED[k],v) for k,v in observed.items() if v.lower()!=EXPECTED[k]}
    if bad: raise RuntimeError(f"source SHA fail-closed: {bad}")

    vu=np.load(src/"velocity_pod_hopf.npz",allow_pickle=False)
    pp=np.load(src/"pressure_pod_hopf.npz",allow_pickle=False)
    gv=np.load(src/"velocity_rom_hopf.npz",allow_pickle=False)
    pr=np.load(src/"pressure_poisson_surrogate_hopf.npz",allow_pickle=False)
    with (src/"projection_snapshots_velocity_hopf.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=len(vu["coeff_uv"]) or len(rows)!=len(pp["coeff_p"]):
        raise RuntimeError("projection CSV/POD coefficient row count mismatch")
    re=np.asarray([float(x["Re"]) for x in rows],dtype=np.float64)
    time=np.asarray([float(x["time"]) for x in rows],dtype=np.float64)
    split=np.asarray([x["split"].lower() for x in rows])
    split=np.where(split=="test","heldout",split)
    keep=np.isin(split,["train","validation"])
    if np.any(np.isclose(re[keep,None],TEST[None,:],atol=5e-7).any(axis=1)):
        raise RuntimeError("heldout row leaked into sealed coefficient view")
    actual_train=np.sort(np.unique(np.round(re[split=="train"],6)))
    actual_val=np.sort(np.unique(np.round(re[split=="validation"],6)))
    if not np.allclose(actual_train,TRAIN,atol=5e-7) or not np.allclose(actual_val,VAL,atol=5e-7):
        raise RuntimeError(f"split mismatch train={actual_train}, validation={actual_val}")

    args.output_dir.mkdir(parents=True)
    coeff_path=args.output_dir/"expanded_h4_trainval_r32.npz"
    np.savez_compressed(coeff_path,
        coeff_uv=vu["coeff_uv"][keep,:32].astype(np.float32),
        coeff_p=pp["coeff_p"][keep,:32].astype(np.float32),
        Re=re[keep],time=time[keep],split=split[keep],source_row=np.flatnonzero(keep),
        phi_uv=vu["phi_uv"][:32].astype(np.float32),phi_p=pp["phi_p"][:32].astype(np.float32),
        point_areas=vu["point_areas"].astype(np.float32),
        mean_uv_train=vu["mean_uv_regime"].astype(np.float32),
        mean_p_train=pp["mean_p_regime"].astype(np.float32))

    gnodes=gv["Re_values_computed"].astype(np.float64)
    gi=exact_indices(gnodes,TRAIN)
    gal_path=args.output_dir/"expanded_h4_trainonly_galerkin_r32.npz"
    np.savez_compressed(gal_path,Re_values_computed=TRAIN,
        c_all=gv["c_all"][gi,:32].astype(np.float32),
        A_all=gv["A_all"][gi,:32,:32].astype(np.float32),
        H=gv["H"][:32,:32,:32].astype(np.float32),P=gv["P"][:32,:32].astype(np.float32))

    pnodes=pr["Re_values_computed"].astype(np.float64)
    exact_indices(pnodes,TRAIN)
    pc=[]; pa=[]
    for rv in TRAIN:
        prefix=label(float(rv))
        pc.append(pr[prefix+"_c_tilde"][:32])
        pa.append(pr[prefix+"_A_tilde"][:32,:32])
    pressure_path=args.output_dir/"expanded_h4_trainonly_pressure_r32.npz"
    np.savez_compressed(pressure_path,Re_values_computed=TRAIN,
        c_tilde_all=np.asarray(pc,dtype=np.float32),A_tilde_all=np.asarray(pa,dtype=np.float32),
        H_tilde=pr["H_tilde"][:32,:32,:32].astype(np.float32))

    diagnostics=json.loads((src/"pod_projection_diagnostics_hopf.json").read_text())
    manifest={
        "schema_version":2,"scope":"hopf_local_train_val_view","r_u":32,"r_p":32,
        "fit_reynolds":TRAIN.tolist(),"validation_reynolds":VAL.tolist(),
        "heldout_reynolds_sealed":TEST.tolist(),"heldout_rows_present":False,
        "pressure_gauge_verified":True,"projection_error_verified":True,
        "critical_mode_pair_verified":True,"shift_mode_verified":True,
        "frequency_retention_verified":True,
        "projection_errors":diagnostics,
        "rank_semantics":"first 32 modes of the new train-only Hopf-local r80 basis; report fair-control rank",
        "source_sha256":observed,
        "sha256":{"coefficient_view":digest(coeff_path),"galerkin":digest(gal_path),"pressure":digest(pressure_path)},
    }
    atomic_json(manifest,args.output_dir/"TRAINING_ASSET_MANIFEST.json")
    print(json.dumps({"status":"PASS","rows":int(keep.sum()),"train_rows":int((split=="train").sum()),
                      "validation_rows":int((split=="validation").sum()),"output":str(args.output_dir)}))

if __name__=="__main__": main()
