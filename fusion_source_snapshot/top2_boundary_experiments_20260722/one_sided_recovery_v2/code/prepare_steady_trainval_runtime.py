"""Materialize a test-sealed Steady runtime view from the existing train/val bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


ROOT = Path("/root/panxy/particalMOE/steady_specialist_v1")


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def main() -> None:
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise RuntimeError(f"refusing non-empty output: {a.output_dir}")
    art=a.output_dir/"artifacts"; compat=art/"Global_POD_AreaWeighted_L2"; code=a.output_dir/"code"; compat.mkdir(parents=True); code.mkdir()
    bundle_path=ROOT/"data/steady_trainval_modal_r32.npz"
    with np.load(bundle_path,allow_pickle=False) as z: bundle={k:z[k] for k in z.files}
    if set(bundle["split"].tolist()) != {"train","validation"}: raise RuntimeError("trainval bundle contains another split")
    labels=[]
    for value in bundle["Re_label"].tolist():
        if value not in labels: labels.append(value)
    label_to_split={label:str(bundle["split"][np.flatnonzero(bundle["Re_label"]==label)[0]]) for label in labels}
    src_art=ROOT/"source_artifacts/steady"
    for kind,coeff_key,raw_key in (("velocity","coeff_uv","a_raw"),("pressure","coeff_p","b_raw")):
        src=src_art/f"{kind}_pod_steady.npz"
        with np.load(src,allow_pickle=False) as z:
            forbidden={"coeff_uv","coeff_p","snapshot_splits","snapshot_Re_labels","mean_uv_by_Re","mean_p_by_Re","Re_values","Re_labels","regimes","split_by_Re"}
            out={k:z[k] for k in z.files if k not in forbidden}
        coeff=np.zeros((len(bundle[raw_key]),80),dtype=np.float32); coeff[:,:32]=bundle[raw_key]
        out[coeff_key]=coeff
        out["snapshot_splits"]=bundle["split"].astype(str); out["snapshot_Re_labels"]=bundle["Re_label"].astype(str)
        out["Re_labels"]=np.asarray(labels); out["Re_values"]=np.asarray([np.mean(bundle["Re"][bundle["Re_label"]==label]) for label in labels],dtype=np.float64)
        out["regimes"]=np.asarray(["steady"]*len(labels)); out["split_by_Re"]=np.asarray([label_to_split[label] for label in labels])
        mean_key="mean_uv_regime" if kind=="velocity" else "mean_p_regime"
        by_key="mean_uv_by_Re" if kind=="velocity" else "mean_p_by_Re"
        out[by_key]=np.repeat(out[mean_key][None],len(labels),axis=0)
        np.savez_compressed(art/f"{kind}_pod_steady.npz",**out)
    source_csv=src_art/"Global_POD_AreaWeighted_L2/pod_snapshot_index.csv"
    allowed=set(bundle["source_snapshot_id"].astype(int).tolist())
    with source_csv.open(newline="",encoding="utf-8") as f: rows=[row for row in csv.DictReader(f) if int(row["snapshot_id"]) in allowed]
    rows.sort(key=lambda row: bundle["source_snapshot_id"].tolist().index(int(row["snapshot_id"])))
    if len(rows)!=len(bundle["source_snapshot_id"]): raise RuntimeError("filtered indexed phase rows mismatch")
    for i,row in enumerate(rows): row["snapshot_id"]=str(i); row["split"]=str(bundle["split"][i]); row["local_snapshot_index"]=str(int(bundle["local_snapshot_index"][i]))
    with (compat/"pod_snapshot_index.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    source_to_new={int(source):i for i,source in enumerate(bundle["source_snapshot_id"].tolist())}
    with np.load(ROOT/"data/perturbation_bank_train_validation.npz",allow_pickle=False) as z: bank={k:z[k] for k in z.files}
    original_labels=[]
    for row in rows:
        label=row["Re_label"]
        if label not in original_labels: original_labels.append(label)
    # Bank label ids refer to the original 20-label POD ordering; recover that ordering
    # from non-field metadata and remap to this 16-label sealed view.
    with np.load(src_art/"velocity_pod_steady.npz",allow_pickle=False) as z: full_labels=z["Re_labels"].astype(str).tolist()
    label_id_map={i:labels.index(label) for i,label in enumerate(full_labels) if label in labels}
    for key in ("train_fixed_id","validation_fixed_id","train_fixed_ids","validation_fixed_ids"):
        bank[key]=np.asarray([source_to_new[int(v)] for v in bank[key]],dtype=np.int64)
    # train_schedule indexes rows of the perturbation bank (not snapshots).
    for key in ("train_label_id","validation_label_id","train_fixed_labels","validation_fixed_labels"):
        bank[key]=np.asarray([label_id_map[int(v)] for v in bank[key]],dtype=np.int64)
    np.savez_compressed(a.output_dir/"perturbation_bank_train_validation_reindexed.npz",**bank)
    os.symlink("../velocity_pod_steady.npz",compat/"global_velocity_pod_area_weighted_l2.npz")
    os.symlink("../pressure_pod_steady.npz",compat/"global_pressure_pod_area_weighted_l2.npz")
    for name in ("normalization_steady.npz","velocity_rom_steady.npz","pressure_poisson_surrogate_steady.npz"):
        os.symlink(src_art/name,art/name)
    trainer_src=(ROOT/"code/train_s2b_3090.py").read_text()
    old='EXPECTED_SPLITS = {"steady": (14, 2, 4)}'; new='EXPECTED_SPLITS = {"steady": (14, 2, 0)}'
    if old not in trainer_src: raise RuntimeError("trainer split constant drift")
    (code/"train_s2b_3090_trainval_only.py").write_text(trainer_src.replace(old,new,1))
    cfg=json.loads((ROOT/"code/training_s2b_portable.json").read_text()); cfg["artifact_dir"]=str(art); cfg["checkpoint_root"]=str(a.output_dir/"runtime_checkpoints")
    (a.output_dir/"training_s2b_trainval_only.json").write_text(json.dumps(cfg,indent=2))
    manifest={"status":"TEST_SEALED_TRAINVAL_RUNTIME_READY","bundle":str(bundle_path),"bundle_sha256":sha256(bundle_path),"snapshots":len(rows),"split_counts":{s:int(np.sum(bundle["split"]==s)) for s in ("train","validation")},"heldout_coefficients_loaded":False,"heldout_fields_loaded":False,"phase_contract":"exact indexed phase rows filtered by source_snapshot_id","bank_reindexed_from_train_validation_only":True,"specialist_checkpoint_modified":False}
    (a.output_dir/"MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
