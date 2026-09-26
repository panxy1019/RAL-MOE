"""Periodic train/validation-only adaptation of the tested H polynomial pilot."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[1]
ROOT=R.parents[1]
BASE=ROOT/'periodic_specialist_r32/assets'
OUT=R/'OpInf_P_development'

def main():
 data=R/'OpInf_P_trainval.npz'
 if not data.exists():
  u=np.load(BASE/'Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz')
  p=np.load(BASE/'Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz')
  with (BASE/'provenance/projection_snapshots_velocity_periodic.csv').open() as f:rows=list(csv.DictReader(f))
  split=np.array([r['split'].lower() for r in rows]);keep=np.isin(split,['train','validation'])
  assert set(split[keep])=={'train','validation'}
  re=np.array([float(r['Re']) for r in rows]);t=np.array([float(r['time']) for r in rows])
  assert len(rows)==len(u['coeff_uv'])==len(p['coeff_p'])
  assert len(np.unique(re[split=='train']))==53 and len(np.unique(re[split=='validation']))==6
  np.savez_compressed(data,coeff_uv=u['coeff_uv'][keep,:32],coeff_p=p['coeff_p'][keep,:32],
    Re=re[keep],time=t[keep],split=split[keep],phi_uv=u['phi_uv'][:32],phi_p=p['phi_p'][:32],
    mean_uv_train=u['mean_uv_regime'],mean_p_train=p['mean_p_regime'],point_areas=u['point_areas'])
 spec=importlib.util.spec_from_file_location('opinf_development',R/'code/opinf_hopf_development.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT;m.DATA=data;m.main()

if __name__=='__main__':main()
