#!/usr/bin/env python3
"""Fit a low-order discrete residual around the stable FVM-Galerkin core."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluate_fvm_grom import pressure_features, project_case


VALIDATION_RE = np.asarray([94.5, 95.25, 95.5, 97.5, 99.0, 101.5])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--pressure-pod", type=Path, required=True)
    parser.add_argument("--fvm-tensors", type=Path, required=True)
    parser.add_argument("--pressure-closure", type=Path, required=True)
    parser.add_argument("--mean-calibration", type=Path, required=True)
    parser.add_argument("--canonical-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-gamma", type=float, default=0.1)
    parser.add_argument("--max-step", type=float, default=0.1)
    return parser.parse_args()


def interpolate(nodes, array, value):
    if value <= nodes[0]: left, right = 0, 1
    elif value >= nodes[-1]: left, right = len(nodes)-2, len(nodes)-1
    else: right = int(np.searchsorted(nodes, value)); left = right-1
    fraction = (value-nodes[left])/(nodes[right]-nodes[left])
    return (1-fraction)*array[left] + fraction*array[right]


def residual_features(state, viscosity, kind):
    value = np.atleast_2d(state)
    nu = np.full((len(value), 1), viscosity)
    columns = [np.ones((len(value), 1)), nu, value, nu*value]
    i, j = np.triu_indices(value.shape[1])
    quadratic = value[:, i]*value[:, j]
    if kind in {"quadratic", "quadratic_nu"}: columns.append(quadratic)
    if kind == "quadratic_nu": columns.append(nu*quadratic)
    result = np.concatenate(columns, axis=1)
    return result[0] if np.ndim(state)==1 else result


def rk4_batch(state, duration, max_step, rhs):
    count = max(1, int(math.ceil(duration/max_step))); step = duration/count
    value = state.copy()
    for _ in range(count):
        k1=rhs(value); k2=rhs(value+0.5*step*k1); k3=rhs(value+0.5*step*k2); k4=rhs(value+step*k3)
        value += step*(k1+2*k2+2*k3+k4)/6
    return value


def main():
    args=parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    canonical=json.loads(args.canonical_cases.read_text())
    with np.load(args.velocity_pod) as pod:
        train_a=np.asarray(pod["coefficients"],dtype=np.float64)[:,:11]
        tags=np.asarray(pod["snapshot_case_tags"]).astype(str)
        times=np.asarray(pod["snapshot_times"],dtype=np.float64)
        u_mean=np.asarray(pod["mean"],dtype=np.float64); u_weights=np.asarray(pod["weights"],dtype=np.float64); u_modes=np.asarray(pod["weighted_modes"],dtype=np.float64)[:11]
    with np.load(args.pressure_pod) as pod:
        p_mean=np.asarray(pod["mean"],dtype=np.float64); p_weights=np.asarray(pod["weights"],dtype=np.float64); p_modes=np.asarray(pod["weighted_modes"],dtype=np.float64)[:11]
    with np.load(args.fvm_tensors) as source: tensor={key:np.asarray(source[key]) for key in source.files}
    with np.load(args.pressure_closure) as source:
        p_kind=str(source["kind"]); p_scale=np.asarray(source["feature_scale"]); p_weights_map=np.asarray(source["weights"])
    with np.load(args.mean_calibration) as source:
        nodes=np.asarray(source["Re_nodes"]); centers=np.asarray(source["center"]); corrections=np.asarray(source["constant_correction"])

    def core_factory(re_value):
        viscosity=1/re_value; center=interpolate(nodes,centers,re_value); correction=interpolate(nodes,corrections,re_value)
        def rhs(value):
            px=pressure_features(value,viscosity,p_kind)/p_scale[None]
            pressure=px@p_weights_map
            physical=(tensor["c_conv"][None]+viscosity*tensor["c_diff"][None]+tensor["c_pressure"][None]
                +value@(tensor["A_conv"]+viscosity*tensor["A_diff"]).T
                +np.einsum("ijk,nj,nk->ni",tensor["H_conv"],value,value,optimize=True)
                +pressure@tensor["P"].T)
            return physical+correction[None]-args.core_gamma*(value-center[None])
        return rhs

    train_inputs=[]; train_nu=[]; train_targets=[]
    for tag in dict.fromkeys(tags.tolist()):
        indices=np.flatnonzero(tags==tag); state=train_a[indices]; case_times=times[indices]
        re_value=float(tag[2:].replace("p",".")); rhs=core_factory(re_value)
        physical_next=rk4_batch(state[:-1],float(case_times[1]-case_times[0]),args.max_step,rhs)
        train_inputs.append(state[:-1]); train_nu.append(np.full(len(state)-1,1/re_value)); train_targets.append(state[1:]-physical_next)
    train_inputs=np.concatenate(train_inputs); train_nu=np.concatenate(train_nu); train_targets=np.concatenate(train_targets)

    assets=[]
    for kind in ["linear","quadratic","quadratic_nu"]:
        raw=residual_features(train_inputs,train_nu,kind) if np.ndim(train_nu)==0 else None
        # residual_features accepts one viscosity, so construct variable-nu features explicitly.
        value=train_inputs; nu=train_nu[:,None]; columns=[np.ones((len(value),1)),nu,value,nu*value]
        i,j=np.triu_indices(value.shape[1]); quad=value[:,i]*value[:,j]
        if kind in {"quadratic","quadratic_nu"}: columns.append(quad)
        if kind=="quadratic_nu": columns.append(nu*quad)
        raw=np.concatenate(columns,axis=1)
        scale=np.std(raw,axis=0); scale[scale<1e-12]=1; scale[0]=1; x=raw/scale[None]
        gram=x.T@x; right=x.T@train_targets
        for ridge in [1e-10,1e-8,1e-6,1e-4,1e-2,1.0,100.0]:
            reg=np.eye(x.shape[1])*ridge; reg[0,0]=0
            weights=np.linalg.solve(gram+reg,right)
            assets.append({"kind":kind,"ridge":ridge,"scale":scale,"weights":weights,
                "train_residual_relative_l2":float(np.linalg.norm(x@weights-train_targets)/np.linalg.norm(train_targets))})

    squared_error=np.zeros(len(assets)); squared_truth=np.zeros(len(assets)); alive_global=np.ones(len(assets),dtype=bool)
    one_step_error=np.zeros(len(assets)); one_step_truth=np.zeros(len(assets)); case_reports=[]
    for target_re in VALIDATION_RE:
        entry=next(item for item in canonical if abs(float(item["Re"])-target_re)<5e-7)
        re_value,times_case,truth,_=project_case(Path(entry["path"]),u_mean,u_weights,u_modes,p_mean,p_weights,p_modes)
        viscosity=1/re_value; rhs=core_factory(re_value)
        state=np.repeat(truth[2][None],len(assets),axis=0); alive=np.ones(len(assets),dtype=bool)
        predictions=np.full((len(assets),len(times_case),11),np.nan); predictions[:,2]=state
        for index in range(2,len(times_case)-1):
            physical_next=rk4_batch(state,float(times_case[index+1]-times_case[index]),args.max_step,rhs)
            next_state=np.empty_like(state)
            for asset_index,asset in enumerate(assets):
                correction=residual_features(state[asset_index],viscosity,asset["kind"])/asset["scale"]@asset["weights"]
                next_state[asset_index]=physical_next[asset_index]+correction
            alive &= np.all(np.isfinite(next_state),axis=1)&(np.linalg.norm(next_state,axis=1)<100)
            next_state[~alive]=0; state=next_state; predictions[alive,index+1]=state[alive]
        alive_global &= alive
        # One-step validation from truth states.
        physical_truth=rk4_batch(truth[2:-1],4.0,args.max_step,rhs)
        for asset_index,asset in enumerate(assets):
            correction=residual_features(truth[2:-1],viscosity,asset["kind"])/asset["scale"][None]@asset["weights"]
            step_prediction=physical_truth+correction
            one_step_error[asset_index]+=np.sum((step_prediction-truth[3:])**2); one_step_truth[asset_index]+=np.sum(truth[3:]**2)
            if alive[asset_index]:
                squared_error[asset_index]+=np.sum((predictions[asset_index,3:]-truth[3:])**2); squared_truth[asset_index]+=np.sum(truth[3:]**2)
        case_reports.append({"Re":re_value,"full_horizon_count":int(np.sum(alive))})
    scores=np.sqrt(squared_error/np.maximum(squared_truth,1e-30)); scores[~alive_global]=np.inf
    step_scores=np.sqrt(one_step_error/np.maximum(one_step_truth,1e-30))
    selected=int(np.argmin(scores)); best=assets[selected]
    reports=[]
    for index,asset in enumerate(assets):
        reports.append({"kind":asset["kind"],"ridge":asset["ridge"],"train_residual_relative_l2":asset["train_residual_relative_l2"],
            "validation_rollout_relative_l2":None if not np.isfinite(scores[index]) else float(scores[index]),
            "validation_one_step_relative_l2":float(step_scores[index]),"full_horizon":bool(alive_global[index])})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,kind=np.asarray(best["kind"]),ridge=np.asarray(best["ridge"]),feature_scale=best["scale"],weights=best["weights"],core_gamma=np.asarray(args.core_gamma))
    payload={"schema_version":1,"status":"PASS","selected":reports[selected],"top10":sorted(reports,key=lambda x: float("inf") if x["validation_rollout_relative_l2"] is None else x["validation_rollout_relative_l2"])[:10],
        "candidate_count":len(assets),"case_reports":case_reports,"validation_loaded":True,"heldout_loaded":False}
    args.output.with_suffix(".json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
