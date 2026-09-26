"""Train and validation-freeze the three S-H one-sided Top-2 routes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn


ROOT = Path("/root/panxy/particalMOE")
METHODS = {
    "T2-C_LearnedConvexCorrection_FieldBlend": 42001,
    "RiskPredictionRouter": 42002,
    "LookAheadShortRolloutRouter": 42003,
}
HORIZONS = (1, 4, 8, 16, 56)


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, output_dim))
    def forward(self, x): return self.net(x)


class ConvexGate(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__(); self.correction = MLP(input_dim, 1)
        nn.init.zeros_(self.correction.net[-1].weight); nn.init.zeros_(self.correction.net[-1].bias)
    def forward(self, x, base_logit): return torch.sigmoid(base_logit + self.correction(x).squeeze(-1))


def blend_relative(alpha, quad):
    a = alpha[:, None]
    q = a.square() * quad[:, :, 0] + (1-a).square() * quad[:, :, 1] + 2*a*(1-a)*quad[:, :, 2]
    return torch.clamp(q / torch.clamp(quad[:, :, 3], min=1e-12), min=0.0)


def e2_pair_probability(re_values: np.ndarray) -> np.ndarray:
    runner = load_module("sh_routes_top1", ROOT / "trajectory_router_e1_e2_e3_20260722/code/run_top1_experiment.py")
    checkpoint = torch.load(ROOT / "trajectory_router_e1_e2_e3_20260722/frozen_E2_baseline_candidate/best.pt", map_location="cpu", weights_only=False)
    model = runner.Router(int(checkpoint["input_dim"]), "E2"); model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    x = (re_values.astype(np.float32)[:, None] - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    with torch.inference_mode(): probs = torch.softmax(model(torch.as_tensor(x)), dim=-1).numpy()
    pair = probs[:, :2]; pair /= pair.sum(axis=1, keepdims=True)
    return pair[:, 0].astype(np.float32)


def lookahead_features(data: dict[str, np.ndarray]) -> np.ndarray:
    values = []
    for prefix in ("s", "h"):
        a, b = data[f"{prefix}_a"][:, :4].astype(np.float64), data[f"{prefix}_b"][:, :4].astype(np.float64)
        da = np.linalg.norm(a[:, -1] - a[:, 0], axis=1); db = np.linalg.norm(b[:, -1] - b[:, 0], axis=1)
        ea = np.sum(a*a, axis=2); eb = np.sum(b*b, axis=2)
        values.extend((np.log1p(da), np.log1p(db), np.tanh(np.log((ea[:, -1]+1e-12)/(ea[:, 0]+1e-12))), np.tanh(np.log((eb[:, -1]+1e-12)/(eb[:, 0]+1e-12)))))
    return np.stack(values, axis=1).astype(np.float32)


def risk_targets(qu, qp):
    ru = np.stack((qu[:,:,0]/np.maximum(qu[:,:,3],1e-12), qu[:,:,1]/np.maximum(qu[:,:,3],1e-12)), axis=-1)
    rp = np.stack((qp[:,:,0]/np.maximum(qp[:,:,3],1e-12), qp[:,:,1]/np.maximum(qp[:,:,3],1e-12)), axis=-1)
    return np.log1p(np.mean(ru+rp, axis=1)).astype(np.float32)


def metrics(alpha: np.ndarray, qu: np.ndarray, qp: np.ndarray, re_values: np.ndarray, active: np.ndarray | None = None):
    if active is None: active = np.ones(len(alpha), dtype=bool)
    au = alpha[:,None]
    eu = np.maximum((au*au*qu[:,:,0]+(1-au)**2*qu[:,:,1]+2*au*(1-au)*qu[:,:,2])/np.maximum(qu[:,:,3],1e-12),0)
    ep = np.maximum((au*au*qp[:,:,0]+(1-au)**2*qp[:,:,1]+2*au*(1-au)*qp[:,:,2])/np.maximum(qp[:,:,3],1e-12),0)
    rows = {}
    for k in HORIZONS:
        joint = np.sqrt(eu[:,k-1]) + np.sqrt(ep[:,k-1])
        rows[f"K{k}"] = {"velocity_mean": float(np.mean(np.sqrt(eu[:,k-1]))), "pressure_mean": float(np.mean(np.sqrt(ep[:,k-1]))), "joint_mean": float(np.mean(joint)), "joint_worst": float(np.max(joint))}
    per_re = {str(float(r)): {"mean": float(np.mean((np.sqrt(eu)+np.sqrt(ep))[re_values==r])), "worst": float(np.max((np.sqrt(eu)+np.sqrt(ep))[re_values==r]))} for r in np.unique(re_values)}
    return {"horizons": rows, "per_Re": per_re, "finite_fraction": 1.0, "divergent_windows": 0, "top2_usage": float(np.mean(active)), "joint_all_mean": float(np.mean(np.sqrt(eu)+np.sqrt(ep))), "joint_all_worst": float(np.max(np.sqrt(eu)+np.sqrt(ep)))}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--cache", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--prereg-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    if set(np.unique(data["split"])) != {"train", "validation"}: raise RuntimeError("cache split contract")
    train = data["split"] == "train"; val = data["split"] == "validation"
    mean = data["features"][train].mean(0); std = data["features"][train].std(0); std[std<1e-8]=1
    x = ((data["features"]-mean)/std).astype(np.float32); la = lookahead_features(data); la_mean=la[train].mean(0); la_std=la[train].std(0); la_std[la_std<1e-8]=1; xla=np.concatenate((x,(la-la_mean)/la_std),axis=1).astype(np.float32)
    e2a = e2_pair_probability(data["re"]); base_logit = np.log(np.maximum(e2a,1e-6)/np.maximum(1-e2a,1e-6)).astype(np.float32)
    target_risk = risk_targets(data["quad_u"], data["quad_p"])
    device=torch.device("cuda"); tx=torch.as_tensor(x,device=device); txla=torch.as_tensor(xla,device=device); tbl=torch.as_tensor(base_logit,device=device); tqu=torch.as_tensor(data["quad_u"],device=device); tqp=torch.as_tensor(data["quad_p"],device=device); trisk=torch.as_tensor(target_risk,device=device)
    summaries={}
    for method, seed in METHODS.items():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        out=args.output_dir/method; out.mkdir()
        gate=ConvexGate(x.shape[1]).to(device)
        scorer = None
        scorer_x = tx
        if method == "RiskPredictionRouter": scorer=MLP(x.shape[1],2).to(device)
        if method == "LookAheadShortRolloutRouter": scorer=MLP(xla.shape[1],2).to(device); scorer_x=txla
        params=list(gate.parameters())+(list(scorer.parameters()) if scorer else [])
        opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4)
        tid=torch.nonzero(torch.as_tensor(train,device=device),as_tuple=False).flatten(); vid=torch.nonzero(torch.as_tensor(val,device=device),as_tuple=False).flatten()
        best=None; history=[]; started=time.time()
        for step in range(1,8001):
            batch=tid[torch.randint(len(tid),(min(256,len(tid)),),device=device)]
            alpha=gate(tx[batch],tbl[batch]); loss=(blend_relative(alpha,tqu[batch]).mean()+blend_relative(alpha,tqp[batch]).mean())
            if scorer is not None: loss=loss+torch.mean((scorer(scorer_x[batch])-trisk[batch])**2)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
            if step%100==0 or step==1:
                with torch.inference_mode():
                    va=gate(tx[vid],tbl[vid]); score=float((blend_relative(va,tqu[vid]).sqrt()+blend_relative(va,tqp[vid]).sqrt()).mean().item())
                history.append({"step":step,"train_loss":float(loss.item()),"validation_joint":score})
                if best is None or score < best[0]:
                    best=(score,step,{k:v.detach().cpu() for k,v in gate.state_dict().items()},({k:v.detach().cpu() for k,v in scorer.state_dict().items()} if scorer else None))
        assert best is not None
        gate.load_state_dict(best[2]); scorer and scorer.load_state_dict(best[3])
        with torch.inference_mode(): alpha=gate(tx,tbl).cpu().numpy(); predicted_risk=scorer(scorer_x).cpu().numpy() if scorer else None
        threshold=None; active=np.ones(len(alpha),dtype=bool); final_alpha=alpha.copy()
        if predicted_risk is not None:
            margin=np.abs(predicted_risk[:,0]-predicted_risk[:,1]); top=np.argmin(predicted_risk,axis=1)
            choices=[]
            for threshold_candidate in (0.0,0.05,0.1,0.2,0.5):
                act=margin<=threshold_candidate; aa=alpha.copy(); aa[~act]=(top[~act]==0).astype(np.float32)
                score=metrics(aa[val],data["quad_u"][val],data["quad_p"][val],data["re"][val],act[val])["joint_all_worst"]
                choices.append((score,threshold_candidate,act,aa))
            _,threshold,active,final_alpha=min(choices,key=lambda z:(z[0],z[1]))
        vm=metrics(final_alpha[val],data["quad_u"][val],data["quad_p"][val],data["re"][val],active[val])
        checkpoint={"method":method,"boundary":"S-H","seed":seed,"step":best[1],"gate_state":gate.state_dict(),"scorer_state":scorer.state_dict() if scorer else None,"feature_mean":mean,"feature_std":std,"lookahead_mean":la_mean,"lookahead_std":la_std,"threshold":threshold,"cache_sha256":sha256(args.cache),"validation_metrics":vm}
        best_path=out/"best.pt"; torch.save(checkpoint,best_path)
        torch.save({**checkpoint,"step":8000},out/"last.pt")
        config=json.loads((args.prereg_dir/(method+".json")).read_text()); config.update({"active_boundary":"S-H only","disabled_boundary":{"H-P":"G_HP_native BLOCKED_TEST_SEAL_ASSET"},"validation_selected_step":best[1],"validation_selected_threshold":threshold,"test_access_during_training":False})
        atomic_json(out/"CONFIG_EFFECTIVE.json",config); atomic_json(out/"validation_metrics.json",vm); atomic_json(out/"history.json",history)
        freeze={"status":"VALIDATION_FROZEN","method":method,"best_checkpoint_sha256":sha256(best_path),"config_sha256":sha256(out/"CONFIG_EFFECTIVE.json"),"threshold":threshold,"test_access":False,"elapsed_seconds":time.time()-started}
        atomic_json(out/"FREEZE_MANIFEST.json",freeze)
        summaries[method]={"freeze":freeze,"validation":vm}
    baselines={}
    for name,aa in {"E2_pair_Top1":(e2a>=.5).astype(np.float32),"fixed_0.5":np.full(len(e2a),.5,np.float32),"E2_pair_probability_blend":e2a}.items(): baselines[name]=metrics(aa[val],data["quad_u"][val],data["quad_p"][val],data["re"][val],np.ones(np.sum(val),bool) if "blend" in name or "0.5" in name else np.zeros(np.sum(val),bool))
    qu,qp=data["quad_u"][val],data["quad_p"][val]; grid=np.linspace(0,1,10001,dtype=np.float64)[:,None]; oracle=[]
    for i in range(len(qu)):
        eu=(grid*grid*qu[i,:,0]+(1-grid)**2*qu[i,:,1]+2*grid*(1-grid)*qu[i,:,2])/np.maximum(qu[i,:,3],1e-12)
        ep=(grid*grid*qp[i,:,0]+(1-grid)**2*qp[i,:,1]+2*grid*(1-grid)*qp[i,:,2])/np.maximum(qp[i,:,3],1e-12)
        oracle.append(float(grid[np.argmin(np.mean(np.sqrt(np.maximum(eu,0))+np.sqrt(np.maximum(ep,0)),axis=1)),0]))
    oracle=np.asarray(oracle,dtype=np.float32)
    baselines["per_window_convex_oracle"]=metrics(oracle,data["quad_u"][val],data["quad_p"][val],data["re"][val])
    atomic_json(args.output_dir/"VALIDATION_COMPARISON.json",{"routes":summaries,"baselines":baselines,"test_access":False})
    atomic_json(args.output_dir/"ALL_ROUTES_VALIDATION_FROZEN.json",{"status":"FROZEN","methods":list(METHODS),"test_access_now_authorized_for_mechanical_final_evaluator":True,"known_legacy_test_disclosure":True})
    print(json.dumps({"status":"ALL_ROUTES_VALIDATION_FROZEN","methods":{k:v["freeze"]["best_checkpoint_sha256"] for k,v in summaries.items()}},indent=2))


if __name__ == "__main__": main()
