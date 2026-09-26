#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, os, random, socket, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

EPS = 1e-8
EXPECTED_SPLITS = {"steady": (14, 2, 4)}


def emit(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def atomic_json(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)); tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def load_vendor(path):
    spec = importlib.util.spec_from_file_location("frozen_v17_vendor_v2", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def fit(values):
    mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(values, axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def scaled_mse(pred, target, scale):
    return torch.mean(((pred.float() - target.float()) / scale.float()) ** 2)


class Sampler:
    def __init__(self, regime, candidates, a, phase, seed):
        self.regime, self.rng = regime, np.random.default_rng(seed)
        self.labels = np.asarray(sorted(candidates), dtype=np.int64); self.groups = {}
        for label, ids in candidates.items():
            ids = np.asarray(ids, dtype=np.int64)
            if regime == "hopf":
                radius = np.sqrt(np.sum(a[ids, :2] ** 2, axis=1) + EPS)
                edges = np.unique(np.quantile(radius, np.linspace(0, 1, 7)))
                code = np.digitize(radius, edges[1:-1]) if len(edges) > 2 else np.zeros(len(ids), int)
            elif regime == "periodic":
                code = np.clip(np.floor((phase[ids] % 1) * 8).astype(int), 0, 7)
            else:
                code = np.zeros(len(ids), int)
            self.groups[int(label)] = [ids[code == group] for group in sorted(set(code.tolist())) if np.any(code == group)]

    def sample(self, count):
        labels = self.rng.choice(self.labels, count, replace=True); result = np.empty(count, np.int64)
        for i, label in enumerate(labels.tolist()):
            groups = self.groups[label]; group = groups[int(self.rng.integers(len(groups)))]
            result[i] = group[int(self.rng.integers(len(group)))]
        return result


class Experiment:
    def __init__(self, cli, cfg):
        self.cli, self.cfg = cli, cfg; self.e = cfg["training"]
        self.experiment = "S2-B"; self.regime = "steady"; self.mechanism = "pressure_fixed_point_anchor"
        self.ru, self.rp = cfg["r_u"], cfg["r_p"]
        self.run = Path(cli.run_dir); self.run.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(cfg["checkpoint_root"]); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0")
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_float32_matmul_precision("high")
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
        seed = cfg["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self.vendor = load_vendor(Path(cfg["vendor_trainer"]))
        rd = Path(cfg["artifact_dir"]); compat = rd / "Global_POD_AreaWeighted_L2"
        self.paths = {
            "velocity_pod": rd / f"velocity_pod_{self.regime}.npz", "pressure_pod": rd / f"pressure_pod_{self.regime}.npz",
            "normalization": rd / f"normalization_{self.regime}.npz", "velocity_rom": rd / f"velocity_rom_{self.regime}.npz",
            "pressure_rom": rd / f"pressure_poisson_surrogate_{self.regime}.npz",
        }
        for path in [*self.paths.values(), compat / "pod_snapshot_index.csv", Path(cfg["vendor_trainer"])]:
            if not path.is_file(): raise FileNotFoundError(path)
        self._audit_artifacts()
        args = SimpleNamespace(data_root=compat, tensor_path=self.paths["velocity_rom"], pressure_surrogate_path=self.paths["pressure_rom"],
            r_u=self.ru, r_p=self.rp, history_len=cfg["model"]["history_len"], phase_harmonics=cfg["model"]["phase_harmonics"], recon_dim=0, seed=seed)
        self.a, self.meta = self.vendor.build_arrays(args)
        with np.load(self.paths["velocity_pod"]) as pod:
            self.split = pod["snapshot_splits"].astype(str); labels = pod["snapshot_Re_labels"].astype(str)
        if not np.array_equal(labels, self.a["labels"][self.a["label_id"]]): raise AssertionError("snapshot/POD label order mismatch")
        self.windows = self.build_windows(16)
        counts = tuple(len(self.windows[name]) for name in ("train", "validation", "heldout"))
        if counts != EXPECTED_SPLITS[self.regime]: raise AssertionError(f"split Re count {counts}")
        sets = [set(self.windows[name]) for name in ("train", "validation", "heldout")]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]: raise AssertionError("Re leakage")
        train_ids = np.flatnonzero(self.split == "train")
        self.train_ids = train_ids
        self.train_rollout_start_ids = np.unique(np.concatenate(list(self.windows["train"].values())))
        if len(train_ids) != 894: raise AssertionError(f"expected 894 Steady train snapshots, got {len(train_ids)}")
        fit_ids = np.unique(np.concatenate(list(self.build_windows(1)["train"].values())))
        self.scaler_fit_ids = fit_ids
        self.xm, self.xs = fit(self.a["x"][fit_ids]); self.rm, self.rs = fit(self.a["residual"][fit_ids]); self.pm, self.ps = fit(self.a["pressure_residual"][fit_ids])
        with np.load(self.paths["normalization"]) as norm:
            if str(norm["fit_split"].item()) != "train": raise AssertionError("non-train normalization")
            self.av = np.maximum(norm["velocity_coeff_std"][:self.ru].astype(np.float32), 1e-7)
            self.bv = np.maximum(norm["pressure_coeff_std"][:self.rp].astype(np.float32), 1e-7)
        for name in ("xm", "xs", "rm", "rs", "pm", "ps", "av", "bv"):
            setattr(self, name + "t", torch.tensor(getattr(self, name), device=self.device))
        velocity_rom, pressure_rom = np.load(self.paths["velocity_rom"]), np.load(self.paths["pressure_rom"])
        self.gal = self.vendor.build_galerkin_torch(velocity_rom, self.a["labels"], self.ru, self.rp, self.device)
        self.sur = self.vendor.build_pressure_surrogate_torch(pressure_rom, self.a["labels"], self.ru, self.rp, self.device)
        self.cache = self._cache_arrays()
        self.validation_windows = {k:{label:torch.as_tensor(ids,dtype=torch.long,device=self.device) for label,ids in self.build_windows(k)["validation"].items()} for k in (1,4,8,16)}
        self.model = self._build_model().to(self.device); self._safe_zero_initial_output(); self.initial_model_sha256 = model_digest(self.model)
        self.bound_scale = self._build_pressure_bound(fit_ids) if self.mechanism == "bounded_pressure_residual" else torch.ones(self.rp,device=self.device)
        self.trainable_audit = self._configure_trainable_parameters()
        self.opt = self._build_optimizer(); self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.e["max_steps"], eta_min=cfg["training"]["learning_rate"] * .05)
        self.cpu_sampler = Sampler(self.regime, self.windows["train"], self.a["a"], self.a["phase"], seed + 91)
        self.sample_schedule = None; self.micro_batch = None; self.grad_accum = None
        self.calibration = None; self.swan = None; self.early_state = {"stale": 0, "best_pressure": float("inf")}

    def _cache_arrays(self):
        cache = {}
        for key in ("a", "b", "rhs_g", "x", "residual", "pressure_residual", "re", "phase", "dt_next", "label_id", "next_idx", "hist_idx"):
            value = np.asarray(self.a[key])
            dtype = torch.long if np.issubdtype(value.dtype, np.integer) else torch.float32
            cache[key] = torch.as_tensor(value, dtype=dtype, device=self.device)
        with np.load(self.paths["velocity_pod"]) as velocity:
            cache["velocity_pod_basis"] = torch.as_tensor(velocity["phi_uv"][:self.ru], dtype=torch.float32, device=self.device)
            cache["velocity_mean"] = torch.as_tensor(velocity["mean_uv_regime"], dtype=torch.float32, device=self.device)
        with np.load(self.paths["pressure_pod"]) as pressure:
            cache["pressure_pod_basis"] = torch.as_tensor(pressure["phi_p"][:self.rp], dtype=torch.float32, device=self.device)
            cache["pressure_mean"] = torch.as_tensor(pressure["mean_p_regime"], dtype=torch.float32, device=self.device)
        return cache

    @staticmethod
    def tensor_bytes(value):
        if torch.is_tensor(value): return value.numel()*value.element_size()
        if isinstance(value,dict): return sum(Experiment.tensor_bytes(v) for v in value.values())
        if isinstance(value,(list,tuple)): return sum(Experiment.tensor_bytes(v) for v in value)
        return 0

    def prepare_schedule(self, micro_batch, grad_accum):
        effective = micro_batch * grad_accum
        if effective != 64: raise AssertionError(f"effective batch must be 64, got {effective}")
        sampler = Sampler(self.regime, self.windows["train"], self.a["a"], self.a["phase"], self.cfg["seed"] + 91)
        schedule = np.stack([sampler.sample(effective) for _ in range(self.e["max_steps"] + 1)])
        self.sample_schedule = torch.as_tensor(schedule, dtype=torch.long, device=self.device)
        self.micro_batch, self.grad_accum = micro_batch, grad_accum

    def _audit_artifacts(self):
        with np.load(self.paths["velocity_pod"]) as velocity, np.load(self.paths["pressure_pod"]) as pressure, np.load(self.paths["velocity_rom"]) as rom:
            if velocity["phi_uv"].shape[0] < self.ru or pressure["phi_p"].shape[0] < self.rp: raise AssertionError("POD rank too small")
            rom_rank=(int(rom["r_u"]),int(rom["r_p"]))
            if rom_rank[0]<self.ru or rom_rank[1]<self.rp: raise AssertionError("ROM rank too small")
            if (self.ru, self.rp) != (32, 32): raise AssertionError("S2-B requires ru=32/rp=32")
            if str(velocity["fit_split"].item()) != "train" or str(pressure["fit_split"].item()) != "train": raise AssertionError("POD not train-only")
            if str(pressure["pressure_gauge"].item()) != "subtract_area_mean_per_snapshot": raise AssertionError("pressure gauge mismatch")

    def _build_model(self):
        m = self.cfg["model"]
        return self.vendor.OperatorSpaceMoEROM(in_dim=self.a["x"].shape[1], out_dim=self.ru, pressure_dim=self.rp,
            hidden_dim=m["hidden_dim"], expert_hidden=m["expert_hidden"], num_blocks=m["num_blocks"], num_experts=m["experts_per_group"],
            num_operator_spaces=m["shared_experts_per_group"], num_regime_groups=m["num_regime_groups"], experts_per_group=m["experts_per_group"],
            top_k=m["top_k"], group_top_k=m["group_top_k"], dropout=m["dropout"], temperature=m["temperature"], gate_floor=m["gate_floor"],
            group_temperature=m["group_temperature"], group_gate_floor=m["group_gate_floor"], shared_scale=m["shared_scale"], routed_scale=m["routed_scale"],
            expert_blocks=m["expert_blocks"], quadratic_rank=m["quadratic_rank"], quadratic_scale=m["quadratic_scale"], phase_harmonics=m["phase_harmonics"],
            closure_mode="baseline", pressure_base_mode="static", film_base_hidden=64, film_base_scale=.2, attractor_conditioned=False)

    def _safe_zero_initial_output(self):
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, self.vendor.PhysicsAwareExpert):
                    module.linear.weight.zero_(); module.mlp_head[-1].weight.zero_(); module.mlp_head[-1].bias.zero_()
                    if module.quad_left is not None: module.quad_left.zero_(); module.quad_right.zero_()

    def _build_pressure_bound(self, train_ids):
        residual = self.a["pressure_residual"][train_ids, :self.rp].astype(np.float64)
        q = np.quantile(np.abs(residual), self.cfg["training"]["bound_quantile"], axis=0)
        std = np.std(residual, axis=0); global_floor = self.cfg["training"]["bound_floor_global_fraction"] * max(float(np.median(q[q > 0])) if np.any(q > 0) else 0., 1e-7)
        floor = np.maximum(self.cfg["training"]["bound_floor_std_fraction"] * std, global_floor)
        factor=self.cfg["training"]["bound_train_only_recalibration_factor"][self.regime]
        scale = (factor*np.maximum(self.cfg["training"]["bound_multiplier"] * q, floor)).astype(np.float32)
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0): raise AssertionError("invalid train-only pressure bound")
        return torch.tensor(scale, device=self.device)

    def _configure_trainable_parameters(self):
        groups = {"frozen": [], "base_lr": [], "low_lr": []}
        for name, param in self.model.named_parameters():
            if self.regime == "steady" and (name.startswith("velocity_expert_groups") or name.startswith("velocity_shared_experts")):
                param.requires_grad_(False); groups["frozen"].append(name)
            elif self.regime == "steady" and (name.startswith("encoder") or name.startswith("refine_blocks") or "router" in name):
                groups["low_lr"].append(name)
            elif name.startswith("closure_confidence_head"):
                param.requires_grad_(False); groups["frozen"].append(name)
            else: groups["base_lr"].append(name)
        return {key: {"names": names, "parameters": sum(self.model.get_parameter(name).numel() for name in names)} for key, names in groups.items()}

    def _build_optimizer(self):
        base_lr = self.cfg["training"]["learning_rate"]; low_names = set(self.trainable_audit["low_lr"]["names"])
        base, low = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad: (low if name in low_names else base).append(param)
        groups = [{"params": base, "lr": base_lr, "initial_lr": base_lr}]
        if low: groups.append({"params": low, "lr": base_lr * self.cfg["training"]["steady_shared_lr_factor"], "initial_lr": base_lr * self.cfg["training"]["steady_shared_lr_factor"]})
        return torch.optim.AdamW(groups, weight_decay=self.cfg["training"]["weight_decay"])

    def build_windows(self, k):
        grouped = {name: {} for name in ("train", "validation", "heldout")}; nxt = self.a["next_idx"]
        for sid in sorted(self.a["sample_ids"].tolist()):
            split = str(self.split[sid]); cur = sid
            if split not in grouped: continue
            for _ in range(k):
                cur = int(nxt[cur])
                if cur < 0 or self.a["label_id"][cur] != self.a["label_id"][sid] or str(self.split[cur]) != split: break
            else: grouped[split].setdefault(int(self.a["label_id"][sid]), []).append(sid)
        return {split: {label: np.asarray(ids, np.int64) for label, ids in labels.items()} for split, labels in grouped.items()}

    def tensor(self, key, ids, dtype=None):
        value = self.cache[key][ids]
        return value if dtype is None or value.dtype == dtype else value.to(dtype=dtype)

    def indices(self, starts, k):
        parts = [starts]
        for _ in range(k): parts.append(self.cache["next_idx"][parts[-1]])
        return torch.stack(parts, dim=1)

    def _model_call(self, x, k):
        def call(value):
            velocity, pressure, _, _ = self.model(value, return_expert_stack=False); return velocity, pressure
        return call(x)

    def rollout(self, starts, k, apply_bound=None, initial_a=None, initial_b=None):
        ix = self.indices(starts, k); cur = ix[:, 0]
        a = self.tensor("a", cur) if initial_a is None else initial_a; b = self.tensor("b", cur) if initial_b is None else initial_b
        a0, b0 = a, b; label = self.tensor("label_id", cur, torch.long); hist = self.tensor("hist_idx", cur, torch.long)
        ah, bh, rh = self.tensor("a", hist), self.tensor("b", hist), self.tensor("rhs_g", hist)
        pred_a=[]; pred_b=[]; true_a=[]; true_b=[]; raw_deltas=[]; bounded_deltas=[]
        bounded = self.mechanism == "bounded_pressure_residual" if apply_bound is None else apply_bound
        for _ in range(k):
            base_rhs = self.vendor.galerkin_rhs_torch(a, b, label, self.gal)
            base_x = self.vendor.make_features_torch(a, b, base_rhs, self.tensor("re", cur), self.tensor("phase", cur), self.cfg["model"]["phase_harmonics"])
            x = self.vendor.make_history_features_from_states_torch(base_x, a, b, base_rhs, ah, bh, rh)
            raw_u, raw_p = self._model_call((x - self.xmt) / self.xst, k)
            rhs = base_rhs + raw_u.float() * self.rst + self.rmt; an = a + self.tensor("dt_next", cur).unsqueeze(1) * rhs
            raw_delta = raw_p.float() * self.pst + self.pmt
            bounded_delta = self.bound_scale * torch.tanh(raw_delta / self.bound_scale) if bounded else raw_delta
            bn = self.vendor.pressure_surrogate_torch(an, label, self.sur) + bounded_delta
            target = ix[:, len(pred_a) + 1]; at, bt = self.tensor("a", target), self.tensor("b", target)
            pred_a.append(an); pred_b.append(bn); true_a.append(at); true_b.append(bt); raw_deltas.append(raw_delta); bounded_deltas.append(bounded_delta)
            if ah.shape[1] > 1:
                ah=torch.cat([an[:,None],a[:,None],ah[:,1:-1]],1); bh=torch.cat([bn[:,None],b[:,None],bh[:,1:-1]],1); rh=torch.cat([rhs[:,None],base_rhs[:,None],rh[:,1:-1]],1)
            a,b,cur=an,bn,target
        pa,pb,ta,tb,raw_delta,bound_delta = map(torch.stack, (pred_a,pred_b,true_a,true_b,raw_deltas,bounded_deltas))
        one_u=scaled_mse(pa[0],ta[0],self.avt); one_p=scaled_mse(pb[0],tb[0],self.bvt)
        roll_u=scaled_mse(pa,ta,self.avt); roll_p=scaled_mse(pb,tb,self.bvt); rollout=roll_u+self.cfg["training"]["rollout_pressure_weight"]*roll_p
        anchor=torch.mean(torch.sum((pb-b0.unsqueeze(0))**2,dim=-1)/(torch.sum(b0**2,dim=-1).unsqueeze(0)+EPS))
        saturation=torch.zeros((),device=self.device)
        return {"one":one_u+one_p,"one_u":one_u,"one_p":one_p,"rollout":rollout,"roll_u":roll_u,"roll_p":roll_p,"anchor":anchor,
            "pa":pa,"pb":pb,"ta":ta,"tb":tb,"raw_delta":raw_delta,"bound_delta":bound_delta,"saturation":saturation,"a0":a0,"b0":b0}

    def stage(self, step):
        for stage in self.e["curriculum"]:
            if stage["start"] <= step <= stage["end"]: return stage
        raise AssertionError(f"no curriculum stage for {step}")

    def ramp(self, step, stage): return min(1., max(0., (step-stage["start"]+1)/self.cfg["training"]["ramp_steps"]))

    def _grad_norm(self, loss, retain=True):
        params=[p for p in self.model.parameters() if p.requires_grad]
        grads=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
        return float(torch.sqrt(sum((g.float()**2).sum() for g in grads if g is not None)+EPS).cpu())

    def calibrate_pair(self):
        starts=self.sample_schedule[0, :self.micro_batch]; out=self.rollout(starts,4,apply_bound=False)
        one_norm=self._grad_norm(out["one"]); rollout_norm=self._grad_norm(out["rollout"])
        lambda_roll=float(np.clip(self.cfg["training"]["rollout_gradient_ratio"]*one_norm/max(rollout_norm,EPS),1e-12,1e6))
        report={"schema_version":1,"regime":self.regime,"source_experiment":self.experiment,"fixed_batch":starts.tolist(),"initial_model_sha256":self.initial_model_sha256,
            "gradient_norms":{"one":one_norm,"rollout":rollout_norm},"lambda_rollout":lambda_roll,"lambda_anchor":0.0}
        if self.regime=="steady":
            base=out["one"]+lambda_roll*out["rollout"]; base_norm=self._grad_norm(base); anchor_norm=self._grad_norm(out["anchor"],retain=False)
            report["gradient_norms"].update({"base":base_norm,"anchor":anchor_norm}); report["lambda_anchor"]=float(np.clip(self.cfg["training"]["anchor_gradient_ratio"]*base_norm/max(anchor_norm,EPS),1e-12,1e6))
        self.opt.zero_grad(set_to_none=True); return report

    def _validation_starts(self, k):
        return self.validation_windows[k]

    def controlled_contraction(self):
        ratios = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.amp_dtype):
            for label, ids in self._validation_starts(16).items():
                start=ids[:1]; a0=self.tensor("a",start); b0=self.tensor("b",start)
                signs=torch.where(torch.arange(self.ru,device=self.device)%2==0,1.,-1.)
                da=.01*self.avt*signs; db=-.01*self.bvt*signs
                out=self.rollout(start,16,initial_a=a0+da,initial_b=b0+db)
                num=torch.sqrt(torch.mean(((out["pa"][-1]-a0)/self.avt)**2)+torch.mean(((out["pb"][-1]-b0)/self.bvt)**2))
                den=torch.sqrt(torch.mean((da/self.avt)**2)+torch.mean((db/self.bvt)**2))
                ratios[str(self.a["labels"][label])]=float((num/den).float().cpu())
        return ratios

    def validate_horizon(self, step, k):
        self.model.eval(); by={}; total=finite_count=divergent=0; worst_drift=worst_fixed=0.
        bs=self.e["validation_batch"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.amp_dtype):
            for label, ids in self._validation_starts(k).items():
                nu=du=np_=dp=0.
                for offset in range(0, len(ids), bs):
                    out=self.rollout(ids[offset:offset+bs], k); pa,pb,ta,tb=[out[x].float() for x in ("pa","pb","ta","tb")]
                    batch=pa.shape[1]; total+=batch
                    finite=(torch.isfinite(pa).all((0,2)) & torch.isfinite(pb).all((0,2))); finite_count+=int(finite.sum().cpu())
                    ra=torch.linalg.vector_norm(pa,dim=2)/(torch.linalg.vector_norm(ta,dim=2)+1e-6); rb=torch.linalg.vector_norm(pb,dim=2)/(torch.linalg.vector_norm(tb,dim=2)+1e-6)
                    divergent+=int((((ra>20)|(rb>20)|(~torch.isfinite(ra))|(~torch.isfinite(rb))).any(0)).sum().cpu())
                    nu+=float(((pa-ta)**2).sum().cpu()); du+=float((ta**2).sum().cpu()); np_+=float(((pb-tb)**2).sum().cpu()); dp+=float((tb**2).sum().cpu())
                    drift=torch.linalg.vector_norm(pb[-1]-out["b0"],dim=1)/(torch.linalg.vector_norm(out["b0"],dim=1)+EPS)
                    fixed=torch.sqrt(torch.sum((pb-out["b0"].unsqueeze(0))**2,dim=-1)/(torch.sum(out["b0"]**2,dim=-1).unsqueeze(0)+EPS))
                    worst_drift=max(worst_drift,float(drift.max().cpu())); worst_fixed=max(worst_fixed,float(fixed.max().cpu()))
                by[str(self.a["labels"][label])]={"velocity_rel_l2":math.sqrt(nu/max(du,EPS)),"pressure_rel_l2":math.sqrt(np_/max(dp,EPS))}
        contraction=self.controlled_contraction() if k==16 else None
        report={"step":step,"k":k,"worst_velocity":max(v["velocity_rel_l2"] for v in by.values()),"worst_pressure":max(v["pressure_rel_l2"] for v in by.values()),
            "finite_fraction":finite_count/max(total,1),"divergent_windows":divergent,"pressure_drift":worst_drift,"fixed_point_residual":worst_fixed,
            "contraction_ratio":max(contraction.values()) if contraction else None,"contraction_by_re":contraction,"by_re":by}
        self.model.train(); return report

    def validate_all(self, step):
        reports={f"k{k}":self.validate_horizon(step,k) for k in (1,4,8,16)}
        with (self.run/"validation_history.jsonl").open("a") as stream: stream.write(json.dumps(reports)+"\n")
        emit("validation_complete", experiment=self.experiment, step=step, reports=reports)
        return reports

    def checkpoint_payload(self, step, validation):
        return {"schema_version":3,"experiment":self.experiment,"step":step,"current_horizon":self.stage(max(step,1))["k"],"validation":validation,
            "model":self.model.state_dict(),"optimizer":self.opt.state_dict(),"scheduler":self.sched.state_dict(),"amp_scaler":self.grad_scaler.state_dict(),
            "rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()},
            "early_stop":self.early_state,"micro_batch":self.micro_batch,"grad_accum":self.grad_accum,"calibration":self.calibration,
            "initial_model_sha256":self.initial_model_sha256,"asset_sha256":{k:digest(v) for k,v in self.paths.items()},"config":self.cfg}

    def save_checkpoint(self, step, validation, name):
        target=self.ckpt_dir/name; tmp=target.with_suffix(target.suffix+".tmp")
        torch.save(self.checkpoint_payload(step,validation),tmp); tmp.replace(target)

    def load_checkpoint(self, path):
        ckpt=torch.load(path,map_location="cpu",weights_only=False)
        self.model.load_state_dict(ckpt["model"]); self.opt.load_state_dict(ckpt["optimizer"]); self.sched.load_state_dict(ckpt["scheduler"]); self.grad_scaler.load_state_dict(ckpt["amp_scaler"])
        for state in self.opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value): state[key]=value.to(self.device)
        random.setstate(ckpt["rng"]["python"]); np.random.set_state(ckpt["rng"]["numpy"]); torch.set_rng_state(ckpt["rng"]["torch"]); torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        self.early_state=ckpt["early_stop"]; self.calibration=ckpt["calibration"]; self.prepare_schedule(ckpt["micro_batch"],ckpt["grad_accum"])
        return int(ckpt["step"])

    def train_step(self, step, k, ramp):
        self.model.train(); self.opt.zero_grad(set_to_none=True); totals={x:torch.zeros((),device=self.device) for x in ("loss","one","rollout","anchor")}
        starts=self.sample_schedule[step]
        for part in range(self.grad_accum):
            batch=starts[part*self.micro_batch:(part+1)*self.micro_batch]
            with torch.autocast("cuda",dtype=self.amp_dtype):
                out=self.rollout(batch,k); total=out["one"]
                if k>1: total=total+ramp*self.calibration["lambda_rollout"]*out["rollout"]+ramp*self.calibration["lambda_anchor"]*out["anchor"]
                loss=total/self.grad_accum
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}")
            self.grad_scaler.scale(loss).backward()
            totals["loss"]+=total.detach()/self.grad_accum; totals["one"]+=out["one"].detach()/self.grad_accum
            totals["rollout"]+=out["rollout"].detach()/self.grad_accum; totals["anchor"]+=out["anchor"].detach()/self.grad_accum
        self.grad_scaler.unscale_(self.opt); grad=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],self.e["grad_clip"])
        if not torch.isfinite(grad): raise FloatingPointError(f"non-finite grad at step {step}")
        self.grad_scaler.step(self.opt); self.grad_scaler.update(); self.sched.step()
        return {k:float(v.cpu()) for k,v in totals.items()}|{"grad_norm":float(grad.cpu())}

    def smoke(self):
        reports=[]
        for k in (1,4,8,16):
            self.model.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda",dtype=self.amp_dtype):
                out=self.rollout(self.sample_schedule[0,:self.micro_batch],k); loss=out["one"]+(self.calibration["lambda_rollout"]*out["rollout"]+self.calibration["lambda_anchor"]*out["anchor"] if k>1 else 0.)
            self.grad_scaler.scale(loss).backward(); self.grad_scaler.unscale_(self.opt)
            grad=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],self.e["grad_clip"])
            if not torch.isfinite(loss) or not torch.isfinite(grad): raise FloatingPointError(f"K{k} smoke failed")
            reports.append({"k":k,"loss":float(loss.detach().cpu()),"grad_norm":float(grad.cpu()),"peak_gpu_gib":torch.cuda.max_memory_allocated()/1024**3})
        self.opt.zero_grad(set_to_none=True)
        self.save_checkpoint(0,{},"smoke.pt"); digest_before=model_digest(self.model); self.load_checkpoint(self.ckpt_dir/"smoke.pt")
        if model_digest(self.model)!=digest_before: raise AssertionError("checkpoint save-load mismatch")
        return reports

    def benchmark(self):
        candidates=[]; initial={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        for c in self.e["benchmark_candidates"]:
            try:
                self.model.load_state_dict(initial); self.opt=self._build_optimizer(); self.sched=torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,self.e["max_steps"]); self.grad_scaler=torch.amp.GradScaler("cuda",enabled=self.amp_dtype==torch.float16)
                self.prepare_schedule(c["micro_batch"],c["grad_accum"]); torch.cuda.empty_cache()
                for s in range(1,self.e["benchmark_warmup_steps"]+1): self.train_step(s,16,1.)
                torch.cuda.synchronize(); started=time.perf_counter()
                for s in range(1,self.e["benchmark_timed_steps"]+1): self.train_step(s,16,1.)
                torch.cuda.synchronize(); elapsed=time.perf_counter()-started
                candidates.append({**c,"status":"PASS","seconds_per_optimizer_step":elapsed/self.e["benchmark_timed_steps"],"peak_gpu_gib":torch.cuda.max_memory_allocated()/1024**3})
            except (torch.OutOfMemoryError,FloatingPointError) as exc:
                candidates.append({**c,"status":"FAIL","error":str(exc)}); torch.cuda.empty_cache()
        passed=[x for x in candidates if x["status"]=="PASS"]
        if not passed: raise RuntimeError("all benchmark candidates failed")
        best=min(passed,key=lambda x:x["seconds_per_optimizer_step"])
        self.model.load_state_dict(initial); self.opt=self._build_optimizer(); self.sched=torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,self.e["max_steps"],eta_min=self.e["learning_rate"]*.05); self.grad_scaler=torch.amp.GradScaler("cuda",enabled=self.amp_dtype==torch.float16)
        self.prepare_schedule(best["micro_batch"],best["grad_accum"])
        atomic_json(self.run/"benchmark.json",{"candidates":candidates,"selected":best}); return best

    def gpu_test(self):
        started=time.perf_counter(); steps=0
        while time.perf_counter()-started<self.e["gpu_test_seconds"]:
            self.model.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=self.amp_dtype): out=self.rollout(self.sample_schedule[0,:self.micro_batch],16); loss=out["one"]+self.calibration["lambda_rollout"]*out["rollout"]+self.calibration["lambda_anchor"]*out["anchor"]
            self.grad_scaler.scale(loss).backward(); steps+=1
        self.model.zero_grad(set_to_none=True); torch.cuda.synchronize()
        return {"seconds":time.perf_counter()-started,"iterations":steps,"peak_gpu_gib":torch.cuda.max_memory_allocated()/1024**3}

    def preflight(self):
        self.prepare_schedule(16,4); self.calibration=self.calibrate_pair(); atomic_json(self.run/"calibration.json",self.calibration)
        smoke=self.smoke(); validation=self.validate_all(0); benchmark=self.benchmark(); gpu=self.gpu_test()
        report={"status":"PASS","amp_dtype":str(self.amp_dtype),"activation_checkpointing":False,"train_snapshots":len(self.train_ids),
            "cache_audit":{"mode":"all_gpu","arrays_gib":self.tensor_bytes(self.cache)/1024**3,"galerkin_gib":self.tensor_bytes(self.gal)/1024**3,"pressure_operator_gib":self.tensor_bytes(self.sur)/1024**3,"disk_reads_inside_training_loop":False,"dataloader_workers":0},
            "smoke":smoke,"validation":validation,"benchmark":benchmark,"gpu_test":gpu,"initial_model_sha256":self.initial_model_sha256,"trainable_parameter_audit":self.trainable_audit}
        atomic_json(self.run/"preflight_report.json",report); emit("preflight_pass",**report)

    def init_swan(self):
        import swanlab
        if not swanlab.login(api_key=os.environ.get("SWANLAB_API_KEY") or None,relogin=False,save=False,timeout=20): raise RuntimeError("SwanLab login")
        log_dir=self.run/"swanlog"; log_dir.mkdir(exist_ok=True)
        self.swan=swanlab.init(project=self.cfg["swanlab"]["project"],name=self.cfg["swanlab"]["experiment"],group=self.cfg["swanlab"]["group"],mode="online",
            config={"experiment":self.experiment,"training":self.e,"model":self.cfg["model"],"calibration":self.calibration,"initial_model_sha256":self.initial_model_sha256,"micro_batch":self.micro_batch,"grad_accum":self.grad_accum},
            tags=["V17","S2-B","RTX3090","from-scratch","K1-K4-K8-K16",str(self.amp_dtype)],log_dir=str(log_dir),reinit=True,parallel="shared")
        swanlab.log({"lifecycle/swanlab_initialized":1},step=0); emit("swanlab_ready",experiment=self.experiment)

    def train(self):
        preflight=json.loads((self.run/"preflight_report.json").read_text()); selected=preflight["benchmark"]
        self.prepare_schedule(selected["micro_batch"],selected["grad_accum"]); self.calibration=json.loads((self.run/"calibration.json").read_text())
        start_step=0
        if self.cli.resume is not None: start_step=self.load_checkpoint(self.cli.resume)
        self.init_swan(); import swanlab
        best_key=None; last_reports={}; started=time.time()
        atomic_json(self.run/"RUNNING.json",{"status":"RUNNING","experiment":self.experiment,"pid":os.getpid(),"host":socket.gethostname(),"started_unix":started,"initial_model_sha256":self.initial_model_sha256})
        emit("training_started",experiment=self.experiment,pid=os.getpid(),max_steps=self.e["max_steps"]); swanlab.log({"lifecycle/training_started":1},step=0)
        boundary_steps={stage["end"] for stage in self.e["curriculum"][:-1]}
        for step in range(start_step+1,self.e["max_steps"]+1):
            stage=self.stage(step); ramp=self.ramp(step,stage) if stage["use_rollout"] else 0.; metrics=self.train_step(step,stage["k"],ramp)
            if step==1 or step%self.e["log_every"]==0:
                logged={"train/loss":metrics["loss"],"train/one_step":metrics["one"],f"train/rollout_k{stage['k']}":metrics["rollout"],"train/fixed_point_anchor":metrics["anchor"],
                    "train/grad_norm":metrics["grad_norm"],"train/lr":self.opt.param_groups[0]["lr"],"train/ramp":ramp,"train/horizon":stage["k"],"runtime/peak_gpu_gib":torch.cuda.max_memory_allocated()/1024**3}
                swanlab.log(logged,step=step); emit("optimizer_step",experiment=self.experiment,step=step,**logged)
            validate_now=step%self.e["validation_every"]==0 or step in boundary_steps
            if validate_now:
                last_reports=self.validate_all(step); flat={}
                for name,report in last_reports.items():
                    prefix=f"validation/{name}_"; flat.update({prefix+key:report[key] for key in ("worst_velocity","worst_pressure","finite_fraction","divergent_windows","pressure_drift","fixed_point_residual")}); flat[prefix+"contraction_ratio"]=report["contraction_ratio"] or 0.
                swanlab.log(flat,step=step); self.save_checkpoint(step,last_reports,"latest.pt")
                k16=last_reports["k16"]; qualified=k16["finite_fraction"]==1. and k16["divergent_windows"]==0 and k16["contraction_ratio"]<1.
                key=(k16["worst_pressure"],k16["worst_velocity"],k16["fixed_point_residual"])
                if qualified and (best_key is None or key<best_key): best_key=key; self.save_checkpoint(step,last_reports,"best-qualified.pt")
                previous=self.early_state["best_pressure"]; improved=k16["worst_pressure"]<previous*(1-self.e["relative_improvement_threshold"]) if math.isfinite(previous) else True
                self.early_state["best_pressure"]=min(previous,k16["worst_pressure"]); self.early_state["stale"]=0 if improved else self.early_state["stale"]+1
                if step>=self.e["early_stop_start"] and self.early_state["stale"]>=self.e["patience_validations"]: emit("early_stop",experiment=self.experiment,step=step); break
            elif step%self.e["latest_every"]==0: self.save_checkpoint(step,last_reports,"latest.pt")
            if step==self.e["base_budget_end"]: emit("base_budget_complete",experiment=self.experiment,step=step)
        status="QUALIFIED_CHECKPOINT" if best_key is not None else "NO_QUALIFIED_CHECKPOINT"; final={"status":status,"experiment":self.experiment,"step":step,"best_qualified_selector":best_key}
        self.save_checkpoint(step,last_reports,"final.pt"); atomic_json(self.run/"final_checkpoint.json",final); atomic_json(self.run/"DONE.json",{"status":"DONE","experiment":self.experiment,"step":step,"checkpoint_status":status,"finished_unix":time.time()})
        swanlab.log({"lifecycle/training_complete":1,"runtime/seconds":time.time()-started},step=step); swanlab.finish(state="success"); emit("training_complete",**final)


def startup_manifest(exp):
    return {"status":"READY","experiment":exp.experiment,"regime":exp.regime,"mechanism":exp.mechanism,"initial_model_sha256":exp.initial_model_sha256,
        "vendor_sha256":digest(exp.cfg["vendor_trainer"]),
        "split_re_counts":{name:len(labels) for name,labels in exp.windows.items()},"split_re_labels":{name:[str(exp.a['labels'][label]) for label in labels] for name,labels in exp.windows.items()},
        "split_intersections":{"train_validation":[],"train_heldout":[],"validation_heldout":[]},"pressure_gauge":"subtract_area_mean_per_snapshot","ru":exp.ru,"rp":exp.rp,
        "train_snapshot_count":len(exp.train_ids),"dynamics_scaler_fit_count":len(exp.scaler_fit_ids),"assets":{name:{"path":str(path),"sha256":digest(path)} for name,path in exp.paths.items()},"trainable_parameter_audit":exp.trainable_audit,"config":exp.e}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--resume",type=Path)
    parser.add_argument("--mode",choices=("audit","preflight","train"),default="train"); cli=parser.parse_args(); cfg=json.loads(cli.config.read_text())
    emit("startup",experiment="S2-B",mode=cli.mode,host=socket.gethostname(),pid=os.getpid(),python=sys.executable)
    exp=Experiment(cli,cfg); atomic_json(Path(cli.run_dir)/"startup_manifest.json",startup_manifest(exp))
    if cli.mode=="audit": emit("audit_pass",**startup_manifest(exp))
    elif cli.mode=="preflight": exp.preflight()
    else: exp.train()


if __name__=="__main__":
    try: main()
    except Exception:
        try:
            run=Path(next((sys.argv[i+1] for i,x in enumerate(sys.argv) if x=="--run-dir"),".")); atomic_json(run/"FAILED.json",{"status":"FAILED","traceback":traceback.format_exc(),"failed_unix":time.time()})
        except Exception: pass
        raise
