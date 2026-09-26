from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


EPS = 1e-8
HORIZONS = (1, 4, 8, 16, 56)
SELECTION_SCALES = (0.005, 0.01, 0.02, 0.05)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {key: float(np.quantile(array, q)) for key, q in (("median", .5), ("p90", .9), ("p95", .95), ("worst", 1.0))} | {"count": int(array.size)}


class S4:
    def __init__(self, args):
        self.args = args; self.run = Path(args.run_dir); self.run.mkdir(parents=True, exist_ok=True)
        self.ckpt = self.run / "checkpoints"; self.ckpt.mkdir(exist_ok=True)
        module = load_module("s4_s3_base", args.s3_trainer)
        base_args = SimpleNamespace(
            experiment="S3-B", trainer=args.trainer, finalizer=args.finalizer, config=args.config,
            checkpoint=args.s2b_checkpoint, bank=args.bank, run_dir=str(self.run / "runtime"),
            validation_lock=str(self.run / "validation.lock"), learning_rate=args.learning_rate,
        )
        self.base = module.S3(base_args); self.base.init_eval_assets(); self.exp = self.base.exp
        source = torch.load(args.s3b_checkpoint, map_location="cpu", weights_only=False)
        self.exp.model.load_state_dict(source["model"], strict=True); self.exp.model.eval()
        self.source_step = int(source["step"]); self.source_sha = sha256(args.s3b_checkpoint)
        self.source_model_sha = self.base.trainer.model_digest(self.exp.model)
        self.lambda_pair = float(source["lambda_new"])
        self.base_calibration = copy.deepcopy(source["base_calibration"])
        self.exp.opt = self.exp._build_optimizer()
        self.exp.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.exp.opt, 3600, eta_min=args.learning_rate * .05)
        self.exp.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.exp.amp_dtype == torch.float16)
        self.lambda_fp = None; self.fp_target_ratio = .075; self.fp_ramp_steps = 300
        self.baseline = None; self.best_fp = math.inf; self.best_step = None; self.stale = 0
        self.previous_fp = None; self.rise_count = 0; self.top = []

    def fixed_point_loss(self, batch):
        fixed = batch["fixed_id"].long(); a = self.exp.tensor("a", fixed); b = self.exp.tensor("b", fixed)
        pa, pb = self.base.free_rollout(fixed, a, b, 1)
        du, dp = pa[0].float() - a, pb[0].float() - b
        velocity = torch.mean(torch.sum(du * du, dim=1) / self.exp.ru)
        pressure = torch.mean(torch.sum(dp * dp, dim=1) / self.exp.rp)
        return velocity + pressure, {"velocity_raw_mse": velocity, "pressure_raw_mse": pressure}

    def exact_grad_norm(self, loss, retain=True):
        params = [parameter for parameter in self.exp.model.parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        squared = sum((gradient.float() ** 2).sum() for gradient in gradients if gradient is not None)
        return float(torch.sqrt(squared).cpu())

    def original_total(self, clean_ids, batch, k=4):
        out = self.exp.rollout(clean_ids, 16)
        clean = out["one"] + self.base_calibration["lambda_rollout"] * out["rollout"] + self.base_calibration["lambda_anchor"] * out["anchor"]
        pair, detail = self.base.pair_loss(batch, k)
        return clean + self.lambda_pair * pair, {"clean": clean, "pair": pair, **detail}

    def calibrate_anchor(self):
        clean_ids = self.exp.sample_schedule[0, :self.base.micro]
        batch = self.base.bank_batch(0, 0)
        with torch.autocast("cuda", dtype=self.exp.amp_dtype):
            original, original_detail = self.original_total(clean_ids, batch, 4)
            anchor, anchor_detail = self.fixed_point_loss(batch)
        original_norm = self.exact_grad_norm(original)
        anchor_norm = self.exact_grad_norm(anchor, False)
        if not math.isfinite(anchor_norm) or anchor_norm <= 0:
            raise FloatingPointError("fixed-point anchor has zero or invalid exact gradient norm")
        self.lambda_fp = self.fp_target_ratio * original_norm / anchor_norm
        if not math.isfinite(self.lambda_fp) or self.lambda_fp <= 0:
            raise FloatingPointError("invalid fixed-point calibration weight")
        self.exp.opt.zero_grad(set_to_none=True)
        report = {
            "schema_version": 1, "fixed_calibration_clean_ids": clean_ids.cpu().tolist(),
            "fixed_calibration_bank_ids": self.base.bank["train_schedule"][0, :self.base.micro].cpu().tolist(),
            "target_gradient_ratio": self.fp_target_ratio, "original_s3b_total_grad_norm": original_norm,
            "unweighted_fixed_point_grad_norm": anchor_norm, "lambda_fixed_point": self.lambda_fp,
            "realized_initial_gradient_ratio": self.lambda_fp * anchor_norm / original_norm,
            "original_detail": {k: float(v.detach().cpu()) for k, v in original_detail.items()},
            "anchor_detail": {k: float(v.detach().cpu()) for k, v in anchor_detail.items()},
        }
        atomic_json(self.run / "gradient_calibration.json", report)
        return report

    def train_step(self, step):
        self.exp.model.train(); self.exp.opt.zero_grad(set_to_none=True)
        k = self.base.perturb_k(step); pair_ramp = min(1., step / 300); fp_ramp = min(1., step / self.fp_ramp_steps)
        sums = {key: 0. for key in ("loss", "original", "fixed_point", "pair", "clean", "fp_velocity_raw_mse", "fp_pressure_raw_mse")}
        for part in range(self.base.accum):
            clean_ids = self.exp.sample_schedule[step, part * self.base.micro:(part + 1) * self.base.micro]
            batch = self.base.bank_batch(step, part)
            with torch.autocast("cuda", dtype=self.exp.amp_dtype):
                out = self.exp.rollout(clean_ids, 16)
                clean = out["one"] + self.base_calibration["lambda_rollout"] * out["rollout"] + self.base_calibration["lambda_anchor"] * out["anchor"]
                pair, _ = self.base.pair_loss(batch, k)
                fp, fp_detail = self.fixed_point_loss(batch)
                original = clean + pair_ramp * self.lambda_pair * pair
                total = original + fp_ramp * self.lambda_fp * fp
            if not torch.isfinite(total): raise FloatingPointError("non-finite total loss")
            self.exp.grad_scaler.scale(total / self.base.accum).backward()
            values = {"loss": total, "original": original, "fixed_point": fp, "pair": pair, "clean": clean,
                      "fp_velocity_raw_mse": fp_detail["velocity_raw_mse"], "fp_pressure_raw_mse": fp_detail["pressure_raw_mse"]}
            for key, value in values.items(): sums[key] += float(value.detach().cpu()) / self.base.accum
        self.exp.grad_scaler.unscale_(self.exp.opt)
        grad = torch.nn.utils.clip_grad_norm_([p for p in self.exp.model.parameters() if p.requires_grad], self.exp.e["grad_clip"])
        if not torch.isfinite(grad): raise FloatingPointError("non-finite gradient")
        self.exp.grad_scaler.step(self.exp.opt); self.exp.grad_scaler.update(); self.exp.sched.step()
        return sums | {"grad_norm": float(grad.cpu()), "k": k, "pair_ramp": pair_ramp, "fp_ramp": fp_ramp, "lr": self.exp.opt.param_groups[0]["lr"]}

    def _physical_component_gain(self, delta_coeff, initial_coeff, basis, weight, gauge=False):
        predicted = torch.matmul(delta_coeff.float(), basis.float())
        initial = torch.matmul(initial_coeff.float(), basis.float())
        if gauge:
            area = weight.float().square(); total = area.sum()
            predicted = predicted - torch.sum(predicted * area, dim=-1, keepdim=True) / total
            initial = initial - torch.sum(initial * area, dim=-1, keepdim=True) / total
        numerator = torch.linalg.vector_norm(predicted * weight, dim=-1)
        denominator = torch.linalg.vector_norm(initial * weight, dim=-1)
        gain = torch.where(denominator > 1e-30, numerator / denominator, torch.nan)
        return gain, numerator, denominator

    def contraction_validation(self, k):
        bank = self.base.bank; result = {"k": k, "records": []}
        vb = self.exp.cache["velocity_pod_basis"]; pbasis = self.exp.cache["pressure_pod_basis"]
        vw, pw = self.base.velocity_weight, self.base.pressure_weight
        for scale in (.001, .005, .01, .02, .05):
            mask = torch.isclose(bank["validation_scale"].float(), torch.tensor(scale, device=self.exp.device))
            fixed = bank["validation_fixed_id"][mask].long(); count = len(fixed)
            a = self.exp.tensor("a", fixed); b = self.exp.tensor("b", fixed)
            da = scale * self.exp.avt * bank["validation_du"][mask].float(); db = scale * self.exp.bvt * bank["validation_dp"][mask].float()
            with torch.inference_mode(), torch.autocast("cuda", dtype=self.exp.amp_dtype):
                ca, cb = self.base.free_rollout(fixed, a, b, k); pa, pp = self.base.free_rollout(fixed, a + da, b + db, k)
            delta_a, delta_b = pa[-1].float() - ca[-1].float(), pp[-1].float() - cb[-1].float()
            norm_den = torch.sqrt(torch.mean((da / self.exp.avt) ** 2, 1) + torch.mean((db / self.exp.bvt) ** 2, 1))
            norm_num = torch.sqrt(torch.mean((delta_a / self.exp.avt) ** 2, 1) + torch.mean((delta_b / self.exp.bvt) ** 2, 1))
            gv, _, dv = self._physical_component_gain(delta_a, da, vb, vw)
            gp, _, dpden = self._physical_component_gain(delta_b, db, pbasis, pw, True)
            raw_v = torch.where(torch.linalg.vector_norm(da, dim=1) > 1e-30, torch.linalg.vector_norm(delta_a, dim=1) / torch.linalg.vector_norm(da, dim=1), torch.nan)
            raw_p = torch.where(torch.linalg.vector_norm(db, dim=1) > 1e-30, torch.linalg.vector_norm(delta_b, dim=1) / torch.linalg.vector_norm(db, dim=1), torch.nan)
            finite = torch.isfinite(pa).all((0, 2)) & torch.isfinite(pp).all((0, 2)) & torch.isfinite(ca).all((0, 2)) & torch.isfinite(cb).all((0, 2))
            labels = bank["validation_label_id"][mask].cpu().tolist(); categories = bank["validation_category"][mask.cpu().numpy()].astype(str).tolist()
            for i in range(count):
                result["records"].append({
                    "re": str(self.exp.a["labels"][labels[i]]), "scale": scale, "direction_id": i % 20,
                    "category": categories[i], "fixed_id": int(fixed[i].cpu()), "finite": bool(finite[i].cpu()),
                    "normalized_state_gain": float((norm_num[i] / norm_den[i]).cpu()),
                    "raw_velocity_gain": float(raw_v[i].cpu()) if torch.isfinite(raw_v[i]) else None,
                    "raw_pressure_gain": float(raw_p[i].cpu()) if torch.isfinite(raw_p[i]) else None,
                    "physical_velocity_gain": float(gv[i].cpu()) if torch.isfinite(gv[i]) else None,
                    "physical_pressure_gain": float(gp[i].cpu()) if torch.isfinite(gp[i]) else None,
                    "initial_physical_velocity_norm": float(dv[i].cpu()), "initial_physical_pressure_norm": float(dpden[i].cpu()),
                })
        selected = [r for r in result["records"] if r["scale"] in SELECTION_SCALES and r["physical_pressure_gain"] is not None]
        result["selection_pressure_physical"] = quantiles([r["physical_pressure_gain"] for r in selected])
        result["selection_pressure_raw"] = quantiles([r["raw_pressure_gain"] for r in selected])
        result["finite_fraction"] = sum(r["finite"] for r in result["records"]) / len(result["records"])
        result["divergent_windows"] = sum(not r["finite"] for r in result["records"])
        return result

    def fixed_point_validation(self, k):
        ids = self.base.bank["validation_fixed_ids"].long(); a = self.exp.tensor("a", ids); b = self.exp.tensor("b", ids)
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.exp.amp_dtype): pa, pb = self.base.free_rollout(ids, a, b, k)
        by_re = {}
        for i, label in enumerate(self.base.bank["validation_fixed_labels"].cpu().tolist()):
            un, ud = self.base.finalizer.field_sums(pa[-1:, i:i+1], a[None, i:i+1], self.exp.cache["velocity_pod_basis"], self.exp.cache["velocity_mean"], self.base.velocity_weight)
            pn, pd = self.base.finalizer.field_sums(pb[-1:, i:i+1], b[None, i:i+1], self.exp.cache["pressure_pod_basis"], self.exp.cache["pressure_mean"], self.base.pressure_weight)
            by_re[str(self.exp.a["labels"][label])] = {"physical_velocity": math.sqrt(un / max(ud, EPS)), "physical_pressure": math.sqrt(pn / max(pd, EPS))}
        return {"k": k, "by_re": by_re, "worst_physical_velocity": max(v["physical_velocity"] for v in by_re.values()), "worst_physical_pressure": max(v["physical_pressure"] for v in by_re.values())}

    def validate(self, step):
        self.exp.model.eval(); clean = {}
        for k in HORIZONS: clean[f"k{k}"] = self.base.finalizer.evaluate_horizon(self.exp, "validation", k)
        contraction = {f"k{k}": self.contraction_validation(k) for k in (16, 56)}
        fixed = {f"k{k}": self.fixed_point_validation(k) for k in (1, 16, 56)}
        value = {"step": step, "clean": clean, "contraction": contraction, "fixed_point": fixed}
        with (self.run / "validation_history.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps(value) + "\n")
        return value

    @staticmethod
    def clean_value(validation, k, field):
        return validation["clean"][f"k{k}"]["overall_worst"]["physical_reconstruction_area_weighted"][field]

    def protection(self, validation):
        reasons = []
        for k in (1, 16, 56):
            current = validation["clean"][f"k{k}"]["overall_worst"]
            baseline = self.baseline["clean"][f"k{k}"]["overall_worst"]
            for field in ("velocity_relative_l2", "pressure_relative_l2"):
                if current["physical_reconstruction_area_weighted"][field] > 1.05 * baseline["physical_reconstruction_area_weighted"][field]: reasons.append(f"clean_k{k}_{field}_gt_5pct")
            if current["finite_fraction"] != 1 or current["divergent_windows"] != 0: reasons.append(f"clean_k{k}_finite_or_divergence")
        for k in (16, 56):
            current = validation["contraction"][f"k{k}"]["selection_pressure_physical"]
            baseline = self.baseline["contraction"][f"k{k}"]["selection_pressure_physical"]
            if current["median"] >= 1: reasons.append(f"k{k}_pressure_gain_median_ge_1")
            if current["p95"] > 1.05 * baseline["p95"]: reasons.append(f"k{k}_pressure_gain_p95_gt_5pct")
            if validation["contraction"][f"k{k}"]["finite_fraction"] != 1: reasons.append(f"k{k}_perturb_nonfinite")
        return {"pass": not reasons, "reasons": reasons}

    def checkpoint_payload(self, step, validation, protection):
        return {
            "schema_version": 1, "experiment": "S4-Steady", "step": step, "model": self.exp.model.state_dict(),
            "optimizer": self.exp.opt.state_dict(), "scheduler": self.exp.sched.state_dict(), "amp_scaler": self.exp.grad_scaler.state_dict(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
            "validation": validation, "protection": protection, "baseline": self.baseline, "lambda_pair": self.lambda_pair,
            "lambda_fixed_point": self.lambda_fp, "source_checkpoint_sha256": self.source_sha, "source_checkpoint_step": self.source_step,
            "early_state": {"best_fp": self.best_fp, "best_step": self.best_step, "stale": self.stale, "rise_count": self.rise_count},
        }

    def save_checkpoint(self, step, validation):
        protection = self.protection(validation); fp = validation["fixed_point"]["k16"]["worst_physical_pressure"]
        previous_best = self.best_fp
        selected = protection["pass"] and fp < self.best_fp
        meaningful = protection["pass"] and fp < previous_best * .99
        if selected: self.best_fp = fp; self.best_step = step
        if meaningful: self.stale = 0
        elif protection["pass"]: self.stale += 1
        if self.previous_fp is not None and fp > 1.05 * self.previous_fp: self.rise_count += 1
        else: self.rise_count = 0
        self.previous_fp = fp
        path = self.ckpt / f"validation_step_{step:04d}.pt"; tmp = path.with_suffix(".pt.tmp")
        torch.save(self.checkpoint_payload(step, validation, protection), tmp); tmp.replace(path)
        latest = self.ckpt / "latest.pt"; latest_tmp = self.ckpt / "latest.pt.tmp"; latest_tmp.unlink(missing_ok=True); os.link(path, latest_tmp); latest_tmp.replace(latest)
        if selected:
            target = self.ckpt / "best_protected.pt"; link_tmp = self.ckpt / "best_protected.pt.tmp"; link_tmp.unlink(missing_ok=True); os.link(path, link_tmp); link_tmp.replace(target)
        self.top.append((0 if protection["pass"] else 1, fp, path)); self.top.sort(key=lambda x: (x[0], x[1])); self.top = self.top[:5]
        keep = {p for _, _, p in self.top}
        for old in self.ckpt.glob("validation_step_*.pt"):
            if old not in keep: old.unlink()
        status = {"step": step, "protection": protection, "fixed_point_k16_worst_pressure": fp, "best_step": self.best_step, "best_fp": self.best_fp, "stale": self.stale, "rise_count": self.rise_count, "top5": [str(p) for _, _, p in self.top]}
        atomic_json(self.run / "checkpoint_status.json", status)
        return status

    def preregister(self, calibration):
        manifest = {
            "schema_version": 1, "experiment": "V17 S4-Steady anchor-only controlled branch", "created_before_optimizer_step": True,
            "immutable_inputs": {"s3b_checkpoint": {"path": self.args.s3b_checkpoint, "sha256": self.source_sha, "step": self.source_step}, "training_bank": {"path": self.args.bank, "sha256": sha256(self.args.bank)}},
            "single_variable": "add one-step raw-POD/area-weighted-equivalent fixed-point anchor to unchanged S3-B training objective",
            "unchanged_s3b": {"architecture": True, "split": True, "pod": True, "normalization": True, "perturbation_bank": True, "normalized_pair_training_loss": True, "lambda_pair": self.lambda_pair, "learning_rate": self.args.learning_rate, "micro_batch": 16, "grad_accum": 4, "effective_batch": 64, "curriculum": {"K4": 600, "K8": 1000, "K16": 2000}},
            "fixed_point_anchor": {"formula": "mean(||F_a(z*)-a*||_2^2/ru)+mean(||F_b(z*)-b*||_2^2/rp)", "pressure_gauge": "subtract_area_mean_per_snapshot; pressure POD basis is gauge-consistent", "target_gradient_ratio": self.fp_target_ratio, "lambda": self.lambda_fp, "ramp_steps": self.fp_ramp_steps, "calibration": calibration},
            "training": {"initialization": "model parameters only from S3-B best_contraction step 600; optimizer/scheduler/AMP/early state reset", "max_steps": 3600, "minimum_steps_before_early_stop": 2400, "validation_every": 200, "checkpoint_every": 200, "early_stop": "after step 2400, stop on 5 protected validations without >=1% K16 pressure FP improvement or 3 consecutive >5% FP rises"},
            "protection_gate": {"clean_k": [1, 16, 56], "clean_physical_velocity_pressure_max_relative_degradation": .05, "finite_fraction": 1, "divergent_windows": 0, "paired_metric": "raw/area-weighted physical pressure; scales 0.5%-5%; same validation bank", "k16_k56_pressure_gain_median_max_exclusive": 1, "k16_k56_pressure_gain_p95_max_relative_degradation": .05},
            "selection": "among protection-pass validation checkpoints, minimize worst K16 physical pressure fixed-point residual; tie by K56 FP then K16 pressure gain P95",
            "success": {"validation_and_one-time-heldout_k16_pressure_fp_worst_relative_improvement_min": .10, "k16_k56_pressure_gain_median_max_exclusive": 1, "p95_max_relative_degradation": .05, "clean_max_relative_degradation": .05, "finite_fraction": 1, "divergent_windows": 0, "aspirational_heldout_re_below_5pct": "at least 3 of 4; reported but not required"},
            "heldout": "forbidden during training/selection; exactly once after selected checkpoint frozen",
            "failure_policy": "do not replace S3-B; freeze current S3-B as final Steady sub-MoE", "success_policy": "freeze S4 only as candidate; do not start other training",
        }
        path = self.run / "PREREGISTRATION.json"; atomic_json(path, manifest); path.chmod(0o444)
        return manifest

    def restore_initial(self):
        source = torch.load(self.args.s3b_checkpoint, map_location="cpu", weights_only=False); self.exp.model.load_state_dict(source["model"], strict=True)
        self.exp.opt = self.exp._build_optimizer(); self.exp.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.exp.opt, 3600, eta_min=self.args.learning_rate * .05)
        self.exp.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.exp.amp_dtype == torch.float16)

    def run_train(self):
        calibration = self.calibrate_anchor(); self.restore_initial(); self.baseline = self.validate(0)
        prereg = self.preregister(calibration)
        if self.args.preflight_only:
            metrics = self.train_step(1); self.restore_initial()
            digest_before = self.base.trainer.model_digest(self.exp.model)
            protection = self.protection(self.baseline)
            tmp = self.run / "preflight_checkpoint.pt"; torch.save(self.checkpoint_payload(0, self.baseline, protection), tmp)
            loaded = torch.load(tmp, map_location="cpu", weights_only=False); self.exp.model.load_state_dict(loaded["model"]); digest_after = self.base.trainer.model_digest(self.exp.model); tmp.unlink()
            if digest_before != digest_after: raise AssertionError("checkpoint smoke mismatch")
            atomic_json(self.run / "PREFLIGHT_DONE.json", {"status": "PASS", "train_step": metrics, "protection_baseline": protection, "model_sha256": digest_after})
            return
        status0 = self.save_checkpoint(0, self.baseline)
        import swanlab
        swanlab.login(api_key=os.environ.get("SWANLAB_API_KEY"), relogin=False, save=False, timeout=20)
        run = swanlab.init(project="V17indepentMOEV2", name="V17-S4-Steady-AnchorOnly-RTX3090", group="S4-Steady-controlled", mode="online", config=prereg, log_dir=str(self.run / "swanlog"), reinit=True)
        atomic_json(self.run / "RUNNING.json", {"status": "RUNNING", "pid": os.getpid(), "host": socket.gethostname(), "started_unix": time.time(), "source_sha256": self.source_sha})
        stop_reason = "max_steps_3600"
        for step in range(1, 3601):
            metrics = self.train_step(step)
            if step == 1 or step % 20 == 0:
                swanlab.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
                print(json.dumps({"event": "optimizer_step", "step": step, **metrics}), flush=True)
            if step % 200 == 0:
                validation = self.validate(step); status = self.save_checkpoint(step, validation)
                log = {"validation/fp_k16_pressure_worst": status["fixed_point_k16_worst_pressure"], "validation/protection_pass": int(status["protection"]["pass"]), "validation/pressure_gain_k16_median": validation["contraction"]["k16"]["selection_pressure_physical"]["median"], "validation/pressure_gain_k16_p95": validation["contraction"]["k16"]["selection_pressure_physical"]["p95"], "validation/pressure_gain_k56_median": validation["contraction"]["k56"]["selection_pressure_physical"]["median"], "validation/pressure_gain_k56_p95": validation["contraction"]["k56"]["selection_pressure_physical"]["p95"]}
                swanlab.log(log, step=step); print(json.dumps({"event": "validation_complete", "step": step, **log}), flush=True)
                if step >= 2400 and status["stale"] >= 5: stop_reason = "protected_fp_patience_5"; break
                if step >= 2400 and status["rise_count"] >= 3: stop_reason = "fp_rise_3"; break
        atomic_json(self.run / "DONE.json", {"status": "TRAINING_COMPLETE", "stop_step": step, "stop_reason": stop_reason, "best_step": self.best_step, "best_fp": self.best_fp, "finished_unix": time.time()})
        swanlab.finish(state="success")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True); parser.add_argument("--finalizer", required=True); parser.add_argument("--s3-trainer", required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--s2b-checkpoint", required=True); parser.add_argument("--s3b-checkpoint", required=True)
    parser.add_argument("--bank", required=True); parser.add_argument("--run-dir", required=True); parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--preflight-only", action="store_true"); args = parser.parse_args()
    try: S4(args).run_train()
    except Exception as exc:
        atomic_json(Path(args.run_dir) / "FAILED.json", {"status": "FAILED", "error": repr(exc), "time": time.time()}); raise


if __name__ == "__main__": main()
