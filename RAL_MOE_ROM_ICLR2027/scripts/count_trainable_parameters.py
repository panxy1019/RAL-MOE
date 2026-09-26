"""Exact architecture/dispatch counts from archived definitions and checkpoint metadata.

This is not a pretrained-weight replay. No checkpoint, source model, or data is
modified. Fresh CPU modules are shape-checked against the full checkpoint
inventory; forced routing logits exercise actual forward paths. Counts depend
on the architecture and selected support, not learned weight values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import itertools
import json
import platform
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def shape_check(model, inventory):
    actual = {n: {"shape": list(v.shape), "dtype": str(v.dtype)}
              for n, v in model.state_dict().items()}
    assert actual == inventory["tensors"], {
        "missing": sorted(set(inventory["tensors"]) - set(actual)),
        "unexpected": sorted(set(actual) - set(inventory["tensors"])),
        "mismatched": [n for n in actual if n in inventory["tensors"]
                       and actual[n] != inventory["tensors"][n]],
    }
    assert sum(v.numel() for v in model.state_dict().values()) == inventory["elements"]
    return len(actual)


def unique_parameters(modules, trainable=True):
    return {id(p): p for m in modules for p in m.parameters()
            if not trainable or p.requires_grad}


def count(params):
    return sum(p.numel() for p in params.values())


def dispatch_counts(model, in_dim, torch):
    """Trace direct parameters, including functional quadratics/residual scales.

    Source inspection confirms every direct parameter in each called module is
    consumed by that module's forward. No descendant parameters are charged
    merely because a parent module is called; object identity deduplicates reuse.
    All local routers execute, including the routers of unselected groups.
    """
    model.eval()
    # The Hopf constructor zero-initializes its physical residual. Give only
    # these fresh synthetic expert heads nonzero weights so output-equivalence
    # checks do not pass trivially on zero corrections. No checkpoint is loaded.
    with torch.no_grad():
        for channel in ('velocity', 'pressure'):
            experts = [e for group in getattr(model, channel + '_expert_groups') for e in group]
            experts += list(getattr(model, channel + '_shared_experts'))
            for expert in experts:
                expert.linear.weight.normal_(std=.01)
                expert.mlp_head[-1].weight.normal_(std=.01)
    name_of = {id(p): n for n, p in model.named_parameters()}
    expert_roots = ["velocity_expert_groups", "pressure_expert_groups",
                    "velocity_shared_experts", "pressure_shared_experts"]
    non_expert = {id(p): p for n, p in model.named_parameters()
                  if p.requires_grad and n.split(".")[0] not in expert_roots}
    pairs = list(itertools.combinations(range(model.experts_per_group), model.top_k))
    all_expected_counts = set()
    # Exhaustive count proof over the entire group x u-pair x p-pair product.
    for group, upair, ppair in itertools.product(range(model.num_regime_groups), pairs, pairs):
        selected = [model.velocity_shared_experts[group], model.pressure_shared_experts[group]]
        selected += [model.velocity_expert_groups[group][e] for e in upair]
        selected += [model.pressure_expert_groups[group][e] for e in ppair]
        all_expected_counts.add(count(non_expert | unique_parameters(selected)))
    assert len(all_expected_counts) == 1
    expected = next(iter(all_expected_counts))
    seen, called, hooks = {}, set(), []
    for module_name, module in model.named_modules():
        def observe(m, args, name=module_name):
            called.add(name)
            seen.update({id(p): p for p in m.parameters(recurse=False) if p.requires_grad})
        hooks.append(module.register_forward_pre_hook(observe))
    x = torch.randn(1, in_dim)
    samples, first_names, native_names = [], None, None

    def force_logits(indices):
        def hook(module, args, output):
            forced = torch.full_like(output, -8.0)
            for rank, index in enumerate(indices):
                forced[:, index] = 8.0 - rank
            return forced
        return hook

    try:
        # Every pair in both channels, and every group, is executed. The
        # combinatorial enumeration above covers all 225 pair cross-products.
        for group in range(model.num_regime_groups):
            for pair_index, upair in enumerate(pairs):
                ppair = pairs[(pair_index + 3) % len(pairs)]
                routing_hooks = [model.group_router.register_forward_hook(force_logits([group]))]
                for router in model.velocity_group_routers:
                    routing_hooks.append(router.register_forward_hook(force_logits(upair)))
                for router in model.pressure_group_routers:
                    routing_hooks.append(router.register_forward_hook(force_logits(ppair)))
                try:
                    seen.clear(); called.clear()
                    with torch.no_grad():
                        output = model(x, return_expert_stack=False, return_closure_params=True)
                    observed = count(seen)
                    assert observed == expected, (group, upair, ppair, observed, expected)
                    expected_modules = {
                        f"velocity_expert_groups.{group}.{e}" for e in upair
                    } | {f"pressure_expert_groups.{group}.{e}" for e in ppair}
                    actual_modules = {n for n in called if n.startswith((
                        "velocity_expert_groups.", "pressure_expert_groups.")) and len(n.split(".")) == 3}
                    assert actual_modules == expected_modules
                    for channel in ("velocity", "pressure"):
                        assert all(f"{channel}_group_routers.{g}" in called
                                   for g in range(model.num_regime_groups))
                    names = sorted(name_of[pid] for pid in seen)
                    if first_names is None:
                        first_names = names
                    samples.append({"group": group, "velocity_pair": upair, "pressure_pair": ppair,
                                    "active_trainable": observed})
                    if group == 0 and pair_index == 0:
                        sparse_result = [output[0].clone(), output[1].clone(),
                                         output[4]["alpha"].clone()]
                        assert all(torch.count_nonzero(v).item() > 0 for v in sparse_result)
                        seen.clear(); called.clear()
                        with torch.no_grad():
                            dense = model(x, return_expert_stack=True, return_closure_params=True)
                        assert all(torch.equal(a, b) for a, b in zip(sparse_result,
                                   [dense[0], dense[1], dense[4]["alpha"]]))
                        diagnostic_count = count(seen)
                        native_names = sorted(name_of[pid] for pid in seen)
                        # Only velocity diagnostic stacks are evaluated densely.
                        expected_diagnostic = (non_expert | unique_parameters([
                            model.velocity_expert_groups,
                            model.velocity_shared_experts[group],
                            model.pressure_shared_experts[group],
                            *[model.pressure_expert_groups[group][e] for e in ppair]]))
                        assert diagnostic_count == count(expected_diagnostic)
                finally:
                    for hook in routing_hooks:
                        hook.remove()
    finally:
        for hook in hooks:
            hook.remove()
    breakdown = {}
    for name, parameter in model.named_parameters():
        item = breakdown.setdefault(name.split(".")[0], {"trainable": 0, "nontrainable": 0})
        item["trainable" if parameter.requires_grad else "nontrainable"] += parameter.numel()
    return {"total_trainable": count(unique_parameters([model])),
            "prediction_active_trainable": expected,
            "prediction_active_fraction": expected / count(unique_parameters([model])),
            "diagnostic_stack_active_trainable": diagnostic_count,
            "nontrainable_parameter_elements": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "buffer_elements": sum(b.numel() for b in model.buffers()),
            "executed_synthetic_decisions": samples,
            "exhaustive_architecture_supports": model.num_regime_groups * len(pairs) ** 2,
            "active_count_invariant": True,
            "prediction_vs_diagnostic_outputs_bitwise_equal": True,
            "synthetic_probe_nonzero_outputs": True,
            "first_prediction_active_names": first_names,
            "first_diagnostic_active_names": native_names,
            "parameter_breakdown": breakdown}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--analysis-dir", type=Path, default=project.parent / "iclr_expert_routing_analysis")
    parser.add_argument("--runtime-dir", type=Path, default=project / "build/parameter_count_runtime")
    parser.add_argument("--output", type=Path, default=project / "build/parameter_counts.json")
    args = parser.parse_args()
    if args.runtime_dir.is_dir():
        sys.path.insert(0, str(args.runtime_dir))
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20260906)
    source = args.analysis_dir / "source"
    inventory_path = args.analysis_dir / "logs/checkpoint_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    periodic_path = source / "periodic_specialist_r32/code/train_periodic_moe.py"
    hopf_path = source / "Hopf/migrated_h4_expanded/code/train_hopf_moe_expanded.py"
    h4_path = source / "Hopf/migrated_h4_expanded/code/train_h4_expanded.py"
    periodic = load("parameter_count_periodic", periodic_path)
    package = ModuleType("periodic_moe_3090")
    package.train_periodic_moe = periodic
    sys.modules["periodic_moe_3090"] = package
    hopf = load("parameter_count_hopf", hopf_path)
    h4 = load("parameter_count_h4", h4_path)
    report = {
        "status": "PASS", "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "pretrained_weight_replay": False,
        "scope": "Fresh actual model classes, recorded checkpoint architecture, exact full state-shape matching, source training flags, actual forward dispatch tracing.",
        "definition": "Unique requires_grad=True parameter objects consumed by called modules in one batch-size-one Level-1 forward; includes direct functional parameters and all executed routers. Excludes buffers/fixed operators and frozen parameters.",
        "macro_step_caveat": "A single routing decision is not the union of four RK4 stages. Batch/trajectory unions and FLOPs/timing are not measured.",
        "inventory_sha256": sha(inventory_path), "script_sha256": sha(__file__),
        "source_sha256": {str(p.relative_to(args.analysis_dir)): sha(p)
                          for p in [periodic_path, hopf_path, h4_path]}, "models": {},
        "excluded_models": {
            "Steady": "Available S4 checkpoint is not established as the paper's Steady identity.",
            "Vanilla-FNN-MoE": "Source launcher is available but no complete model-specific frozen checkpoint/config inventory; not estimated.",
            "DataOnly-MoE": "No complete model-specific frozen checkpoint/config inventory; not estimated.",
            "Global MoE": "No complete model-specific frozen checkpoint/config inventory; not estimated."},
        "parameter_matched_claim": False,
    }
    for chart in ["hopf", "periodic"]:
        checkpoint = inventory["checkpoints"][chart]
        verify_path = args.analysis_dir / f"logs/{chart}_routing_verification.json"
        verification = json.loads(verify_path.read_text(encoding="utf-8"))
        assert checkpoint["sha256"] == verification["checkpoint_sha256"]
        assert sha(periodic_path) in verification["source_sha256"].values()
        if chart == "hopf":
            assert sha(hopf_path) in verification["source_sha256"].values()
        state = checkpoint["state_dicts"]["model_state"]
        in_dim = state["tensors"]["encoder.net.0.weight"]["shape"][1]
        if chart == "hopf":
            previous_argv = sys.argv
            try:
                sys.argv = [str(hopf_path), "--variant", "common"]
                for key in ["coefficient-view", "galerkin-path", "pressure-path", "asset-manifest", "output-root"]:
                    sys.argv += ["--" + key, "UNUSED_ARCHITECTURE_AUDIT"]
                config = hopf.parse_args()
            finally:
                sys.argv = previous_argv
            for key, value in checkpoint["metadata"]["args"].items():
                if hasattr(config, key) and not isinstance(value, dict):
                    setattr(config, key, value)
            stats = {"rhs_mean": torch.zeros(32), "rhs_scale": torch.ones(32),
                     "pressure_mean": torch.zeros(32), "pressure_scale": torch.ones(32)}
            model = hopf.build_model(in_dim, config, stats, torch.device("cpu"))
            normal_form = h4.NormalForm(torch.tensor([45.5, 59.201432]))
            nf_tensors = shape_check(normal_form, checkpoint["state_dicts"]["normal_form_state"])
            normal_form_audit = {"trainable_parameters": count(unique_parameters([normal_form])),
                                 "buffers": sum(b.numel() for b in normal_form.buffers()),
                                 "verified_state_tensors": nf_tensors,
                                 "scope": "Separate training-loss-only normal-form head; absent from deployment forward."}
        else:
            recorded = checkpoint["metadata"]["args"]
            aliases = {"in_dim": in_dim, "out_dim": recorded["r_u"], "pressure_dim": recorded["r_p"],
                       "num_operator_spaces": recorded["num_shared_experts"]}
            kwargs = {k: aliases.get(k, recorded.get(k, v.default))
                      for k, v in inspect.signature(periodic.OperatorSpaceMoEROM).parameters.items()}
            assert all(v is not inspect.Parameter.empty for v in kwargs.values())
            model = periodic.OperatorSpaceMoEROM(**kwargs)
        verified_shapes = shape_check(model, state)
        result = dispatch_counts(model, in_dim, torch)
        result.update({"checkpoint_sha256": checkpoint["sha256"],
                       "checkpoint_remote_path": checkpoint["path"],
                       "recorded_state_elements": state["elements"],
                       "verified_state_tensors": verified_shapes, "input_dim": in_dim,
                       "routing_config": verification["routing_config"],
                       "source_hash_matches_original_routing_audit": True,
                       "native_audit_macro_stage1_stack": chart == "hopf"})
        if chart == "hopf":
            result["training_only_normal_form"] = normal_form_audit
        result["native_audit_macro_stage1_active_trainable"] = result[
            "diagnostic_stack_active_trainable" if chart == "hopf" else "prediction_active_trainable"]
        result["native_audit_macro_stage1_active_fraction"] = (
            result["native_audit_macro_stage1_active_trainable"] / result["total_trainable"])
        report["models"][chart] = result
        print(json.dumps({"chart": chart, **{k: result[k] for k in (
            "total_trainable", "prediction_active_trainable", "prediction_active_fraction",
            "native_audit_macro_stage1_active_trainable", "nontrainable_parameter_elements")}}), flush=True)
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "total_trainable", "prediction_active_trainable", "prediction_active_fraction",
                  "native_audit_macro_stage1_active_trainable", "native_audit_macro_stage1_active_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for name, result in report["models"].items():
            writer.writerow({"model": name, **{k: result[k] for k in fields[1:]}})


if __name__ == "__main__":
    main()
