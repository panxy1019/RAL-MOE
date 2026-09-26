"""Diagnostic-only profiler for the archived Circular P K48 benchmark.

Runs a single window in an isolated output directory. Original sources,
checkpoints, and timing artifacts are never modified. Profiling overhead is
reported separately and is not a replacement for the native benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
ROUND2 = ROOT / 'experiments/round2_20260917'
sys.path.insert(0, str(ROUND2 / 'code'))
import periodic_controls


INJECT = '''
    # Diagnostic scopes and call counters. No numerical operation is changed.
    import collections
    from torch.profiler import ProfilerActivity, profile, record_function
    with torch.inference_mode():
        audit_reference = rollout(starts[0])
    audit_calls = collections.Counter()
    audit_transfer = collections.Counter()
    audit_hooks = []
    audit_scopes = []
    def audit_wrap_function(obj, name, label):
        original = getattr(obj, name)
        def wrapped(*args, **kwargs):
            audit_calls[label] += 1
            with record_function('audit.' + label):
                return original(*args, **kwargs)
        setattr(obj, name, wrapped)
        audit_scopes.append((obj, name, original))
    for function, label in [
        ('galerkin_rhs_by_label','galerkin_rhs'),
        ('pressure_surrogate_by_label','pressure_surrogate'),
        ('make_features_np','features'),
        ('model_outputs_from_states_np','model_outputs'),
        ('integrate_autonomous_step_np','integration_step')]:
        if hasattr(trainer, function):
            audit_wrap_function(trainer, function, label)
    for name, module in model.named_modules():
        category = None
        if name == 'encoder': category = 'encoder'
        elif name == 'group_router': category = 'group_router'
        elif name.startswith('velocity_group_routers.') and name.count('.') == 1: category = 'velocity_channel_router'
        elif name.startswith('pressure_group_routers.') and name.count('.') == 1: category = 'pressure_channel_router'
        elif name.startswith('velocity_shared_experts.') and name.count('.') == 1: category = 'velocity_shared_expert'
        elif name.startswith('pressure_shared_experts.') and name.count('.') == 1: category = 'pressure_shared_expert'
        elif name.startswith('velocity_expert_groups.') and name.count('.') == 2: category = 'velocity_routed_expert'
        elif name.startswith('pressure_expert_groups.') and name.count('.') == 2: category = 'pressure_routed_expert'
        elif name == 'velocity_expert': category = 'velocity_structured_expert'
        elif name == 'pressure_expert': category = 'pressure_structured_expert'
        elif name.startswith('structured_correction.') and name.count('.') == 1: category = 'structured_expert'
        elif name == 'dense_closure': category = 'dense_closure'
        elif name == 'dense_correction': category = 'dense_correction'
        elif name == 'closure_confidence_head': category = 'closure_confidence_head'
        if category:
            def hook(mod, inputs, category=category, name=name):
                audit_calls[category] += 1
                audit_calls['module:' + name] += 1
            audit_hooks.append(module.register_forward_pre_hook(hook))
            original_forward = module.forward
            def wrapped_forward(*a, _f=original_forward, _category=category, **kw):
                with record_function('audit.' + _category):
                    return _f(*a, **kw)
            module.forward = wrapped_forward
            audit_scopes.append((module, 'forward', original_forward))
    audit_wrap_function(trainer, 'topk_router_from_logits', 'topk_router')
    audit_original_cpu = torch.Tensor.cpu
    audit_original_tensor = torch.tensor
    def audit_cpu(self, *args, **kwargs):
        if self.is_cuda:
            audit_transfer['device_to_host_calls'] += 1
            audit_transfer['device_to_host_bytes'] += self.numel() * self.element_size()
        return audit_original_cpu(self, *args, **kwargs)
    def audit_tensor(data, *args, **kwargs):
        output = audit_original_tensor(data, *args, **kwargs)
        if output.is_cuda and isinstance(data, np.ndarray):
            audit_transfer['host_to_device_calls'] += 1
            audit_transfer['host_to_device_bytes'] += output.numel() * output.element_size()
        return output
    torch.Tensor.cpu = audit_cpu
    torch.tensor = audit_tensor
    native_rollout_original = native_rollout
    def native_rollout(start):
        audit_calls['native_rollout'] += 1
        with record_function('audit.native_rollout'):
            return native_rollout_original(start)
    # Use the same 48-step wrapper, including all-step field decode and gauge.
    torch.cuda.synchronize()
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False) as prof:
            import time as audit_time
            audit_t0 = audit_time.perf_counter()
            audit_result = rollout(starts[0])
            torch.cuda.synchronize()
            audit_wall = audit_time.perf_counter() - audit_t0
    for hook in audit_hooks: hook.remove()
    for obj, name, original in reversed(audit_scopes): setattr(obj, name, original)
    torch.Tensor.cpu = audit_original_cpu
    torch.tensor = audit_original_tensor
    assert len(audit_result) == 48
    audit_max_modal_difference = max(float(np.max(np.abs(a[i]-b[i])))
        for a,b in zip(audit_reference,audit_result) for i in (0,1))
    audit_events = []
    for item in prof.key_averages():
        if item.key.startswith('audit.') or item.key in (
            'aten::item','aten::_local_scalar_dense','aten::to','aten::_to_copy',
            'aten::copy_','aten::topk','aten::nonzero','aten::cat','aten::index'):
            audit_events.append(dict(key=item.key,count=item.count,
                self_cpu_us=item.self_cpu_time_total,cpu_total_us=item.cpu_time_total,
                self_device_us=getattr(item,'self_device_time_total',None),
                device_total_us=getattr(item,'device_time_total',None)))
    write(cli.output/'profile.json',dict(
        profiler_wall_seconds=audit_wall, warning='Profiler overhead; not a speed benchmark. Nested inclusive ranges and CPU/GPU time must not be summed.',
        calls=dict(audit_calls), transfer=dict(audit_transfer), events=audit_events,
        max_modal_difference_from_uninstrumented=audit_max_modal_difference,
        source_sha256=__import__('hashlib').sha256(open(__file__,'rb').read()).hexdigest()))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=['proposed', 'dense', 'structured'])
    parser.add_argument('--tag', default='profile', choices=['profile', 'transfer', 'decode'])
    cli = parser.parse_args()
    kind = cli.kind
    output = ROUND2 / f'P_latency_{cli.tag}_20260923' / kind
    if output.exists():
        raise FileExistsError(output)
    checkpoint = ROUND2 / f'P_{kind}/1248' / f'Circular_P_{kind}_round2_Re_70p314635_checkpoint.pt'
    source = ROOT / 'experiments/strengthening_20260915/code/evaluate_periodic_dense.py'
    sys.path.insert(0, str(source.parent))
    text = source.read_text()
    modifications = [
        ("    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)",
         "    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)\n    ck['args']=json.loads((checkpoint_path.parent/'protocol.json').read_text())['config']"),
        ("        patch(trainer, cli.output/'architecture.json')", "        control_patch(trainer)"),
        ("    results = []\n", "    native_rollout = rollout\n    phi_u = vel['phi_uv'][:args.r_u].astype('float64')\n    phi_p = pre['phi_p'][:args.r_p].astype('float64')\n    mean_u = vel['mean_uv_regime'].astype('float64')\n    mean_p = pre['mean_p_regime'].astype('float64')\n    area = vel['point_areas'].astype('float64')\n    def rollout(start):\n        result = native_rollout(start)\n        if result:\n            decoded_u = np.asarray([v[0] for v in result]) @ phi_u + mean_u\n            decoded_p = np.asarray([v[1] for v in result]) @ phi_p + mean_p\n            decoded_p -= (decoded_p @ area / area.sum())[:,None]\n            if not (np.isfinite(decoded_u).all() and np.isfinite(decoded_p).all()):\n                raise FloatingPointError('Nonfinite physical decode')\n        return result\n    results = []\n"),
        ("    for re in helper.HELDOUT_RE:", "    for re in helper.HELDOUT_RE[:1]:"),
        ("        starts.extend(candidates[j] for j in chosen)", "        starts.extend(candidates[j] for j in chosen[:1])"),
        ("    if cli.benchmark:", INJECT + "\n    if False:"),
    ]
    for old, new in modifications:
        if text.count(old) != 1:
            raise RuntimeError(f'Unexpected archived evaluator anchor: {old[:60]}')
        text = text.replace(old, new)
    decode_old = """            decoded_u = np.asarray([v[0] for v in result]) @ phi_u + mean_u
            decoded_p = np.asarray([v[1] for v in result]) @ phi_p + mean_p
            decoded_p -= (decoded_p @ area / area.sum())[:,None]
            if not (np.isfinite(decoded_u).all() and np.isfinite(decoded_p).all()):
                raise FloatingPointError('Nonfinite physical decode')"""
    decode_new = "            with torch.profiler.record_function('audit.physical_decode'):\n" + "\n".join(
        "    " + line for line in decode_old.splitlines())
    if text.count(decode_old) != 1:
        raise RuntimeError('Physical-decode scope anchor changed')
    text = text.replace(decode_old, decode_new)
    module = types.ModuleType('isolated_profile')
    module.__file__ = str(source)
    module.control_patch = lambda trainer: periodic_controls.patch(trainer, kind, output)
    sys.modules[module.__name__] = module
    exec(compile(text, str(source), 'exec'), module.__dict__)
    sys.argv = [sys.argv[0], '--method', 'full' if kind == 'proposed' else 'dense',
                '--checkpoint', str(checkpoint), '--output', str(output)]
    module.main()
    (output / 'profile_manifest.json').write_text(json.dumps({
        'profile_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'archived_evaluator_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'diagnostic_only': True,
    }, indent=2))


if __name__ == '__main__':
    main()
