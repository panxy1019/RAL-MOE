"""Evaluate exact frozen models using their original runtimes; no training."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from inspect_checkpoints import CHECKPOINTS, sha256
from verify_steady import load, relocate
from routing_logger import RoutingLogger, model_digest


def checked_pair(model, logger, run, re_value, label, starts, times, horizon):
    with torch.inference_mode():
        baseline = run()
        logger.begin(re_value, label, starts, times, horizon)
        logger.attach()
        try:
            observed = run()
            logger.finish_window_batch(horizon)
        finally:
            logger.detach()
    for key in ('pred_a', 'pred_b'):
        assert np.isfinite(observed[key]).all(), (label, key, 'nonfinite')
        assert np.array_equal(baseline[key], observed[key]), (label, key, 'logger changed prediction')
    return observed


def steady(root, out, ckpt):
    base = root / 'steady_specialist_v1'
    trainer = load('routing_steady_trainer', base / 'code/train_s2b_3090.py')
    evaluator = load('routing_steady_evaluator', base / 'code/finalize_s2b_3090.py')
    cfg = relocate(json.loads((base / 'code/training_s2b_portable.json').read_text()), root)
    runtime = out / 'raw/steady_routing_runtime'
    runtime.mkdir(exist_ok=True)
    cfg['checkpoint_root'] = str(runtime / 'unused_checkpoint_directory')
    exp = trainer.Experiment(SimpleNamespace(run_dir=str(runtime), resume=None), cfg)
    exp.model.load_state_dict(ckpt['model'], strict=True)
    model = exp.model.eval().requires_grad_(False)
    logger = RoutingLogger(model, 'steady', 1)
    state_hash = model_digest(model)
    metrics = evaluator.evaluate_horizon(exp, 'heldout', 56)
    rows, predictions = [], {}
    for label_id, starts in exp.build_windows(56)['heldout'].items():
        label = str(exp.a['labels'][label_id])
        ids = torch.as_tensor(starts, device=exp.device)
        def run():
            with torch.autocast('cuda', dtype=exp.amp_dtype):
                result = exp.rollout(ids, 56)
            return {dst: result[src].transpose(0, 1).float().cpu().numpy()
                    for dst, src in [('pred_a', 'pa'), ('pred_b', 'pb'), ('true_a', 'ta'), ('true_b', 'tb')]}
        re_value = float(exp.a['re'][starts[0]])
        result = checked_pair(model, logger, run, re_value, label, starts, exp.a['time'][starts], 56)
        for k, v in result.items():
            predictions[f'{label}_{k}'] = v
        physical = metrics['by_re'][label]['physical_reconstruction_area_weighted']
        rows.append({'Re': re_value, 'label': label, 'windows': len(starts),
                     'starts': np.asarray(starts).tolist(),
                     'Eu_percent': 100 * physical['velocity_relative_l2'],
                     'Ep_percent': 100 * physical['pressure_relative_l2'],
                     'finite_fraction': metrics['by_re'][label]['finite_fraction'],
                     'divergent_windows': metrics['by_re'][label]['divergent_windows']})
        print(json.dumps({'event': 'steady_routing_verified', **rows[-1]}), flush=True)
    assert model_digest(model) == state_hash
    return model, logger, rows, predictions, {
        'integrator': 'Euler; native additive algebraic pressure closure',
        'precision': str(exp.amp_dtype), 'state_sha256': state_hash,
        'source_files': [base / 'code/train_s2b_3090.py', base / 'code/finalize_s2b_3090.py',
                         Path(cfg['vendor_trainer'])] + list(exp.paths.values()),
        'full_native_metrics': metrics,
        'window_policy': 'All legal heldout K56 windows; six per Re; native batching',
        'paper_identity': 'Independent specified S4 diagnostic, not the current paper Steady baseline',
    }


def hopf(root, out, ckpt):
    base = root / 'Hopf/migrated_h4_expanded'
    trainer = load('routing_hopf_trainer', base / 'code/train_hopf_moe_expanded.py')
    evaluator = load('routing_hopf_evaluator', base / 'code/evaluate_h4_expanded.py')
    audit_path = base / 'native_attractor_audit_20260723_v1/code/audit_hopf_native_attractor.py'
    audit = load('routing_hopf_audit', audit_path)
    args = SimpleNamespace(**ckpt['args'])
    for k in ('baseline_trainer', 'coefficient_view', 'galerkin_path', 'pressure_path', 'asset_manifest', 'contract'):
        value = Path(getattr(args, k))
        setattr(args, k, value if value.is_absolute() else base / value)
    defaults = dict(history_len=3, hidden_dim=256, expert_hidden=1024, num_blocks=3,
                    experts=6, top_k=2, expert_blocks=4, quadratic_rank=4, dropout=.04,
                    temperature=.8, adaptive_gate_initial_logit=6., lambda_scale_amplitude=1.,
                    lambda_scale_growth=.5, lambda_scale_sign=.1, scale_floor_quantile=.1)
    for k, value in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, value)
    device = torch.device('cuda')
    args.device = 'cuda'
    train_data = trainer.load_coefficients(args)
    rom_np = trainer.load_train_rom(args)
    norm, _ = trainer.fit_stats(train_data, rom_np, args)
    norm_diff = {k: float(np.max(np.abs(np.asarray(v) - np.asarray(ckpt['norm_stats'][k]))))
                 for k, v in asdict(norm).items()}
    stats = {k: torch.as_tensor(v, device=device) for k, v in asdict(norm).items()}
    rom = {k: torch.as_tensor(v, device=device) for k, v in rom_np.items()}
    probe = trainer.batch_from_ids(train_data, train_data['train_ids'][:2], device)
    rhs = trainer.galerkin(probe['ah'].reshape(-1, 32), probe['bh'].reshape(-1, 32),
                           probe['re'][:, None].expand(-1, args.history_len).reshape(-1), rom).reshape_as(probe['ah'])
    in_dim = trainer.state_features(probe['a'], probe['b'], probe['re'], probe['ah'], probe['bh'], rhs, rom, stats)[0].shape[1]
    model = trainer.build_model(in_dim, args, stats, device)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval().requires_grad_(False)
    state_hash = model_digest(model)
    data = audit.build_all_split_data(trainer, root / 'Hopf/artifacts/hopf', args.history_len)
    ug = evaluator.weighted_geometry(data['phi_u'], data['mean_u'], data['areas'], True)
    pg = evaluator.weighted_geometry(data['phi_p'], data['mean_p'], data['areas'], False)
    old_path = base / 'native_attractor_audit_20260723_v1/results/HOPF_NATIVE_ATTRACTOR_AUDIT.json'
    old = json.loads(old_path.read_text())
    assert old['checkpoint_sha256'] == sha256(root / CHECKPOINTS['hopf'])
    logger = RoutingLogger(model, 'hopf', 4)
    rows, predictions = [], {}
    for old_row in [r for r in old['per_Re'] if r['split'] == 'heldout']:
        re_value = old_row['Re']
        ids = data['valid'][(data['split'][data['valid']] == 'heldout') &
                            (np.abs(data['re'][data['valid']] - re_value) <= 5.e-6)]
        legal = trainer.legal_starts(data, ids, 56)
        starts = audit.evenly(legal, old['evaluation']['K56_windows_per_Re'])
        label = f'Re_{re_value:.6f}'
        def run():
            return audit.rollout_arrays(trainer, model, data, starts, 56, rom, stats, device)
        result = checked_pair(model, logger, run, re_value, label, starts, data['time'][starts], 56)
        met = audit.horizon_metrics(evaluator, result, 56, ug, pg)
        old_met = old_row['horizons']['56']
        keys = ['velocity_area_weighted_physical_relative_l2', 'pressure_area_weighted_physical_relative_l2']
        row = {'Re': re_value, 'label': label, 'windows': len(starts), 'legal_window_count': len(legal),
               'starts': starts.tolist(), 'Eu_percent': 100 * met[keys[0]], 'Ep_percent': 100 * met[keys[1]],
               'reference_Eu_percent': 100 * old_met[keys[0]], 'reference_Ep_percent': 100 * old_met[keys[1]],
               'finite_fraction': met['finite_fraction'], 'divergent_windows': met['divergent_windows']}
        rows.append(row)
        for k, v in result.items():
            predictions[f'{label}_{k}'] = v
        print(json.dumps({'event': 'hopf_routing_verified', **row}), flush=True)
    assert model_digest(model) == state_hash
    return model, logger, rows, predictions, {
        'integrator': 'Native RK4; pressure from stage 1 closure with predicted next velocity',
        'precision': 'float32; no autocast in original native audit', 'state_sha256': state_hash,
        'normalization_vs_checkpoint_max_absolute_difference': norm_diff,
        'source_files': [Path(trainer.__file__), Path(evaluator.__file__), audit_path,
                         Path(trainer.v16.__file__), old_path, args.coefficient_view,
                         args.galerkin_path, args.pressure_path, args.contract] +
                        [Path(str(data[k])) for k in ('velocity_asset', 'pressure_asset', 'index_asset')],
        'window_policy': 'Original native audit: five evenly spaced legal K56 windows per heldout Re',
        'paper_identity': 'Specified H4 checkpoint; compare against same-checkpoint native K56 audit',
    }


def periodic(root, out, ckpt):
    base = root / 'periodic_specialist_r32'
    evaluator = load('routing_periodic_evaluator', base / 'code/evaluate_periodic_r32_portable.py')
    trainer = evaluator.load_training_module(base / 'code/train_periodic_moe.py')
    args = SimpleNamespace(**ckpt['args'])
    args.device = 'cuda'
    args.data_root = base / 'assets/Global_POD_AreaWeighted_L2'
    args.tensor_path = base / 'assets/velocity_rom_periodic.npz'
    args.pressure_surrogate_path = base / 'assets/pressure_poisson_surrogate_periodic.npz'
    assert args.integrator == 'rk4' and args.pressure_input_mode == 'pressure_only'
    arrays, _ = trainer.build_arrays(args)
    tensors, pressure_tensors = np.load(args.tensor_path), np.load(args.pressure_surrogate_path)
    scalers = {k: trainer.Standardizer(mean=np.asarray(v['mean'], dtype=np.float32),
                                      scale=np.asarray(v['scale'], dtype=np.float32)) for k, v in ckpt['scalers'].items()}
    device = torch.device('cuda')
    model = evaluator.instantiate_model(trainer, args, arrays, ckpt['model_state'], device).requires_grad_(False)
    state_hash = model_digest(model)
    logger = RoutingLogger(model, 'periodic', 4)
    upath = args.data_root / 'global_velocity_pod_area_weighted_l2.npz'
    ppath = args.data_root / 'global_pressure_pod_area_weighted_l2.npz'
    vel, pre = np.load(upath), np.load(ppath)
    ug = evaluator.weighted_geometry(vel['phi_uv'][:args.r_u], vel['mean_uv_regime'], vel['point_areas'], True)
    pg = evaluator.weighted_geometry(pre['phi_p'][:args.r_p], pre['mean_p_regime'], pre['point_areas'], False)
    old_path = base / 'reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json'
    old = json.loads(old_path.read_text())
    valid = set(arrays['sample_ids'].tolist())
    re_by_label = np.array([arrays['re'][np.where(arrays['label_id'] == i)[0][0]] for i in range(len(arrays['labels']))])
    rows, predictions = [], {}
    for requested_re in evaluator.HELDOUT_RE:
        label_id = int(np.argmin(abs(re_by_label - requested_re)))
        re_value = float(re_by_label[label_id])
        assert abs(re_value - requested_re) < 2.e-5
        label = str(arrays['labels'][label_id])
        idx = np.where(arrays['label_id'] == label_id)[0]
        idx = idx[np.argsort(arrays['time'][idx])]
        starts = [int(idx[p]) for p in range(1, len(idx) - 48 - 1, 8) if int(idx[p]) in valid]
        windows = []
        for start in starts:
            def run():
                a_cur, b_cur = arrays['a'][start].copy(), arrays['b'][start].copy()
                ah, bh, rh = trainer.init_history_states_np(start, arrays)
                cur = start
                result = {k: [] for k in ('pred_a', 'pred_b', 'true_a', 'true_b', 'times')}
                for _ in range(48):
                    nxt = int(arrays['next_idx'][cur])
                    dt = float(arrays['time'][nxt] - arrays['time'][cur])
                    an, bn, rhs_g = trainer.integrate_autonomous_step_np(
                        model, a_cur, b_cur, cur, dt, ah, bh, rh, arrays, scalers, tensors, pressure_tensors, args, device)
                    for k, v in [('pred_a', an), ('pred_b', bn), ('true_a', arrays['a'][nxt]),
                                 ('true_b', arrays['b'][nxt]), ('times', arrays['time'][nxt])]:
                        result[k].append(np.array(v).copy())
                    ah = np.concatenate([an[None, None, :], ah[:, :-1, :]], axis=1)
                    bh = np.concatenate([bn[None, None, :], bh[:, :-1, :]], axis=1)
                    rh = np.concatenate([rhs_g[None, None, :], rh[:, :-1, :]], axis=1)
                    a_cur, b_cur, cur = an, bn, nxt
                return {k: np.asarray(v) for k, v in result.items()}
            result = checked_pair(model, logger, run, re_value, label, [start], [arrays['time'][start]], 48)
            windows.append(result)
        stacked = {k: np.stack([w[k] for w in windows]) for k in windows[0]}
        pa, pb, ta, tb = [stacked[k].reshape(-1, 32) for k in ('pred_a', 'pred_b', 'true_a', 'true_b')]
        Eu = np.sqrt(np.sum(evaluator.field_error_energy(ta, pa, ug)) / (np.sum(evaluator.field_energy(ta, ug)) + evaluator.EPS))
        Ep = np.sqrt(np.sum(evaluator.field_error_energy(tb, pb, pg)) / (np.sum(evaluator.field_energy(tb, pg)) + evaluator.EPS))
        divergent = 0
        for w in windows:
            ar = np.linalg.norm(w['pred_a'], axis=1) / (np.max(np.linalg.norm(w['true_a'], axis=1)) + evaluator.EPS)
            br = np.linalg.norm(w['pred_b'], axis=1) / (np.max(np.linalg.norm(w['true_b'], axis=1)) + evaluator.EPS)
            divergent += int(np.any((ar > 10) | (br > 10)))
        ref = next(r for r in old['results'] if abs(r['Re'] - re_value) < 2.e-5)['horizons']['48']
        rows.append({'Re': re_value, 'label': label, 'windows': len(starts), 'starts': starts,
                     'Eu_percent': 100 * float(Eu), 'Ep_percent': 100 * float(Ep),
                     'reference_Eu_percent': 100 * ref['velocity_area_weighted_l2'],
                     'reference_Ep_percent': 100 * ref['pressure_area_weighted_l2'],
                     'finite_fraction': 1., 'divergent_windows': divergent})
        for k, v in stacked.items():
            predictions[f'{label}_{k}'] = v
        print(json.dumps({'event': 'periodic_routing_verified', **rows[-1]}), flush=True)
    assert model_digest(model) == state_hash
    return model, logger, rows, predictions, {
        'integrator': 'Native RK4; pressure from stage 1 closure with predicted next velocity',
        'precision': 'float32; original portable evaluator without autocast', 'state_sha256': state_hash,
        'source_files': [Path(trainer.__file__), Path(evaluator.__file__), old_path, upath, ppath,
                         args.tensor_path, args.pressure_surrogate_path] + list(args.data_root.glob('*.csv')),
        'window_policy': 'Original portable evaluator: K48-capable starts at stride 8; 13 per Re',
        'known_input_contract': 'Predicted a/b/history; native Re and dataset-clock phase/time retained',
        'paper_identity': 'Specified Periodic epoch85 checkpoint; compare against archived portable K48 results',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--chart', choices=['steady', 'hopf', 'periodic'], required=True)
    a = p.parse_args()
    out = a.root / 'iclr_expert_routing_analysis'
    start = time.monotonic()
    path = a.root / CHECKPOINTS[a.chart]
    checksum = sha256(path)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model, logger, rows, predictions, details = globals()[a.chart](a.root, out, ckpt)
    assert sha256(path) == checksum
    count = logger.save(out / f'raw/{a.chart}_routing.jsonl.gz')
    np.savez_compressed(out / f'raw/{a.chart}_predictions.npz', **predictions)
    details['source_sha256'] = {str(p): sha256(p) for p in details.pop('source_files')}
    report = {'chart': a.chart, 'checkpoint': str(path), 'checkpoint_sha256': checksum,
              'checkpoint_unchanged': True, 'model_state_unchanged': True,
              'hook_on_off_predictions_bitwise_equal_all_windows': True,
              'strict_load_missing_unexpected_keys': [[], []], 'routing_config': logger.config,
              'rows': rows, 'raw_routing_rows': count, 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
              'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'device': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'numpy': np.__version__, 'elapsed_seconds': time.monotonic() - start,
              **details}
    (out / f'logs/{a.chart}_routing_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'event': 'complete', 'chart': a.chart, 'routing_rows': count,
                      'seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
