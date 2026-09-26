"""Fixed-cache descriptor ablation; no specialist/E2 training or paper edits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from common import ConvexGate, atomic_json, router_probabilities, sha256
from evaluate_heldout_rollout import evaluate_boundary, metrics

SEEDS = (42001, 42002, 42003)


def data_at(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def evaluate_gate(path, router, data):
    c = torch.load(path, map_location="cpu", weights_only=False)
    assert c['router_checkpoint_sha256'] == sha256(router)
    assert np.all(data['pair_indices'] == c['pair_indices'])
    x = ((data['features'] - c['feature_mean']) / c['feature_std']).astype(np.float32)
    if c['input_mode'] == 'mu_only':
        x[:, 1:] = 0
        assert np.count_nonzero(x[:, 1:]) == 0
    p = router_probabilities(router, data['re'])[:, c['pair_indices']]
    p /= p.sum(axis=1, keepdims=True)
    base = np.clip(p[:, 0], 1e-6, 1 - 1e-6)
    logit = np.log(base / (1 - base)).astype(np.float32)
    gate = ConvexGate(x.shape[1])
    gate.load_state_dict(c['gate_state'], strict=True)
    gate.eval()
    with torch.inference_mode():
        alpha = gate(torch.as_tensor(x), torch.as_tensor(logit)).numpy()
    assert np.isfinite(alpha).all() and ((alpha >= 0) & (alpha <= 1)).all()
    result = metrics(alpha, data['quad_u'], data['quad_p'], data['re'])
    assert all(np.isfinite(v) for v in result['by_horizon']['K24'].values())
    result.update(seed=c['seed'], input_mode=c['input_mode'], best_step=c['best_step'],
                  checkpoint_sha256=sha256(path), validation_metrics=c['validation_metrics'],
                  alpha_by_window=alpha.tolist(), re_by_window=data['re'].tolist())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run_root, args.output
    out.mkdir(parents=True, exist_ok=False)
    router = root / 'e2_router/best.pt'
    old = json.loads((root / 'heldout_evaluation_20260730_V1/comparison/HELDOUT_ROLLOUT_COMPARISON.json').read_text())
    inventory = json.loads((root / 'FINAL_FUSION_SUMMARY.json').read_text())['artifact_inventory']
    original_hashes = {rel: sha256(root / rel) for rel in inventory}
    assert all(original_hashes[k] == v['sha256'] for k, v in inventory.items())
    manifest = dict(status='PREREGISTERED', seeds=SEEDS, modes=['full', 'mu_only'],
        boundaries=['SH', 'HP'], steps=8000, eval_every=100, selection='validation only',
        original_asset_hashes=original_hashes, created_unix=time.time(),
        user_authorized_deferrals=['Table4 provenance', 'paper descriptor dimension'],
        code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
        torch_version=torch.__version__, numpy_version=np.__version__,
        cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    atomic_json(out / 'preregistration.json', manifest)
    replay = {}
    devs, tests = {}, {}
    for boundary, expected in [('SH', '1.1958'), ('HP', '0.8465')]:
        lower = boundary.lower()
        devpath = root / f'cache_{lower}/{lower}_development_cache.npz'
        testpath = root / f'heldout_evaluation_20260730_V1/cache_{lower}/{lower}_heldout_cache.npz'
        assert sha256(testpath) == old['boundaries'][boundary]['cache_sha256']
        dev, test = data_at(devpath), data_at(testpath)
        devs[boundary], tests[boundary] = dev, test
        assert set(test['split']) == {'heldout'}
        assert set(dev['split']) == {'train', 'validation'}
        train_re = set(dev['re'][dev['split'] == 'train'])
        val_re = set(dev['re'][dev['split'] == 'validation'])
        assert not train_re & val_re and not (train_re | val_re) & set(test['re'])
        assert np.all(dev['pair_indices'] == test['pair_indices'][0])
        for d in (dev, test):
            for i in (1, 2):
                assert d[f'candidate_{i}_finite'].all()
                assert not d[f'candidate_{i}_divergent'].any()
            assert d['features'].shape[1] == 9 and int(d['horizon']) == 24
        original = torch.load(root / f'gate_{lower}/best.pt', map_location='cpu', weights_only=False)
        assert original['cache_sha256'] == sha256(devpath)
        mean = dev['features'][dev['split'] == 'train'].mean(0)
        std = dev['features'][dev['split'] == 'train'].std(0)
        std[std < 1e-8] = 1
        assert np.array_equal(mean, original['feature_mean'])
        assert np.array_equal(std, original['feature_std'])
        history = json.loads((root / f'gate_{lower}/history.json').read_text())
        assert max(h['step'] for h in history) == 8000
        assert [h['step'] for h in history] == [1] + list(range(100, 8001, 100))
        assert original['seed'] == SEEDS[0]
        replay[boundary] = evaluate_boundary(root, testpath, boundary)
        got = replay[boundary]['K24_summary']['T2-C_joint_mean'] * 100
        assert f'{got:.4f}' == expected, (boundary, got, expected)
        # Check complete original method components, not just the rounded headline.
        for name, m in replay[boundary]['methods'].items():
            for k, v in m['by_horizon']['K24'].items():
                assert abs(v - old['boundaries'][boundary]['methods'][name]['by_horizon']['K24'][k]) < 1e-7
        print(f'REPLAY PASS {boundary}: {got:.10f}%', flush=True)
    atomic_json(out / 'original_replay.json', replay)
    jobs = [(b, mode, seed) for b in ('SH', 'HP') for seed in SEEDS for mode in ('full', 'mu_only')]
    results = []
    for index, (boundary, mode, seed) in enumerate(jobs, 1):
        lower = boundary.lower()
        job = out / 'centered_square' / lower / mode / f'seed_{seed}'
        job.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).parent / 'train_t2c_gate.py'),
            '--cache', str(root / f'cache_{lower}/{lower}_development_cache.npz'),
            '--router-checkpoint', str(router), '--output-dir', str(job),
            '--steps', '8000', '--eval-every', '100', '--seed', str(seed),
            '--device', 'cuda', '--input-mode', mode, '--swanlab-mode', 'disabled']
        print(f'TRAIN {index}/{len(jobs)} {boundary} {mode} {seed}', flush=True)
        atomic_json(out / 'status.json', dict(status='running', job=index, total=len(jobs),
                    boundary=boundary, input_mode=mode, seed=seed, updated_unix=time.time()))
        with (job.parent / f'seed_{seed}.log').open('w') as stream:
            subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
        c = torch.load(job / 'best.pt', map_location='cpu', weights_only=False)
        orig = torch.load(root / f'gate_{lower}/best.pt', map_location='cpu', weights_only=False)
        assert np.array_equal(c['feature_mean'], orig['feature_mean'])
        assert np.array_equal(c['feature_std'], orig['feature_std'])
        assert {k: tuple(v.shape) for k, v in c['gate_state'].items()} == {k: tuple(v.shape) for k, v in orig['gate_state'].items()}
        result = evaluate_gate(job / 'best.pt', router, tests[boundary])
        result.update(boundary=boundary, evaluated_unix=time.time(),
                      config=json.loads((job / 'config.json').read_text()))
        atomic_json(job / 'test_metrics.json', result)
        results.append(result)
        atomic_json(out / 'results.json', results)
        k24 = result['by_horizon']['K24']
        print(f"DONE {boundary} {mode} {seed} step={c['best_step']} joint={100*k24['joint_mean']:.8f}%", flush=True)
    assert all(sha256(root / rel) == digest for rel, digest in original_hashes.items())
    atomic_json(out / 'status.json', dict(status='complete', jobs=len(results), updated_unix=time.time(),
                original_assets_unchanged=True))
    print('ALL 12 RUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
