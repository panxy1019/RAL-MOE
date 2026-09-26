"""Post-training checks: selected checkpoint, fixed inputs and fresh inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, sha256
from run_study import data_at, evaluate_gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    root, study = args.run_root, args.study
    assert json.loads((study / 'status.json').read_text())['status'] == 'complete'
    manifest = json.loads((study / 'preregistration.json').read_text())
    checks = []
    def check(ok, label):
        checks.append({'check': label, 'passed': bool(ok)})
        if not ok:
            raise AssertionError(label)
    for rel, digest in manifest['original_asset_hashes'].items():
        check(sha256(root / rel) == digest, 'Original unchanged: ' + rel)
    for rel, digest in manifest['code_sha256'].items():
        check(sha256(Path(__file__).parent / rel) == digest, 'Training code unchanged: ' + rel)
    results = json.loads((study / 'results.json').read_text())
    check(len(results) == 12, '12 completed runs')
    for result in results:
        b, mode, seed = result['boundary'], result['input_mode'], result['seed']
        key = f'{b}/{mode}/{seed}'
        job = study / 'centered_square' / b.lower() / mode / f'seed_{seed}'
        c = torch.load(job / 'best.pt', map_location='cpu', weights_only=False)
        history = json.loads((job / 'history.json').read_text())
        selected = min(history, key=lambda h: (h['worst_Re_mean'], h['joint_mean']))
        check(selected['step'] == c['best_step'], key + ': validation-only argmin')
        check(all(np.isfinite(h['train_loss']) and np.isfinite(h['grad_norm']) for h in history), key + ': finite logged training')
        check([h['step'] for h in history] == [1] + list(range(100, 8001, 100)), key + ': fixed steps and validation schedule')
        check(sum(v.numel() for v in c['gate_state'].values()) == 4865, key + ': matched 4865 parameters')
        check(sha256(job / 'best.pt') == result['checkpoint_sha256'], key + ': evaluated checkpoint identity')
        check(c['cache_sha256'] == sha256(root / f'cache_{b.lower()}/{b.lower()}_development_cache.npz'), key + ': fixed development cache')
        data = data_at(root / f'heldout_evaluation_20260730_V1/cache_{b.lower()}/{b.lower()}_heldout_cache.npz')
        fresh = evaluate_gate(job / 'best.pt', root / 'e2_router/best.pt', data)
        check(np.array_equal(fresh['alpha_by_window'], result['alpha_by_window']), key + ': fresh checkpoint inference')
        check(all(abs(fresh['by_horizon']['K24'][k] - v) < 1e-14 for k, v in result['by_horizon']['K24'].items()), key + ': K24 metrics reproduced')
        if mode == 'mu_only':
            changed = dict(data)
            changed['features'] = data['features'].copy()
            changed['features'][:, 1:] += np.float32(1000)
            probe = evaluate_gate(job / 'best.pt', root / 'e2_router/best.pt', changed)
            check(np.array_equal(probe['alpha_by_window'], fresh['alpha_by_window']), key + ': invariant to arbitrary history perturbation')
            a = np.asarray(fresh['alpha_by_window'])
            check(all(np.ptp(a[data['re'] == re]) == 0 for re in np.unique(data['re'])), key + ': equal alpha for same Re')
    atomic_json(study / 'verification.json', {'status': 'PASS', 'checks': checks, 'count': len(checks)})
    print(f'PASS: {len(checks)} post-training checks')


if __name__ == '__main__':
    main()
