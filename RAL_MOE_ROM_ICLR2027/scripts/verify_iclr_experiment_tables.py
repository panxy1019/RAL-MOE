"""Read-only preconditions for P1/P2/P3; fail closed on unresolved provenance.

This initial audit does NOT certify revision8 as the paper's raw data, rerun
T2-C, generate publishable mean/SD values, or implement the pending seed study.
Run from any directory with Python 3.10+ (standard library only).
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'
SOURCES = OUT / 'experiment_p123_20260911' / 'sources'
RUN = ROOT.parent / 'centeredsquare_fusion_v1/results/E2_T2C_K24_20260730_STRICT_V3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npy_header(archive, name):
    raw = archive.read(name + '.npy')
    assert raw[:6] == b'\x93NUMPY'
    version = raw[6]
    assert version in (1, 2, 3)
    size = 2 if version == 1 else 4
    count = int.from_bytes(raw[8:8 + size], 'little')
    offset = 8 + size + count
    return ast.literal_eval(raw[8 + size:offset].decode().strip()), raw[offset:]


def vector(archive, name):
    meta, raw = npy_header(archive, name)
    assert len(meta['shape']) == 1 and not meta['fortran_order']
    dtype = meta['descr']
    if dtype.startswith('<U'):
        width = int(dtype[2:]) * 4
        return [raw[i:i + width].decode('utf-32-le').rstrip('\0')
                for i in range(0, len(raw), width)]
    return list(struct.unpack('<' + {'<f4': 'f', '<f8': 'd'}[dtype] * meta['shape'][0], raw))


def main():
    OUT.mkdir(exist_ok=True)
    candidate = SOURCES / 'REVISION8_DETAILED_ROWS.csv'
    rows = list(csv.DictReader(candidate.open(encoding='utf-8-sig', newline='')))
    tex = (ROOT / 'sections/experiments.tex').read_text(encoding='utf-8')
    table = tex.split('\\label{tab:compact_ablation}', 1)[1].split('\\bottomrule', 1)[0]
    methods = [('RAL-MoE-ROM', 'Proposed Specialist MoE'),
               ('Vanilla-FNN-MoE', 'Vanilla-FNN-MoE'),
               ('DataOnly-MoE', 'DataOnly-MoE'), ('Global MoE', 'Global MoE†')]
    comparisons = []
    for paper_name, raw_name in methods:
        segment = table.split('\n' + paper_name + '\n', 1)[1].split('\\addlinespace', 1)[0]
        cells = re.findall(r'\\shortstack\{([^}]+)\}|&\s*(N/E)', segment)
        assert len(cells) == 3, (paper_name, cells)
        for (regime, count, horizon), (cell, ne) in zip(
                [('Steady', 4, 'K56'), ('Hopf', 3, 'K56'), ('Periodic', 4, 'K48')], cells):
            selected = [r for r in rows if r['Method'] == raw_name and r['Regime'] == regime]
            for metric, endpoints in zip(('Velocity_field', 'Pressure_field'),
                                         ('N/E', 'N/E') if ne else cell.split('\\\\')):
                values = [float(r[metric].rstrip('%')) for r in selected if r[metric].endswith('%')]
                observed = f'{min(values):.4f}--{max(values):.4f}' if values else 'N/E'
                complete = len(values) == count and len({r['Test_Re'] for r in selected}) == count
                if ne:
                    complete = not values and len(selected) == count
                horizon_ok = {r['Horizon'] for r in selected} == {horizon}
                comparisons.append(dict(model=paper_name, regime=regime, metric=metric,
                    paper=endpoints, candidate=observed, n_numeric=len(values),
                    count_ok=complete, horizon_ok=horizon_ok,
                    matches=complete and horizon_ok and observed == endpoints))

    report_path = RUN / 'heldout_evaluation_20260730_V1/comparison/HELDOUT_ROLLOUT_COMPARISON.json'
    stored = json.loads(report_path.read_text(encoding='utf-8'))
    square = {}
    for boundary, target in [('SH', '1.1958'), ('HP', '0.8465')]:
        b = stored['boundaries'][boundary]
        k24 = b['methods']['T2-C']['by_horizon']['K24']
        lower = boundary.lower()
        split_sets = {}
        shapes = {}
        for role, path in [('development', RUN / f'cache_{lower}/{lower}_development_cache.npz'),
                           ('heldout', RUN / f'heldout_evaluation_20260730_V1/cache_{lower}/{lower}_heldout_cache.npz')]:
            with zipfile.ZipFile(path) as archive:
                shapes[role] = npy_header(archive, 'features')[0]['shape']
                splits, re_values = vector(archive, 'split'), vector(archive, 're')
                for split in set(splits):
                    split_sets[split] = sorted({r for s, r in zip(splits, re_values) if s == split})
        assert set(split_sets) == {'train', 'validation', 'heldout'}
        disjoint = all(not set(split_sets[a]) & set(split_sets[b]) for a, b in
                       [('train', 'validation'), ('train', 'heldout'), ('validation', 'heldout')])
        square[boundary] = dict(stored_K24_percent={k: 100 * k24[k] for k in
            ('velocity_mean', 'pressure_mean', 'joint_mean')},
            matches_paper_rounding=f"{100 * k24['joint_mean']:.4f}" == target,
            joint_sum_ok=abs(k24['joint_mean'] - k24['velocity_mean'] - k24['pressure_mean']) < 1e-12,
            gate_hash_ok=sha(RUN / f'gate_{lower}/best.pt') == b['gate_checkpoint_sha256'],
            cache_hash_ok=sha(RUN / f'heldout_evaluation_20260730_V1/cache_{lower}/{lower}_heldout_cache.npz') == b['cache_sha256'],
            router_hash_ok=sha(RUN / 'e2_router/best.pt') == b['router_checkpoint_sha256'],
            feature_shapes=shapes, Re_by_split=split_sets, cached_gate_splits_disjoint=disjoint,
            full_pipeline_split_audit='NOT_COMPLETED', evaluation_replayed=False)

    manifest = json.loads((ROOT / 'build/method_intro_20260909/upload_manifest.json').read_text())
    changed = [name for name, digest in manifest.items() if sha(ROOT / name) != digest]
    summary = dict(status='BLOCKED_PROVENANCE', scope='Precondition audit only; no new experiment',
        blockers=['Table4 current-paper per-Re raw/checkpoint identity unresolved; revision8 fails regression',
                  'Paper descriptor dimension 6 versus original cache history dimension 8'],
        candidate_source=dict(path=str(candidate), sha256=sha(candidate),
            qualification='Historical rounded per-Re report; NOT approved as current-paper raw data'),
        table4_candidate_checks=comparisons, table4_candidate_matches=sum(c['matches'] for c in comparisons),
        table4_candidate_total=len(comparisons), centered_square=square,
        original_report_sha256=sha(report_path), manuscript_files_checked=len(manifest),
        manuscript_changed_vs_20260909=changed,
        missing_deliverables=['approved table4 raw data and mean/std', 'descriptor ablation',
            'seed study', 'complete benchmark protocol and split verification', 'revised paper/PDF'])
    (OUT / 'iclr_experiment_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (OUT / 'experiment_p123_20260911/table4_candidate_regression.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps({k: summary[k] for k in ['status', 'table4_candidate_matches', 'table4_candidate_total',
                     'manuscript_files_checked', 'manuscript_changed_vs_20260909']}, indent=2))
    return 2  # Deliberate nonzero: do not run downstream training or publication.


if __name__ == '__main__':
    if sys.argv[1:] == ['--t2c-study-only']:
        # User explicitly deferred Table4 and manuscript descriptor edits.
        # Preserve the historical audit; verify the completed study separately.
        sys.exit(subprocess.call([sys.executable, str(ROOT / 'scripts/summarize_t2c_seed_study.py')]))
    sys.exit(main())
