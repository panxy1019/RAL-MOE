"""Generate seed/component CSVs from completed fixed-cache runs (stdlib)."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'experiments/iclr_seed_study_20260911/results_v1'
ART = ROOT / 'artifacts'


def write_csv(name, rows):
    with (ART / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    results = json.loads((STUDY / 'results.json').read_text())
    replay = json.loads((STUDY / 'original_replay.json').read_text())
    verify = json.loads((STUDY / 'verification.json').read_text())
    assert verify['status'] == 'PASS'
    expected = {(b, m, s) for b in ('SH', 'HP') for m in ('full', 'mu_only') for s in (42001, 42002, 42003)}
    assert {(r['boundary'], r['input_mode'], r['seed']) for r in results} == expected and len(results) == 12
    rows, per_re = [], []
    for r in results:
        job = STUDY / 'centered_square' / r['boundary'].lower() / r['input_mode'] / f"seed_{r['seed']}"
        assert hashlib.sha256((job / 'best.pt').read_bytes()).hexdigest() == r['checkpoint_sha256']
        assert json.loads((job / 'config.json').read_text()) == r['config']
        k = r['by_horizon']['K24']
        assert abs(k['joint_mean'] - k['velocity_mean'] - k['pressure_mean']) < 1e-12
        rows.append(dict(overlap=r['boundary'], mode=r['input_mode'], seed=r['seed'],
            best_step=r['best_step'], Eu_percent=100*k['velocity_mean'], Ep_percent=100*k['pressure_mean'],
            Ejoint_percent=100*k['joint_mean'], alpha_min=r['alpha_min'], alpha_max=r['alpha_max']))
        for re, v in r['per_Re'].items():
            per_re.append(dict(overlap=r['boundary'], mode=r['input_mode'], seed=r['seed'], Re=re,
                windows=v['windows'], Eu_percent=100*v['K24_velocity_mean'], Ep_percent=100*v['K24_pressure_mean'],
                Ejoint_percent=100*v['K24_joint_mean'], alpha_mean=v['alpha_mean']))
    summaries, paired, components = [], [], []
    for b in ('SH', 'HP'):
        for mode in ('full', 'mu_only'):
            group = sorted([r for r in rows if r['overlap'] == b and r['mode'] == mode], key=lambda r: r['seed'])
            for metric in ('Eu_percent', 'Ep_percent', 'Ejoint_percent'):
                values = [r[metric] for r in group]
                summaries.append(dict(overlap=b, mode=mode, metric=metric, n_seeds=3,
                    seed_42001=values[0], seed_42002=values[1], seed_42003=values[2],
                    mean=st.mean(values), SD_seed=st.stdev(values), ddof=1,
                    minimum=min(values), maximum=max(values)))
        full = {r['seed']: r['Ejoint_percent'] for r in rows if r['overlap'] == b and r['mode'] == 'full'}
        mu = {r['seed']: r['Ejoint_percent'] for r in rows if r['overlap'] == b and r['mode'] == 'mu_only'}
        delta = [mu[s]-full[s] for s in sorted(full)]
        paired.append(dict(overlap=b, mean_mu_minus_full_pp=st.mean(delta), SD_paired_difference_pp=st.stdev(delta),
            history_improves_seed_count=sum(d > 0 for d in delta), n_seeds=3,
            relative_reduction_of_mean_percent=100*(st.mean(mu.values())-st.mean(full.values()))/st.mean(mu.values())))
        for name, entry in replay[b]['methods'].items():
            k = entry['by_horizon']['K24']
            components.append(dict(overlap=b, method=name, source='original_checkpoint_replay', seed='',
                Eu_percent=100*k['velocity_mean'], Ep_percent=100*k['pressure_mean'], Ejoint_percent=100*k['joint_mean']))
        mu_fixed = next(r for r in rows if r['overlap'] == b and r['mode'] == 'mu_only' and r['seed'] == 42001)
        components.append(dict(overlap=b, method='T2-C_mu_only', source='new_run_fixed_seed_42001', seed=42001,
            **{k: mu_fixed[k] for k in ('Eu_percent', 'Ep_percent', 'Ejoint_percent')}))
    ART.mkdir(exist_ok=True)
    write_csv('t2c_seed_results.csv', rows)
    write_csv('t2c_seed_per_re.csv', per_re)
    write_csv('t2c_seed_summary.csv', summaries)
    write_csv('t2c_descriptor_ablation.csv', components)
    write_csv('t2c_paired_seed_differences.csv', paired)
    manifest = json.loads((ROOT / 'build/method_intro_20260909/upload_manifest.json').read_text())
    changed = [p for p, digest in manifest.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest]
    assert not changed, changed
    summary = dict(status='T2C_ABLATION_AND_THREE_SEEDS_COMPLETE', manuscript_changed=False,
        manuscript_assets_checked=len(manifest), table4='DEFERRED_BY_USER', descriptor_paper_edit='DEFERRED_BY_USER',
        scope='Centered-square fixed-cache gate study only; no specialist seed study',
        seed_summary=summaries, paired_comparisons=paired, fixed_run_components=components,
        post_training_verification_count=verify['count'],
        results_sha256=hashlib.sha256((STUDY / 'results.json').read_bytes()).hexdigest())
    (ART / 't2c_experiment_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
