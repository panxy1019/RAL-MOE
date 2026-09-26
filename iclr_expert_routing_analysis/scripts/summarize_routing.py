"""Recompute all diagnostics from saved routing decisions, independently of models."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair(record):
    return (record['group_id'], *sorted(record['expert_top_ids']))


def summarize(records, config, chart, channel, population, re_scope):
    G, E = config['num_regime_groups'], config['experts_per_group']
    N = len(records)
    assert N > 0
    counts = np.zeros((G, E), dtype=np.int64)
    masses = np.zeros((G, E), dtype=np.float64)
    group_counts = np.zeros(G, dtype=np.int64)
    shared = np.zeros(G, dtype=np.float64)
    pairs = Counter()
    timelines = defaultdict(list)
    for r in records:
        g = r['group_id']
        counts[g, r['expert_top_ids']] += 1
        masses[g, r['expert_top_ids']] += r['expert_top_weights']
        group_counts[g] += 1
        shared[g] += r['shared_weight']
        pairs[pair(r)] += 1
        timelines[r['window_id']].append(r)
    f, m = counts / counts.sum(), masses / N
    assert counts.sum() == 2 * N and abs(f.sum() - 1) < 1.e-12
    nonzero = f[f > 0]
    H = float(-np.sum(nonzero * np.log(nonzero)) / np.log(G * E))
    transitions = group_switches = pair_switches = 0
    for timeline in timelines.values():
        ordered = sorted(timeline, key=lambda r: (r['rollout_step'], r['stage']))
        for left, right in zip(ordered, ordered[1:]):
            transitions += 1
            group_switches += left['group_id'] != right['group_id']
            pair_switches += pair(left) != pair(right)
    common = dict(chart=chart, channel=channel, population=population, Re=re_scope)
    experts = [{**common, 'group_id': g, 'expert_id': e, 'expert_label': f'g{g}:e{e}',
                'decisions': N, 'activation_count': int(counts[g, e]),
                'normalized_activation_frequency': float(f[g, e]),
                'selection_probability': float(counts[g, e] / N),
                'mean_routing_mass': float(m[g, e])}
               for g in range(G) for e in range(E)]
    groups = [{**common, 'group_id': g, 'decisions': N, 'count': int(group_counts[g]),
               'frequency': float(group_counts[g] / N), 'shared_mass_in_group': float(shared[g] / N)} for g in range(G)]
    dominant_pair, dominant_pair_count = pairs.most_common(1)[0]
    summary = {**common, 'decisions': N, 'groups_available': G, 'groups_used': int(np.count_nonzero(group_counts)),
               'routed_experts_available': G * E, 'routed_experts_used': int(np.count_nonzero(counts)),
               'dominant_group': int(group_counts.argmax()), 'dominant_group_fraction': float(group_counts.max() / N),
               'pair_configurations_used': len(pairs), 'dominant_pair': ':'.join(map(str, dominant_pair)),
               'dominant_pair_fraction': dominant_pair_count / N,
               'mean_shared_mass': float(shared.sum() / N), 'total_routed_mass': float(m.sum()),
               'mean_total_mass': float(shared.sum() / N + m.sum()), 'normalized_utilization_entropy': H,
               'within_window_transitions': transitions, 'group_switch_fraction': group_switches / max(transitions, 1),
               'pair_switch_fraction': pair_switches / max(transitions, 1),
               'multi_group_concentration_over_95pct': bool(G > 1 and group_counts.max() / N > .95),
               'pair_concentration_over_95pct': dominant_pair_count / N > .95}
    pair_rows = [{**common, 'group_id': k[0], 'expert_id_1': k[1], 'expert_id_2': k[2],
                  'count': count, 'fraction': count / N} for k, count in pairs.most_common()]
    return experts, groups, summary, pair_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.analysis_dir
    (out / 'stats').mkdir(exist_ok=True)
    (out / 'reports').mkdir(exist_ok=True)
    expert_rows, group_rows, summaries, pair_rows, steps, errors, checks, joint = [], [], [], [], [], [], [], []
    for chart in ('steady', 'hopf', 'periodic'):
        report = json.loads((out / f'logs/{chart}_routing_verification.json').read_text())
        config = report['routing_config']
        with gzip.open(out / f'raw/{chart}_routing.jsonl.gz', 'rt', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
        K, stages = (56, 1) if chart == 'steady' else ((56, 4) if chart == 'hopf' else (48, 4))
        expected = sum(r['windows'] for r in report['rows']) * K * stages * 2
        assert len(records) == expected == report['raw_routing_rows']
        seen = set()
        max_mass_error = 0.
        for r in records:
            key = (r['window_id'], r['rollout_step'], r['stage'], r['channel'])
            assert key not in seen
            seen.add(key)
            gates = np.array(r['combined_gates']).reshape(config['num_regime_groups'], -1)
            assert r['group_id'] == int(gates.sum(1).argmax())
            actual = set(np.flatnonzero(gates[r['group_id'], 1:] > 0).tolist())
            assert actual == set(r['expert_top_ids']) and len(actual) == 2
            assert r['used_in_update'] == (r['channel'] == 'u' or r['stage'] == 1)
            max_mass_error = max(max_mass_error, abs(float(gates.sum()) - 1))
        macro = [r for r in records if r['stage'] == 1]
        lookup = {(r['window_id'], r['rollout_step'], r['channel']): r for r in macro}
        joint_counts = Counter()
        differing = 0
        for r in [r for r in macro if r['channel'] == 'u']:
            q = lookup[(r['window_id'], r['rollout_step'], 'p')]
            assert r['group_id'] == q['group_id']
            joint_counts[(pair(r), pair(q))] += 1
            differing += pair(r) != pair(q)
        joint.append({'chart': chart, 'macro_decisions': len(macro) // 2,
                      'u_p_different_pair_fraction': differing / (len(macro) // 2),
                      'joint_configurations': len(joint_counts),
                      'dominant_joint_configuration_fraction': max(joint_counts.values()) / (len(macro) // 2)})
        for channel in ('u', 'p'):
            for population in ('macro_stage1', 'used_forward_calls'):
                selected = [r for r in records if r['channel'] == channel and
                            (r['stage'] == 1 if population == 'macro_stage1' else r['used_in_update'])]
                for re_scope in ['pooled'] + sorted(set(r['Re'] for r in selected)):
                    subset = selected if re_scope == 'pooled' else [r for r in selected if r['Re'] == re_scope]
                    ex, gr, su, pr = summarize(subset, config, chart, channel, population, re_scope)
                    expert_rows.extend(ex); group_rows.extend(gr); summaries.append(su); pair_rows.extend(pr)
            selected = [r for r in macro if r['channel'] == channel]
            for k in range(1, K + 1):
                subset = [r for r in selected if r['rollout_step'] == k]
                gates = np.mean([r['combined_gates'] for r in subset], axis=0).reshape(config['num_regime_groups'], -1)
                for g in range(config['num_regime_groups']):
                    for e in range(config['experts_per_group']):
                        steps.append({'chart': chart, 'channel': channel, 'step': k, 'normalized_step': k / K,
                                      'group_id': g, 'expert_id': e, 'mean_routing_mass': float(gates[g, e + 1])})
        for row in report['rows']:
            errors.append({'chart': chart, 'Re': row['Re'], 'windows': row['windows'], 'horizon': K,
                           'Eu_percent': row['Eu_percent'], 'Ep_percent': row['Ep_percent'],
                           'reference_Eu_percent': row.get('reference_Eu_percent', ''),
                           'reference_Ep_percent': row.get('reference_Ep_percent', ''),
                           'finite_fraction': row['finite_fraction'], 'divergent_windows': row['divergent_windows']})
        checks.append({'chart': chart, 'raw_rows': len(records), 'expected_rows': expected,
                       'all_identifiers_unique': True, 'actual_gate_top2_agrees': True,
                       'max_absolute_mass_sum_error': max_mass_error,
                       'hook_on_off_bitwise_equal_all_windows': report['hook_on_off_predictions_bitwise_equal_all_windows'],
                       'model_and_checkpoint_unchanged': report['model_state_unchanged'] and report['checkpoint_unchanged']})
    datasets = {'circular_expert_utilization_table': expert_rows, 'group_usage': group_rows,
                'routing_summary': summaries, 'expert_pair_usage': pair_rows,
                'rollout_step_mass': steps, 'error_reproduction': errors, 'joint_channel_configurations': joint}
    for name, rows in datasets.items():
        csv_write(out / f'stats/{name}.csv', rows)
    for name, cols in [
        ('expert_activation_frequency', ('chart', 'channel', 'population', 'Re', 'group_id', 'expert_id', 'activation_count', 'normalized_activation_frequency', 'selection_probability')),
        ('expert_routing_mass', ('chart', 'channel', 'population', 'Re', 'group_id', 'expert_id', 'mean_routing_mass'))]:
        csv_write(out / f'stats/{name}.csv', [{k: r[k] for k in cols} for r in expert_rows])
    for name, cols in [('shared_expert_mass', ('chart', 'channel', 'population', 'Re', 'mean_shared_mass')),
                       ('routing_entropy', ('chart', 'channel', 'population', 'Re', 'normalized_utilization_entropy'))]:
        csv_write(out / f'stats/{name}.csv', [{k: r[k] for k in cols} for r in summaries])
    pooled = [r for r in summaries if r['Re'] == 'pooled' and r['population'] == 'macro_stage1']
    (out / 'stats/validation_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    (out / 'stats/summary.json').write_text(json.dumps({'primary_summary': pooled, 'joint': joint, 'checks': checks}, indent=2) + '\n')
    lines = ['# Appendix: complete pooled routed-expert utilization', '',
             'Primary population: macro-step stage 1. Zero-use experts are retained. Steady is the independently diagnosed S4 checkpoint.', '',
             '| Chart | Channel | Group:expert | Activation frequency | Mean routing mass |',
             '|---|---|---|---:|---:|']
    for r in expert_rows:
        if r['Re'] == 'pooled' and r['population'] == 'macro_stage1':
            lines.append(f"| {r['chart']} | {r['channel']} | {r['expert_label']} | {r['normalized_activation_frequency']:.6f} | {r['mean_routing_mass']:.6f} |")
    (out / 'reports/APPENDIX_ROUTING_TABLE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'primary_summary': pooled, 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
