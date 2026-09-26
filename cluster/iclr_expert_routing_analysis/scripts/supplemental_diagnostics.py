"""Mass concentration and within-step RK4 sensitivity from existing raw logs."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from summarize_routing import csv_write, pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.analysis_dir
    result, sensitivity = [], []
    for chart in ('steady', 'hopf', 'periodic'):
        with gzip.open(out / f'raw/{chart}_routing.jsonl.gz', 'rt') as f:
            records = [json.loads(line) for line in f]
        report = json.loads((out / f'logs/{chart}_routing_verification.json').read_text())
        G = report['routing_config']['num_regime_groups']
        E = report['routing_config']['experts_per_group']
        for channel in ('u', 'p'):
            selected = [r for r in records if r['channel'] == channel and r['stage'] == 1]
            for re_scope in ['pooled'] + sorted(set(r['Re'] for r in selected)):
                subset = selected if re_scope == 'pooled' else [r for r in selected if r['Re'] == re_scope]
                combined = np.array([r['combined_gates'] for r in subset]).reshape(len(subset), G, E + 1)
                mass = combined[:, :, 1:].mean(0)
                q = mass.reshape(-1) / mass.sum()
                top1_share = [max(r['expert_top_weights']) / sum(r['expert_top_weights']) for r in subset]
                result.append({'chart': chart, 'channel': channel, 'Re': re_scope,
                               'largest_mass_expert': f'g{int(q.argmax()) // E}:e{int(q.argmax()) % E}',
                               'largest_expert_fraction_of_routed_mass': float(q.max()),
                               'mass_effective_expert_count_inverse_simpson': float(1 / np.sum(q*q)),
                               'mean_per_decision_top1_share_of_routed_mass': float(np.mean(top1_share)),
                               'min_per_decision_top1_share': float(min(top1_share)),
                               'max_per_decision_top1_share': float(max(top1_share))})
            if chart != 'steady':
                lookup = {(r['window_id'], r['rollout_step']): r for r in selected}
                comparisons = changed_pair = changed_group = 0
                l1 = []
                for r in records:
                    if r['channel'] == channel and r['stage'] > 1:
                        q = lookup[r['window_id'], r['rollout_step']]
                        comparisons += 1
                        changed_pair += pair(r) != pair(q)
                        changed_group += r['group_id'] != q['group_id']
                        l1.append(float(np.sum(np.abs(np.array(r['combined_gates']) - q['combined_gates']))))
                sensitivity.append({'chart': chart, 'channel': channel, 'stage2_3_4_vs_stage1_comparisons': comparisons,
                                    'pair_changed_fraction': changed_pair / comparisons,
                                    'group_changed_fraction': changed_group / comparisons,
                                    'mean_combined_gate_l1_change': float(np.mean(l1)),
                                    'stage2_3_4_used_in_own_channel_update': channel != 'p'})
    csv_write(out / 'stats/mass_concentration.csv', result)
    csv_write(out / 'stats/rk4_stage_sensitivity.csv', sensitivity)
    print(json.dumps({'pooled_mass_concentration': [r for r in result if r['Re'] == 'pooled'],
                      'rk4_stage_sensitivity': sensitivity}, indent=2))


if __name__ == '__main__':
    main()
