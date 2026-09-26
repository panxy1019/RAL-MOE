"""Paper-only routing figures from existing verified macro-step statistics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def save(fig, root, name):
    fig.savefig(root / 'figures' / (name + '.pdf'), bbox_inches='tight', pad_inches=.035)
    fig.savefig(root / 'build/routing_preview' / (name + '.png'), dpi=180, bbox_inches='tight', pad_inches=.035)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    (root / 'build/routing_preview').mkdir(parents=True, exist_ok=True)
    inputs = {name: args.analysis_dir / 'stats' / f'{name}.csv'
              for name in ('group_usage', 'expert_routing_mass', 'rollout_step_mass')}
    groups, experts, steps = [read(inputs[n]) for n in inputs]
    group_rows = [r for r in groups if r['chart'] == 'periodic' and r['channel'] == 'u'
                  and r['population'] == 'macro_stage1' and r['Re'] != 'pooled']
    res = sorted(set(r['Re'] for r in group_rows), key=float)
    assert len(res) == 4
    labels = [f'{float(v):.2f}' for v in res]
    group_matrix = np.array([[next(float(r['frequency']) for r in group_rows if r['Re'] == re and int(r['group_id']) == g)
                              for g in range(3)] for re in res])
    assert np.allclose(group_matrix.sum(1), 1)
    matrices = {}
    for channel in ('u', 'p'):
        rows = [r for r in experts if r['chart'] == 'periodic' and r['channel'] == channel
                and r['population'] == 'macro_stage1' and r['Re'] != 'pooled']
        matrix = np.zeros((18, 4))
        for i, re in enumerate(res):
            for r in rows:
                if r['Re'] == re:
                    matrix[int(r['group_id']) * 6 + int(r['expert_id']), i] = float(r['mean_routing_mass'])
        assert np.allclose(matrix.sum(0), 3/7, atol=1.e-7)
        matrices[channel] = matrix
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 7.6,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(7.2, 2.65))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.16, 1, 1, .07],
                           left=.075, right=.94, bottom=.2, top=.90, wspace=.5)
    ax = fig.add_subplot(grid[0])
    bottom = np.zeros(4)
    for g, color in enumerate(('#327da8', '#e39b43', '#5d9c82')):
        ax.bar(range(4), group_matrix[:, g], bottom=bottom, width=.65, color=color, label=f'g{g}')
        bottom += group_matrix[:, g]
    ax.set_xticks(range(4), labels, rotation=35, ha='right')
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylim(0, 1.16)
    ax.set_xlabel('Held-out Re', labelpad=1)
    ax.set_ylabel('Top-1 group fraction', labelpad=2)
    ax.set_title('(a) Group selection', loc='left', pad=6)
    ax.legend(loc='upper center', ncol=3, frameon=False, fontsize=8,
              columnspacing=.7, handlelength=1.1, handletextpad=.3, borderaxespad=0)
    for col, channel in enumerate(('u', 'p'), 1):
        ax = fig.add_subplot(grid[col])
        im = ax.imshow(matrices[channel], aspect='auto', interpolation='nearest',
                       origin='upper', cmap='viridis', vmin=0, vmax=.20)
        ax.set_yticks(range(18), [f'g{i//6}:e{i%6}' for i in range(18)])
        ax.tick_params(axis='y', length=0, pad=2, labelsize=7)
        ax.set_xticks(range(4), labels, rotation=35, ha='right')
        ax.set_xlabel('Held-out Re', labelpad=1)
        ax.set_title('(b) Velocity' if channel == 'u' else '(c) Pressure', loc='left', pad=6)
        for y in (5.5, 11.5):
            ax.axhline(y, color='white', lw=.6, alpha=.8)
    cax = fig.add_subplot(grid[3])
    cb = fig.colorbar(im, cax=cax, ticks=[0, .05, .10, .15, .20])
    cb.set_label('Mean routing mass', labelpad=4)
    cb.ax.tick_params(labelsize=8, length=2)
    save(fig, root, 'periodic_internal_routing')

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, channel in zip(axes, ('u', 'p')):
        matrix = np.zeros((18, 48))
        for r in steps:
            if r['chart'] == 'periodic' and r['channel'] == channel:
                matrix[int(r['group_id']) * 6 + int(r['expert_id']), int(r['step']) - 1] = float(r['mean_routing_mass'])
        assert np.allclose(matrix.sum(0), 3/7, atol=1.e-7)
        im = ax.imshow(matrix, aspect='auto', interpolation='nearest', origin='upper',
                       cmap='viridis', vmin=0, vmax=.20, extent=(.5, 48.5, 17.5, -.5))
        ax.set_yticks(range(18), [f'g{i//6}:e{i%6}' for i in range(18)])
        ax.tick_params(axis='y', length=0, labelsize=7)
        ax.set_xticks([1,12,24,36,48], ['1/48','1/4','1/2','3/4','1'])
        ax.set_xlabel('Normalized rollout position k/48')
        ax.set_title('(a) Velocity' if channel == 'u' else '(b) Pressure', loc='left')
        for y in (5.5,11.5):
            ax.axhline(y,color='white',lw=.6,alpha=.8)
    fig.subplots_adjust(left=.075,right=.88,bottom=.17,top=.9,wspace=.28)
    cax=fig.add_axes([.91,.17,.018,.73])
    fig.colorbar(im,cax=cax,ticks=[0,.05,.10,.15,.20],label='Mean routing mass')
    save(fig,root,'periodic_rollout_step_routing')
    payload = {'population': 'macro_stage1', 'res': list(map(float,res)),
               'group_matrix': group_matrix.tolist(), 'velocity_mass': matrices['u'].tolist(),
               'pressure_mass': matrices['p'].tolist(),
               'source_sha256': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n,p in inputs.items()}}
    (root / 'build/routing_figure_data.json').write_text(json.dumps(payload,indent=2)+'\n')


if __name__ == '__main__':
    main()
