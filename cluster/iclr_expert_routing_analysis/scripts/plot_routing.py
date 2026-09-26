"""Static scientific figures from immutable CSV statistics; no model execution."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#2878a0', '#d88729', '#499a7c', '#ad608b']
NAMES = {'steady': 'Steady S4 (independent diagnostic)', 'hopf': 'Hopf H4', 'periodic': 'Periodic epoch 85'}


def read(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def save(fig, folder, name):
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(folder / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.analysis_dir
    dest = out / 'figures'
    dest.mkdir(exist_ok=True)
    ex = read(out / 'stats/circular_expert_utilization_table.csv')
    groups = read(out / 'stats/group_usage.csv')
    summary = json.loads((out / 'stats/summary.json').read_text())['primary_summary']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.8), sharey=True)
    for col, chart in enumerate(('steady', 'hopf', 'periodic')):
        for row, channel in enumerate(('u', 'p')):
            ax = axes[row, col]
            rows = [r for r in ex if r['chart'] == chart and r['channel'] == channel
                    and r['population'] == 'macro_stage1' and r['Re'] != 'pooled']
            res = sorted(set(r['Re'] for r in rows), key=float)
            n = 6 if chart == 'hopf' else 18
            width = .8 / len(res)
            for j, re_value in enumerate(res):
                values = np.zeros(n)
                for r in rows:
                    if r['Re'] == re_value:
                        values[int(r['group_id']) * 6 + int(r['expert_id'])] = float(r['mean_routing_mass'])
                ax.bar(np.arange(n) + (j - (len(res) - 1) / 2) * width, values,
                       width=width, color=COLORS[j], label=f'Re {float(re_value):.3f}', linewidth=0)
            ax.set_xticks(np.arange(n), [f'{i // 6}:{i % 6}' for i in range(n)], rotation=90 if n > 6 else 0)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_ylim(0, .48)
            ax.grid(axis='y', alpha=.18)
            ax.set_axisbelow(True)
            ax.set_xlabel('Chart-local group:expert (zero-based)')
            if col == 0:
                ax.set_ylabel(f"{'Velocity' if channel == 'u' else 'Pressure'}: mean routing mass")
            s = next(r for r in summary if r['chart'] == chart and r['channel'] == channel)
            ax.text(.98 if row == 1 else .02, .96, f"Fixed shared mass = {s['mean_shared_mass']:.4f}\n"
                    f"Observed pairs = {s['pair_configurations_used']}", transform=ax.transAxes,
                    ha='right' if row == 1 else 'left', va='top', fontsize=9,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
            if n > 6:
                ax.axvline(5.5, color='#dddddd', lw=.8)
                ax.axvline(11.5, color='#dddddd', lw=.8)
            if row == 0:
                gr = [r for r in groups if r['chart'] == chart and r['channel'] == 'u'
                      and r['population'] == 'macro_stage1' and r['Re'] == 'pooled']
                fractions = ' / '.join(f"g{r['group_id']}: {100 * float(r['frequency']):.1f}%" for r in gr)
                ax.set_title(NAMES[chart] + '\n' + fractions, fontsize=11, pad=10)
                ax.legend(loc='upper right', fontsize=8, frameon=False)
    fig.suptitle('Circular-cylinder internal routing on held-out rollout windows', fontsize=15, y=.99)
    fig.subplots_adjust(top=.85, bottom=.15, hspace=.55, wspace=.18)
    fig.text(.5, .025, 'One sample per macro-step (stage 1). All zero-use experts retained. '
             'Hopf has one group by design and fixed selected pairs.\n'
             'S4 is not the current paper Steady checkpoint. Shared mass is a fixed coefficient, not measured output contribution.',
             ha='center', fontsize=9, color='#444444')
    save(fig, dest, 'circular_internal_expert_routing')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels, chart_labels, fractions = [], [], []
    for chart in ('steady', 'hopf', 'periodic'):
        selected = [r for r in groups if r['chart'] == chart and r['channel'] == 'u'
                    and r['population'] == 'macro_stage1' and r['Re'] != 'pooled']
        for re_value in sorted(set(r['Re'] for r in selected), key=float):
            labels.append(f"{chart[0].upper()}\n{float(re_value):.2f}")
            fractions.append([sum(float(r['frequency']) for r in selected if r['Re'] == re_value and int(r['group_id']) == g) for g in range(3)])
    data = np.array(fractions)
    bottom = np.zeros(len(labels))
    for g in range(3):
        axes[0].bar(np.arange(len(labels)), data[:, g], bottom=bottom, color=COLORS[g], label=f'group {g}')
        bottom += data[:, g]
    axes[0].set_xticks(np.arange(len(labels)), labels, fontsize=8)
    axes[0].set_ylabel('Group Top-1 fraction')
    axes[0].set_ylim(0, 1.18)
    axes[0].legend(ncol=3, frameon=False, loc='upper center', fontsize=8)
    axes[0].set_title('Group selection by held-out Re')
    vals = [next(r for r in summary if r['chart'] == c and r['channel'] == 'u')['mean_shared_mass'] for c in ('steady', 'hopf', 'periodic')]
    axes[1].bar(np.arange(3), vals, color='#636c78', label='Shared coefficient')
    axes[1].bar(np.arange(3), 1 - np.array(vals), bottom=vals, color='#aec4d0', label='Total routed coefficient')
    for i, v in enumerate(vals):
        axes[1].text(i, v / 2, f'{v:.4f}', color='white', ha='center')
        axes[1].text(i, v + (1 - v) / 2, f'{1-v:.4f}', ha='center')
    axes[1].set_xticks(np.arange(3), ['Steady S4\nindependent', 'Hopf H4', 'Periodic'])
    axes[1].set_ylim(0, 1.18)
    axes[1].set_ylabel('Mixing coefficient (same for u and p)')
    axes[1].legend(ncol=2, frameon=False, loc='upper center', fontsize=8)
    axes[1].set_title('Shared/routed balance is fixed by configuration')
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.text(.5, .02, 'S = independent Steady S4 diagnosis; H = Hopf; P = Periodic. Hopf group 0 is the only available group.', ha='center', fontsize=9)
    save(fig, dest, 'group_usage_and_fixed_shared_mass')

    step_rows = read(out / 'stats/rollout_step_mass.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, channel in zip(axes, ('u', 'p')):
        selected = [r for r in step_rows if r['chart'] == 'periodic' and r['channel'] == channel]
        active = []
        for g in range(3):
            for e in range(6):
                sequence = sorted([r for r in selected if int(r['group_id']) == g and int(r['expert_id']) == e], key=lambda r: int(r['step']))
                values = [float(r['mean_routing_mass']) for r in sequence]
                if max(values) > 0:
                    active.append((g, e, values))
        matrix = np.array([v for _, _, v in active])
        im = ax.imshow(matrix, aspect='auto', origin='lower', vmin=0, vmax=.35, cmap='viridis',
                       extent=(.5, 48.5, -.5, len(active) - .5), interpolation='nearest')
        ax.set_yticks(range(len(active)), [f'{g}:{e}' for g, e, _ in active], fontsize=8)
        ax.set_xticks([1, 12, 24, 36, 48], ['1/48', '1/4', '1/2', '3/4', '1'])
        ax.set_xlabel('Normalized rollout step k / 48 (not physical phase)')
        ax.set_ylabel('Chart-local group:expert')
        ax.set_title('Periodic ' + ('velocity' if channel == 'u' else 'pressure'))
    fig.subplots_adjust(right=.88, bottom=.2, wspace=.25)
    cax = fig.add_axes([.9, .2, .018, .68])
    fig.colorbar(im, cax=cax, label='Mean routing mass')
    fig.text(.5, .035, 'Stage-1 mass averaged over all 52 fixed windows. Only nonzero rows shown; complete zero rows remain in CSV.\n'
             'Different window starting states are pooled; horizontal position is not vortex-shedding phase.', ha='center', fontsize=9)
    save(fig, dest, 'periodic_rollout_step_routing')


if __name__ == '__main__':
    main()
