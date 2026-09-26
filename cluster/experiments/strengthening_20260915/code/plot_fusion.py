from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parents[1] / 'constant_alpha_v1'
out = base / 'figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
rows = list(csv.DictReader((base/'router_fusion_windows.csv').open()))
fig, axes = plt.subplots(1, 2, figsize=(8, 2.9), constrained_layout=True)
for ax, overlap in zip(axes, ['SH', 'HP']):
    subset = [r for r in rows if r['overlap'] == overlap]
    for split, marker in [('train', 'o'), ('validation', 's'), ('test', '^')]:
        rs = [r for r in subset if r['split'] == split]
        if not rs:
            continue
        ax.scatter([float(r['Re']) for r in rs], [float(r['T2C_alpha']) for r in rs],
                   s=16, alpha=.6, marker=marker, label=f'T2-C ({split})', color='#0072B2')
    re = sorted(set(float(r['Re']) for r in subset))
    # Show evaluated points, not a fabricated dense Re sweep.
    e2 = [np.mean([float(r['E2_pair_probability']) for r in subset if float(r['Re']) == x]) for x in re]
    ax.scatter(re, e2, marker='x', color='#D55E00', s=28, label='E2 pair probability')
    ax.axhline(float(subset[0]['constant_alpha']), color='#009E73', ls='--', label='Train-fitted constant')
    ax.set(xlabel='Re', ylabel='Weight of S' if overlap == 'SH' else 'Weight of P', ylim=(-.04, 1.04),
           title='S–H overlap' if overlap == 'SH' else 'H–P overlap')
axes[0].legend(fontsize=7, loc='best')
fig.savefig(out/'router_fusion_weights.pdf')
fig.savefig(out/'router_fusion_weights.png', dpi=220)
plt.close(fig)
rows = list(csv.DictReader((base/'fusion_error_curves.csv').open()))
fig, axes = plt.subplots(2, 3, figsize=(10, 5), constrained_layout=True)
methods = ['Equal', 'E2', 'Constant-alpha_train', 'T2C', 'Oracle']
available = sorted(set(r['method'] for r in rows))
print('Available methods:', available)
colors = ['#999999', '#E69F00', '#009E73', '#0072B2', '#CC79A7']
for row, overlap in enumerate(['SH', 'HP']):
    for col, metric in enumerate(['Eu_percent', 'Ep_percent', 'Ejoint_percent']):
        ax = axes[row, col]
        for method, color in zip(methods, colors):
            rs = sorted([r for r in rows if r['overlap'] == overlap and r['method'] == method], key=lambda r: int(r['step']))
            if not rs:
                continue
            ax.plot([int(r['step']) for r in rs], [float(r[metric]) for r in rs],
                    label=method.replace('_train', ''), color=color, lw=1.5,
                    ls=':' if method == 'Oracle' else '-')
        ax.set(xlabel='Rollout step k', ylabel=metric.replace('_percent', '')+' (%)', title=overlap)
axes[0, 0].legend(fontsize=7)
fig.savefig(out/'fusion_error_growth.pdf')
fig.savefig(out/'fusion_error_growth.png', dpi=220)
