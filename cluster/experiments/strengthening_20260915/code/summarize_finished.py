"""Produce an objective terminal report, only after all four benchmarks finish."""
import csv
import json
from dense_periodic import ROOT,write
work=ROOT/'experiments/strengthening_20260915'
methods={'Dense correction':'final_periodic_dense_v1','RAL local specialist':'final_periodic_full_v1',
         'POD-Galerkin':'final_periodic_galerkin_v1','Global MoE':'global_common_truth_v1'}
rows={m:list(csv.DictReader((work/p/'window_step_errors.csv').open())) for m,p in methods.items()}
keys=[{(round(float(r['Re']),4),int(r['start']),int(r['step'])) for r in rs} for rs in rows.values()]
assert all(k==keys[0] for k in keys)
table=[]
for method,rs in rows.items():
    rr=[r for r in rs if int(r['step'])==48]
    table.append(dict(method=method,windows=len(rr),**{m:sum(float(r[m]) for r in rr)/len(rr)
        for m in ['Eu_percent','Ep_percent','Ejoint_percent']}))
write(work/'FINAL_PERIODIC_K48.json',table)
timing={m:json.loads((work/f'final_periodic_{m}_v1/runtime.json').read_text()) for m in ['dense','full','global','galerkin']}
write(work/'FINAL_NATIVE_RUNTIME.json',timing)
lines=['# Circular Periodic supplemental experiments — completed',
    '', 'Accuracy: 12 aligned windows, K48 terminal mean; common local POD-reconstructed physical truth; pressure area-mean removed. Not FOM snapshot error.',
    '', '| Method | Eu (%) | Ep (%) | Ejoint (%) |','|---|---:|---:|---:|']
for r in table:lines.append(f"| {r['method']} | {r['Eu_percent']:.6f} | {r['Ep_percent']:.6f} | {r['Ejoint_percent']:.6f} |")
lines += ['', '## Native implementation runtime', '',
    'Same host; neural correction on GPU, native NumPy Galerkin on CPU. Modal rollout and pressure reconstruction only; excludes field decoding and loading. This is not a full Top-2 fusion speed comparison.',
    '', '| Method | K48 median seconds | Peak GPU allocated MiB |','|---|---:|---:|']
for m,r in timing.items():lines.append(f"| {m} | {r['median_seconds']:.6f} | {r['peak_allocated_bytes']/2**20:.2f} |")
lines += ['', 'Scope: one Circular Periodic total-parameter-matched Dense seed. No claim of full multi-benchmark, multi-regime, multi-seed completion. Preserve unfavourable comparisons; do not conclude universal superiority.',
    '', 'See per-run protocol.json, architecture.json, checkpoint hashes and runtime.json for exact provenance.']
(work/'FINAL_PERIODIC_REPORT.md').write_text('\n'.join(lines)+'\n')
