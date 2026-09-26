"""Run the unmodified, sealed train/validation H4 trainer in isolated directories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
BASE = ROOT/'Hopf/migrated_h4_expanded'
OUT = ROOT/'experiments/round2_20260917'
SEEDS = [1248, 1600, 2026]

def save(name, value):
    path = OUT/name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((BASE/'runs/HopfExpanded34_H4_NormalFormRadial_r32/config.json').read_text())
    source = BASE/'code/train_h4_expanded.py'
    report = dict(seeds=SEEDS, original_config=cfg,
        trainer_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        protocol='Native H4, fresh physical-zero initialization; fixed train-only sampler and assets; only initialization/dropout seed varied; no heldout evaluation',
        comparison_scope='Proposed H arm only; Dense/structured arms use same seeds and data contract',
        smoke=args.smoke, runs=[])
    for seed in ([SEEDS[0]] if args.smoke else SEEDS):
        output = OUT/('smoke_H_native_stage' if args.smoke else 'H_proposed')/str(seed)
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')
        output.mkdir(parents=True)
        cmd = [sys.executable, str(source), '--variant','h4',
               '--experiment-name',cfg['experiment_name'], '--seed',str(seed),
               '--output-root',str(output),'--swanlab-mode','disabled']
        for key in ('baseline_trainer','coefficient_view','galerkin_path','pressure_path','asset_manifest','contract'):
            cmd += ['--'+key.replace('_','-'), str(BASE/cfg[key])]
        for key in ('micro_batch','grad_accum','max_steps','lr','weight_decay','grad_clip',
                    'eval_every','long_eval_every','validation_windows_per_re','early_stop_patience_evals'):
            cmd += ['--'+key.replace('_','-'), str(cfg[key])]
        # Native --smoke-only compresses the 8000-step curriculum to 8 updates,
        # which is not a valid stability check of a cold start at long horizons.
        # Test eight real stage-1 updates instead; retain all numerical safeguards.
        if args.smoke: cmd += ['--benchmark-steps','8','--benchmark-horizon','1']
        record = dict(seed=seed,command=cmd,start_unix=time.time(),status='running')
        report['runs'].append(record)
        save('H_smoke_status.json' if args.smoke else 'H_proposed_status.json',report)
        with (output/'training.log').open('w') as log:
            proc = subprocess.Popen(cmd,cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
            record['pid']=proc.pid
            save('H_smoke_status.json' if args.smoke else 'H_proposed_status.json',report)
            rc=proc.wait()
        record.update(returncode=rc,end_unix=time.time(),status='completed' if rc==0 else 'failed')
        save('H_smoke_status.json' if args.smoke else 'H_proposed_status.json',report)
        if rc: raise RuntimeError(f'Seed {seed} failed; see {output}')

if __name__ == '__main__': main()
