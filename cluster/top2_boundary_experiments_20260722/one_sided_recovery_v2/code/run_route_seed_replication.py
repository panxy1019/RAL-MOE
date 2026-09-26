"""Run supplementary route-module seeds on the same frozen development cache."""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed-offset',type=int,required=True); p.add_argument('--cache',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--prereg-dir',type=Path,required=True); a=p.parse_args()
    source=Path(__file__).with_name('train_sh_routes.py'); spec=importlib.util.spec_from_file_location('seed_routes',source); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    module.METHODS={name:seed+a.seed_offset for name,seed in module.METHODS.items()}
    sys.argv=['train_sh_routes.py','--cache',str(a.cache),'--output-dir',str(a.output_dir),'--prereg-dir',str(a.prereg_dir)]
    module.main()

if __name__=='__main__': main()
