"""Re-execute train-only fit and held-out scores in a new immutable output directory."""
import types,sys
from pathlib import Path
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
p=R/'experiments/round2_20260917/code/reevaluate_fusion.py'
t=p.read_text();t=t.replace("ROOT/'experiments/round2_20260917/fusion_reevaluation'","ROOT/'experiments/round3_20260921/a05_refit'")
m=types.ModuleType('a05_round3');m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(t,str(p),'exec'),m.__dict__);m.main()
