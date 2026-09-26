"""Evaluate selected P checkpoints, retaining native modal trajectories for FOM audit."""
import argparse
from pathlib import Path
import sys
import types
import periodic_controls

R=Path(__file__).resolve().parents[1]
ROOT=periodic_controls.ROOT
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['proposed','dense','structured'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--quadratic-fixed',action='store_true');a=p.parse_args()
 out=R/('P_evaluation_fixed' if a.quadratic_fixed else 'P_evaluation_v2')/a.kind/str(a.seed)
 if out.exists():raise FileExistsError(out)
 checkpoint=R/f'P_{a.kind}'/str(a.seed)/f'Circular_P_{a.kind}_round2_Re_70p314635_checkpoint.pt'
 if a.quadratic_fixed:checkpoint=R/'quadratic_fixed/P'/a.kind/str(a.seed)/checkpoint.name
 source=ROOT/'experiments/strengthening_20260915/code/evaluate_periodic_dense.py'
 sys.path.insert(0,str(source.parent))
 text=source.read_text()
 text=text.replace("    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)","    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)\n    ck['args'] = json.loads((checkpoint_path.parent/'protocol.json').read_text())['config']")
 text=text.replace("        patch(trainer, cli.output/'architecture.json')","        control_patch(trainer)")
 text=text.replace('    results = []\n','    results = []\n    modal_records = []\n')
 text=text.replace('            result = rollout(start)\n','            result = rollout(start)\n            modal_records.append(dict(start=start,Re=float(arrays["re"][start]),initial_time=float(arrays["time"][start]),values=result))\n')
 text=text.replace("    write(cli.output/'COMPLETED.json',{'completed':True})", "    torch.save(modal_records,cli.output/'modal_rollouts.pt')\n    write(cli.output/'COMPLETED.json',{'completed':True})")
 m=types.ModuleType('periodic_round2_evaluation');m.__file__=str(source);sys.modules[m.__name__]=m
 m.control_patch=lambda trainer:periodic_controls.patch(trainer,a.kind,out)
 exec(compile(text,str(source),'exec'),m.__dict__)
 sys.argv=[sys.argv[0],'--method','full' if a.kind=='proposed' else 'dense','--checkpoint',str(checkpoint),'--output',str(out)]
 m.main()
if __name__=='__main__':main()
