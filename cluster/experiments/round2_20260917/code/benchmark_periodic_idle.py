"""Isolated native-modal plus physical-decode cost. Does not claim full encoder timing."""
import argparse
import sys
import types
from pathlib import Path
import periodic_controls
R=Path(__file__).resolve().parents[1];ROOT=periodic_controls.ROOT
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['proposed','dense','structured'],required=True);a=p.parse_args()
 out=R/'P_isolated_runtime'/a.kind
 if out.exists():raise FileExistsError(out)
 cp=R/f'P_{a.kind}/1248'/f'Circular_P_{a.kind}_round2_Re_70p314635_checkpoint.pt'
 source=ROOT/'experiments/strengthening_20260915/code/evaluate_periodic_dense.py';sys.path.insert(0,str(source.parent));text=source.read_text()
 text=text.replace("    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)","    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)\n    ck['args']=json.loads((checkpoint_path.parent/'protocol.json').read_text())['config']")
 text=text.replace("        patch(trainer, cli.output/'architecture.json')","        control_patch(trainer)")
 anchor='    results = []\n'
 addition='''    native_rollout = rollout
    phi_u = vel['phi_uv'][:args.r_u].astype('float64')
    phi_p = pre['phi_p'][:args.r_p].astype('float64')
    mean_u = vel['mean_uv_regime'].astype('float64')
    mean_p = pre['mean_p_regime'].astype('float64')
    area = vel['point_areas'].astype('float64')
    def rollout(start):
        result = native_rollout(start)
        if result:
            decoded_u = np.asarray([v[0] for v in result]) @ phi_u + mean_u
            decoded_p = np.asarray([v[1] for v in result]) @ phi_p + mean_p
            decoded_p -= (decoded_p @ area / area.sum())[:,None]
            if not (np.isfinite(decoded_u).all() and np.isfinite(decoded_p).all()):
                raise FloatingPointError('Nonfinite physical decode')
        return result
'''
 text=text.replace(anchor,addition+anchor)
 text=text.replace("includes='native modal rollout, feature building, host/device transfers, pressure reconstruction in modal coordinates'","includes='native modal rollout, features, transfers, algebraic pressure, all-step physical decode and pressure gauge'")
 text=text.replace("excludes='checkpoint/data loading, physical-field decoding, error computation'","excludes='checkpoint/data loading, physical-history encoding, descriptor/fusion, error computation; CPU decoded arrays not included in GPU peak'")
 text=text.replace('    for re in helper.HELDOUT_RE:', '    for re in helper.HELDOUT_RE[:1]:')
 text=text.replace('        starts.extend(candidates[j] for j in chosen)','        starts.extend(candidates[j] for j in chosen[:1])')
 m=types.ModuleType('isolated_benchmark');m.__file__=str(source);m.control_patch=lambda trainer:periodic_controls.patch(trainer,a.kind,out);sys.modules[m.__name__]=m;exec(compile(text,str(source),'exec'),m.__dict__)
 sys.argv=[sys.argv[0],'--method','full' if a.kind=='proposed' else 'dense','--checkpoint',str(cp),'--output',str(out),'--benchmark'];m.main()
if __name__=='__main__':main()
