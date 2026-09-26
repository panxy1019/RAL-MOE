"""Warm Circular P physical-history to physical-field timing, Top-1 specialist only."""
import sys, types
from pathlib import Path
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
source=R/'experiments/strengthening_20260915/code/evaluate_periodic_dense.py'
sys.path.insert(0,str(source.parent));text=source.read_text()
cp=R/'experiments/round2_20260917/P_proposed/1248/Circular_P_proposed_round2_Re_70p314635_checkpoint.pt'
out=R/'experiments/round3_20260921/timing_circular_P_v2'
if out.exists():raise FileExistsError(out)
text=text.replace("    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)","    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)\n    ck['args']=json.loads((checkpoint_path.parent/'protocol.json').read_text())['config']")
text=text.replace('    for re in helper.HELDOUT_RE:', '    for re in helper.HELDOUT_RE[:1]:')
text=text.replace('        starts.extend(candidates[j] for j in chosen)','        starts.extend(candidates[j] for j in chosen[:1])')
addition='''    import platform, subprocess
    torch.set_num_threads(1)
    start=starts[0]
    input_ids=np.unique(np.concatenate([[start],arrays['hist_idx'][start][arrays['hist_idx'][start]>=0]]))
    phi_u=vel['phi_uv'][:args.r_u].astype('float64');phi_p=pre['phi_p'][:args.r_p].astype('float64')
    mu=vel['mean_uv_regime'].astype('float64');mp=pre['mean_p_regime'].astype('float64');area=vel['point_areas'].astype('float64')
    wu=np.concatenate([area,area])
    # Actual physical-size inputs reside in host memory before each timed call.
    # Reconstructed history has the identical grid and projection workload;
    # this is timing evidence, not a raw-FOM accuracy evaluation.
    history_u=arrays['a'][input_ids].astype('float64')@phi_u+mu
    history_p=arrays['b'][input_ids].astype('float64')@phi_p+mp
    projector_u=(phi_u*wu) .T @ np.linalg.inv((phi_u*wu)@phi_u.T)
    projector_p=(phi_p*area).T @ np.linalg.inv((phi_p*area)@phi_p.T)
    au=(history_u-mu)@projector_u;bp=(history_p-mp)@projector_p
    assert np.max(abs(au-arrays['a'][input_ids]))<1e-5
    assert np.max(abs(bp-arrays['b'][input_ids]))<1e-5
    end=start
    for _ in range(48):end=int(arrays['next_idx'][end])
    write(cli.output/'pairing.json',dict(Re=float(arrays['re'][start]),start=int(start),end=int(end),initial_time=float(arrays['time'][start]),final_time=float(arrays['time'][end]),duration=float(arrays['time'][end]-arrays['time'][start]),history_ids=input_ids.tolist(),history_frames=len(input_ids),physical_points=len(area),velocity_modes=args.r_u,pressure_modes=args.r_p,history_source='POD reconstructed full-size physical history; not raw FOM accuracy',cpu=platform.processor(),cpu_lscpu=subprocess.check_output(['lscpu'],text=True),gpu_before=subprocess.check_output(['nvidia-smi'],text=True),numpy_threads=1,torch_threads=1))
    native_rollout=rollout
    def rollout(start):
        arrays['a'][input_ids]=((history_u-mu)@projector_u).astype('float32')
        arrays['b'][input_ids]=((history_p-mp)@projector_p).astype('float32')
        arrays['rhs_g'][input_ids]=trainer.galerkin_rhs_by_label(tensors,arrays['a'][input_ids],arrays['b'][input_ids],arrays['label_id'][input_ids],arrays['labels'],args.r_u,args.r_p)
        result=native_rollout(start)
        if len(result)!=48:raise FloatingPointError('Incomplete prediction')
        u=np.asarray([r[0] for r in result])@phi_u+mu
        p=np.asarray([r[1] for r in result])@phi_p+mp
        p-=(p@area/area.sum())[:,None]
        assert np.isfinite(u).all() and np.isfinite(p).all()
        return result
'''
text=text.replace('    results = []\n',addition+'    results = []\n')
text=text.replace("includes='native modal rollout, feature building, host/device transfers, pressure reconstruction in modal coordinates'","includes='full-size physical history POD encoding, history Galerkin RHS, native rollout, features, transfers, algebraic pressure, all 48 physical velocity/pressure decodes and pressure gauge'")
text=text.replace("excludes='checkpoint/data loading, physical-field decoding, error computation'","excludes='checkpoint and input disk I/O, cold start, original history generation, E2 selection and T2-C fusion (specialist-only), output disk I/O, error computation'")
m=types.ModuleType('physical_benchmark');m.__file__=str(source);sys.modules[m.__name__]=m;exec(compile(text,str(source),'exec'),m.__dict__)
sys.argv=[sys.argv[0],'--method','full','--checkpoint',str(cp),'--output',str(out),'--benchmark'];m.main()
