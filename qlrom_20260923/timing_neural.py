"""Common physical-history-to-fields timing, original neural checkpoint, seed1248."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,sys,types
from pathlib import Path
ROOT=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
R=ROOT/'experiments/round2_20260917'
sys.path.insert(0,str(R/'code'))
import periodic_controls

BLOCK='''
    import resource, platform
    start=976
    assert starts[0]==start
    ids=arrays['hist_idx'][start]
    phi_u=vel['phi_uv'][:32].astype(float);phi_p=pre['phi_p'][:32].astype(float)
    wu=np.r_[vel['point_areas'],vel['point_areas']].astype(float)
    wp=pre['point_areas'].astype(float)
    mu=vel['mean_uv_regime'].astype(float);mp=pre['mean_p_regime'].astype(float)
    phi_p-= (phi_p@wp/wp.sum())[:,None];mp-=mp@wp/wp.sum()
    Gu=(phi_u*wu)@phi_u.T;Gp=(phi_p*wp)@phi_p.T
    encoding_u=(phi_u*wu).T@np.linalg.inv(Gu)
    encoding_p=(phi_p*wp).T@np.linalg.inv(Gp)
    input_u=arrays['a'][ids].astype(float)@phi_u+mu
    input_p=arrays['b'][ids].astype(float)@phi_p+mp
    def end_to_end():
        aa=(input_u-mu)@encoding_u
        pin=input_p-(input_p@wp/wp.sum())[:,None]
        bb=(pin-mp)@encoding_p
        arrays['a'][ids]=aa.astype(np.float32);arrays['b'][ids]=bb.astype(np.float32)
        arrays['rhs_g'][ids]=trainer.galerkin_rhs_by_label(tensors,arrays['a'][ids],arrays['b'][ids],
            arrays['label_id'][ids],arrays['labels'],32,32)
        rr=rollout(start)
        pa=np.asarray([v[0] for v in rr]);pb=np.asarray([v[1] for v in rr])
        u=pa@phi_u+mu;p=pb@phi_p+mp;p-=(p@wp/wp.sum())[:,None]
        return u,p,pa,pb
    with torch.inference_mode():
        for _ in range(3): end_to_end()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        samples=[]
        for _ in range(10):
            torch.cuda.synchronize();t=time.perf_counter()
            u,p,pa,pb=end_to_end();torch.cuda.synchronize();samples.append(time.perf_counter()-t)
            assert len(pa)==48 and np.isfinite(u).all() and np.isfinite(p).all()
    cached=torch.load(audit_reference,weights_only=False)[0]['values']
    delta=max(float(np.max(np.abs(pa-np.array([v[0] for v in cached])))),
              float(np.max(np.abs(pb-np.array([v[1] for v in cached])))))
    write(cli.output/'timing.json',dict(method=audit_kind,seed=1248,start=start,Re=float(arrays['re'][start]),
        seconds=samples,median=float(np.median(samples)),iqr=float(np.percentile(samples,75)-np.percentile(samples,25)),
        warmups=3,repeats=10,batch=1,K=48,dtype='FP32 neural, FP64 physical encode/decode',
        cpu=platform.processor(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        intraop_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_rss_scope='whole process high-water mark, including loaded evaluation assets',
        scope='in-memory common POD-reconstructed physical history -> all decoded and gauge-fixed velocity/pressure fields; includes encoding, history RHS, rollout, transfers; excludes loading and disk IO',
        synchronization='CUDA synchronize before and after each repetition',max_modal_difference_from_cached=delta,
        checkpoint=str(checkpoint_path),checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()))
    return
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['proposed','dense','structured'],required=True)
    kind=parser.parse_args().kind
    output=ROOT/'experiments/qlrom_20260923/timing'/kind
    if output.exists():raise FileExistsError(output)
    cp=R/f'P_{kind}/1248/Circular_P_{kind}_round2_Re_70p314635_checkpoint.pt'
    source=ROOT/'experiments/strengthening_20260915/code/evaluate_periodic_dense.py'
    sys.path.insert(0,str(source.parent));text=source.read_text()
    old="    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)"
    text=text.replace(old,old+"\n    ck['args']=json.loads((checkpoint_path.parent/'protocol.json').read_text())['config']")
    text=text.replace("        patch(trainer, cli.output/'architecture.json')","        control_patch(trainer)")
    assert text.count('    results = []\n')==1
    text=text.replace('    results = []\n',BLOCK+'    results = []\n')
    m=types.ModuleType('qlrom_paired_timing');m.__file__=str(source)
    m.control_patch=lambda trainer:periodic_controls.patch(trainer,kind,output)
    m.audit_kind=kind;m.audit_reference=R/f'P_evaluation_v2/{kind}/1248/modal_rollouts.pt'
    sys.modules[m.__name__]=m;exec(compile(text,str(source),'exec'),m.__dict__)
    sys.argv=['timing','--method','full' if kind=='proposed' else 'dense','--checkpoint',str(cp),'--output',str(output)]
    m.main()

if __name__=='__main__':main()
