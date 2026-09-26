"""Match one frozen ROM window to production FOM solver clock records."""
from pathlib import Path
import json,re,hashlib,statistics
D=Path(__file__).parent/'timing';p=json.loads((D/'pairing.json').read_text());r=json.loads((D/'runtime.json').read_text());path=D/'log.retain.pimpleFoam';text=path.read_text(errors='replace')
records=[];physical=None
for line in text.splitlines():
 m=re.match(r'^Time = ([0-9.eE+-]+)s?$',line)
 if m:physical=float(m[1])
 m=re.search(r'ExecutionTime = ([0-9.eE+-]+) s\s+ClockTime = ([0-9.eE+-]+) s',line)
 if m and physical is not None:records.append((physical,float(m[1]),float(m[2])))
assert records and 'End' in text
chosen=[min(range(len(records)),key=lambda j:abs(records[j][0]-p[k])) for k in ['initial_time','final_time']]
a,b=[records[j] for j in chosen];errors=[a[0]-p['initial_time'],b[0]-p['final_time']]
assert max(abs(x) for x in errors)<.001,errors
fom=b[2]-a[2];lat=r['seconds'];out={
 'source_url':'https://huggingface.co/datasets/panxy1019/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/resolve/main/Re_70p314635_logs/log.retain.pimpleFoam',
 'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'solver':'OpenFOAM 13 incompressibleFluid','fom_original_host':'ray-virtual-machine','fom_hardware_evidence':'2026-09-20 inventory in Circular_CFD运行时间核查.md, not contemporaneous benchmark; current VM no longer exposes original path','fom_cpu':'Intel Core i5-14600K; 16-vCPU VMware; one solver process; at most five concurrent cases (production setting)',
 'Re':p['Re'],'rom_initial_time':p['initial_time'],'rom_final_time':p['final_time'],'fom_initial_time':a[0],'fom_final_time':b[0],'boundary_alignment_error':errors,'fom_steps':chosen[1]-chosen[0],
 'fom_clock_initial':a[2],'fom_clock_final':b[2],'fom_interval_seconds':fom,'fom_repeats':1,'fom_warmup':'retained production trajectory after initialization; no repeated timing experiment',
 'rom_median_seconds':statistics.median(lat),'rom_mean_seconds':statistics.mean(lat),'rom_sd_seconds':statistics.stdev(lat),'rom_min_seconds':min(lat),'rom_max_seconds':max(lat),'rom_warmups':3,'rom_repeats':10,
 'production_FOM_to_warm_specialist_ratio':fom/statistics.median(lat),
 'scope':'matched physical interval; production FOM to warm physical-history/physical-output specialist; NOT complete E2/Top-2 framework speedup',
 'io':'FOM includes native solver field writes within interval; excludes startup/mesh/detect/VTK/NPZ. ROM input histories/checkpoint already resident, no disk writes. Not equal-I/O compute-only comparison.',
 'reference_history':'POD reconstructed full-size physical histories, same encoding workload; excludes initial history acquisition',
 'remaining_Q17':'complete E2/Top-2 and matched-output-I/O timing are unmeasured; do not extrapolate ratio to them'
}
(D/'paired_timing.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
