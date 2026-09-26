import csv
import numpy as np
from dense_periodic import ROOT,write
from evaluate_periodic_dense import load
work=ROOT/'experiments/strengthening_20260915'
rows=list(csv.DictReader((work/'periodic_global_interface_smoke_v2/window_step_errors.csv').open()))
helper=load('geometry_helper',ROOT/'periodic_specialist_r32/code/evaluate_periodic_r32_portable.py')
pod=ROOT/'V16_1_SteadyPressureAnchor32/assets/common_global_data/Global_POD_AreaWeighted_L2'
v=np.load(pod/'global_velocity_pod_area_weighted_l2.npz');p=np.load(pod/'global_pressure_pod_area_weighted_l2.npz')
cache=ROOT/'paper_experiments/revisions/revision10_missing_metric_completion_20260725/evaluations/global_periodic_k48_arrays_retry'
diff=[]
for path in sorted(cache.glob('*rollout_arrays.npz')):
    z=np.load(path);re=float(z['Re'][0]);ri=int(np.argmin(abs(v['Re_values']-re)))
    rs=[r for r in rows if abs(float(r['Re'])-re)<2e-5]
    starts=sorted(set(int(r['start']) for r in rs));assert len(starts)==len(z['times'])
    for j,start in enumerate(starts):
        rr=sorted([r for r in rs if int(r['start'])==start],key=lambda r:int(r['step']))
        for c,poddata,ph,mn,vec,metric in [('a',v,'phi_uv','mean_uv_by_Re',True,'Eu_percent'),('b',p,'phi_p','mean_p_by_Re',False,'Ep_percent')]:
            geo=helper.weighted_geometry(poddata[ph][:32],poddata[mn][ri],poddata['point_areas'],vec)
            expected=100*np.sqrt(np.maximum(helper.field_error_energy(z['true_'+c][j],z['pred_'+c][j],geo),0)/helper.field_energy(z['true_'+c][j],geo))
            actual=np.array([float(r[metric]) for r in rr])
            diff.append(float(np.max(abs(expected-actual))))
            np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=1e-4)
write(work/'GLOBAL_REPLAY_VERIFIED.json',dict(passed=True,max_difference_percentage_points=max(diff),
    check='fresh native Global rollout vs frozen cached native Eu/Ep, all 12 windows and all 48 steps',
    note='native pressure gauge used only for replay verification; final comparison separately centers pressure'))
print(max(diff))
