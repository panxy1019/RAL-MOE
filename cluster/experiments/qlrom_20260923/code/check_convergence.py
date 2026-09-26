"""Validation-only numerical refinement diagnostic, run before held-out test."""
import numpy as np
import json,time
from qlrom_experiment import OUT,SEEDS,load_data,windows,load_model,evaluate,aggregate,write,sha
d=load_data();val,_=windows(d);s=json.loads((OUT/'selected_config.json').read_text())
allrows=[]
for seed in SEEDS:
    m=load_model(s['J'],s['r'],seed);previous=None
    for sub in [1,2,4,8,16]:
        rows,res=evaluate(d,m,val,sub);item=dict(seed=seed,substeps=sub,**aggregate(rows,48))
        if previous:
            delta=np.concatenate([res[k]['pred_a']-previous[k]['pred_a'] for k in res])
            item['max_velocity_weighted_change_from_half_substeps']=float(np.max(np.linalg.norm(delta@d['S'].T,axis=1)))
        allrows.append(item);previous=res
write(OUT/'time_refinement_validation.json',dict(test_seen=False,selected_sha256=sha(OUT/'selected_config.json'),
    created_at=time.strftime('%FT%T%z'),rows=allrows))
print(json.dumps(allrows,indent=2),flush=True)
