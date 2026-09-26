import hashlib
import json
import torch
from dense_periodic import ROOT,write
work=ROOT/'experiments/strengthening_20260915'
results={}
for name,path in [('dense',work/'dense_periodic_v1/best_validation.pt'),
                  ('full',ROOT/'periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt')]:
    ck=torch.load(path,map_location='cpu',weights_only=False)
    st=ck['model_state']
    n=sum(v.numel() for v in st.values())
    result=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        selected_epoch=ck.get('best_epoch',ck.get('epoch')),best_val_score=ck.get('best_val_score'),
        state_tensor_elements=n,finite_weights=all(torch.isfinite(v).all().item() for v in st.values()))
    if name=='full':
        def count(prefix):return sum(v.numel() for k,v in st.items() if k.startswith(prefix))
        groups=['velocity_expert_groups.','pressure_expert_groups.','velocity_shared_experts.','pressure_shared_experts.']
        common=n-sum(count(g) for g in groups)
        active=common+count('velocity_shared_experts.0.')+count('pressure_shared_experts.0.')+2*count('velocity_expert_groups.0.0.')+2*count('pressure_expert_groups.0.0.')
        result.update(common_parameter_elements=common,inference_active_parameter_estimate=active,
            active_definition='one selected regime group; one shared plus two routed experts for each velocity/pressure channel; all common/router parameters counted; no training diversity stacks')
    else:
        result['inference_active_parameter_estimate']=n
        assert not any('expert_groups' in k or 'group_router' in k for k in st)
    results[name]=result
write(work/'FINAL_CHECKPOINT_AUDIT.json',results)
print(json.dumps(results,indent=2))
