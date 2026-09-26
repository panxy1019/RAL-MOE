"""Assemble machine-readable evidence and invariant checks; never edit paper sources."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1];R=P.parent;O=P/'build/network_architecture_audit_20260912'
read=lambda n:json.loads((O/n).read_text(encoding='utf-8'))
local=read('local_checkpoints.json')['local'];remote=read('remote_checkpoints.json');models=read('instantiated_models.json');counts=read('parameter_counts.json');pinball=read('pinball_candidates.json')
checks={}
checks['original_gates_input9']=all(local['square_'+b+'_original']['input_dim']==9 for b in ('sh','hp'))
checks['all_14_gates_4865']=all(v['total_trainable']==4865 for k,v in local.items() if k!='square_e2')
checks['E2_input1_count123']=local['square_e2']['input_dim']==1 and local['square_e2']['total_trainable']==123
for boundary in ('sh','hp'):
 for seed in (42001,42002,42003):
  full=local[f'seed_study/{boundary}/full/seed_{seed}'];mu=local[f'seed_study/{boundary}/mu_only/seed_{seed}']
  checks[f'{boundary}_{seed}_same_shapes']=full['state_dict']==mu['state_dict']
  cfg1,cfg2=full['resolved_config'],mu['resolved_config']
  keys=['batch_size','steps','seed','lr','weight_decay','optimizer','gradient_clip_norm','eval_every','selection','cache_sha256','router_checkpoint_sha256','input_dim','parameter_count']
  checks[f'{boundary}_{seed}_matched_training']=all(cfg1[k]==cfg2[k] for k in keys)
  checks[f'{boundary}_{seed}_mask_after_standardization']=cfg2['mask_after_standardization'] and not cfg1['mask_after_standardization']
for chart,total,active in [('hopf',31810392,14165816),('periodic',48617547,8308959)]:
 checks[chart+'_counts']=models[chart]['total_trainable']==counts['models'][chart]['total_trainable']==total and counts['models'][chart]['prediction_active_trainable']==active
 checks[chart+'_all_supports']=counts['models'][chart]['active_count_invariant']
old=json.loads((P/'build/protocol_polish_20260912/upload_manifest.json').read_text())
checks['all_31_paper_files_unchanged']=all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in old.items())
expected={
 'fusion_source_snapshot/CenteredSquare_Hopf_H4_20260728/code/train_centeredsquare_h4.py':'cf038f75a21635673d58ecdd7d9360faacf0af9536a35924e2049d00f844e967',
 'fusion_source_snapshot/centered_square_periodic_v1/code/train_square_periodic_moe_optimized.py':'85142d79bc818b353696e19e1f1d569e1b43f7ec10ae5ba9b8ffcf69b67bd48a',
 'fusion_source_snapshot/centeredsquare_steady_specialist_v1/code/train_s2b_3090.py':'63d72a8d596f180b2bbbbc90614f24e914a11837a372bde79c8642aaa934896e'}
for k,v in expected.items(): checks['live_source_hash_'+Path(k).name]=hashlib.sha256((R/k).read_bytes()).hexdigest()==v
for name,row in models.items(): checks['all_state_shapes_'+name]=row['shape_check']=='PASS'
unresolved={
 'circular_steady':'UNRESOLVED: available S4 checkpoint not linked to Table 4 Steady row; excluded from confirmed table.',
 'circular_vanilla_fnn_moe':'UNRESOLVED: source launcher found, frozen per-chart final checkpoint/config mapping absent.',
 'circular_dataonly_moe':'UNRESOLVED: candidate source differs from a structured-expert-only physics removal; no final Table 4 checkpoint mapping.',
 'global_moe':'UNRESOLVED: final checkpoint/rank/architecture/active counts not established.',
 'circular_E2':'UNRESOLVED: frozen E2 identity and complete configuration not audited in this run.',
 'pinball_E2_T2C':'Formal Pinball outer checkpoint shapes independently recovered (1-24-3 E2, 9-64-64-1 gates); UNRESOLVED: complete Appendix H run-to-row mapping and actual batch settings. Do not extrapolate Square training.',
 'all_training_completed_steps':'Budget and selected checkpoint epoch/step are reported separately; full executed duration is UNRESOLVED when no terminal log was verified.',
 'square_active_counts':'UNRESOLVED: total trainable was recalculated; full deployed physical rollout active count not independently traced for these three models.',
 'pinball_hopf_publication_mapping':'UNRESOLVED: found validation record is not Table 2 test result; no speculative replacement.',
 'pinball_steady_publication_mapping':'UNRESOLVED in current structured evidence bundle; B1 training report was a lead, not a full output/hash chain.'}
resolved={'date':'2026-09-12','scope':'Audit only; paper sources unchanged; no training','confirmed_t2c':{'input_dim':9,'descriptor_count':8,'parameter_inputs':1,'hidden_dims':[64,64],'output_dim':1,'activation':'SiLU','total_trainable':4865,'first_weight_key':'gate_state.correction.net.0.weight','first_weight_shape':[64,9],'paper_status':'PAPER-CODE MISMATCH','descriptor_order':['0.5 log(current velocity energy / area)','0.5 log(current gauge-pressure energy / area)','log1p(current velocity increment rate / current velocity norm)','log1p(previous velocity increment rate / previous velocity norm)','tanh(current log velocity-energy growth)','tanh(current minus previous growth)','log1p(current pressure increment rate / current pressure norm)','log1p(norm of difference between velocity rates / current velocity norm)']},'local_checkpoints':local,'remote_checkpoints':remote,'instantiated_models':models,'circular_active_count_recheck':counts,'pinball_candidates':pinball,'pinball_instantiated':read('pinball_instantiated.json'),'live_source_hashes':expected,'unresolved':unresolved,'verification':checks}
(P/'resolved_architecture.json').write_text(json.dumps(resolved,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
assert all(checks.values()),checks
(O/'verification.json').write_text(json.dumps({'status':'PASS','checks':checks,'unresolved_are_not_passes':unresolved},indent=2)+'\n',encoding='utf-8')
print('PASS',len(checks),'checks; paper unchanged; unresolved scopes explicitly separated')
