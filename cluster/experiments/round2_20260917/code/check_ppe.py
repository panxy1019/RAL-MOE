import json
from pathlib import Path
import numpy as np
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
z=np.load(R/'periodic_specialist_r32/assets/pressure_poisson_surrogate_periodic.npz')
np.testing.assert_allclose(z['L_pinv'],np.linalg.pinv(z['L'],rcond=1e-10),rtol=1e-9,atol=1e-10)
np.testing.assert_allclose(z['H_tilde'],np.einsum('ij,jkl->ikl',z['L_pinv'],z['H_p']),rtol=1e-9,atol=1e-10)
rng=np.random.default_rng(20260917);maximum=0.
for label in z['Re_labels_computed']:
 c,A=z[str(label)+'_c_p'],z[str(label)+'_A_p']
 ct,At=z[str(label)+'_c_tilde'],z[str(label)+'_A_tilde']
 np.testing.assert_allclose(ct,z['L_pinv']@c,rtol=1e-9,atol=1e-10)
 np.testing.assert_allclose(At,z['L_pinv']@A,rtol=1e-9,atol=1e-10)
 a=rng.normal(size=32)
 direct=z['L_pinv']@(c+A@a+np.einsum('ijk,j,k->i',z['H_p'],a,a))
 effective=ct+At@a+np.einsum('ijk,j,k->i',z['H_tilde'],a,a)
 np.testing.assert_allclose(direct,effective,rtol=1e-9,atol=1e-10)
 maximum=max(maximum,float(np.max(np.abs(direct-effective))))
result=dict(passed=True,parameter_nodes=len(z['Re_labels_computed']),max_absolute_replay_difference=maximum,
 scope='saved weak-form coefficients to effective online polynomial; does not reassemble mesh derivatives',metadata=json.loads(str(z['metadata_json'])))
(R/'experiments/round2_20260917/ppe_replay_test.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
