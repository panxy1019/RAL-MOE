"""Attach source-backed unit/seed metadata without modifying numerical arrays."""
from pathlib import Path
import json,hashlib
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
D=R/'experiments/field_redraw_20260922/data'
def record(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for p in D.glob('*.json'):
 m=json.loads(p.read_text());key=m['case_id']
 m['pressure_definition']='native kinematic pressure p (pressure divided by density), no density multiplication; weighted zero-mean gauge'
 m['nondimensional_scales']='native POD physical scale retained; no additional u/U or p/U^2 rescaling; independent case color scales, not a common nondimensional comparison'
 if key=='circular_p':
  m['unit_evidence']=[record(R/'periodic_specialist_r32/assets/provenance'/n) for n in ['velocity_rom_periodic.md','pressure_poisson_surrogate_periodic.md']]
  m['native_kinematic_viscosity']=.001
  m['scale_warning']='Circular source declares raw physical POD modes and fixed nu=0.001; do not label its arrays as sharing the Square/Pinball nondimensional scale.'
 if key=='square_sh':
  src=R/'centeredsquare_steady_specialist_v1/code/training_centeredsquare_steady_rank999.json'
  m['training_seed'][0]=json.loads(src.read_text())['seed'];m['training_seed_source']=record(src)
 if key.startswith('pinball'):
  src=R.parent/'Pinball/fluidicPinball_v2/data_npz/mesh/mesh_production.npz';m['geometry_coordinate_evidence']=record(src)
 p.write_text(json.dumps(m,indent=2,allow_nan=False))
