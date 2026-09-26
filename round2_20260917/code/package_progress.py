"""Export small evaluation evidence; never bundle large training checkpoints."""
from pathlib import Path
import zipfile
R=Path(__file__).resolve().parents[1]
folders=['fusion_reevaluation','fusion_seed_reevaluation','fusion_energy_deficit','H_evaluation','P_evaluation_v2',
 'OpInf_H_development_v2','OpInf_P_development','OpInf_H_heldout','OpInf_P_heldout']
with zipfile.ZipFile(R/'evaluation_evidence_20260918.zip','w',compression=zipfile.ZIP_DEFLATED) as out:
 for folder in folders:
  for path in sorted((R/folder).rglob('*')):
   if path.is_file():out.write(path,path.relative_to(R))
 for path in sorted(R.glob('*.json')):out.write(path,path.name)
print(R/'evaluation_evidence_20260918.zip')
