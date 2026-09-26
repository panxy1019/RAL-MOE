"""Reuse the diagnostic renderer without recomputing or altering measurements."""
from pathlib import Path
import csv,json,numpy as np
O=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/experiments/round3_20260921/field_diagnostics')
def read(name):
 data=list(csv.DictReader((O/name).open()))
 for r in data:
  for k,v in r.items():
   if v in ['True','False']:r[k]=v=='True'
   else:
    try:r[k]=float(v)
    except ValueError:pass
 return data
def dump(name,data):(O/name).write_text(json.dumps(data,indent=2,allow_nan=False))
rows=read('field_curves.csv');phase_rows=read('phase_curves.csv')
source=Path('/tmp/field_diagnostics_round3.py').read_text()
exec(source.split('# Fixed-window, all-case diagnostic plots. No outcome-based frame selection.',1)[1])
