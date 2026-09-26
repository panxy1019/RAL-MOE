"""Package the portable plotting inputs and outputs, excluding installed libraries."""
from pathlib import Path
import hashlib,json,zipfile
R=Path(__file__).resolve().parents[1]
files=[p for p in R.rglob('*') if p.is_file() and not any(s in p.relative_to(R).parts for s in ['.runtime','__pycache__','preview_pages']) and p.suffix not in ['.zip','.log']]
inventory={str(p.relative_to(R)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='FILE_SHA256.json'}
(R/'FILE_SHA256.json').write_text(json.dumps(inventory,indent=2))
if R/'FILE_SHA256.json' not in files:files.append(R/'FILE_SHA256.json')
target=R.parent/(R.name+'_portable.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,Path(R.name)/p.relative_to(R))
with zipfile.ZipFile(target) as z:assert z.testzip() is None
print(target, target.stat().st_size,'bytes',len(files),'files')
