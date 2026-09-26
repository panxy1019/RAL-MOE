"""Fetch four public held-out references; never used for fitting or selection."""
import json, urllib.request, hashlib
from pathlib import Path
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/experiments/qlrom_20260923/raw_reference')
R.mkdir(parents=True,exist_ok=True)
repo='panxy1019/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re'
manifest=[]
for label in ['70p314635','100p352251','149p059229','189p862278']:
    name=f'Re_{label}_uvp_pointData.npz';path=R/name
    url=f'https://huggingface.co/datasets/{repo}/resolve/main/{name}'
    try:
        if not path.exists():
            tmp=path.with_suffix('.download')
            with urllib.request.urlopen(url,timeout=45) as response,tmp.open('wb') as f:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    f.write(chunk)
            tmp.rename(path)
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        import numpy as np
        z=np.load(path)
        entry=dict(path=str(path),url=url,sha256=h,bytes=path.stat().st_size,
                   arrays={k:dict(shape=z[k].shape,dtype=str(z[k].dtype)) for k in z.files})
        print(json.dumps(entry),flush=True);manifest.append(entry)
    except Exception as e:
        manifest.append(dict(url=url,error=repr(e)));print('FETCH_FAILED',url,repr(e),flush=True)
(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
