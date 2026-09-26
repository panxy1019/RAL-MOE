from pathlib import Path
import re, json, hashlib, shutil
import pymupdf

ROOT=Path(__file__).parent
SRC=Path(r'C:\Users\panxy1019\Desktop\STABLEMOE\PMD_Galerkin_Pan_ICLR (3)')
# Mechanical removal of an unresolved redundant citation; preserve the supported statement.
p=SRC/'sections/related_work.tex'
s=p.read_text(encoding='utf-8').replace('diaz2024,bhat2025','diaz2024')
p.write_text(s,encoding='utf-8')

def flatten(root, name='main.tex'):
    s=(root/name).read_text(encoding='utf-8')
    def sub(m):
        q=m.group(1)
        if not q.endswith('.tex'): q+='.tex'
        return flatten(root,q)
    return re.sub(r'\\input\{([^}]+)\}',sub,s)

build=ROOT/'diff_build'
build.mkdir(exist_ok=True)
for p in SRC.iterdir():
    if p.is_dir(): shutil.copytree(p,build/p.name,dirs_exist_ok=True)
    else: shutil.copy2(p,build/p.name)
(ROOT/'old_flat.tex').write_text(flatten(ROOT/'original'),encoding='utf-8')
(ROOT/'new_flat.tex').write_text(flatten(SRC),encoding='utf-8')

tex=flatten(SRC)
bib='\n'.join(p.read_text(encoding='utf-8') for p in SRC.glob('*.bib'))
keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',bib))
cites={k.strip() for c in re.findall(r'\\cite\w*\s*\{([^}]+)\}',tex) for k in c.split(',')}
labels=re.findall(r'\\label\{([^}]+)\}',tex)
refs={r for r in re.findall(r'\\(?:ref|eqref|plaineqref)\{([^}]+)\}',tex) if not r.startswith('#')}
audit={'missing_citations':sorted(cites-keys),'missing_references':sorted(refs-set(labels)),
       'duplicate_labels':sorted({k for k in labels if labels.count(k)>1}),
       'changed_files':[]}
for p in SRC.rglob('*'):
    if p.is_file():
        old=ROOT/'original'/p.relative_to(SRC)
        if not old.exists() or p.read_bytes()!=old.read_bytes():
            audit['changed_files'].append(str(p.relative_to(SRC)))
(ROOT/'source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))

qa=ROOT/'qa'; qa.mkdir(exist_ok=True)
doc=pymupdf.open(Path(r'C:\Users\panxy1019\Downloads\ICLR.pdf'))
print('Original PDF pages:',len(doc))
for n in (11,15,16):
    doc[n].get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(qa/f'original_{n+1}.png')
