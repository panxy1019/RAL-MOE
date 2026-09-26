from pathlib import Path
import re,json,zipfile
R=Path(__file__).parent
S=Path(r'C:\Users\panxy1019\Desktop\STABLEMOE\PMD_Galerkin_Pan_ICLR (3)')
active=[]
def read(name):
    p=S/name
    if p.suffix!='.tex': p=p.with_suffix('.tex')
    active.append(str(p.relative_to(S)))
    # Ignore ordinary LaTeX comments, preserving escaped percent signs.
    t=re.sub(r'(?<!\\)%[^\n]*','',p.read_text(encoding='utf-8'))
    return re.sub(r'\\input\{([^}]+)\}',lambda m:read(m.group(1)),t)
t=read('main.tex')
labels=re.findall(r'\\label\{([^}]+)\}',t)
refs={x for x in re.findall(r'\\(?:ref|eqref|plaineqref)\{([^}]+)\}',t) if not x.startswith('#')}
keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)', '\n'.join(p.read_text(encoding='utf-8') for p in S.glob('*.bib'))))
cites={k.strip() for x in re.findall(r'\\cite\w*\{([^}]+)\}',t) for k in x.split(',')}
out={'missing_refs':sorted(refs-set(labels)),'duplicate_labels':sorted({x for x in labels if labels.count(x)>1}),
     'missing_cites':sorted(cites-keys),'circular_appendix_active':'appendix\\circular_cylinder.tex' in active,
     'gate_input_dimension':7,'gate_trainable_parameters':7*64+64+64*64+64+64+1,
     'compiled':False,'experiments_started':False,'changed_files':[]}
for p in S.rglob('*'):
    if p.is_file():
        old=R/'before_changes'/p.relative_to(S)
        if not old.exists() or old.read_bytes()!=p.read_bytes():out['changed_files'].append(str(p.relative_to(S)))
assert not out['missing_refs'] and not out['missing_cites'] and not out['duplicate_labels']
assert not out['circular_appendix_active']
print(json.dumps(out,ensure_ascii=False,indent=2))
(R/'static_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(R/'论文六维定义_PPE补充_精简附录_待实验更新.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in S.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(S))
