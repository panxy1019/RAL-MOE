from pathlib import Path
import json, hashlib, shutil, re, difflib, zipfile

ROOT = Path(__file__).parent
SRC = Path(r'C:\Users\panxy1019\Desktop\STABLEMOE\PMD_Galerkin_Pan_ICLR (3)')
PLOT = ROOT.parent / 'centeredsquare_fusion_v1'
OLD = PLOT / 'results/E2_T2C_K24_20260730_STRICT_V3/paper_boundary_figures_20260730_V1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checks = {}
dest = SRC / 'figures/verified_20260920'
dest.mkdir(exist_ok=True)
for boundary in ['SH','HP']:
    old = json.loads((OLD/boundary/'FIELD_FIGURE_MANIFEST.json').read_text())
    new = json.loads((ROOT/boundary/'FIELD_FIGURE_MANIFEST.json').read_text())
    assert old['selected'] == new['selected'], boundary
    for key in ['bundle_sha256','template_vtk_sha256','weights_csv_sha256']:
        assert old['inputs'][key] == new['inputs'][key], (boundary,key)
    assert old['inputs']['bases'] == new['inputs']['bases'], boundary
    for kind, suffix in [('fields','physical_fields'),('errors','pointwise_errors')]:
        image = ROOT/boundary/f'CenteredSquare_{boundary}_boundary_T2C_{suffix}_ultra_compact_wide.png'
        record = next(x for x in new['outputs'] if Path(x['path']).name == image.name)
        assert sha(image) == record['sha256'], image
        shutil.copy2(image, dest/f'{boundary}_{kind}.png')
    checks[boundary] = {'selected_and_metrics_unchanged':True,'cache_weights_mesh_bases_hashes_unchanged':True,
        'output_hashes_verified':True,'selected':new['selected']}

def flatten(root, name='main.tex'):
    text = (root/name).read_text(encoding='utf-8')
    def sub(m):
        filename = m.group(1)
        return flatten(root, filename if filename.endswith('.tex') else filename+'.tex')
    return re.sub(r'\\input\{([^}]+)\}',sub,text)

text = flatten(SRC)
keys = set(re.findall(r'@\w+\s*\{\s*([^,]+)', '\n'.join(p.read_text(encoding='utf-8') for p in SRC.glob('*.bib'))))
cites = {k.strip() for c in re.findall(r'\\cite\w*\s*\{([^}]+)\}',text) for k in c.split(',')}
labels = re.findall(r'\\label\{([^}]+)\}',text)
refs = {r for r in re.findall(r'\\(?:ref|eqref|plaineqref)\{([^}]+)\}',text) if not r.startswith('#')}
missing_images = [s for s in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text) if not (SRC/s).is_file()]
audit = {'figures':checks,'missing_citations':sorted(cites-keys),'missing_references':sorted(refs-set(labels)),
    'duplicate_labels': sorted(k for k in set(labels) if labels.count(k)>1),'missing_images':missing_images,'changed_files':[]}
patches = []
for path in SRC.rglob('*'):
    if not path.is_file(): continue
    rel = path.relative_to(SRC)
    before = ROOT/'paper_before'/rel
    if not before.exists() or sha(before) != sha(path):
        audit['changed_files'].append(str(rel))
        if path.suffix == '.tex':
            patches.extend(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(True) if before.exists() else [],
                path.read_text(encoding='utf-8').splitlines(True),fromfile='before/'+str(rel),tofile='after/'+str(rel)))
(ROOT/'changes.patch').write_text(''.join(patches),encoding='utf-8')
(ROOT/'verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
assert not any(audit[k] for k in ['missing_citations','missing_references','duplicate_labels','missing_images']), audit
(ROOT/'old_flat.tex').write_text(flatten(ROOT/'paper_before'),encoding='utf-8')
(ROOT/'new_flat.tex').write_text(text,encoding='utf-8')
shutil.copytree(SRC,ROOT/'diff_build',dirs_exist_ok=True)
with zipfile.ZipFile(ROOT/'RAL_MoE_ROM_review_source_20260920.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in SRC.rglob('*'):
        if path.is_file() and path.suffix.lower() not in {'.aux','.log','.out','.synctex','.gz'}:
            if path.suffix.lower()=='.pdf' and 'figures' not in path.parts: continue
            z.write(path,path.relative_to(SRC))
print(json.dumps(audit,ensure_ascii=False,indent=2))
