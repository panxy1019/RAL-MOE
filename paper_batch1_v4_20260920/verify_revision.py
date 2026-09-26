from pathlib import Path
import hashlib, json, re, difflib
import pymupdf

OUT = Path(__file__).resolve().parent
ROOT = Path(r'C:\Users\panxy1019\Downloads\PMD_Galerkin_Pan_ICLR (4)')
ORIG = OUT / 'original'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
protected = {}
for name in ['sections/introduction.tex', 'sections/related_work.tex']:
    protected[name] = sha(ROOT/name) == sha(ORIG/name)
pat = rb'\\begin\{abstract\}.*?\\end\{abstract\}'
protected['abstract'] = re.search(pat,(ROOT/'main.tex').read_bytes(),re.S).group() == re.search(pat,(ORIG/'main.tex').read_bytes(),re.S).group()
assert all(protected.values()), protected
texts = {str(p.relative_to(ROOT)):p.read_text(encoding='utf-8') for p in ROOT.rglob('*.tex')}
combined = '\n'.join(texts.values())
labels = re.findall(r'\\label\{([^}]+)\}',combined)
refs = re.findall(r'\\(?:ref|eqref|plaineqref)\{([^}]+)\}',combined)
missing = sorted({x for x in refs if not x.startswith('#')}-set(labels))
duplicates = sorted({x for x in labels if labels.count(x)>1})
assert not missing, missing
assert not duplicates, duplicates
log = (OUT/'main.log').read_text(errors='replace')
assert not re.search(r'undefined|multiply defined|Overfull \\hbox|Overfull \\vbox|^!',log,re.M|re.I)
changes = [f for f in texts if not (ORIG/f).exists() or sha(ROOT/f)!=sha(ORIG/f)]
audit = Path(r'C:\Users\panxy1019\Documents\CHANNEL\round2_20260917')
numeric_checks = 0
for filename,tag in [('seed_summary_current.json','original'),('final_audit_evidence_20260920/seed_summary_fixed_fp32.json','repaired')]:
    for row in json.loads((audit/filename).read_text()):
        if row['regime']!='P' or row['method'] not in ['proposed','structured','dense']:
            continue
        target=texts['sections\\experiments.tex'] if row['K']==24 else texts['appendix\\circular_cylinder.tex']
        for metric in ['Eu','Ep','Ejoint']:
            mean=row[metric+'_percent_window_mean_mean']
            std=row[metric+'_percent_window_mean_sample_std']
            expected=f'{mean:.4f}\\pm{std:.4f}'
            assert expected in target, (tag,row['method'],row['K'],metric,expected)
            numeric_checks+=1
for filename in ['seed_summary_current.json','final_audit_evidence_20260920/seed_summary_fixed_fp32.json']:
    for row in json.loads((audit/filename).read_text()):
        if row['regime']=='H' and row['method'] in ['proposed','structured','dense']:
            mean=row['Ejoint_percent_window_mean_mean']
            std=row['Ejoint_percent_window_mean_sample_std']
            assert f'{mean:.5f}\\pm{std:.5f}' in texts['appendix\\circular_cylinder.tex']
            numeric_checks+=1
for row in json.loads((audit/'fusion_reevaluation/macro_Re_scores.json').read_text()):
    if row['K']==24 and row['method'] not in ['candidate_1','candidate_2']:
        for metric in ['Eu_window_mean','Ep_window_mean','Ejoint_window_mean']:
            assert f'{row[metric]:.4f}' in texts['appendix\\square_cylinder.tex']
            numeric_checks+=1
        assert f'{row["squared_objective"]*1e4:.4f}' in texts['appendix\\square_cylinder.tex']
        numeric_checks+=1
patch=[]
for path in sorted(ROOT.rglob('*')):
    if path.suffix not in ['.tex','.bib']:
        continue
    rel=path.relative_to(ROOT)
    before=(ORIG/rel).read_text(encoding='utf-8').splitlines(keepends=True) if (ORIG/rel).exists() else []
    after=path.read_text(encoding='utf-8').splitlines(keepends=True)
    patch.extend(difflib.unified_diff(before,after,fromfile='before/'+str(rel),tofile='after/'+str(rel)))
(OUT/'changes.diff').write_text(''.join(patch),encoding='utf-8')
doc = pymupdf.open(OUT/'main.pdf')
render = OUT/'render'
render.mkdir(exist_ok=True)
selected = {}
terms = ['Full-window joint errors', 'Circular Periodic controls', 'ordered raw descriptor', 'weak-form construction', 'Hopf accepted-only', 'Train-', 'Representative held-out prediction', 'Local native Circular']
for i,p in enumerate(doc):
    t=p.get_text()
    for term in terms:
        if term in t:
            selected.setdefault(i+1,[]).append(term)
    if i+1 in selected:
        p.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(render/f'page_{i+1:02}.png')
for start in range(0,len(doc),12):
    contact=pymupdf.open()
    sheet=contact.new_page(width=1000,height=1400)
    for i in range(start,min(start+12,len(doc))):
        x=((i-start)%3)*333+35
        y=((i-start)//3)*350
        sheet.show_pdf_page(pymupdf.Rect(x,y+15,x+257,y+347),doc,i)
        sheet.insert_text((x,y+12),str(i+1))
    sheet.get_pixmap().save(render/f'overview_{start+1:02}.png')
result = {'protected_unchanged':protected,'numeric_checks':numeric_checks,'changed_tex':changes,'pages':len(doc),'missing_refs':missing,'duplicate_labels':duplicates,'selected_pages':selected,'compile':'passed; no undefined references or overfull boxes'}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
