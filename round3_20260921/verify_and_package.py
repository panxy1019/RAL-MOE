from pathlib import Path
import csv,json,math,re,zipfile,hashlib,difflib
import pymupdf
D=Path(__file__).parent;paper=Path(r'C:\Users\panxy1019\Downloads\PMD_Galerkin_Pan_ICLR (4)');baseline=D.parent/'paper_batch1_v4_20260920/diff_build'
v={}
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['sections/introduction.tex','sections/related_work.tex']:
 assert (paper/name).read_bytes()==(baseline/name).read_bytes();v[name+'_unchanged']=True
extract=lambda s:re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',s,re.S).group(1)
assert extract((paper/'main.tex').read_text())==extract((baseline/'main.tex').read_text());v['abstract_unchanged']=True
old=json.loads((D.parent/'round2_20260917/fusion_reevaluation/macro_Re_scores.json').read_text());new=json.loads((D/'a05_refit/macro_Re_scores.json').read_text());assert old==new;v['A05_exact_replay']=True
rows=list(csv.DictReader((D/'field_diagnostics/field_curves.csv').open()));assert len(rows)==6144
for r in rows:
 for key in ['energy','energy_relative_error','momentum_rms','ppe_rms','divergence_rms','alpha']:assert math.isfinite(float(r[key]))
assert len({(r['pair'],r['window']) for r in rows})==32;v['field_rows']=len(rows);v['all_windows']=32
ids=list(csv.DictReader((D/'field_diagnostics/identity_checks.csv').open()));v['identity_max_errors']={k:max(float(r[k]) for r in ids) for k in ['energy_identity_max_abs','momentum_identity_max_abs','ppe_identity_max_abs']};assert max(v['identity_max_errors'].values())<1e-10
pr=list(csv.DictReader((D/'field_diagnostics/phase_curves.csv').open()));v['resolved_phase']={p:sum(r['pair']==p and r['step']=='1' and r['method']=='fusion' and r['phase_resolved']=='True' for r in pr) for p in ['sh','hp']};assert v['resolved_phase']=={'sh':0,'hp':15}
st=[r for r in json.loads((D/'H_status.json').read_text()) if not r['smoke']];v['H_FP32']={k:{s:sum(r['kind']==k and r['status']==s for r in st) for s in ['completed','failed']} for k in ['proposed','structured']};assert all(z=={'completed':2,'failed':1} for z in v['H_FP32'].values())
time=json.loads((D/'timing/paired_timing.json').read_text());assert time['fom_interval_seconds']==1996 and len(json.loads((D/'timing/runtime.json').read_text())['seconds'])==10;v['Q17_scope']='paired warm specialist only; complete Top-2 and equivalent I/O NOT completed'
log=(D/'main.log').read_text(errors='replace');assert not re.search(r'Overfull|undefined references|undefined citations|LaTeX Error',log);v['compile_no_overfull_or_undefined']=True
pdf=pymupdf.open(D/'main.pdf');v['paper_pages']=len(pdf)
for i in range(len(pdf)-5,len(pdf)):pdf[i].get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(D/f'qa_page_{i+1}.png')
diff=[]
for f in paper.rglob('*.tex'):
 b=baseline/f.relative_to(paper)
 if b.exists():diff.extend(difflib.unified_diff(b.read_text().splitlines(True),f.read_text().splitlines(True),fromfile='batch1/'+f.relative_to(paper).as_posix(),tofile='batch3/'+f.relative_to(paper).as_posix()))
(D/'third_batch_changes.diff').write_text(''.join(diff))
with zipfile.ZipFile(D/'Overleaf_after_batch3_20260921.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in paper.rglob('*'):
  if f.is_file():z.write(f,f.relative_to(paper))
with zipfile.ZipFile(D/'Overleaf_after_batch3_20260921.zip') as z:
 assert 'main.tex' in z.namelist()
 for f in paper.rglob('*'):
  if f.is_file():assert hashlib.sha256(z.read(f.relative_to(paper).as_posix())).hexdigest()==sh(f)
v['source_archive_verified']=True
(D/'verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
