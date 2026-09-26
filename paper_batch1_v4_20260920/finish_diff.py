from pathlib import Path
import re,zipfile
R=Path(__file__).parent;d=R/'diff_build';p=d/'review_diff.tex';s=p.read_text()
# Floats reset their color on entry; put block labels inside each changed float.
pat=r'(\{\\color\{(red|blue)\}\s*\\begin\{(table\*?|figure\*?)\}(?:\[[^\]]*\])?)'
def fix(m):
 return m[1]+'\n\\color{'+m[2]+'}\\noindent\\textbf{'+('REMOVED' if m[2]=='red' else 'ADDED / REPLACEMENT')+' BLOCK}\\par\n'
s=re.sub(pat,fix,s);p.write_text(s)
with zipfile.ZipFile(R/'First_batch_diff_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in d.rglob('*'):
  if f.is_file() and (f.suffix in ['.tex','.bib','.sty','.bst'] or 'figures' in f.parts):z.write(f,f.relative_to(d))
print('Diff source packaged; select review_diff.tex as main')
