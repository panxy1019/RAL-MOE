from pathlib import Path
import re
p=Path(__file__).parent/'diff_build/review_diff.tex'
s=p.read_text(encoding='utf-8')
# The removed old key has no bibliographic record; retain that fact visibly.
def cite(m):
    keys=[x.strip() for x in m.group(2).split(',')]
    if 'bhat2025' not in keys: return m.group(0)
    keys.remove('bhat2025')
    return '\\'+m.group(1)+'{'+','.join(keys)+'} [removed unresolved key: bhat2025]'
s=re.sub(r'\\(cite\w*)\{([^}]+)\}',cite,s)
legend='\\begin{center}\\small Review copy: {\\color{blue}additions}; {\\color{red}deletions}.\\end{center}'
if legend not in s: s=s.replace('\\maketitle','\\maketitle\n'+legend,1)
# latexdiff splits braces across paired \\left/\\right in this aligned formula.
# Replace only this marked block with verbatim old/new equations in color groups.
root=Path(__file__).parent
old=(root/'original/appendix/specialist_details.tex').read_text(encoding='utf-8')
new=Path(r'C:\Users\panxy1019\Desktop\STABLEMOE\PMD_Galerkin_Pan_ICLR (3)\appendix\specialist_details.tex').read_text(encoding='utf-8')
def formula(t):
    a=t.index('\\begin{align}',t.index('For output component'))
    b=t.index('\\end{align}',a)+len('\\end{align}')
    return t[a:b]
old_eq=formula(old).replace('align}','align*}')
old_eq=re.sub(r'\\label\{[^}]+\}','',old_eq)
old_eq=re.sub(r'\n\s*\n','\n',old_eq)
a=s.index('\\DIFdelbegin',s.index('For output component'))
b=s.index('Thus each expert',a)
s=s[:a]+'{\\color{red}\n'+old_eq+'\n}\n{\\color{blue}\n'+formula(new)+'\n}\n'+s[b:]
p.write_text(s,encoding='utf-8')
