"""Build an isolated review copy; never modifies the user's manuscript."""
from pathlib import Path
import re, shutil, subprocess
R=Path(__file__).resolve().parent
SRC=Path(r'C:\Users\panxy1019\Downloads\PMD_Galerkin_Pan_ICLR (4)')
BUILD=R/'diff_build'
BUILD.mkdir(exist_ok=True)
def flatten(root,name='main.tex'):
    text=(root/name).read_text(encoding='utf-8')
    def sub(m):
        q=m.group(1)
        if not q.endswith('.tex'):q+='.tex'
        return flatten(root,q)
    return re.sub(r'\\input\{([^}]+)\}',sub,text)
for p in SRC.iterdir():
    if p.is_dir():shutil.copytree(p,BUILD/p.name,dirs_exist_ok=True)
    else:shutil.copy2(p,BUILD/p.name)
for root,out in [(R/'original','old_flat.tex'),(SRC,'new_flat.tex')]:
    (R/out).write_text(flatten(root),encoding='utf-8')
print('Isolated diff inputs prepared')
