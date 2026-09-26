"""Prepare and restore robust whole-block LaTeX review diffs."""
from pathlib import Path
import argparse, json, re, shutil, zipfile

p=argparse.ArgumentParser()
p.add_argument('mode',choices=['prepare','restore','package'])
p.add_argument('--old',type=Path)
p.add_argument('--new',type=Path)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--title',default='Review diff')
a=p.parse_args(); out=a.out.resolve()

def flatten(root,name='main.tex'):
 text=(root/name).read_text(encoding='utf-8')
 def sub(m):
  q=m.group(1);q=q if q.endswith('.tex') else q+'.tex'
  return flatten(root,q)
 return re.sub(r'\\input\{([^}]+)\}',sub,text)

pat=re.compile(r'\\begin\{(table\*?|figure\*?|equation\*?|align\*?|gather\*?|tikzpicture)\}.*?\\end\{\1\}',re.S)

if a.mode=='prepare':
 assert a.old and a.new
 out.mkdir(parents=True,exist_ok=True)
 for f in a.new.iterdir():
  if f.is_dir():shutil.copytree(f,out/f.name,dirs_exist_ok=True)
  else:shutil.copy2(f,out/f.name)
 mapping={};reverse={}
 def protect(m):
  block=m.group()
  if block not in reverse:
   key=f'REVIEWBLOCK{len(mapping):05d}';mapping[key]=block;reverse[block]=key
  return '\n\n'+reverse[block]+'\n\n'
 for label,root in [('old',a.old),('new',a.new)]:
  text=flatten(root)
  labels=''
  fp=root/'figures/field_panel_labels.tex'
  if fp.exists():
   labels=fp.read_text(encoding='utf-8');text=text.replace(labels,'')
  pre,body=text.split('\\begin{document}',1)
  (out/f'{label}.tex').write_text(pre+labels+'\n\\begin{document}'+pat.sub(protect,body),encoding='utf-8')
 (out/'blocks.json').write_text(json.dumps(mapping),encoding='utf-8')
 print(out)
elif a.mode=='restore':
 target=out/'review_diff.tex';text=target.read_text(encoding='utf-8');blocks=json.loads((out/'blocks.json').read_text())
 def render(m):
  mode,key=m.group(1),m.group(2);block=blocks[key]
  if mode=='del':block=re.sub(r'\\label\{([^}]+)\}',r'\\label{old:\1}',block)
  color='red' if mode=='del' else 'blue';tag='REMOVED' if mode=='del' else 'ADDED / REPLACEMENT'
  if re.match(r'\\begin\{(?:table|figure)',block):
   block=re.sub(r'(\\begin\{(?:table\*?|figure\*?)\}(?:\[[^\]]*\])?)',r'\1\n\\color{'+color+r'}\\noindent\\textbf{'+tag+r' BLOCK}\\par',block,count=1)
  return '\n{\\color{'+color+'}\n'+block+'\n}\n'
 text=re.sub(r'\\DIF(add|del)(?:FL)?\{\s*(REVIEWBLOCK\d+)\s*\}',render,text)
 for key,block in blocks.items():text=text.replace(key,block)
 text=text.replace('\\hypersetup{hidelinks}','\\hypersetup{hidelinks,hypertexnames=false}')
 legend='\\begin{center}\\small '+a.title+': {\\color{blue}added}; {\\color{red}deleted}. Tables, figures and equations are compared as complete blocks.\\end{center}'
 text=text.replace('\\maketitle','\\maketitle\n'+legend,1);target.write_text(text,encoding='utf-8')
elif a.mode=='package':
 zpath=out.parent/(out.name+'_Overleaf.zip')
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
  for f in out.rglob('*'):
   if f.is_file() and (f.suffix.lower() in ['.tex','.bib','.sty','.bst','.png','.jpg','.jpeg','.pdf']):z.write(f,f.relative_to(out))
 print(zpath)
