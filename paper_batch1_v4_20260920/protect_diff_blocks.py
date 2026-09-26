from pathlib import Path
import re, json, sys
R=Path(__file__).parent
pat=re.compile(r'\\begin\{(table\*?|figure\*?|equation\*?|align\*?|gather\*?|tikzpicture)\}.*?\\end\{\1\}',re.S)
if len(sys.argv)==1:
    mapping={}; reverse={}
    def replace(m):
        block=m.group()
        if block not in reverse:
            key=f'REVIEWBLOCK{len(mapping):05d}';mapping[key]=block;reverse[block]=key
        return '\n\n'+reverse[block]+'\n\n'
    for name in ['old','new']:
        text=(R/(name+'_flat.tex')).read_text(encoding='utf-8')
        # Keep new command definitions intact by treating them as preamble.
        if name=='new':
            defs=(R/'diff_build/figures/field_panel_labels.tex').read_text(encoding='utf-8')
            text=text.replace(defs,'').replace('\\begin{document}',defs+'\n\\begin{document}')
        pre,body=text.split('\\begin{document}',1)
        (R/(name+'_protected.tex')).write_text(pre+'\\begin{document}'+pat.sub(replace,body),encoding='utf-8')
    (R/'diff_blocks.json').write_text(json.dumps(mapping),encoding='utf-8')
else:
    p=R/'diff_build/review_diff.tex';text=p.read_text(encoding='utf-8')
    blocks=json.loads((R/'diff_blocks.json').read_text())
    def render(m):
        mode,key=m.group(1),m.group(2);block=blocks[key]
        if mode=='del':block=re.sub(r'\\label\{([^}]+)\}',r'\\label{old:\1}',block)
        return '\n{\\color{'+('red' if mode=='del' else 'blue')+'}\n'+block+'\n}\n'
    text=re.sub(r'\\DIF(add|del)(?:FL)?\{\s*(REVIEWBLOCK\d+)\s*\}',render,text)
    for key,block in blocks.items():text=text.replace(key,block)
    text=text.replace('\\maketitle','\\maketitle\n\\begin{center}\\small First-batch review: {\\color{blue}added}; {\\color{red}deleted}. Tables, figures and equations are compared as complete blocks.\\end{center}',1)
    p.write_text(text,encoding='utf-8')
    print('Restored whole-block comparisons')
