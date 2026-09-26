from pathlib import Path
import json,re,zipfile,shutil
import pymupdf
from PIL import Image,ImageDraw
R=Path(__file__).parent
S=Path(r'C:\Users\panxy1019\Desktop\STABLEMOE\PMD_Galerkin_Pan_ICLR (3)')
B=R/'diff_build'
doc=pymupdf.open(B/'review_diff.pdf')
qa=R/'qa'; qa.mkdir(exist_ok=True)
flags=[]
for i,page in enumerate(doc):
    pix=page.get_pixmap(matrix=pymupdf.Matrix(1.1,1.1))
    pix.save(qa/f'diff_{i+1:02d}.png')
    if '\ufffd' in page.get_text(): flags.append({'page':i+1,'issue':'replacement character'})
for start in range(0,len(doc),8):
    sheet=Image.new('RGB',(1200,1700),'#dddddd')
    draw=ImageDraw.Draw(sheet)
    for j in range(start,min(start+8,len(doc))):
        im=Image.open(qa/f'diff_{j+1:02d}.png').convert('RGB')
        im.thumbnail((580,395))
        x=(j-start)%2*600+(600-im.width)//2; y=(j-start)//2*425+22
        sheet.paste(im,(x,y));draw.text((x,y-18),f'PAGE {j+1}',fill='black')
    sheet.save(qa/f'contact_{start+1:02d}.png')
log=(B/'review_diff.log').read_text(encoding='utf-8',errors='replace')
blg=(B/'review_diff.blg').read_text(encoding='utf-8',errors='replace')
summary={'diff_pages':len(doc),'rendered_pages':len(doc),'pdf_text_flags':flags,
    'undefined_citations_or_references':re.findall(r'.*(?:undefined|multiply defined).*',log),
    'bibtex_warnings':re.findall(r'.*Warning.*',blg),
    'overfull_boxes':re.findall(r'.*Overfull.*',log),
    'normal_revised_pdf_compiled':False}
(R/'qa_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
out=R/'output/pdf';out.mkdir(parents=True,exist_ok=True)
shutil.copy2(B/'review_diff.pdf',out/'RAL_MoE_ROM_round1_diff.pdf')
with zipfile.ZipFile(R/'revised_source_round1.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in S.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(S))
with zipfile.ZipFile(R/'review_diff_source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in B.rglob('*'):
        if p.is_file() and (p.suffix in ('.tex','.sty','.bst','.bib') or 'figures' in p.parts):
            z.write(p,p.relative_to(B))
