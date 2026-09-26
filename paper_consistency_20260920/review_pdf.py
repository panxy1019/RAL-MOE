from pathlib import Path
import pymupdf
root=Path(__file__).parent
doc=pymupdf.open(root/'diff_build/review_diff.pdf')
qa=root/'qa'; qa.mkdir(exist_ok=True)
for start in range(0,len(doc),9):
    montage=pymupdf.open()
    sheet=montage.new_page(width=1050,height=1455)
    for index in range(start,min(start+9,len(doc))):
        pix=doc[index].get_pixmap(matrix=pymupdf.Matrix(.55,.55))
        x=((index-start)%3)*350; y=((index-start)//3)*485
        sheet.insert_image(pymupdf.Rect(x,y+22,x+340,y+477),stream=pix.tobytes('png'))
        sheet.insert_text((x+8,y+14),f'Page {index+1}')
    sheet.get_pixmap().save(qa/f'contact_{start+1}.png')
need=['Selected qualitative centered-square','Selected held-out prediction','Deep-FNN-H3','Chart-specific provenance','eight-dimensional','unreconciled']
for index,page in enumerate(doc):
    if any(term in page.get_text() for term in need):
        page.get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(qa/f'page_{index+1}.png')
print('Pages:',len(doc))
print('QA:',[p.name for p in qa.glob('*.png')])
print('Missing glyph count:',sum(page.get_text().count('\ufffd') for page in doc))
