"""Internal visual QA contact sheets from Poppler-rendered final pages."""
from pathlib import Path
from PIL import Image, ImageDraw

folder = Path(__file__).resolve().parents[1] / 'build/t2c_integration_20260911/render'
pages = sorted(folder.glob('final-*.png'))
for start in range(0, len(pages), 9):
    canvas = Image.new('RGB', (1410, 1860), '#dddddd')
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(pages[start:start+9]):
        x, y = index % 3 * 470, index // 3 * 620
        with Image.open(path) as page:
            canvas.paste(page.convert('RGB'), (x, y + 20))
        draw.text((x + 10, y + 3), str(start + index + 1), fill='black')
    canvas.save(folder / f'contact-{start//9+1}.png')
