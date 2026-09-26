"""Compare the separately extracted upload build with the working PDF."""
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader


def main():
    root = Path(__file__).resolve().parents[1]
    original = PdfReader(root / 'build/routing_compile/main.pdf')
    packaged = PdfReader(root / 'build/routing_package_compile/main.pdf')
    assert len(original.pages) == len(packaged.pages) == 21
    for n, (a, b) in enumerate(zip(original.pages, packaged.pages), 1):
        assert a.extract_text() == b.extract_text(), n
        assert a.get_contents().get_data() == b.get_contents().get_data(), n
        first = root / f'build/routing_render/final-{n:02d}.png'
        second = root / f'build/routing_render/package-{n:02d}.png'
        assert first.read_bytes() == second.read_bytes(), n
    result = {'status': 'PASS', 'pages': 21, 'all_page_text_equal': True,
              'all_page_content_streams_equal': True, 'all_85dpi_page_renders_byte_equal': True,
              'zip_sha256': hashlib.sha256((root / 'build/overleaf_upload_routing_20260905.zip').read_bytes()).hexdigest()}
    (root / 'build/routing_package_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
