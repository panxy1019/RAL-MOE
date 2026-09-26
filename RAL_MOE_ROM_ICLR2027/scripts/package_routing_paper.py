"""Package only the recursively referenced anonymous paper sources/assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path,
                   help='Optional versioned manifest path; defaults to legacy routing manifest')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = set()
    def visit(relative):
        path = (root / relative).resolve()
        assert path.is_relative_to(root) and path.is_file(), path
        rel = path.relative_to(root)
        if rel in files:
            return
        files.add(rel)
        if path.suffix == '.tex':
            text = path.read_text(encoding='utf-8-sig')
            text = re.sub(r'(?m)^\s*%.*$', '', text)
            for command, value in re.findall(r'\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}', text):
                candidate = Path(value)
                if command == 'input' and not candidate.suffix:
                    candidate = candidate.with_suffix('.tex')
                visit(candidate)
    visit(Path('main.tex'))
    for rel in ('references.bib', 'legacy_references.bib', 'iclr2027_conference.sty',
                'iclr2027_conference.bst', 'natbib.sty', 'fancyhdr.sty'):
        visit(Path(rel))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in sorted(files):
            archive.write(root / rel, rel.as_posix())
    manifest = {rel.as_posix(): hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in sorted(files)}
    manifest_path = a.manifest or root / 'build/routing_upload_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': len(files), 'package': str(a.output), 'bytes': a.output.stat().st_size}))


if __name__ == '__main__':
    main()
