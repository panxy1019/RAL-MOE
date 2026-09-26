"""Restore verified data from the dated RAL-MOE GitHub Release (stdlib only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request

RELEASE = 'https://github.com/panxy1019/RAL-MOE/releases/download/channel-archive-20260926/'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def safe_path(root, name):
    target = (root / name).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f'Unsafe archive path: {name}')
    return target

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['all', 'local', 'cluster'], default='all')
    parser.add_argument('--remove-downloads', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scopes = ['local', 'cluster'] if args.scope == 'all' else [args.scope]
    for scope in scopes:
        manifest = json.loads((root / 'archive' / f'{scope}-data-manifest.json').read_text(encoding='utf8'))
        rows = {r['path']: r for r in manifest['files']}
        if not args.verify_only:
            downloads = root / '.archive_downloads'
            downloads.mkdir(exist_ok=True)
            for asset in manifest['assets']:
                required = [r for r in rows.values() if r['storage'] == asset['name']]
                if all(safe_path(root, r['path']).is_file() and digest(safe_path(root, r['path'])) == r['sha256'] for r in required):
                    continue
                path = downloads / asset['name']
                if not path.exists():
                    print('Downloading', asset['name'], flush=True)
                    partial = path.with_suffix(path.suffix + '.partial')
                    with urllib.request.urlopen(RELEASE + asset['name'], timeout=120) as response, partial.open('wb') as output:
                        shutil.copyfileobj(response, output, 4 * 1024 * 1024)
                    if digest(partial) != asset['sha256']:
                        raise RuntimeError(f'Checksum mismatch: {partial}')
                    os.replace(partial, path)
                if digest(path) != asset['sha256']:
                    raise RuntimeError(f'Checksum mismatch: {path}')
                with tarfile.open(path, 'r:gz') as archive:
                    for member in archive:
                        if not member.isfile() or member.name not in rows or rows[member.name]['storage'] != asset['name']:
                            raise ValueError(f'Unexpected archive member: {member.name}')
                        target = safe_path(root, member.name)
                        expected = rows[member.name]['sha256']
                        if target.exists():
                            if digest(target) != expected:
                                raise RuntimeError(f'Refusing to overwrite modified file: {target}')
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_name(target.name + '.restoring')
                        with archive.extractfile(member) as source, temporary.open('wb') as output:
                            shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                        if digest(temporary) != expected:
                            raise RuntimeError(f'Checksum mismatch: {target}')
                        os.replace(temporary, target)
                if args.remove_downloads:
                    path.unlink()
            for row in rows.values():
                if row['storage'] != 'duplicate':
                    continue
                source = safe_path(root, row['duplicate_of'])
                target = safe_path(root, row['path'])
                if digest(source) != row['sha256']:
                    raise RuntimeError(f'Invalid duplicate source: {source}')
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        failed = [r['path'] for r in rows.values() if not safe_path(root, r['path']).is_file() or digest(safe_path(root, r['path'])) != r['sha256']]
        if failed:
            raise RuntimeError(f'{len(failed)} files missing or changed; first: {failed[:10]}')
        print(scope, len(rows), 'files verified')

if __name__ == '__main__':
    main()
