"""Supplement main fingerprints with original data-view/index/config files."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    report = {}
    for chart, folder in [('steady', 'steady_specialist_v1/source_artifacts/steady/Global_POD_AreaWeighted_L2'),
                          ('periodic', 'periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2')]:
        paths = [v for v in (a.root / folder).iterdir() if v.is_file()]
        if chart == 'steady':
            paths.append(a.root / 'steady_specialist_v1/code/training_s2b_portable.json')
        report[chart] = {str(v): {'resolved_path': str(v.resolve()), 'bytes': v.stat().st_size,
                                  'sha256': sha256(v)} for v in sorted(paths)}
    (a.root / 'iclr_expert_routing_analysis/logs/input_view_fingerprints.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
