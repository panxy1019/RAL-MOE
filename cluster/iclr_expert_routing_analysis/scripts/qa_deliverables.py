"""Independent saved-data checks and delivered-file manifest, without PyTorch."""
import argparse
import ast
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.analysis_dir
    checks = {}
    for path in sorted((out / 'scripts').glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    checks['all_analysis_scripts_parse'] = True
    with (out / 'stats/circular_expert_utilization_table.csv').open(newline='', encoding='utf-8') as f:
        stored = list(csv.DictReader(f))
    for chart in ('steady', 'hopf', 'periodic'):
        report = json.loads((out / f'logs/{chart}_routing_verification.json').read_text())
        G, E = report['routing_config']['num_regime_groups'], report['routing_config']['experts_per_group']
        total, counts, masses = defaultdict(int), defaultdict(int), defaultdict(float)
        with gzip.open(out / f'raw/{chart}_routing.jsonl.gz', 'rt') as f:
            for line in f:
                r = json.loads(line)
                if r['stage'] != 1:
                    continue
                channel = r['channel']
                total[channel] += 1
                for e, weight in zip(r['expert_top_ids'], r['expert_top_weights']):
                    key = (channel, r['group_id'], e)
                    counts[key] += 1
                    masses[key] += weight
        for row in [r for r in stored if r['chart'] == chart and r['population'] == 'macro_stage1' and r['Re'] == 'pooled']:
            key = row['channel'], int(row['group_id']), int(row['expert_id'])
            assert abs(counts[key] / (2 * total[key[0]]) - float(row['normalized_activation_frequency'])) < 1e-12
            assert abs(masses[key] / total[key[0]] - float(row['mean_routing_mass'])) < 1e-12
        predictions = np.load(out / f'raw/{chart}_predictions.npz')
        assert all(np.isfinite(predictions[k]).all() for k in predictions.files)
        if chart != 'steady':
            assert all(r['Eu_percent'] == r['reference_Eu_percent'] and r['Ep_percent'] == r['reference_Ep_percent'] for r in report['rows'])
        checks[chart] = {'saved_predictions_all_finite': True, 'stream_recomputed_primary_statistics_match_csv': True,
                         'macro_rows_per_channel': dict(total), 'checkpoint_sha256': report['checkpoint_sha256']}
    pdfs = []
    for path in sorted((out / 'figures').glob('*.pdf')):
        reader = PdfReader(path)
        assert len(reader.pages) == 1
        text = reader.pages[0].extract_text()
        assert len(text.strip()) > 100
        pdfs.append({'file': path.name, 'pages': 1, 'text_characters': len(text)})
    assert len(pdfs) == 3
    checks['pdf_structure'] = pdfs
    checks['pdf_visual_review'] = 'Three PDFs rendered with Poppler and inspected; main pressure annotations moved to avoid bar overlap.'
    for name in ('EXPERT_ROUTING_ANALYSIS_REPORT.md', 'model_asset_inventory.md',
                 'PAPER_INTEGRATION_SNIPPETS.md', 'APPENDIX_ROUTING_TABLE.md'):
        assert (out / 'reports' / name).stat().st_size > 500
    (out / 'logs/final_delivery_qa.json').write_text(json.dumps(checks, indent=2) + '\n')
    paths = [out / 'README.md']
    for folder in ('scripts', 'reports', 'stats', 'figures', 'logs'):
        paths.extend(p for p in (out / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(p for p in (out / 'raw').iterdir() if p.is_file())
    manifest = {str(p.relative_to(out)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(paths))}
    (out / 'DELIVERABLE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'delivered_files_in_manifest': len(manifest), 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
