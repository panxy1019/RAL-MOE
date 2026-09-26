"""Targeted source, numerical-regression, compile and upload checks for final P0/P1.

This does not re-evaluate models or infer experimental performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from pypdf import PdfReader


def read(path):
    return path.read_text(encoding='utf-8-sig')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--compile-dir', type=Path, required=True)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--package-pdf', type=Path)
    parser.add_argument('--render-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checks = []

    def check(value, description):
        assert value, description
        checks.append(description)

    baseline = args.baseline.resolve()
    current = read(root / 'main.tex')
    previous = read(baseline / 'main.tex')
    for name, pattern in (
        ('Title', r'\\title\{(.*?)\}\s*\n'),
        ('Abstract', r'\\begin\{abstract\}(.*?)\\end\{abstract\}'),
    ):
        check(re.search(pattern, current, re.S).group(1) ==
              re.search(pattern, previous, re.S).group(1), f'{name} unchanged verbatim')

    unchanged = ['sections/method.tex', 'sections/limitations.tex',
                 'sections/conclusion.tex', 'appendix/rom_derivation.tex',
                 'appendix/specialist_details.tex', 'appendix/routing_details.tex',
                 'appendix/implementation.tex', 'appendix/circular_cylinder.tex',
                 'appendix/square_cylinder.tex', 'iclr2027_conference.sty',
                 'iclr2027_conference.bst', 'math_commands.tex', 'legacy_references.bib']
    for rel in unchanged:
        check(sha(root / rel) == sha(baseline / rel), f'Unchanged source/style: {rel}')
    intro = read(root / 'sections/introduction.tex')
    remove_figure = lambda t: re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', t, flags=re.S)
    check(remove_figure(intro) == remove_figure(read(baseline / 'sections/introduction.tex')),
          'Introduction outside Figure 1 unchanged')
    references = read(root / 'references.bib')
    check(read(baseline / 'references.bib').strip() in references,
          'Every pre-existing bibliography entry preserved verbatim')
    bib_keys = re.findall(r'@\w+\s*\{\s*([^,\s]+)', references + read(root / 'legacy_references.bib'))
    check(len(bib_keys) == len(set(bib_keys)), 'No duplicate bibliography keys')
    related = read(root / 'sections/related_work.tex')
    for key in ('ICLR2024_9aeda582', 'NEURIPS2024_b83fae17', 'NEURIPS2025_2d23a999', 'ICLR2026_754612bd'):
        check(key in bib_keys and key in related, f'Verified recent work cited: {key}')

    table_pattern = r'\\begin\{tabular\}.*?\\end\{tabular\}'
    old_tables = 0
    for rel in ['sections/experiments.tex'] + [
        'appendix/' + name for name in ('rom_derivation.tex', 'specialist_details.tex',
        'routing_details.tex', 'implementation.tex', 'circular_cylinder.tex',
        'square_cylinder.tex', 'fluidic_pinball.tex')]:
        before = re.findall(table_pattern, read(baseline / rel), re.S)
        after = re.findall(table_pattern, read(root / rel), re.S)
        after = [table.replace(r'Best single$^\ast$', 'Best single') for table in after]
        check(before == after, f'Original table data and bolding unchanged: {rel}')
        old_tables += len(before)
    check(old_tables == 9, 'All nine original experimental/protocol tables retained')
    for path in (baseline / 'figures').rglob('*'):
        if path.is_file():
            check(sha(path) == sha(root / path.relative_to(baseline)),
                  f'Existing figure asset retained unchanged: {path.name}')
    experiments = read(root / 'sections/experiments.tex')
    old_experiments = read(baseline / 'sections/experiments.tex')
    check(experiments.split(r'\paragraph{Internal routing behavior.}')[1] ==
          old_experiments.split(r'\paragraph{Internal routing behavior.}')[1],
          'Figure 3 and the entire main internal-routing result unchanged')

    source_files = [root / 'main.tex'] + list((root / 'sections').glob('*.tex')) + list((root / 'appendix').glob('*.tex'))
    source = '\n'.join(read(path) for path in source_files)
    forbidden = [r'AI use statement.*TODO', r'finite evaluation windows',
                 r'Periodic core is a second', r'Best single is the lowest',
                 r'key innovation', r'single latent chart',
                 r'ral_moe_rom_framework_user\.png', r'The history connection denotes']
    for pattern in forbidden:
        check(not re.search(pattern, source, re.I), f'Obsolete wording/reference absent: {pattern}')
    check('TODO' not in current, 'AI disclosure has no TODO')
    check(all(s in current for s in ('ChatGPT and Codex', 'analysis scripting',
          'schematic figure preparation', 'full responsibility')), 'AI assistance and responsibility disclosed')
    check('all numerically finite windows' in experiments and '141 of 320' in experiments,
          'Table 2 distinguishes numerical finiteness from the divergence threshold')
    check('stride-four matched final-test evaluation' in experiments.replace('\n', ' '),
          'Table 2 defines Periodic core')
    pinball = read(root / 'appendix/fluidic_pinball.tex')
    check(all(s in pinball for s in ('stride-four', r'\Delta t=1', '$K=56$', '0.25')),
          'Appendix H records complete Periodic-core cadence')
    check(all(s in experiments for s in ('Best single$^\\ast$', 'a posteriori fixed-specialist',
          'not a routing strategy available during prediction')), 'Best single explicitly nondeployable')
    check(all(s in experiments for s in ('1.5921', '23.3447', '4,228', 'one of 28',
          'not a matched long-horizon')), 'Main DeepONet contrast preserves values and cadence limitation')
    appendix = read(root / 'appendix/internal_routing.tex')
    check(all(s in appendix for s in ('0.387', '0.929', '0.784', '99.93', '84.16')),
          'Appendix I uses requested rounded diagnostics')
    check(not re.search(r'\d+\.\d{5,}', appendix), 'No five/six-decimal routing diagnostics in Appendix I')
    figures = PdfReader(root / 'figures/ral_moe_rom_overview.pdf')
    check(len(figures.pages) == 1, 'Vector overview is one page')
    check(not list(figures.pages[0].images), 'Overview has no embedded raster images')
    count_data = json.loads(read(root / 'build/parameter_counts.json'))
    check(count_data['status'] == 'PASS' and not count_data['pretrained_weight_replay'],
          'Parameter architecture/dispatch audit passes with honest weight-replay scope')
    check(count_data['script_sha256'] == sha(root / 'scripts/count_trainable_parameters.py'),
          'Parameter results match current audit script')
    analysis_root = root.parent / 'iclr_expert_routing_analysis'
    check(count_data['inventory_sha256'] == sha(analysis_root / 'logs/checkpoint_inventory.json'),
          'Parameter audit checkpoint inventory unchanged')
    for rel, digest in count_data['source_sha256'].items():
        check(sha(analysis_root / rel) == digest, f'Actual model source unchanged: {rel}')
    count_tex = read(root / 'appendix/parameter_counts.tex')
    for model_name, model in count_data['models'].items():
        for key in ('total_trainable', 'prediction_active_trainable', 'native_audit_macro_stage1_active_trainable'):
            check(f"{model[key]:,}" in count_tex, f'{model_name} table agrees with executed count: {key}')
        check(model['synthetic_probe_nonzero_outputs'] and model['prediction_vs_diagnostic_outputs_bitwise_equal'],
              f'{model_name} diagnostic toggle preserves nonzero synthetic outputs')
        check(model['active_count_invariant'], f'{model_name} active count invariant over all supports')
    check(not count_data['parameter_matched_claim'], 'No unsupported parameter-matched claim')

    aux = read(args.compile_dir / 'main.aux')
    labels = {k: (v, int(pg)) for k, v, pg in re.findall(
        r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}', aux)}
    for label, number in {'fig:ral_overview': '1', 'fig:periodic_internal_routing': '3',
                          'app:pinball': 'H', 'app:internal_routing': 'I',
                          'app:parameter_counts': 'J'}.items():
        check(label in labels and labels[label][0] == number, f'Expected label {label} = {number}')
    conclusion = re.search(r'\\numberline \{6\}Conclusion\}\{(\d+)\}', aux)
    check(conclusion is not None and int(conclusion.group(1)) <= 9,
          'Scientific main text concludes within nine pages')
    log = read(args.compile_dir / 'main.log')
    check(not re.search(r'Overfull \\[hv]box|(?:Reference|Citation).*undefined|There were undefined|multiply defined|Font shape .* undefined', log),
          'No overfull boxes, unresolved references/citations, duplicate labels or missing font shapes')
    pdf = PdfReader(args.pdf)
    pdf_text = '\n'.join(p.extract_text() for p in pdf.pages)
    check('TODO' not in pdf_text, 'Compiled PDF contains no TODO')
    if args.package:
        with zipfile.ZipFile(args.package) as archive:
            check('main.tex' in archive.namelist(), 'Upload main.tex is at ZIP root')
            for rel in archive.namelist():
                check(archive.read(rel) == (root / rel).read_bytes(), f'Packaged file matches: {rel}')
                check(not rel.startswith(('build/', 'source_snapshot/', 'scripts/')),
                      f'No internal build/source-audit path in upload: {rel}')
    if args.package_pdf:
        packaged = PdfReader(args.package_pdf)
        check(len(pdf.pages) == len(packaged.pages), 'Independent upload compile has same page count')
        for n, (a, b) in enumerate(zip(pdf.pages, packaged.pages), 1):
            check(a.extract_text() == b.extract_text(), f'Upload page {n} text matches')
            check(a.get_contents().get_data() == b.get_contents().get_data(), f'Upload page {n} content stream matches')
            if args.render_dir:
                check((args.render_dir / f'final-{n:02}.png').read_bytes() ==
                      (args.render_dir / f'package-{n:02}.png').read_bytes(), f'Upload page {n} render matches')
    result = {'status': 'PASS', 'pages': len(pdf.pages), 'checks_passed': len(checks),
              'original_performance_numbers_changed': False, 'checks': checks,
              'labels': labels, 'pdf_sha256': sha(args.pdf)}
    if args.package:
        result['zip_sha256'] = sha(args.package)
    output = root / 'build/final_p0_p1_verification.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'labels')}, indent=2))


if __name__ == '__main__':
    main()
