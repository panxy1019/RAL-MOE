"""Regression checks for the routing integration; no inference or metrics changes."""
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
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--pdf', type=Path, required=True)
    p.add_argument('--analysis-dir', type=Path, required=True)
    p.add_argument('--user-image', type=Path, required=True)
    p.add_argument('--package', type=Path)
    p.add_argument('--expected-pages', type=int,
                   help='Optional page count established by this round of visual review, not a legacy fixed count')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    checks = []
    def check(condition, message):
        assert condition, message
        checks.append(message)
    baseline = args.baseline.resolve()
    main_before, main_after = read(baseline/'main.tex'), read(root/'main.tex')
    abstract = lambda t: re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S).group(1)
    check(abstract(main_before) == abstract(main_after), 'Abstract unchanged verbatim')
    unchanged = ['sections/method.tex','sections/related_work.tex','sections/conclusion.tex',
                 'appendix/rom_derivation.tex','appendix/specialist_details.tex','appendix/routing_details.tex',
                 'appendix/implementation.tex','appendix/circular_cylinder.tex',
                 'appendix/square_cylinder.tex','appendix/fluidic_pinball.tex']
    for rel in unchanged:
        check(sha(root/rel)==sha(baseline/rel), f'Byte-identical original source: {rel}')
    original_tables=0
    for rel in ['sections/experiments.tex'] + [v for v in unchanged if v.startswith('appendix/')]:
        pattern = r'\\begin\{tabular\}.*?\\end\{tabular\}'
        before = re.findall(pattern, read(baseline/rel), re.S)
        after = re.findall(pattern, read(root/rel), re.S)
        check(before==after, f'Original tabular data unchanged verbatim: {rel}')
        original_tables+=len(before)
    check(original_tables==9, 'All nine original tabular blocks retained')
    old_experiments = re.sub(r'\\FloatBarrier\s*','',read(baseline/'sections/experiments.tex')).strip()
    retained = read(root/'sections/experiments.tex').split(r'\paragraph{Internal routing behavior.}')[0]
    retained = re.sub(r'\\FloatBarrier\s*','',retained).replace(r'\begin{table}[!ht]',r'\begin{table}[t]').strip()
    check(old_experiments==retained, 'All pre-existing experiment prose and performance numbers unchanged')
    for path in (baseline/'figures').rglob('*'):
        if path.is_file():
            check(sha(path)==sha(root/path.relative_to(baseline)), f'Existing figure asset unchanged: {path.relative_to(baseline)}')
    check(sha(args.user_image)==sha(root/'figures/ral_moe_rom_framework_user.png'), 'User framework PNG copied byte-for-byte')
    introduction = read(root/'sections/introduction.tex')
    remove_fig = lambda t: re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', t, flags=re.S)
    check(remove_fig(introduction)==remove_fig(read(baseline/'sections/introduction.tex')), 'Introduction outside replaced framework figure unchanged')
    include = re.search(r'\\includegraphics\[([^]]*)\]\{figures/ral_moe_rom_framework_user\.png\}', introduction)
    check(include is not None and 'trim' not in include.group(1) and 'clip' not in include.group(1),
          'Full user framework included without legacy viewport cropping')
    check('whereas E2 uses only' in introduction and 'pressure uses the algebraic map' in introduction,
          'Framework caption retains parameter-only E2 and algebraic pressure definition')
    added = read(root/'sections/experiments.tex').split(r'\paragraph{Internal routing behavior.}')[1]
    appendix = read(root/'appendix/internal_routing.tex')
    check('Steady' not in added and 'S4' not in added+appendix, 'S4 routing excluded from paper integration')
    check(all(v in added for v in ('18 routed experts','33 distinct','14 of 18','16 configurations')), 'Main paragraph contains verified Periodic counts')
    check(all(v in appendix for v in ('99.9344','84.1630','0.3869','0.9292','0.7838','16.27','31.25')), 'Appendix reports verified Hopf/Periodic statistics')
    check('first integration stage' in appendix and 'not a count of all' in appendix, 'Macro-step stage-1 population specified')
    check('not interpreted as physical oscillation phase' in appendix, 'Rollout position not identified with physical phase')
    check('fixed by architecture' in appendix and 'causal contribution' in appendix, 'Fixed scaling and noncausal interpretation disclosed')
    check('shared-expert index' in appendix, 'Routed diagnostic identifiers distinguished from shared index')
    check('not used for model selection' in main_after, 'Reproducibility statement updated')
    source = '\n'.join(read(p) for d in ('sections','appendix') for p in (root/d).glob('*.tex'))
    check(not re.search(r'all experts specialize|all charts avoid collapse|learns to balance shared|do not report post-hoc internal',source,re.I), 'No prohibited strong/obsolete routing claims')
    figure_data=json.loads(read(root/'build/routing_figure_data.json'))
    check(figure_data['population']=='macro_stage1', 'Plots use verified macro-stage1 CSV population')
    for name, digest in figure_data['source_sha256'].items():
        check(sha(args.analysis_dir/'stats'/f'{name}.csv')==digest, f'Plot source CSV unchanged: {name}')
    aux=read(root/'build/routing_compile/main.aux')
    labels={k:(v,int(pg)) for k,v,pg in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux)}
    check(labels['fig:periodic_internal_routing'][0]=='3','Main routing figure is Figure 3')
    check(labels['app:internal_routing'][0]=='I','New diagnostic appendix is Appendix I')
    check(labels['fig:periodic_routing_step'][0]=='6','Rollout-position figure is Figure 6 in appendix')
    check(labels['tab:internal_routing_summary'][0]=='10','New compact routing table is Table 10')
    conclusion = re.search(r'\\numberline \{6\}Conclusion\}\{(\d+)\}',aux)
    check(conclusion is not None and int(conclusion.group(1))<=9, 'Scientific main text concludes within nine pages')
    log=read(root/'build/routing_compile/main.log')
    check(not re.search(r'Overfull \\[hv]box|(?:Reference|Citation).*undefined|There were undefined|multiply defined|Font shape .* undefined',log),'No overfull boxes, unresolved refs/citations, duplicate labels, or missing font shapes')
    reader=PdfReader(args.pdf)
    if args.expected_pages is not None:
        check(len(reader.pages)==args.expected_pages, f'Final PDF has the visually reviewed {args.expected_pages} pages')
    pdftext='\n'.join(p.extract_text() for p in reader.pages)
    check('99.9344' in pdftext and '84.1630' in pdftext,'Hopf mass concentration appears in compiled PDF')
    if args.package:
        with zipfile.ZipFile(args.package) as z:
            names=z.namelist()
            check('main.tex' in names and 'appendix/internal_routing.tex' in names,'Upload contains entry point and routing appendix')
            check(all(not n.startswith(('build/','source_snapshot/','scripts/')) for n in names),'Anonymous upload excludes internal reports, build files and scripts')
            for name in names:
                check(z.read(name)==(root/name).read_bytes(), f'Upload matches working source: {name}')
    report={'status':'PASS','checks':checks,'check_count':len(checks),'original_tables':original_tables,
            'existing_performance_numbers_changed':False,'method_definitions_changed':False,
            'pdf_pages':len(reader.pages),'scientific_main_text_last_page':int(conclusion.group(1)),
            'references_start_page':next(i+1 for i,p in enumerate(reader.pages) if 'REFERENCES' in p.extract_text().upper()),
            'labels':{k:list(labels[k]) for k in ('fig:ral_overview','fig:periodic_internal_routing','app:internal_routing','fig:periodic_routing_step','tab:internal_routing_summary')},
            'framework_sha256':sha(args.user_image),'pdf_sha256':sha(args.pdf),'baseline':str(baseline)}
    (root/'build/routing_integration_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','baseline')},indent=2))


if __name__=='__main__':
    main()
