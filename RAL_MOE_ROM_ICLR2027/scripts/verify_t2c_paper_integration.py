"""Check fixed-run preservation, new statistics, anonymity and the upload ZIP."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'build/t2c_integration_20260911'
BASE = RUN / 'before'
checks = []


def read(path):
    return path.read_text(encoding='utf-8-sig')


def check(ok, name):
    checks.append({'check': name, 'passed': bool(ok)})
    if not ok:
        raise AssertionError(name)


def table(tex, label):
    pos = tex.index('\\label{' + label + '}')
    return tex[tex.rfind('\\begin{table}', 0, pos):tex.index('\\end{table}', pos)+len('\\end{table}')]


def number_rows(tex):
    return [re.findall(r'\d+\.\d+', line) for line in tex.splitlines()
            if '&' in line and re.search(r'\d+\.\d+', line)]


def main():
    manifest = json.loads(read(BASE / 'manifest.json'))
    expected = {'main.tex', 'sections/experiments.tex', 'sections/limitations.tex',
                'appendix/routing_details.tex', 'appendix/square_cylinder.tex', 'appendix/implementation.tex'}
    changed = set()
    for rel in manifest:
        same = (ROOT / rel).read_bytes() == (BASE / rel).read_bytes()
        if not same:
            changed.add(rel)
        if rel not in expected:
            check(same, 'Unchanged source/asset: ' + rel)
    check(changed == expected, 'Only six intended manuscript files changed')
    old, new = read(BASE / 'sections/experiments.tex'), read(ROOT / 'sections/experiments.tex')
    check(table(old, 'tab:compact_ablation') == table(new, 'tab:compact_ablation'), 'Table4 byte-equivalent text preserved')
    check(table(old, 'tab:pinball_autonomous') == table(new, 'tab:pinball_autonomous'), 'Table2 unchanged')
    before = number_rows(table(old, 'tab:routing_fusion'))
    after = number_rows(table(new, 'tab:routing_fusion'))
    check(len(after) == 2 and all(len(r) == 7 for r in after), 'Table3 has seven method columns')
    check([r[:4]+r[5:] for r in after] == before, 'All original Table3 values unchanged')
    check([r[4] for r in after] == ['2.1846', '1.1504'], 'Table3 mu-only values correct')
    check(re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', old, re.S) ==
          re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', new, re.S), 'Figures 2/3 source blocks unchanged')
    old_main, main_tex = read(BASE / 'main.tex'), read(ROOT / 'main.tex')
    check(re.search(r'\\begin\{abstract\}.*?\\end\{abstract\}', old_main, re.S).group() ==
          re.search(r'\\begin\{abstract\}.*?\\end\{abstract\}', main_tex, re.S).group(), 'Abstract unchanged')
    check(read(ROOT / 'appendix/routing_details.tex').startswith(read(BASE / 'appendix/routing_details.tex')),
          'Appendix D original descriptor and equations unchanged; oracle paragraph only appended')
    square = read(ROOT / 'appendix/square_cylinder.tex')
    square_old = read(BASE / 'appendix/square_cylinder.tex')
    old_components = number_rows(table(square_old, 'tab:square_fusion_full'))
    filtered = '\n'.join(line for line in table(square, 'tab:square_fusion_full').splitlines() if '$\\mu$-only &' not in line)
    check(number_rows(filtered) == old_components, 'All 42 original Appendix G components unchanged')
    mu_rows = [re.findall(r'\d+\.\d+', l) for l in table(square, 'tab:square_fusion_full').splitlines() if '$\\mu$-only &' in l]
    check(mu_rows == [['0.7898','1.3948','2.1846'], ['0.5875','0.5629','1.1504']], 'Appendix mu-only components correct')
    check(table(square_old, 'tab:square_parameter') == table(square, 'tab:square_parameter'), 'Original parameter-wise table unchanged')
    summary = json.loads(read(ROOT / 'artifacts/t2c_experiment_summary.json'))
    expected_pairs = []
    for b, mode in [('SH','mu_only'), ('SH','full'), ('HP','mu_only'), ('HP','full')]:
        for metric in ['Eu_percent','Ep_percent','Ejoint_percent']:
            r = next(x for x in summary['seed_summary'] if (x['overlap'],x['mode'],x['metric']) == (b,mode,metric))
            expected_pairs.append((f"{r['mean']:.5f}", f"{r['SD_seed']:.5f}"))
    observed = re.findall(r'(\d+\.\d{5})\\pm(\d+\.\d{5})', table(square, 'tab:t2c_seed_variability'))
    check(observed == expected_pairs, 'All 24 seed mean/SD numbers match unrounded data at five decimals')
    raw = json.loads(read(ROOT / 'experiments/iclr_seed_study_20260911/results_v1/results.json'))
    expected_rows = [[f"{next(x for x in raw if (x['boundary'],x['input_mode'],x['seed']) == (b,m,s))['by_horizon']['K24']['joint_mean']*100:.5f}" for s in [42001,42002,42003]]
                     for b,m in [('SH','mu_only'),('SH','full'),('HP','mu_only'),('HP','full')]]
    check(number_rows(table(square, 'tab:t2c_seedwise')) == expected_rows, 'All twelve seed-wise values match raw results')
    for phrase in ['terminal-step ($k=24$)', '16 held-out windows', 'eight windows', 'denominator $N-1$',
                   'not across-$Re$', 'Only the lightweight gate', 'without test-based selection']:
        check(phrase in square, 'Protocol stated: ' + phrase)
    source = '\n'.join(read(ROOT / p) for p in manifest if p.endswith('.tex'))
    stripped = re.sub(r'(?m)^\s*%.*$', '', source)
    check('\\iclrfinalcopy' not in stripped and '\\author{Anonymous Authors}' in stripped, 'Anonymous mode retained')
    check('statistically significant' not in stripped.lower(), 'No significance claim')
    check('repeated-seed dispersion and parameter-matched capacity controls are not evaluated' not in stripped, 'Stale no-seed Appendix E statement removed')
    aux = read(RUN / 'compile/main.aux')
    labels = re.findall(r'\\newlabel\{([^}]+)\}', aux)
    refs = re.findall(r'\\(?:ref|eqref|plaineqref)\{([^}]+)\}', stripped)
    check(len(labels) == len(set(labels)), 'No duplicate labels')
    check(not {r for r in refs if not r.startswith('#')}-set(labels), 'All concrete cross-references resolve')
    log = read(RUN / 'compile/main.log')
    for bad in ['Overfull', 'Missing character', 'undefined references', 'undefined citations', 'LaTeX Error']:
        check(bad not in log, 'Compile log excludes: ' + bad)
    package = ROOT / 'output/overleaf_T2C_integrated_20260912.zip'
    new_manifest = json.loads(read(RUN / 'upload_manifest.json'))
    with zipfile.ZipFile(package) as z:
        check(set(z.namelist()) == set(new_manifest), 'ZIP has exactly recursively referenced source/assets')
        for name in z.namelist():
            check(z.read(name) == (ROOT / name).read_bytes(), 'Packaged bytes match: ' + name)
        check('main.tex' in z.namelist() and not any(n.startswith(('build/','scripts/','artifacts/','experiments/')) for n in z.namelist()), 'ZIP excludes internal files and has root main.tex')
    pdf = PdfReader(RUN / 'compile/main.pdf')
    packaged = PdfReader(RUN / 'package_compile/main.pdf')
    check(len(pdf.pages) == len(packaged.pages), 'Independent package compile page count matches')
    for i, (a,b) in enumerate(zip(pdf.pages, packaged.pages),1):
        check(a.extract_text() == b.extract_text(), f'Independent package PDF page {i} text matches')
        check(a.get_contents().get_data() == b.get_contents().get_data(), f'Independent package PDF page {i} content matches')
    page_texts = [re.sub(r'\s+', '', p.extract_text()) for p in pdf.pages]
    check(any('0.85004' in t and '0.00347' in t for t in page_texts), 'Final PDF contains new seed table')
    result = dict(status='PASS', count=len(checks), checks=checks, modified_files=sorted(changed),
        pages=len(pdf.pages), underfull_messages=log.count('Underfull'), package=str(package),
        package_sha256=hashlib.sha256(package.read_bytes()).hexdigest())
    (RUN / 'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
