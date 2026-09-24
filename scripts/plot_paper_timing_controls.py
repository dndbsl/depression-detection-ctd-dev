#!/usr/bin/env python3
"""Render existing nested-control estimates; no model fitting or new analysis.

Run build_paper_numbers.py first; pdfLaTeX and PGF must be available. Exact
printed values share the paper ledger; full-precision audited means and SDs
determine points and whisker lengths.
"""
from pathlib import Path
import hashlib
import json
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ('all24', 'no_ask', 'latency_structure', 'latency', 'structure')
LABELS = ('Full CTD', 'No ask-side', 'Latency + structure', 'Latency', 'Structure')
FIGURE = 'paper/figures/timing-controls.pdf'


def main():
    numbers = json.loads((ROOT/'output/paper_numbers.json').read_text())
    source = 'output/timing_controls.json'
    digest = hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    controls = json.loads((ROOT/source).read_text())
    keys = [cfg+metric+stat for cfg in CONFIGS for metric in ('F', 'A')
            for stat in ('mean', 'sd')]
    for key in keys:
        entry = numbers[key]
        assert entry['source'] == source and entry['source_sha256'] == digest
        assert entry['figure_paths'] == [FIGURE]
        value = controls
        for part in entry['key'].split('.'): value = value[part]
        assert entry['value'] == value and entry['display'] == f'{value:.3f}'

    # Typeset with the same Times font and 9-point size as the paper's tables.
    # Fix the final printed width to 178 mm, matching spconf.sty's text width.
    # Layout dimensions below use physical PDF points (72 per inch).
    width, height = 178 / 25.4 * 72, 104
    drawing = [rf'\useasboundingbox (0,0) rectangle ({width:.6f},{height});']

    def text(x, y, label, anchor='center'):
        drawing.append(rf'\node[anchor={anchor},inner sep=0pt,outer sep=0pt] '
                       rf'at ({x:.6f},{y:.6f}) {{{label}}};')

    def line(x1, y1, x2, y2, weight=0.5):
        drawing.append(rf'\draw[line width={weight}bp] '
                       rf'({x1:.6f},{y1:.6f}) -- ({x2:.6f},{y2:.6f});')

    # Booktabs-style rules and a shared row grid make the values read as a
    # table; the two small plots retain the mean/SD comparison on equal scales.
    for y, weight in ((103, 0.8), (85, 0.5), (1, 0.8)):
        line(36, y, 481, y, weight)
    text(40, 94, 'Configuration', anchor='west')
    row_positions = [75 - 12 * i for i in range(len(CONFIGS))]
    for y, label in zip(row_positions, LABELS):
        text(40, y, label, anchor='west')

    rows = []
    for metric, title, plot_left, value_right in (
            ('F', 'Macro-F1', 151, 306), ('A', 'ROC-AUC', 326, 477)):
        text((plot_left + value_right) / 2, 94, title)

        def position(value):
            return plot_left + (value - 0.45) / (0.73 - 0.45) * 80

        line(position(0.45), 18, position(0.73), 18)
        for tick in (0.5, 0.6, 0.7):
            line(position(tick), 18, position(tick), 16)
            text(position(tick), 14, f'{tick:.1f}', anchor='north')
        for y, cfg in zip(row_positions, CONFIGS):
            mean, sd = numbers[cfg+metric+'mean'], numbers[cfg+metric+'sd']
            label = f"{mean['display']} ± {sd['display']}"
            low, high = mean['value'] - sd['value'], mean['value'] + sd['value']
            assert 0.45 <= low <= high <= 0.73
            line(position(low), y, position(high), y, 0.7)
            for endpoint in (low, high):
                line(position(endpoint), y - 2, position(endpoint), y + 2, 0.6)
            drawing.append(rf'\fill ({position(mean["value"]):.6f},{y}) '
                           r'circle[radius=1.3bp];')
            text(value_right, y, label.replace('±', r'$\pm$'), anchor='east')
            rows.append({'config': cfg, 'metric': title, 'label': label,
                         'mean_key': cfg+metric+'mean', 'sd_key': cfg+metric+'sd',
                         'mean': mean['value'], 'sd': sd['value']})
    pdf = ROOT/FIGURE
    # Ship one tightly bounded vector page; this avoids rescaling the text or
    # relying on system-font substitutes when the paper includes the figure.
    tex = '\n'.join([
        r'\documentclass{article}', r'\usepackage{tikz}',
        r'\renewcommand{\rmdefault}{ptm}',
        r'\pdfinfoomitdate=1', r'\pdftrailerid{}', r'\pdfsuppressptexinfo=-1',
        r'\pdfinfo{/Title (DAIC nested timing controls: means and standard deviations)'
        r'/Creator (scripts/plot_paper_timing_controls.py)}',
        r'\begin{document}', r'\newbox\figurebox',
        r'\setbox\figurebox=\hbox{\begin{tikzpicture}'
        r'[x=1bp,y=1bp,font=\fontsize{9}{11}\selectfont]',
        *drawing, r'\end{tikzpicture}}',
        r'\pdfpagewidth=\wd\figurebox',
        r'\pdfpageheight=\ht\figurebox', r'\advance\pdfpageheight by\dp\figurebox',
        r'\hoffset=-1in', r'\voffset=-1in', r'\shipout\box\figurebox',
        r'\end{document}',
    ])
    with TemporaryDirectory(prefix='ctd-figure-') as directory:
        temp = Path(directory)
        (temp/'figure.tex').write_text(tex+'\n')
        build = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
             '-no-shell-escape', 'figure.tex'],
            cwd=temp, capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(build.stdout + build.stderr)
        pdf.write_bytes((temp/'figure.pdf').read_bytes())
    record = {'source': source, 'source_sha256': digest,
              'command': 'python scripts/plot_paper_timing_controls.py',
              'experiment_rerun': False,
              'uncertainty': 'Across-repeat SD, not CI; repeats share subjects.',
              'n_subjects': controls['n_subjects'],
              'n_repeats': controls['results']['all24']['n_repeats'],
              'figure': FIGURE, 'figure_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
              'number_keys': keys, 'rows': rows}
    (ROOT/'output/paper_timing_controls.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'Rendered {FIGURE}: {len(rows)} mean/SD labels from unchanged result artifact.')


if __name__ == '__main__':
    main()
