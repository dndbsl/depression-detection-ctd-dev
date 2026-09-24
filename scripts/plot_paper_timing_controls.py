#!/usr/bin/env python3
"""Render existing nested-control estimates; no model fitting or new analysis.

Run build_paper_numbers.py first. Exact printed values share the paper ledger;
full-precision audited means and SDs determine points and whisker lengths.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ('all24', 'no_ask', 'latency_structure', 'latency', 'structure')
LABELS = ('Full CTD', 'No ask-side group', 'Latency + structure', 'Latency', 'Structure')
COLORS = ('#174d79', '#a55c00', '#087c7c', '#646464', '#646464')
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

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.size': 9.5, 'axes.titlesize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.linewidth': 0.5})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.12), sharey=True)
    fig.subplots_adjust(left=0.215, right=0.994, top=0.81, bottom=0.14, wspace=0.12)
    fig.text(0.215, 0.985, 'Whiskers: ±SD across matched repeats (not confidence intervals)',
             ha='left', va='top', fontsize=9.5)
    rows = []
    for ax, metric, title in zip(axes, ('F', 'A'), ('Macro-F1', 'ROC-AUC')):
        ax.set_title(title, loc='left', fontweight='bold', pad=8)
        ax.set_xlim(0.45, 0.90)
        ax.set_ylim(4.55, -0.6)
        ax.set_xticks([0.5, 0.6, 0.7])
        ax.set_yticks(range(len(CONFIGS)))
        ax.set_yticklabels(LABELS)
        ax.tick_params(axis='y', length=0, pad=8)
        ax.tick_params(axis='x', length=3, color='#888888')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_bounds(0.45, 0.73)
        ax.spines['bottom'].set_color('#888888')
        for y, (cfg, color) in enumerate(zip(CONFIGS, COLORS)):
            mean, sd = numbers[cfg+metric+'mean'], numbers[cfg+metric+'sd']
            label = f"{mean['display']} ± {sd['display']}"
            ax.hlines(y, 0.45, 0.73, color='#e6e6e6', linewidth=0.55, zorder=0)
            ax.errorbar(mean['value'], y, xerr=sd['value'], fmt='o',
                        color=color, markersize=4.5, capsize=3,
                        elinewidth=1.05, markeredgewidth=0.9)
            ax.text(0.896, y, label, ha='right', va='center', fontsize=9.5,
                    color=color, fontweight='bold' if cfg=='all24' else 'normal')
            rows.append({'config': cfg, 'metric': title, 'label': label,
                         'mean_key': cfg+metric+'mean', 'sd_key': cfg+metric+'sd',
                         'mean': mean['value'], 'sd': sd['value']})
    axes[1].tick_params(axis='y', labelleft=False)
    pdf = ROOT/FIGURE
    fig.savefig(pdf, metadata={'Title': 'DAIC nested timing controls: means and standard deviations',
                              'Creator': 'scripts/plot_paper_timing_controls.py',
                              'CreationDate': None, 'ModDate': None})
    plt.close(fig)
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
