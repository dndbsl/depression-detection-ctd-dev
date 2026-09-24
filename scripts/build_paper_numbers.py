#!/usr/bin/env python3
"""Render manuscript display numbers and their source ledger from audited artifacts.

Run after experiments. TeX, JSON and Markdown share each exact display string.
The ledger records the full-precision source, JSON key and generating command.
"""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/ctd'))
from constants import CTD_FEATURE_NAMES
from feature_groups import FEATURE_CONFIGS, ASK_ONLY, CROSS, RES_ONLY
from ml_splits import N_BOOTSTRAPS, ALPHA
from pdch_experiment import N_SPLITS_OUTER, N_SPLITS_INNER, N_REPEATS, C_GRID

COMMANDS = {
    'output/daic_gate_results.json':'python src/ctd/ml_splits.py (copied aggregate gate artifact)',
    'output/timing_controls.json':'python src/ctd/timing_controls.py',
    'output/ctd_ablation_groups.json':'python src/ctd/ablation_groups.py',
    'output/ctd_interpretability.json':'python src/ctd/interpret.py',
    'output/pdch_ctd_results.json':'python src/ctd/pdch_experiment.py',
    'output/shared_signal_audit.json':'python src/ctd/shared_signal_audit.py',
    'output/official_split_ctd.json':'python scripts/official_split_sensitivity.py',
    'output/revision_review_context.json':'python scripts/revision_context.py',
    'output/fusion_wavlm_seed44.json':'recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun',
    'output/mean_prob_seed44.json':'recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun',
}
DATA = {}
NUMBERS = {}
CONTROL_CONFIGS = ('all24', 'no_ask', 'latency_structure', 'latency', 'structure')
CONTROL_FIGURE = 'paper/figures/timing-controls.pdf'
CONTROL_FIGURE_KEYS = [config+metric+stat for config in CONTROL_CONFIGS
                        for metric in ('F', 'A') for stat in ('mean', 'sd')]


def add(name, value, source, key, digits=3, signed=False, note=''):
    display = (str(value) if isinstance(value,str) else f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}')
    NUMBERS[name] = {'display':display, 'value':value, 'source':source, 'key':key,
                     'command':COMMANDS.get(source,'python scripts/build_paper_numbers.py; fixed protocol/code definition'),
                     'note':note}
    return display


def get(name, source, keys, digits=3, signed=False):
    if source not in DATA:
        DATA[source] = json.loads((ROOT/source).read_text())
    v=DATA[source]
    for key in keys: v=v[key]
    return add(name,v,source,'.'.join(map(str,keys)),digits,signed)


def main():
    # Successful before/after gates are a prerequisite for publishing numbers.
    before=json.loads((ROOT/'output/revision_20260924/gate_before.json').read_text())
    after=json.loads((ROOT/'output/revision_20260924/gate_after.json').read_text())
    assert before['exit_code']==after['exit_code']==0
    assert before['gate_md5']==after['gate_md5']=='ca0b9cb2dd4c799b3a7c12532ceafd7e'
    gate=ROOT/'src/ctd/outputs/ml_splits_results.json'
    (ROOT/'output/daic_gate_results.json').write_bytes(gate.read_bytes())
    src='output/daic_gate_results.json'
    for split in ('train','dev','test'):
        get(split+'N',src,['cohort',split,'n'],0)
        get(split+'Positive',src,['cohort',split,'n_depressed'],0)
    add('cleanN',sum(DATA[src]['cohort'][s]['n'] for s in ('train','dev','test')),src,'sum(cohort.*.n)',0)
    add('fitC',0.3,src,'session_mean_24d.models.logreg_l2.selected_hyperparam',1)
    get('gateDev',src,['session_mean_24d','models','logreg_l2','dev','macro_f1'],6)
    get('gateTest',src,['session_mean_24d','models','logreg_l2','test','macro_f1','point'],6)
    # The richer-functionals result is retained in the audit, not promoted.
    models=DATA[src]['functionals_240d']['models']
    best=max(models,key=lambda k:models[k]['dev']['balanced_accuracy'])
    get('functionalDev',src,['functionals_240d','models',best,'dev','macro_f1'])
    get('functionalTest',src,['functionals_240d','models',best,'test','macro_f1','point'])

    src='output/timing_controls.json'
    get('nestedN',src,['n_subjects'],0)
    for config in ('all24','no_ask','latency','structure','latency_structure'):
        for metric,short in (('macro_f1','F'),('roc_auc','A')):
            for stat in ('mean','sd'):
                get(config+short+stat,src,['results',config,metric,stat])
    get('nestedRepeats',src,['results','all24','n_repeats'],0)

    src='output/ctd_ablation_groups.json'
    for config in FEATURE_CONFIGS:
        get(config+'Dim',src,['configs',config,'n_features'],0)
        get(config+'C',src,['configs',config,'selected_C'],2)
        get(config+'Dev',src,['configs',config,'dev','macro_f1'])
        get(config+'Test',src,['configs',config,'test','macro_f1','point'])
    for split in ('dev','test'):
        for key in ('point','ci_low','ci_high'):
            get('ablation'+split+key,src,['primary_ablation_paired_bootstrap',split,key],signed=True)
    src='output/ctd_interpretability.json'
    for feat in ('ask_d','res_h'):
        for key in ('coef','ci_low','ci_high'):
            get(feat+key,src,['coefficients',feat,key],signed=True)

    src='output/fusion_wavlm_seed44.json'
    rows={'A':'single_wavlm','T':'single_roberta','CTD':'single_ctd','AT':'wconvex[wavlm+roberta]',
          'ACTD':'wconvex[wavlm+ctd]','TCTD':'wconvex[roberta+ctd]'}
    for row,key in rows.items():
        get(row+'Threshold',src,['results',key,'threshold'],4 if row=='T' else 2)
        get(row+'Dev',src,['results',key,'dev','macro_f1','point'])
        for stat in ('point','ci_low','ci_high'):
            get(row+'Test'+stat,src,['results',key,'test','macro_f1',stat])
    # The rerun canonical point estimates must reproduce the recorded CTD row.
    assert NUMBERS['CTDDev']['display']==NUMBERS['all24Dev']['display']
    assert NUMBERS['CTDTestpoint']['display']==NUMBERS['all24Test']['display']
    f=DATA[src]['results']
    add('fusionDelta',f['wconvex[roberta+ctd]']['test']['macro_f1']['point']-f['single_roberta']['test']['macro_f1']['point'],src,'results.wconvex[roberta+ctd].test.macro_f1.point - results.single_roberta.test.macro_f1.point')
    for mod in ('roberta','ctd'):
        get(mod+'Weight',src,['results','wconvex[roberta+ctd]','weights',mod],1)
    get('acousticWeight',src,['results','wconvex[wavlm+roberta+ctd]','weights','wavlm'],1)
    src='output/mean_prob_seed44.json'
    for split in ('dev','test'):
        get('mean'+split,src,['mean_prob[roberta+ctd]',split,'macro_f1','point'])
    get('meanThreshold',src,['mean_prob[roberta+ctd]','threshold'],2)

    src='output/pdch_ctd_results.json'
    for key,short in (('n_sessions','pdchN'),('n_subjects','pdchSubjects'),('n_subjects_with_two_sessions','pdchRepeated'),('n_positive','pdchPositive')):
        get(short,src,['labels',key],0)
    for metric,short in (('macro_f1','F'),('roc_auc','A')):
        for stat in ('mean','sd'):
            get('pdch'+short+stat,src,['classification','all24',metric,stat])
    get('pdchRepeats',src,['classification','all24','n_repeats'],0)
    src='output/shared_signal_audit.json'
    for corpus in ('daic','pdch'):
        for key,raw in (('r','r'),('p','p_perm'),('q','q_bh')):
            get(corpus+'Latency'+key,src,['E4_secondary_targets','F2',corpus,'per_feature','res_h__amean',raw])
    get('binaryHits',src,['E2_concordance','concordance','counts','confirmed_shared'],0)
    get('signAgree',src,['E2_concordance','concordance','sign_agreement','n_agree'],0)
    get('liveDim',src,['E2_concordance','concordance','sign_agreement','n_comparable'],0)
    for key,raw in (('r','r'),('q','q_bh')):
        get('movingLatency'+key,src,['E4_secondary_targets','F3','daic_phq8_moving','per_feature','res_h__amean',raw])
    for key in ('max_abs_r_observed','max_abs_r_permutation_p'):
        get('moving'+key,src,['E4_secondary_targets','F3','daic_phq8_moving',key])
    get('maxBinaryP',src,['E2_concordance','pdch','max_abs_r_permutation_p'])
    get('fdr',src,['settings','fdr_q'],2)
    get('permutations',src,['settings','n_permutations'],0)
    get('silenceSeconds',src,['settings','silence_threshold_s'],1)
    get('rankLive',src,['E0_inventory','algebraic_redundancy','daic','ask','matrix_rank_22_live_centered'],0)
    assert DATA[src]['E0_inventory']['algebraic_redundancy']['pdch']['ask']['matrix_rank_22_live_centered']==NUMBERS['rankLive']['value']
    power = ['E1_confounds', 'e1c_power_calibration']
    get('powerN',src,power+['n_subsamples'],0)
    for metric,short in (('macro_f1','F'),('roc_auc','A')):
        for stat in ('mean','sd'):
            get('power'+short+stat,src,power+[metric,stat])
    get('powerFpercentile',src,power+['pdch_located_in_daic_null','macro_f1',
                                  'percentile_within_daic_at_n62'],0)

    src='output/official_split_ctd.json'
    for split in ('train','dev','test'):
        get('official'+split+'N',src,['cohort',split,'n'],0)
    add('officialN',sum(DATA[src]['cohort'][s]['n'] for s in ('train','dev','test')),src,'sum(cohort.*.n)',0)
    add('excludedN',sum(DATA[src]['cohort'][s]['integrity_exclusions_restored'] for s in ('train','dev','test')),src,'sum(cohort.*.integrity_exclusions_restored)',0)
    add('noPairsN',sum(len(x) for x in DATA[src]['sessions_without_turn_pairs'].values()),src,'sum(length(sessions_without_turn_pairs.*))',0)
    get('officialC',src,['result','selected_C'],1)
    get('officialDev',src,['result','dev','macro_f1'])
    get('officialTest',src,['result','test','macro_f1','point'])
    get('officialAuc',src,['result','test','roc_auc','point'])

    src='output/revision_review_context.json'
    get('agarwalTest',src,['SLT06','test_macro_f1'],2)
    for enc in ('wavlm','roberta'):
        for stat in ('mean','sd'):
            get(enc+'Seed'+stat,src,['SLT08',enc+'_test_seed_'+stat])
        get(enc+'Seeds',src,['SLT08',enc+'_n_seeds'],0)
    get('wavlmSeed',src,['SLT08','wavlm_selected_seed'],0)
    # Fixed settings: export with explicit committed-code provenance too.
    fixed=[('outerFolds',N_SPLITS_OUTER,'src/ctd/pdch_experiment.py','N_SPLITS_OUTER',0),
           ('innerFolds',N_SPLITS_INNER,'src/ctd/pdch_experiment.py','N_SPLITS_INNER',0),
           ('bootstrapN',N_BOOTSTRAPS,'src/ctd/ml_splits.py','N_BOOTSTRAPS',0),
           ('confidence',100-ALPHA,'src/ctd/ml_splits.py','100-ALPHA',0),
           ('threshold',0.5,'src/ctd/ml_splits.py','LogisticRegression.predict decision rule',1),
           ('phqCutoff',10,'src/common/daic_cleaning.py','PHQ8_DEPRESSED_THRESHOLD',0),
           ('correctedSession',409,'src/common/daic_cleaning.py','KNOWN_ERRORS.relabel',0),
           ('hamdCutoff',17,'src/ctd/pdch_labels.py','primary label cutoff',0),
           ('annotationSeconds',1,'src/ctd/pdch_adapter.py','parse_turns MM:SS annotation resolution',0),
           ('cGrid',', '.join(str(c) for c in C_GRID),'src/ctd/pdch_experiment.py','C_GRID',0),
           ('weightStep',0.1,'src/fusion/run_fusion_wavlm_seeds.py','weight grid',1),
           ('thresholdLow',0.05,'src/fusion/run_fusion_wavlm_seeds.py','threshold grid start',2),
           ('thresholdHigh',0.95,'src/fusion/run_fusion_wavlm_seeds.py','threshold grid stop',2),
           ('thresholdStep',0.05,'src/fusion/run_fusion_wavlm_seeds.py','threshold grid step',2)]
    for name,val,source,key,digits in fixed: add(name,val,source,key,digits)
    add('unitSum',1,'output/shared_signal_audit.json','E0_inventory.algebraic_redundancy.daic.ask.identity',0)
    add('crossDim',len(CROSS),'src/ctd/feature_groups.py','len(CROSS)',0)
    for source in {v['source'] for v in NUMBERS.values()}:
        digest=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
        for v in NUMBERS.values():
            if v['source']==source: v['source_sha256']=digest
    manuscript=(ROOT/'paper/main.tex').read_text().splitlines()
    figure_used = any('{figures/timing-controls.pdf}' in line for line in manuscript)
    for key,v in NUMBERS.items():
        v['manuscript_lines']=[i for i,line in enumerate(manuscript,1) if r'\R{'+key+'}' in line]
        v['figure_paths'] = [CONTROL_FIGURE] if figure_used and key in CONTROL_FIGURE_KEYS else []
    (ROOT/'output/paper_numbers.json').write_text(json.dumps(NUMBERS,indent=2)+'\n')
    tex=['% Generated by scripts/build_paper_numbers.py; do not edit display values.',r'\newcommand{\R}[1]{\csname result#1\endcsname}']
    tex += [r'\expandafter\def\csname result'+k+r'\endcsname{'+v['display']+'}'
            for k,v in NUMBERS.items() if v['manuscript_lines']]
    (ROOT/'paper/generated-numbers.tex').write_text('\n'.join(tex)+'\n')
    md=['# Manuscript number ledger','',
        'Exact display strings; full precision, source hashes, keys, manuscript lines and figure paths are in `paper_numbers.json`. Only used TeX macros are emitted. Figure labels use the same display strings; other ledger entries preserve audited checks.','',
        '| Manuscript number key | Display value | Source file → key | Command / provenance | Used in current paper |','|---|---|---|---|---|']
    for key,v in NUMBERS.items():
        locations = []
        if v['manuscript_lines']: locations.append('TeX lines '+', '.join(map(str,v['manuscript_lines'])))
        if v['figure_paths']: locations.extend(v['figure_paths'])
        md.append(f"| `{key}` | {v['display']} | `{v['source']}` → `{v['key']}` | `{v['command']}` | {'; '.join(locations) or 'audit only'} |")
    (ROOT/'output/paper_numbers.md').write_text('\n'.join(md)+'\n')
    print(f'Generated {len(NUMBERS)} display values and provenance records.')

if __name__=='__main__': main()
