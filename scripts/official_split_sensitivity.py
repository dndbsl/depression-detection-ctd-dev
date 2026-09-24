#!/usr/bin/env python3
"""SLT-07: one CTD sensitivity run on all official DAIC 107/35/47 sessions.

Reuse the fixed CTD definitions and deployed C grid/selection/refit protocol.
Retain sessions excluded by the canonical integrity policy; sessions without
Ask/Res pairs receive missing descriptors and training-fitted median imputation.
This is an annotation-quality sensitivity analysis, not a replacement baseline.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/ctd'))
from constants import CTD_FEATURE_NAMES, DEFAULT_DATA_ROOT
from ml_splits import LABEL_FILES, load_transcript, transcript_path
from turn_pairing import build_turn_pairs
from feature_extraction import extract_session_turn_features
from ablation_groups import run_config
from common.daic_cleaning import excluded_ids


def main():
    columns = [f'{f}__amean' for f in CTD_FEATURE_NAMES]
    data, cohort, no_pairs = {}, {}, {}
    for split, filename in LABEL_FILES.items():
        labels = pd.read_csv(Path(DEFAULT_DATA_ROOT) / 'labels' / filename)
        score = 'PHQ8_Score' if 'PHQ8_Score' in labels else 'PHQ_Score'
        rows, empty = [], []
        for _, label in labels.sort_values('Participant_ID').iterrows():
            sid = int(label.Participant_ID)
            transcript = load_transcript(transcript_path(DEFAULT_DATA_ROOT, sid))
            pairs = build_turn_pairs(sid, transcript)
            means = extract_session_turn_features(pairs)[CTD_FEATURE_NAMES].mean() if pairs else pd.Series(np.nan, index=CTD_FEATURE_NAMES)
            if not pairs:
                empty.append(sid)
            rows.append({'session_id': sid, 'PHQ8_Binary': int(label[score] >= 10),
                         **{f'{f}__amean': float(means[f]) for f in CTD_FEATURE_NAMES}})
        data[split] = pd.DataFrame(rows)
        cohort[split] = {'n': len(rows), 'n_positive': int(data[split].PHQ8_Binary.sum()),
                         'integrity_exclusions_restored': len(set(data[split].session_id) & excluded_ids())}
        no_pairs[split] = empty
    assert [cohort[s]['n'] for s in ('train','dev','test')] == [107,35,47]
    result = run_config(columns, data)
    result = {k:v for k,v in result.items() if not k.startswith('_')}
    payload = {
        'protocol': 'All official sessions; PHQ-8 >=10 including corrected session 409; unchanged 24 CTD means; median imputation and scaling fitted on train; C in 0.01/0.03/0.1/0.3/1.0 selected by dev balanced accuracy; threshold 0.5; train+dev refit for test.',
        'cohort': cohort, 'sessions_without_turn_pairs': no_pairs,
        'missing_feature_policy': 'All 24 coordinates missing for no-pair sessions; use training-fitted feature medians. No reconstructed interviewer turns.',
        'cleaning_caveat': 'Nine integrity exclusions restored, including missing interviewer turns and timestamp problems. Results do not establish measurement quality or a matched published-baseline comparison.',
        'neural_scope': 'T and T+CTD cannot be evaluated for the official cohort: frozen caches, selected checkpoints and per-session predictions are unavailable.',
        'result': result,
    }
    out = ROOT / 'output'
    (out/'official_split_ctd.json').write_text(json.dumps(payload, indent=2)+'\n')
    d,t = result['dev'], result['test']
    (out/'official_split_ctd.md').write_text(
        '# SLT-07: official-split CTD sensitivity\n\n'+payload['protocol']+'\n\n'
        '| Cohort | C | Dev macro-F1 | Test macro-F1 | Test AUC |\n|---|---:|---:|---:|---:|\n'
        f"| 107/35/47 | {result['selected_C']} | {d['macro_f1']:.3f} | {t['macro_f1']['point']:.3f} | {t['roc_auc']['point']:.3f} |\n\n"
        +payload['cleaning_caveat']+'\n\n'+payload['missing_feature_policy']
        +f' No-pair sessions: {no_pairs}.\n\n'+payload['neural_scope']+'\n')
    print(json.dumps(payload, indent=2))

if __name__ == '__main__':
    main()
