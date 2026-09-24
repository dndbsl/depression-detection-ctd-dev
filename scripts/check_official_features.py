#!/usr/bin/env python3
"""Validate the official-cohort sensitivity extraction against canonical caches."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/ctd'))
from constants import CTD_FEATURE_NAMES, DEFAULT_DATA_ROOT, OUTPUT_DIR
from ml_splits import load_transcript, transcript_path
from turn_pairing import build_turn_pairs
from feature_extraction import extract_session_turn_features

errors=[]
count=0
for split in ('train','dev','test'):
    reference=pd.read_csv(OUTPUT_DIR/f'functionals_{split}.csv').set_index('session_id')
    for sid,row in reference.iterrows():
        pairs=build_turn_pairs(int(sid),load_transcript(transcript_path(DEFAULT_DATA_ROOT,int(sid))))
        got=extract_session_turn_features(pairs)[CTD_FEATURE_NAMES].mean().to_numpy()
        expected=row[[f'{f}__amean' for f in CTD_FEATURE_NAMES]].to_numpy(float)
        np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
        errors.append(float(np.nanmax(np.abs(got-expected))))
        count+=1
payload={'canonical_sessions_checked':count,'max_absolute_feature_difference':max(errors),'tolerance':1e-12,'result':'pass','fit_performed':False}
(ROOT/'output/revision_20260924/official_feature_check.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload))
