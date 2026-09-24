#!/usr/bin/env python3
"""Check gate integrity, exact manuscript display strings and source hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-dir', default='output/title_author_20260924')
    args = parser.parse_args()
    records=ROOT/'output/revision_20260924'
    before=json.loads((records/'gate_before.json').read_text())
    after=json.loads((records/'gate_after.json').read_text())
    assert before['gate_md5']==after['gate_md5']=='ca0b9cb2dd4c799b3a7c12532ceafd7e'
    assert hashlib.md5((ROOT/'src/ctd/outputs/ml_splits_results.json').read_bytes()).hexdigest()==before['gate_md5']
    initial=json.loads((records/'initial_state.json').read_text())
    for p in ('src/ctd/constants.py','src/ctd/feature_groups.py','src/ctd/feature_extraction.py','src/ctd/turn_pairing.py','src/ctd/ml_splits.py'):
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==initial['initial_sha256'][p],p
    numbers=json.loads((ROOT/'output/paper_numbers.json').read_text())
    tex=(ROOT/'paper/main.tex').read_text()
    generated=(ROOT/'paper/generated-numbers.tex').read_text()
    md=(ROOT/'output/paper_numbers.md').read_text()
    used=set(re.findall(r'\\R\{([^}]+)\}',tex))
    assert used and not (used-set(numbers)),used-set(numbers)
    defined=set(re.findall(r'\\csname result([^\\]+)\\endcsname\{',generated))
    assert defined == used, {'unused':sorted(defined-used),'missing':sorted(used-defined)}
    figure=json.loads((ROOT/'output/paper_timing_controls.json').read_text())
    figure_keys=set(figure['number_keys'])
    assert '{figures/timing-controls.pdf}' in tex
    assert hashlib.sha256((ROOT/figure['figure']).read_bytes()).hexdigest()==figure['figure_sha256']
    assert hashlib.sha256((ROOT/figure['source']).read_bytes()).hexdigest()==figure['source_sha256']
    for row in figure['rows']:
        mean,sd=numbers[row['mean_key']],numbers[row['sd_key']]
        assert row['label']==f"{mean['display']} ± {sd['display']}"
        assert row['mean']==mean['value'] and row['sd']==sd['value']
    for key in used|figure_keys:
        v=numbers[key]
        if key in used:
            assert r'\csname result'+key+r'\endcsname{'+v['display']+'}' in generated,key
        if key in figure_keys:
            assert v['figure_paths']==[figure['figure']],key
        assert f"| `{key}` | {v['display']} |" in md,key
        assert hashlib.sha256((ROOT/v['source']).read_bytes()).hexdigest()==v['source_sha256'],v['source']
    for p in ('output/fusion_wavlm_seed44.json','output/mean_prob_seed44.json','output/summary_wavlm_seeds.json'):
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==initial['initial_sha256'][p],p
    # All scientific results and protocol numerals must be ledger macros. Exempt
    # instrument names, author ORCIDs/affiliations, model notation and citations.
    body=tex.split(r'\begin{abstract}',1)[1]
    stripped=re.sub(r'\\R\{[^}]+\}|\\(?:cite|label|ref|url|href)\{[^}]+\}', '', body)
    stripped=stripped.replace('PHQ-8','').replace('HAMD-17','').replace('PHQ8','').replace('L_2','')
    stripped=re.sub(r'%[^\n]*','',stripped)
    stripped=re.sub(r'\\setlength\{\\tabcolsep\}\{[^}]+\}', '', stripped)
    raw=re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?',stripped)
    assert not raw, f'Unledgered body numerals: {raw}'
    banned=['CTD transfers','replicates on PDCH','generalizes cross-lingually','matches or beats','establishes superiority','proves','0.829','0.779']
    assert not [w for w in banned if w in tex]
    assert r'\title{Can Conversational Temporal Dynamics Improve Depression Detection in Dyads? A Preliminary Investigation in Multi-Modality Perspectives}' in tex
    presentation=json.loads((ROOT/'output/presentation_20260924/initial_state.json').read_text())
    for path,digest in presentation['scientific_and_result_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    result={'status':'pass','gate_md5':before['gate_md5'],'unique_manuscript_number_keys':len(used),
            'unique_figure_number_keys':len(figure_keys),'unique_paper_number_keys':len(used|figure_keys),
            'scientific_source_files_unchanged':True,'recorded_neural_artifacts_unchanged':True,
            'all_prior_scientific_artifacts_unchanged':True,'exact_display_strings_verified':True,
            'no_unused_tex_macros':True,'used_keys':sorted(used|figure_keys)}
    destination=ROOT/args.record_dir
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'number_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='used_keys'},indent=2))

if __name__=='__main__': main()
