#!/usr/bin/env python3
"""Refresh the recorded-run and review-scope evidence without fitting encoders."""
from pathlib import Path
import hashlib
import json
import subprocess
import importlib.util
import torch

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '49520dc65987cf1f686e4aa28e0fd742778f6b31'


def main():
    roots = [Path('/home/exouser/projects'), Path('/home/exouser/data'), Path('/home/exouser/.cache')]
    patterns = ['*.npz', '*.pt', '*.pth', '*.safetensors', '*predictions*.csv']
    matches = sorted({str(p) for root in roots if root.exists() for pat in patterns for p in root.rglob(pat) if p.is_file()})
    sources = ['output/fusion_wavlm_seed44.json', 'output/mean_prob_seed44.json', 'output/summary_wavlm_seeds.json', 'RESULTS.md', 'reviews/slt2026/reviews.md']
    committed = {p: {'commit': COMMIT, 'sha256': hashlib.sha256(subprocess.check_output(['git','show',f'{COMMIT}:{p}'],cwd=ROOT)).hexdigest()} for p in sources}
    data = {
        'compute': {'cuda_available': torch.cuda.is_available(), 'transformers_installed': importlib.util.find_spec('transformers') is not None},
        'artifact_search': {'roots': list(map(str,roots)), 'patterns': patterns, 'matches': matches},
        'recorded_sources': committed,
        'neural_rerun': 'Not rerun: frozen embeddings, selected checkpoints and per-session predictions unavailable. No replacement seed or encoder run generated.',
        'E6': {'status': 'scoped_out', 'reason': 'No raw frozen RoBERTa cache, encoder weights, checkpoint, or predictions found. Downloading the large encoder and rebuilding raw embeddings and nested heads is beyond this bounded CPU revision; aggregate metrics cannot support nested fusion.', 'nested_results': None},
        'SLT06': {'status': 'context_only', 'paper': 'Agarwal, Dias and Dollfus (CLPsych 2024)', 'source': 'https://aclanthology.org/2024.clpsych-1.9.pdf', 'verified_utc_date': '2026-09-24', 'source_tables': [1,2], 'cohort': [107,35,47], 'test_macro_f1': 0.80, 'input': 'patient and therapist text', 'limitation': 'Published result; no matched-protocol reproduction. Author implementation and sentence-encoder artifacts are not available locally; retraining is outside this pass.'},
        'SLT08': {'status': 'probe_specific_disclosure', 'wavlm_selected_seed': 44, 'selection': 'best test macro-F1 at dev-tuned threshold', 'source': f'{COMMIT}:RESULTS.md sections 2.1, 2.2', 'wavlm_test_seed_mean': 0.506, 'wavlm_test_seed_sd': 0.034, 'wavlm_n_seeds': 6, 'roberta_test_seed_mean': 0.604, 'roberta_test_seed_sd': 0.025, 'roberta_n_seeds': 3, 'limitation': 'Frozen probe only; zero selected acoustic weight is not acoustic redundancy. A stronger acoustic rerun is not feasible with missing caches/checkpoints.'},
        'fusion_paired_uncertainty': {'status':'recorded_qualitative_only', 'source': f'{COMMIT}:reviews/slt2026/reviews.md; paper/revision-history/main-before-review-20260924.tex', 'finding':'Paired intervals for fusion minus each component include or touch zero.', 'limitation':'No committed numerical paired-CI artifact or underlying predictions found; exact paired limits cannot be regenerated. Do not infer paired uncertainty from overlapping marginal intervals.'},
        'venue': {'source':'https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php', 'verified_utc_date':'2026-09-24', 'max_content_pages':4, 'max_total_pages':5, 'review':'not double blind unless otherwise specified', 'deadline_check':'CFP lists September 23, 2026; paper kit lists September 16. Verify existing submission/access and applicable timezone; no submission action taken.', 'cfp_source':'https://2027.ieeeicassp.org/call-for-papers/', 'author_policy_source':'https://2027.ieeeicassp.org/sps-policies/', 'author_statements':'Ethics compliance and funding/conflict statements required; author facts requested asynchronously and not inferred.'},
    }
    (ROOT/'output/revision_review_context.json').write_text(json.dumps(data,indent=2)+'\n')
    (ROOT/'output/revision_review_context.md').write_text(
        '# Review-item scope and recorded neural provenance\n\n'
        '- SLT-06: Agarwal et al., published test macro-F1 0.80 on 107/35/47, verified in Tables 1–2 of '+data['SLT06']['source']+'. Both speakers’ text; not a matched-protocol rerun.\n'
        '- SLT-07: see `official_split_ctd.{json,md}`; missing neural inputs prevent official-cohort T or T+CTD evaluation.\n'
        '- SLT-08: WavLM seed 44 was selected using test macro-F1. Recorded test seed mean 0.506 ± 0.034 (6 seeds); RoBERTa 0.604 ± 0.025 (3 seeds). '+data['SLT08']['limitation']+'\n'
        '- E6: '+data['E6']['reason']+' No nested fusion scores.\n'
        '- Fusion paired uncertainty: '+data['fusion_paired_uncertainty']['finding']+' '+data['fusion_paired_uncertainty']['limitation']+'\n'
        '- Original neural JSON artifacts remain byte-identical to the starting commit. Search evidence and hashes are in `revision_review_context.json`.\n')
    print(json.dumps({'matches':matches,'E6':data['E6']},indent=2))

if __name__ == '__main__':
    main()
