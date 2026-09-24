# Review-item scope and recorded neural provenance

- SLT-06: Agarwal et al., published test macro-F1 0.80 on 107/35/47, verified in Tables 1–2 of https://aclanthology.org/2024.clpsych-1.9.pdf. Both speakers’ text; not a matched-protocol rerun.
- SLT-07: see `official_split_ctd.{json,md}`; missing neural inputs prevent official-cohort T or T+CTD evaluation.
- SLT-08: WavLM seed 44 was selected using test macro-F1. Recorded test seed mean 0.506 ± 0.034 (6 seeds); RoBERTa 0.604 ± 0.025 (3 seeds). Frozen probe only; zero selected acoustic weight is not acoustic redundancy. A stronger acoustic rerun is not feasible with missing caches/checkpoints.
- E6: No raw frozen RoBERTa cache, encoder weights, checkpoint, or predictions found. Downloading the large encoder and rebuilding raw embeddings and nested heads is beyond this bounded CPU revision; aggregate metrics cannot support nested fusion. No nested fusion scores.
- Fusion paired uncertainty: Paired intervals for fusion minus each component include or touch zero. No committed numerical paired-CI artifact or underlying predictions found; exact paired limits cannot be regenerated. Do not infer paired uncertainty from overlapping marginal intervals.
- Original neural JSON artifacts remain byte-identical to the starting commit. Search evidence and hashes are in `revision_review_context.json`.
