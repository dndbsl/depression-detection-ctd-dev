# Score-level late fusion (A / T / CTD)

Fuses the deployed per-modality session probabilities — frozen WavLM-large (A),
frozen RoBERTa-large (T), and the 24-D CTD detector — with parameter-free
(`mean_prob`, `mean_logit`) and dev-tuned convex (`wconvex`) rules. All weight
and threshold choices are made on dev; test is evaluated once. Every metric
carries a 2000-bootstrap 95% CI.

## Prerequisites

All three single-modality pipelines must have been run first (see the
repository README): the WavLM feature cache + seed checkpoints, the RoBERTa
seed-43 run outputs, and the CTD environment.

## Run order

```bash
python extract_ctd_roberta.py         # CTD + RoBERTa session embeddings
                                      # -> ../ctd/outputs/fusion/{ctd,roberta}_{train,dev,test}.npz
python check_splits.py                # verify identical sessions + labels across modalities
python run_fusion_wavlm_seeds.py --wavlm-seeds 44 --device cuda
                                      # -> ../ctd/outputs/fusion_by_wavlm_seed/fusion_wavlm_seed44.json
python paired_delta_analysis.py       # paired bootstrap deltas + prediction-change counts
```

| Script | Role |
|---|---|
| `extract_ctd_roberta.py` | Pre-prediction session representations for CTD and RoBERTa |
| `check_splits.py` | Split/label consistency check across all modalities |
| `run_fusion_wavlm_seeds.py` | Canonical fusion evaluation (selectable WavLM seed; paper uses seed 44) |
| `fusion_late.py` | Same fusion logic reading the deployed `linear_probe` WavLM checkpoint |
| `paired_delta_analysis.py` | Paired uncertainty/error analysis of the deployed RoBERTa+CTD rule |

The published aggregate outputs of these scripts are checked in under
[`output/`](../../output/) at the repository root.
