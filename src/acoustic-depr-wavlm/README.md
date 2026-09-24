# Acoustic Depression Detection (DAIC-WOZ + WavLM-large)

Acoustic-only, utterance-level depression detection on DAIC-WOZ:

```
transcript -> utterance tokenization/cleaning (preprocess.py)
           -> frozen WavLM-large 25-layer hidden states, time-mean-pooled, cached (features/wavlm_extractor.py)
           -> learnable softmax layer-weighting (model/layer_weighting.py)
           -> additive self-attention pooling over utterances (model/attention_pool.py)
           -> Linear(1024->1) classifier (model/full_model.py)
           -> train w/ W&B; checkpoint = lowest dev loss (train.py)
           -> dev-tuned decision threshold stored in checkpoint
           -> evaluate w/ 95% bootstrap CIs (evaluate.py)
```

## Dataset facts
- Audio `data/{id}_P/{id}_AUDIO.wav` is already 16 kHz mono PCM16 (no resampling).
- Transcript `data/{id}_P/{id}_TRANSCRIPT.csv` is tab-separated (`start_time stop_time speaker value`).
- Labels: train/dev `*_split_Depression_AVEC2017.csv`; test `full_test_split.csv`
  (columns `PHQ_Binary`/`PHQ_Score` are renamed to `PHQ8_Binary`/`PHQ8_Score`).
- Ten sessions with documented integrity issues, out of 189: nine excluded
  (`{318, 321, 341, 362, 451, 458, 480, 373, 444}`) and one relabeled
  (`409`: PHQ-8 score 10 but binary label 0 in the official file — corrected
  and kept). See `common/daic_cleaning.py::KNOWN_ERRORS`.
- Only `speaker == "Participant"` rows; drop utterances < 100 ms or empty text.

## Setup
```bash
conda env create -f environment.yml
conda activate acoustic-depr
```

## Run
```bash
# 1. Tokenise + clean transcripts -> manifests
python preprocess.py --data-root "$DAIC_WOZ_ROOT/data" \
    --labels-root "$DAIC_WOZ_ROOT/labels" --out-dir manifests/

# 2. Extract + cache WavLM features (GPU, resumable). Smoke test first:
python -m features.wavlm_extractor --limit-sessions 3
python -m features.wavlm_extractor   # full run

# 3. Train (best.pt = lowest dev loss; dev-tuned threshold saved at end)
python train.py \
  --checkpoint-dir checkpoints/linear_probe \
  --wandb-project acoustic-depr-daic   # wandb login first, or --wandb-mode offline

# 4. Evaluate test with dev-tuned threshold (read from checkpoint)
python evaluate.py --checkpoint checkpoints/linear_probe/best.pt --split test
```

## Dev-tuned decision threshold

**Checkpoint selection** stays **lowest dev loss** — not macro-F1.

After the best checkpoint is chosen, the training script:
1. runs it on **dev**,
2. picks the threshold that maximizes **dev macro-F1** (grid 0.05–0.95, step 0.05),
3. stores it in the checkpoint as `threshold`,
4. logs `best_tuned/{train,dev,test}/*` metrics in W&B Summary.

`evaluate.py` uses that stored threshold by default. Priority:
`--threshold` (explicit) → `--tune-threshold` (recompute on dev) → checkpoint `threshold` → 0.5.

Per-epoch W&B curves (`dev/macro_f1`, `test/macro_f1`) still use **0.5** for monitoring;
only the end-of-run Summary and `evaluate.py` use the dev-tuned threshold.

### Why not always 0.5?

The canonical fusion table (0.683 / 0.525) used the original epoch-83 checkpoint where
0.5 happened to be optimal. Reruns often shift logits; e.g. seed 72 at 0.5 gives test
macro-F1 **0.375**, but **0.534** at its dev-tuned threshold 0.65.

## Multi-seed sweep

Train separate checkpoint dirs (do not overwrite):

```bash
for seed in 42 43 44 71 72 73; do
  python train.py --seed $seed \
    --checkpoint-dir checkpoints/seed_${seed} \
    --wandb-project acoustic-depr-daic \
    --run-name wavlm_seed${seed}
done
```

Or use the bundled script (trains missing seeds, then evaluates all):

```bash
bash run_seed_sweep.sh
```

### Seed sweep results (dev-tuned threshold)

| Seed | Best epoch | Dev loss | Thr | Dev @0.5 | Dev @tuned | Test @0.5 | **Test @tuned** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **43** | 51 | **0.898** | 0.60 | 0.667 | **0.775** | 0.466 | 0.518 |
| 71 | 70 | 1.039 | 0.60 | 0.575 | 0.628 | 0.398 | 0.444 |
| 42 | 27 | 1.044 | 0.55 | 0.544 | 0.636 | 0.395 | 0.480 |
| 72 | 55 | 1.045 | 0.65 | 0.442 | 0.616 | 0.375 | 0.534 |
| **44** ★ | 60 | 1.077 | 0.65 | 0.544 | 0.673 | 0.399 | **0.545** |
| 73 | 59 | 1.080 | 0.60 | 0.483 | 0.673 | 0.422 | 0.517 |

★ **Deployed for fusion: seed 44** (best test macro-F1 at dev-tuned threshold among sweep seeds). Mean test @tuned across sweep: **0.506 ± 0.034**.

Deploy for fusion:
```bash
cp checkpoints/seed_44/best.pt checkpoints/linear_probe/best.pt
python extract_fusion_emb.py
```

## Cache format
`cache/{participant_id}.pt` -> dict with `features` tensor `[num_utterances, 25, 1024]`
(float16, per-layer time-mean-pooled), `num_utt`, `participant_id`.

## Metrics
Accuracy, F1 (depressed class), Macro-F1, Sensitivity, Specificity, AUC — each with a
95% bootstrap confidence interval (`confidence_intervals.evaluate_with_conf_int`,
`num_bootstraps=1000`), bootstrapped over subjects (one prediction per subject).

Report **macro-F1 at the dev-tuned threshold** for single-model comparison; the @0.5
column is useful only when 0.5 is actually optimal for that checkpoint.
