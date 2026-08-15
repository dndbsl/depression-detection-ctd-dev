#!/usr/bin/env bash
# Train WavLM linear probe for multiple seeds; evaluate each with dev-tuned threshold.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(71 73 42 43 44)
LOG_DIR="checkpoints/seed_sweep_logs"
mkdir -p "$LOG_DIR"

for seed in "${SEEDS[@]}"; do
  out="checkpoints/seed_${seed}"
  if [[ -f "${out}/best.pt" ]]; then
    echo "[skip] seed ${seed}: ${out}/best.pt already exists"
    continue
  fi
  echo "=================================================================="
  echo "[sweep] seed=${seed} -> ${out}"
  echo "=================================================================="
  python train.py \
    --seed "$seed" \
    --checkpoint-dir "$out" \
    --wandb-project acoustic-depr-daic \
    --run-name "wavlm_seed${seed}" \
    2>&1 | tee "${LOG_DIR}/train_seed${seed}.log"
done

echo ""
echo "=== Evaluation (dev-tuned threshold, test split) ==="
printf "%-6s %10s %8s %12s %12s %12s\n" seed epoch dev_loss thr dev_mf1 test_mf1
printf "%-6s %10s %8s %12s %12s %12s\n" ---- ------ -------- --- ------- --------

for seed in 72 "${SEEDS[@]}"; do
  ckpt="checkpoints/seed_${seed}/best.pt"
  [[ -f "$ckpt" ]] || continue
  out_json="results_test_seed${seed}.json"
  python evaluate.py --checkpoint "$ckpt" --split test --out "$out_json" 2>&1 | tee "${LOG_DIR}/eval_seed${seed}.log"
  python - "$ckpt" "$out_json" <<'PY'
import json, sys, torch
from sklearn.metrics import f1_score
import numpy as np

ckpt_path, json_path = sys.argv[1], sys.argv[2]
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
with open(json_path) as f:
    rep = json.load(f)
thr = rep["threshold"]
test_mf1 = rep["metrics"]["macro_f1"]["center"]
# dev macro-f1 at tuned thr from per-subject would need re-run; use checkpoint threshold + summary
print(f"{ck.get('seed','?')} {ck['epoch']:10d} {ck['dev_loss']:8.4f} {thr:12.3f} {'':12s} {test_mf1:12.4f}")
PY
done

echo ""
echo "Done. See ${LOG_DIR}/ and results_test_seed*.json"
