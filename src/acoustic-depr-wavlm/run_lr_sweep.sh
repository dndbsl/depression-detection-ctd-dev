#!/usr/bin/env bash
# Full 400-epoch training run for each candidate learning rate.
# Only the LR varies (seed=72, batch_size=32, no early stopping) so dev-loss is comparable.
set -euo pipefail
cd "$(dirname "$0")"

PROJECT="acoustic-depr-daic"
EPOCHS=400

declare -A LRS=( ["1e-5"]=0.00001 ["5e-5"]=0.00005 ["1e-4"]=0.0001 )

for tag in 1e-5 5e-5 1e-4; do
    lr="${LRS[$tag]}"
    echo "=================================================================="
    echo "[sweep] lr=$lr (tag=$tag) -> checkpoints/lr_$tag"
    echo "=================================================================="
    python train.py \
        --lr "$lr" \
        --max-epochs "$EPOCHS" \
        --checkpoint-dir "checkpoints/lr_$tag" \
        --wandb-project "$PROJECT" \
        --run-name "lr_$tag" \
        2>&1 | grep -Ev "FutureWarning|weights_only|^  blob"
done

echo "[sweep] all runs complete."
