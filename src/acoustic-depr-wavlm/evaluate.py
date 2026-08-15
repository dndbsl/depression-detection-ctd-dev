"""Evaluate a trained checkpoint on a split with 95% bootstrap confidence intervals.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt \
        --manifest-dir manifests/ --cache-dir cache/ --split test
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CACHE_DIR, MANIFEST_DIR
from dataset import SubjectFeatureDataset, collate_fn
from metrics import best_threshold, compute_metrics_with_ci, format_metrics_table
from model.full_model import build_model


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    pids, labels, probs = [], [], []
    for batch in loader:
        feats = batch["features"].to(device)
        mask = batch["mask"].to(device)
        logits, _ = model(feats, mask)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
        pids.extend(batch["pids"])
    return np.array(pids), np.concatenate(labels), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest-dir", default=MANIFEST_DIR)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed decision threshold. If omitted, uses the dev-tuned "
                         "threshold stored in the checkpoint, falling back to 0.5.")
    ap.add_argument("--tune-threshold", action="store_true",
                    help="Recompute the threshold on DEV (max dev macro-F1) and apply "
                         "it to --split. Never uses the eval split to pick the threshold.")
    ap.add_argument("--num-bootstraps", type=int, default=1000)
    ap.add_argument("--alpha", type=int, default=5)
    ap.add_argument("--out", default=None, help="Output JSON path (default results_{split}.json)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[eval] loaded checkpoint from epoch {ckpt.get('epoch')} "
          f"(dev_loss={ckpt.get('dev_loss'):.4f})")

    # Threshold resolution (priority): explicit --threshold > --tune-threshold on
    # dev > checkpoint's stored dev-tuned threshold > 0.5. The threshold is always
    # chosen on dev (or given), never on the eval split itself.
    if args.threshold is not None:
        threshold = args.threshold
        thr_source = "cli"
    elif args.tune_threshold:
        dev_ds = SubjectFeatureDataset("dev", args.manifest_dir, args.cache_dir)
        dev_loader = DataLoader(dev_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
        _, dev_labels, dev_probs = run_inference(model, dev_loader, device)
        threshold = best_threshold(dev_labels, dev_probs)
        thr_source = "tuned-on-dev"
    elif ckpt.get("threshold") is not None:
        threshold = float(ckpt["threshold"])
        thr_source = "checkpoint(dev-tuned)"
    else:
        threshold = 0.5
        thr_source = "default-0.5"
    print(f"[eval] threshold={threshold:.3f} (source: {thr_source})")

    ds = SubjectFeatureDataset(args.split, args.manifest_dir, args.cache_dir)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    pids, labels, probs = run_inference(model, loader, device)

    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    results = compute_metrics_with_ci(
        labels, probs, threshold=threshold,
        num_bootstraps=args.num_bootstraps, alpha=args.alpha)

    print(f"\n[eval] split={args.split} n_subjects={len(labels)} "
          f"(pos={int(labels.sum())} neg={int((labels==0).sum())})")
    print(f"[eval] confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"\n{format_metrics_table(results)}\n")

    out_path = args.out or f"results_{args.split}.json"
    payload = {
        "split": args.split,
        "checkpoint": os.path.abspath(args.checkpoint),
        "threshold": threshold,
        "threshold_source": thr_source,
        "num_bootstraps": args.num_bootstraps,
        "alpha": args.alpha,
        "n_subjects": int(len(labels)),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "metrics": results,
        "per_subject": [
            {"pid": int(p), "label": int(l), "prob": float(pr), "pred": int(pd_)}
            for p, l, pr, pd_ in zip(pids, labels, probs, preds)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
