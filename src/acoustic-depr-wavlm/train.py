"""Train the layer-weighting + attention-pool + classifier head on cached features.

Model selection: the checkpoint with the lowest dev (eval) loss is saved as
``checkpoints/best.pt``. Per-epoch train/dev/test loss and metrics are logged to
W&B (test is monitoring only; selection uses dev loss). See ``WANDB_METRICS.md``.

Usage:
    python train.py \
        --manifest-dir manifests/ --cache-dir cache/ \
        --checkpoint-dir checkpoints/ --wandb-project acoustic-depr-daic
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    CACHE_DIR,
    CHECKPOINT_DIR,
    DEFAULT_TRAIN_CONFIG,
    FEATURE_DIM,
    MANIFEST_DIR,
    NUM_LAYERS,
)
from dataset import SubjectFeatureDataset, collate_fn
from metrics import best_threshold, compute_metrics
from model.full_model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_inference(model, loader, device):
    """Return (labels, probs) over a split for threshold tuning / metrics."""
    model.eval()
    labels, probs = [], []
    for batch in loader:
        feats = batch["features"].to(device)
        mask = batch["mask"].to(device)
        logits, _ = model(feats, mask)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(labels).astype(int), np.concatenate(probs)


def run_epoch(model, loader, criterion, device, optimizer=None, label_smoothing=0.0,
              grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    all_labels, all_probs = [], []
    for batch in loader:
        feats = batch["features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        # Label smoothing: 1 -> 1 - eps/2, 0 -> eps/2 (loss only; metrics use raw labels).
        target = labels
        if label_smoothing > 0:
            target = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

        with torch.set_grad_enabled(train):
            logits, _ = model(feats, mask)
            loss = criterion(logits, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    metrics = compute_metrics(labels, probs)
    return total_loss / max(n, 1), metrics


def main():
    cfg = DEFAULT_TRAIN_CONFIG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", default=MANIFEST_DIR)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--wandb-project", default="acoustic-depr-daic")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--run-name", default=None)
    # --- model architecture (linear probe) ---
    ap.add_argument("--attn-hidden-dim", type=int, default=cfg.attn_hidden_dim)
    ap.add_argument("--dropout", type=float, default=cfg.dropout)
    # --- optimisation ---
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    ap.add_argument("--label-smoothing", type=float, default=cfg.label_smoothing)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--max-epochs", type=int, default=cfg.max_epochs)
    ap.add_argument("--early-stopping", action="store_true", default=cfg.early_stopping,
                    help="Enable early stopping (disabled by default for this small dataset).")
    ap.add_argument("--patience", type=int, default=cfg.early_stop_patience,
                    help="Early-stop patience; only used when --early-stopping is set.")
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--device", default=cfg.device if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="If set, BCEWithLogitsLoss pos_weight; else class-balanced n_neg/n_pos.")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = args.device

    train_ds = SubjectFeatureDataset("train", args.manifest_dir, args.cache_dir)
    dev_ds = SubjectFeatureDataset("dev", args.manifest_dir, args.cache_dir)
    test_ds = SubjectFeatureDataset("test", args.manifest_dir, args.cache_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn)

    # Class-balanced objective: pos_weight = n_neg / n_pos (inverse class frequency)
    # applied to the positive (depressed) class in BCEWithLogitsLoss, unless overridden.
    train_labels = np.array([lbl for _, lbl in train_ds.subjects])
    n_pos = max(int(train_labels.sum()), 1)
    n_neg = max(int((train_labels == 0).sum()), 1)
    pos_weight_val = args.pos_weight if args.pos_weight is not None else n_neg / n_pos
    print(f"[train] subjects: train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)} | "
          f"pos={n_pos} neg={n_neg} class-balanced pos_weight={pos_weight_val:.3f}")

    model_cfg = {
        "num_layers": NUM_LAYERS, "feature_dim": FEATURE_DIM,
        "attn_hidden_dim": args.attn_hidden_dim, "dropout": args.dropout,
    }
    model = build_model(model_cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model: linear-probe attn_hidden={args.attn_hidden_dim} "
          f"dropout={args.dropout} | trainable params={n_trainable:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    import wandb
    wandb.init(project=args.wandb_project, name=args.run_name, mode=args.wandb_mode,
               settings=wandb.Settings(init_timeout=120),
               config={**vars(args), "pos_weight_val": pos_weight_val,
                       "trainable_params": n_trainable})

    best_dev_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_path = os.path.join(args.checkpoint_dir, "best.pt")

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_m = run_epoch(model, train_loader, criterion, device, optimizer,
                                        label_smoothing=args.label_smoothing,
                                        grad_clip=args.grad_clip)
        dev_loss, dev_m = run_epoch(model, dev_loader, criterion, device, optimizer=None,
                                    label_smoothing=args.label_smoothing)
        # Test split: monitoring only (never used for checkpoint selection).
        test_loss, test_m = run_epoch(model, test_loader, criterion, device, optimizer=None,
                                      label_smoothing=args.label_smoothing)

        log = {"epoch": epoch, "train/loss": train_loss, "dev/loss": dev_loss, "test/loss": test_loss}
        log.update({f"train/{k}": v for k, v in train_m.items()})
        log.update({f"dev/{k}": v for k, v in dev_m.items()})
        log.update({f"test/{k}": v for k, v in test_m.items()})
        log["layer_weights"] = wandb.Histogram(model.layer_weighting.weights().cpu().numpy())
        wandb.log(log)

        improved = dev_loss < best_dev_loss
        marker = " *" if improved else ""
        print(f"[epoch {epoch:3d}] train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
              f"test_loss={test_loss:.4f} dev_f1={dev_m['f1']:.3f} dev_auc={dev_m['auc']:.3f} "
              f"test_f1={test_m['f1']:.3f} test_auc={test_m['auc']:.3f}{marker}")

        if improved:
            best_dev_loss = dev_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "dev_loss": dev_loss,
                "config": model_cfg,
            }, best_path)
        else:
            epochs_no_improve += 1
            if args.early_stopping and epochs_no_improve >= args.patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(no dev-loss improvement for {args.patience} epochs)")
                break

    print(f"[train] best dev_loss={best_dev_loss:.4f} at epoch {best_epoch} -> {best_path}")
    # Reload the selected (lowest-dev-loss) checkpoint. Selection stays dev-loss
    # based; we only tune the DECISION THRESHOLD on dev (never on test) and store
    # it so evaluate.py / fusion use a dev-calibrated threshold instead of 0.5.
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    split_probs = {}
    for split_name, loader in (("train", train_loader), ("dev", dev_loader), ("test", test_loader)):
        labels_s, probs_s = run_inference(model, loader, device)
        split_probs[split_name] = (labels_s, probs_s)

    dev_labels, dev_probs = split_probs["dev"]
    tuned_thr = best_threshold(dev_labels, dev_probs)
    print(f"[train] dev-tuned threshold (max dev macro-F1) = {tuned_thr:.3f}")

    # Persist the tuned threshold inside the checkpoint for downstream use.
    ckpt["threshold"] = tuned_thr
    torch.save(ckpt, best_path)

    wandb.summary["best_dev_loss"] = best_dev_loss
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["tuned_threshold"] = tuned_thr
    for split_name, (labels_s, probs_s) in split_probs.items():
        # Report each split at BOTH 0.5 and the dev-tuned threshold for comparison.
        m_half = compute_metrics(labels_s, probs_s, threshold=0.5)
        m_tuned = compute_metrics(labels_s, probs_s, threshold=tuned_thr)
        for key, value in m_half.items():
            wandb.summary[f"best/{split_name}/{key}"] = value
        for key, value in m_tuned.items():
            wandb.summary[f"best_tuned/{split_name}/{key}"] = value
        print(f"[train] {split_name:>5}: macro_f1@0.5={m_half['macro_f1']:.3f} "
              f"macro_f1@{tuned_thr:.2f}={m_tuned['macro_f1']:.3f}")
    wandb.finish()


if __name__ == "__main__":
    main()
