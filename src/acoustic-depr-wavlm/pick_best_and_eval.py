"""Compare LR-sweep checkpoints by best dev loss, then evaluate the winner on test.

Reads checkpoints/lr_{tag}/best.pt for each tag, picks the one with the lowest
recorded dev_loss, copies it to checkpoints/best.pt, and runs evaluate.py on test.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = []
    for ckpt in sorted(glob.glob(os.path.join(HERE, "checkpoints", "lr_*", "best.pt"))):
        tag = os.path.basename(os.path.dirname(ckpt)).replace("lr_", "")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        rows.append((tag, float(blob["dev_loss"]), int(blob["epoch"]), ckpt))

    if not rows:
        print("No sweep checkpoints found under checkpoints/lr_*/best.pt")
        sys.exit(1)

    print(f"{'lr':>8}{'best_dev_loss':>16}{'best_epoch':>12}")
    print("-" * 36)
    for tag, loss, ep, _ in sorted(rows, key=lambda r: r[1]):
        print(f"{tag:>8}{loss:>16.4f}{ep:>12}")

    best = min(rows, key=lambda r: r[1])
    print(f"\n[pick] winner: lr={best[0]} (dev_loss={best[1]:.4f}, epoch={best[2]})")

    dst = os.path.join(HERE, "checkpoints", "best.pt")
    shutil.copy2(best[3], dst)
    print(f"[pick] copied winning checkpoint -> {dst}")

    print("\n[pick] evaluating winner on test split ...")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "evaluate.py"),
         "--checkpoint", dst, "--split", "test"],
        check=True,
    )


if __name__ == "__main__":
    main()
