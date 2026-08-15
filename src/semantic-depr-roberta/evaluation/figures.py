"""Figure generation for training and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.metrics import pr_points, roc_points


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(metrics_path: str | Path, output_path: str | Path) -> None:
    """Plot train/dev/test loss and macro-F1 curves."""
    metrics = pd.read_csv(metrics_path)
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(metrics["epoch"], metrics["train_loss"], label="train")
    axes[0].plot(metrics["epoch"], metrics["dev_loss"], label="dev")
    if "test_loss" in metrics.columns:
        axes[0].plot(metrics["epoch"], metrics["test_loss"], label="test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(metrics["epoch"], metrics["train_macro_f1"], label="train")
    axes[1].plot(metrics["epoch"], metrics["dev_macro_f1"], label="dev")
    if "test_macro_f1" in metrics.columns:
        axes[1].plot(metrics["epoch"], metrics["test_macro_f1"], label="test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_pr(predictions_path: str | Path, roc_path: str | Path, pr_path: str | Path) -> None:
    """Plot ROC and precision-recall curves from prediction CSV."""
    predictions = pd.read_csv(predictions_path)
    labels = predictions["label"].astype(int).tolist()
    probs = predictions["probability"].astype(float).tolist()
    fpr, tpr, _ = roc_points(labels, probs)
    precision, recall, _ = pr_points(labels, probs)
    plt = _plt()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.tight_layout()
    Path(roc_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    Path(pr_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pr_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(predictions_path: str | Path, output_path: str | Path) -> None:
    """Plot a 2x2 confusion matrix from prediction CSV."""
    from sklearn.metrics import confusion_matrix

    predictions = pd.read_csv(predictions_path)
    cm = confusion_matrix(
        predictions["label"].astype(int),
        predictions["prediction"].astype(int),
        labels=[0, 1],
    )
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (row, col), value in pd.DataFrame(cm).stack().items():
        ax.text(col, row, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["neg", "pos"])
    ax.set_yticks([0, 1], labels=["neg", "pos"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_attention(
    attention_path: str | Path,
    output_dir: str | Path,
    *,
    participant_ids: list[int] | None = None,
    top_k: int = 30,
) -> list[Path]:
    """Plot attention weights for selected participants."""
    attention = pd.read_csv(attention_path)
    if participant_ids is None:
        participant_ids = attention["participant_id"].drop_duplicates().head(5).astype(int).tolist()
    output_paths: list[Path] = []
    plt = _plt()
    for pid in participant_ids:
        subset = attention[attention["participant_id"] == pid].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("attention_weight", ascending=False).head(top_k)
        subset = subset.sort_values("instance_index")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(subset["instance_index"].astype(str), subset["attention_weight"])
        ax.set_xlabel("Instance")
        ax.set_ylabel("Attention")
        ax.set_title(f"Participant {pid}")
        fig.tight_layout()
        out = Path(output_dir) / f"attention_{pid}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
        output_paths.append(out)
    return output_paths

