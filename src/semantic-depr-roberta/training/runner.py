"""Training and evaluation runner for RoBERTa MIL experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from datasets.bags import BagBatch, TextBagDataset, collate_bags
from datasets.daic import BagRecord, class_counts, load_bags, prepare_dataset
from evaluation.figures import plot_confusion_matrix, plot_training_curves
from evaluation.metrics import compute_metrics, select_threshold
from models.roberta_mil import RoBERTaMILClassifier
from training.config import (
    ExperimentConfig,
    config_to_dict,
    ensure_output_dirs,
    experiment_slug,
)
from training.utils import effective_accumulation, gpu_info, set_seed, sigmoid_to_float, write_json
from training.wandb_utils import start_wandb_run


def _run_dir(config: ExperimentConfig, seed: int, *, debug: bool = False) -> Path:
    category = "debug_runs" if debug else "runs"
    return Path(config.output.results_dir) / category / f"{experiment_slug(config)}_seed{seed}"


def _checkpoint_dir(config: ExperimentConfig, seed: int, *, debug: bool = False) -> Path:
    category = "debug_runs" if debug else ""
    return Path(config.output.checkpoints_dir) / category / f"{experiment_slug(config)}_seed{seed}"


def _tokenize(tokenizer: Any, batch: BagBatch, config: ExperimentConfig, device: torch.device) -> dict[str, torch.Tensor]:
    tokenized = tokenizer(
        batch.flat_texts,
        max_length=config.data.max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in tokenized.items()}


@torch.no_grad()
def build_embedding_cache(
    model: RoBERTaMILClassifier,
    tokenizer: Any,
    records: list[BagRecord],
    config: ExperimentConfig,
    device: torch.device,
    *,
    encode_batch: int = 16,
) -> dict[int, torch.Tensor]:
    """Precompute frozen per-instance embeddings once, keyed by participant id.

    Only valid when the encoder is frozen (weights never change), so the same
    embeddings can be reused every epoch and for the best-checkpoint eval.
    """
    was_training = model.training
    model.eval()
    cache: dict[int, torch.Tensor] = {}
    for record in records:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(record.instances), encode_batch):
            texts = record.instances[start : start + encode_batch]
            tokenized = tokenizer(
                texts,
                max_length=config.data.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            embeddings = model.encode_instances(
                input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"]
            )
            chunks.append(embeddings.detach())
        cache[record.participant_id] = torch.cat(chunks, dim=0)
    if was_training:
        model.train()
    return cache


def _cached_batch_embeddings(cache: dict[int, torch.Tensor], batch: BagBatch) -> torch.Tensor:
    """Gather flattened instance embeddings for a batch from the cache."""
    return torch.cat([cache[record.participant_id] for record in batch.records], dim=0)


def disk_feature_dir(config: ExperimentConfig) -> Path:
    """Return the directory holding precomputed frozen features for this encoder/pooling."""
    encoder = config.encoder.replace("/", "-")
    return Path(config.data.processed_dir) / f"features_{encoder}_{config.model.pooling}"


def load_disk_features(
    config: ExperimentConfig, device: torch.device
) -> dict[int, torch.Tensor] | None:
    """Load precomputed per-instance features from disk, or None if not available."""
    feature_dir = disk_feature_dir(config)
    if not feature_dir.exists():
        return None
    cache: dict[int, torch.Tensor] = {}
    for split in ("train", "dev", "test"):
        split_path = feature_dir / f"{split}.pt"
        if not split_path.exists():
            return None
        payload = torch.load(split_path, map_location="cpu")
        for pid, tensor in payload.items():
            cache[int(pid)] = tensor.float().to(device)
    return cache


def _loader(records: list[BagRecord], batch_size: int, *, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TextBagDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        collate_fn=collate_bags,
    )


def _optimizer(model: RoBERTaMILClassifier, config: ExperimentConfig) -> torch.optim.Optimizer:
    head_params = list(model.attention.parameters()) + list(model.classifier.parameters())
    if config.training_strategy == "finetune":
        return torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": config.training.encoder_lr},
                {"params": head_params, "lr": config.training.classifier_lr},
            ],
            weight_decay=config.training.weight_decay,
        )
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.classifier_lr,
        weight_decay=config.training.weight_decay,
    )


def _largest_probe_records(records: list[BagRecord], batch_size: int) -> list[BagRecord]:
    return sorted(records, key=lambda record: len(record.instances), reverse=True)[:batch_size]


def _debug_subset(
    records: list[BagRecord],
    limit: int | None,
    max_instances: int | None = None,
) -> list[BagRecord]:
    if limit is None or limit >= len(records):
        selected = records
    else:
        positives = [record for record in records if record.label == 1]
        negatives = [record for record in records if record.label == 0]
        pos_take = min(len(positives), max(1, limit // 2))
        neg_take = min(len(negatives), max(0, limit - pos_take))
        selected = positives[:pos_take] + negatives[:neg_take]
        if len(selected) < limit:
            selected.extend(record for record in records if record not in selected)
        selected = selected[:limit]
    if max_instances is None:
        return selected
    return [
        BagRecord(
            participant_id=record.participant_id,
            split=record.split,
            label=record.label,
            phq_score=record.phq_score,
            instances=record.instances[:max_instances],
        )
        for record in selected
    ]


def resolve_batching(
    model: RoBERTaMILClassifier,
    tokenizer: Any,
    train_records: list[BagRecord],
    config: ExperimentConfig,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[int, int]:
    """Auto-probe physical participant batch size and accumulation steps."""
    requested = config.training.physical_batch_size
    if isinstance(requested, int):
        physical = requested
    elif not torch.cuda.is_available():
        physical = 1
    else:
        physical = 1
        for candidate in range(min(config.training.max_physical_batch_size, len(train_records)), 0, -1):
            try:
                model.zero_grad(set_to_none=True)
                probe = BagBatch(_largest_probe_records(train_records, candidate))
                tokenized = _tokenize(tokenizer, probe, config, device)
                labels = probe.labels.to(device)
                output = model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    bag_lengths=probe.bag_lengths,
                )
                loss = criterion(output.logits, labels)
                loss.backward()
                model.zero_grad(set_to_none=True)
                physical = candidate
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
    accumulation = effective_accumulation(physical, config.training.min_effective_batch_size)
    return physical, accumulation


@torch.no_grad()
def predict_records(
    model: RoBERTaMILClassifier,
    tokenizer: Any,
    records: list[BagRecord],
    config: ExperimentConfig,
    *,
    threshold: float,
    device: torch.device,
    criterion: nn.Module | None = None,
    embedding_cache: dict[int, torch.Tensor] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Predict participant probabilities, instance scores, and attention weights."""
    model.eval()
    participant_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_count = 0
    loader = _loader(records, config.training.eval_batch_size, shuffle=False, seed=0, num_workers=0)
    for batch in loader:
        labels = batch.labels.to(device)
        if embedding_cache is not None:
            embeddings = _cached_batch_embeddings(embedding_cache, batch)
            output = model.forward_from_embeddings(embeddings, batch.bag_lengths)
        else:
            tokenized = _tokenize(tokenizer, batch, config, device)
            output = model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                bag_lengths=batch.bag_lengths,
            )
        if criterion is not None:
            total_loss += float(criterion(output.logits, labels).item()) * len(batch.records)
            total_count += len(batch.records)
        probs = sigmoid_to_float(output.logits)
        instance_probs = sigmoid_to_float(output.instance_logits)
        cursor = 0
        for idx, record in enumerate(batch.records):
            prob = float(probs[idx])
            participant_rows.append(
                {
                    "split": record.split,
                    "participant_id": record.participant_id,
                    "label": record.label,
                    "phq_score": record.phq_score,
                    "logit": float(output.logits[idx].detach().cpu()),
                    "probability": prob,
                    "threshold": threshold,
                    "prediction": int(prob >= threshold),
                }
            )
            weights = output.attention_weights[idx].detach().cpu().numpy().astype(float).tolist()
            for instance_index, (text, weight) in enumerate(zip(record.instances, weights)):
                instance_prob = float(instance_probs[cursor + instance_index])
                base = {
                    "split": record.split,
                    "participant_id": record.participant_id,
                    "label": record.label,
                    "phq_score": record.phq_score,
                    "instance_index": instance_index,
                    "text": text,
                    "n_words": len(text.split()),
                }
                attention_rows.append({**base, "attention_weight": float(weight)})
                instance_rows.append(
                    {
                        **base,
                        "instance_probability": instance_prob,
                        "instance_prediction": int(instance_prob >= threshold),
                    }
                )
            cursor += len(record.instances)
    avg_loss = total_loss / total_count if total_count else 0.0
    return (
        pd.DataFrame(participant_rows),
        pd.DataFrame(attention_rows),
        pd.DataFrame(instance_rows),
        avg_loss,
    )


def _wandb_epoch_payload(
    *,
    epoch: int,
    train_loss: float,
    train_f1: float,
    train_metrics: Any,
    dev_loss: float,
    dev_f1: float,
    dev_metrics: Any,
    test_loss: float,
    test_f1: float,
    test_metrics: Any,
    threshold: float,
) -> dict[str, Any]:
    """Build a W&B payload with grouped keys for train/dev/test curves."""
    payload = {
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_metrics.accuracy,
        "train/f1": train_f1,
        "train/macro_f1": train_metrics.macro_f1,
        "train/auroc": train_metrics.auroc,
        "train/precision": train_metrics.precision,
        "train/recall": train_metrics.recall,
        "train/sensitivity": train_metrics.sensitivity,
        "train/specificity": train_metrics.specificity,
        "dev/loss": dev_loss,
        "dev/accuracy": dev_metrics.accuracy,
        "dev/f1": dev_f1,
        "dev/macro_f1": dev_metrics.macro_f1,
        "dev/auroc": dev_metrics.auroc,
        "dev/precision": dev_metrics.precision,
        "dev/recall": dev_metrics.recall,
        "dev/sensitivity": dev_metrics.sensitivity,
        "dev/specificity": dev_metrics.specificity,
        "dev/threshold": threshold,
        "test/loss": test_loss,
        "test/accuracy": test_metrics.accuracy,
        "test/f1": test_f1,
        "test/macro_f1": test_metrics.macro_f1,
        "test/auroc": test_metrics.auroc,
        "test/precision": test_metrics.precision,
        "test/recall": test_metrics.recall,
        "test/sensitivity": test_metrics.sensitivity,
        "test/specificity": test_metrics.specificity,
    }
    return payload


def _save_checkpoint(
    path: Path,
    model: RoBERTaMILClassifier,
    config: ExperimentConfig,
    *,
    seed: int,
    epoch: int,
    threshold: float,
    best_dev_macro_f1: float,
    physical_batch_size: int,
    gradient_accumulation_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_to_dict(config),
            "seed": seed,
            "epoch": epoch,
            "threshold": threshold,
            "best_dev_macro_f1": best_dev_macro_f1,
            "physical_batch_size": physical_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
        path,
    )


def load_model_from_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[RoBERTaMILClassifier, dict[str, Any]]:
    """Load a model and checkpoint payload."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    model = RoBERTaMILClassifier(
        raw_config["encoder"],
        freeze_encoder=raw_config["training_strategy"] == "frozen",
        dropout=raw_config["model"]["dropout"],
        gradient_checkpointing=raw_config["model"]["gradient_checkpointing"],
        pooling=raw_config["model"].get("pooling", "mean"),
        mil_pooling=raw_config["model"].get("mil_pooling", "attention"),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload


def train_one_run(
    config: ExperimentConfig,
    *,
    seed: int,
    disable_wandb: bool = False,
    debug_epochs: int | None = None,
    debug_participants: int | None = None,
    debug_max_instances: int | None = None,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    """Train one config/seed run and write all run-level artifacts."""
    ensure_output_dirs(config)
    debug_mode = (
        debug_epochs is not None
        or debug_participants is not None
        or debug_max_instances is not None
    )
    if not (Path(config.data.processed_dir) / "bags.jsonl").exists():
        prepare_dataset(config.data)
    set_seed(seed, config.training.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.training.require_cuda and device.type != "cuda" and not allow_cpu:
        raise RuntimeError("CUDA is required by config; pass allow_cpu=True only for debug smoke tests.")

    train_records = _debug_subset(
        load_bags(config.data.processed_dir, "train"), debug_participants, debug_max_instances
    )
    dev_records = _debug_subset(
        load_bags(config.data.processed_dir, "dev"), debug_participants, debug_max_instances
    )
    test_records = _debug_subset(
        load_bags(config.data.processed_dir, "test"), debug_participants, debug_max_instances
    )
    negatives, positives = class_counts(train_records)
    if positives == 0:
        raise ValueError("Training split has no positive examples.")
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tokenizer = AutoTokenizer.from_pretrained(config.encoder)
    model = RoBERTaMILClassifier(
        config.encoder,
        freeze_encoder=config.training_strategy == "frozen",
        dropout=config.model.dropout,
        gradient_checkpointing=config.model.gradient_checkpointing,
        pooling=config.model.pooling,
        mil_pooling=config.model.mil_pooling,
    ).to(device)
    optimizer = _optimizer(model, config)
    physical_batch_size, accumulation_steps = resolve_batching(
        model, tokenizer, train_records, config, criterion, device
    )
    effective_batch_size = physical_batch_size * accumulation_steps

    embedding_cache: dict[int, torch.Tensor] | None = None
    if config.training_strategy == "frozen":
        embedding_cache = load_disk_features(config, device)
        if embedding_cache is None:
            embedding_cache = build_embedding_cache(
                model, tokenizer, train_records + dev_records + test_records, config, device
            )
        if getattr(config.model, "standardize_features", False):
            train_features = torch.cat(
                [embedding_cache[record.participant_id] for record in train_records], dim=0
            )
            model.set_feature_normalization(
                train_features.mean(dim=0), train_features.std(dim=0)
            )

    run_dir = _run_dir(config, seed, debug=debug_mode)
    checkpoint_dir = _checkpoint_dir(config, seed, debug=debug_mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = start_wandb_run(config, seed=seed, job_type="train", disabled=disable_wandb)
    metadata = {
        "experiment_name": config.experiment_name,
        "encoder": config.encoder,
        "training_strategy": config.training_strategy,
        "seed": seed,
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "pos_weight": float(pos_weight.item()),
        "gpu": gpu_info(),
    }
    write_json(run_dir / "run_metadata.json", metadata)
    logger.log(metadata)

    epochs = debug_epochs or config.training.epochs
    best_macro_f1 = -1.0
    best_dev_loss = float("inf")
    best_threshold = 0.5
    best_epoch = 0
    metric_rows: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []

    for epoch in tqdm(range(1, epochs + 1), desc=f"{experiment_slug(config)} seed {seed}"):
        model.train()
        train_loader = _loader(
            train_records,
            physical_batch_size,
            shuffle=True,
            seed=seed + epoch,
            num_workers=config.training.num_workers,
        )
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_count = 0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.labels.to(device)
            if embedding_cache is not None:
                embeddings = _cached_batch_embeddings(embedding_cache, batch)
                output = model.forward_from_embeddings(embeddings, batch.bag_lengths)
            else:
                tokenized = _tokenize(tokenizer, batch, config, device)
                output = model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    bag_lengths=batch.bag_lengths,
                )
            loss = criterion(output.logits, labels)
            (loss / accumulation_steps).backward()
            train_loss_sum += float(loss.item()) * len(batch.records)
            train_count += len(batch.records)
            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if step % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_loss = train_loss_sum / max(1, train_count)

        dev_predictions, _, _, dev_loss = predict_records(
            model,
            tokenizer,
            dev_records,
            config,
            threshold=0.5,
            device=device,
            criterion=criterion,
            embedding_cache=embedding_cache,
        )
        threshold = select_threshold(dev_predictions["label"], dev_predictions["probability"])
        train_predictions, _, _, _ = predict_records(
            model,
            tokenizer,
            train_records,
            config,
            threshold=threshold,
            device=device,
            criterion=None,
            embedding_cache=embedding_cache,
        )
        train_metrics = compute_metrics(
            train_predictions["label"],
            train_predictions["probability"],
            threshold=threshold,
        )
        train_f1 = float(
            f1_score(
                train_predictions["label"],
                train_predictions["prediction"],
                zero_division=0,
            )
        )
        dev_f1 = float(
            f1_score(
                dev_predictions["label"],
                dev_predictions["prediction"],
                zero_division=0,
            )
        )
        dev_metrics = compute_metrics(
            dev_predictions["label"],
            dev_predictions["probability"],
            threshold=threshold,
        )
        # Test split: monitoring only (never used for checkpoint selection).
        test_predictions_ep, _, _, test_loss = predict_records(
            model,
            tokenizer,
            test_records,
            config,
            threshold=threshold,
            device=device,
            criterion=criterion,
            embedding_cache=embedding_cache,
        )
        test_f1 = float(
            f1_score(
                test_predictions_ep["label"],
                test_predictions_ep["prediction"],
                zero_division=0,
            )
        )
        test_metrics = compute_metrics(
            test_predictions_ep["label"],
            test_predictions_ep["probability"],
            threshold=threshold,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_metrics.macro_f1,
            "train_auroc": train_metrics.auroc,
            "train_accuracy": train_metrics.accuracy,
            "train_precision": train_metrics.precision,
            "train_recall": train_metrics.recall,
            "train_sensitivity": train_metrics.sensitivity,
            "train_specificity": train_metrics.specificity,
            "dev_loss": dev_loss,
            "dev_macro_f1": dev_metrics.macro_f1,
            "dev_auroc": dev_metrics.auroc,
            "dev_accuracy": dev_metrics.accuracy,
            "dev_precision": dev_metrics.precision,
            "dev_recall": dev_metrics.recall,
            "dev_sensitivity": dev_metrics.sensitivity,
            "dev_specificity": dev_metrics.specificity,
            "dev_threshold": threshold,
            "test_loss": test_loss,
            "test_macro_f1": test_metrics.macro_f1,
            "test_auroc": test_metrics.auroc,
            "test_accuracy": test_metrics.accuracy,
            "test_precision": test_metrics.precision,
            "test_recall": test_metrics.recall,
            "test_sensitivity": test_metrics.sensitivity,
            "test_specificity": test_metrics.specificity,
        }
        metric_rows.append(row)
        lr_rows.append(
            {
                "epoch": epoch,
                "encoder_lr": config.training.encoder_lr,
                "classifier_lr": config.training.classifier_lr,
            }
        )
        logger.log(
            _wandb_epoch_payload(
                epoch=epoch,
                train_loss=train_loss,
                train_f1=train_f1,
                train_metrics=train_metrics,
                dev_loss=dev_loss,
                dev_f1=dev_f1,
                dev_metrics=dev_metrics,
                test_loss=test_loss,
                test_f1=test_f1,
                test_metrics=test_metrics,
                threshold=threshold,
            ),
            step=epoch,
        )
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_macro_f1 = dev_metrics.macro_f1
            best_threshold = threshold
            best_epoch = epoch
            _save_checkpoint(
                checkpoint_dir / "best_checkpoint.pt",
                model,
                config,
                seed=seed,
                epoch=epoch,
                threshold=threshold,
                best_dev_macro_f1=best_macro_f1,
                physical_batch_size=physical_batch_size,
                gradient_accumulation_steps=accumulation_steps,
            )

    metrics_csv = run_dir / "metrics.csv"
    lr_csv = run_dir / "lr_history.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)
    pd.DataFrame(lr_rows).to_csv(lr_csv, index=False)
    _save_checkpoint(
        checkpoint_dir / "final_checkpoint.pt",
        model,
        config,
        seed=seed,
        epoch=epochs,
        threshold=best_threshold,
        best_dev_macro_f1=best_macro_f1,
        physical_batch_size=physical_batch_size,
        gradient_accumulation_steps=accumulation_steps,
    )

    best_model, _ = load_model_from_checkpoint(checkpoint_dir / "best_checkpoint.pt", device)
    final_dev_predictions, _, _, _ = predict_records(
        model, tokenizer, dev_records, config, threshold=best_threshold, device=device, criterion=criterion,
        embedding_cache=embedding_cache,
    )
    best_dev_predictions, best_dev_attention, best_dev_instances, _ = predict_records(
        best_model, tokenizer, dev_records, config, threshold=best_threshold, device=device, criterion=criterion,
        embedding_cache=embedding_cache,
    )
    test_predictions, test_attention, test_instances, _ = predict_records(
        best_model, tokenizer, test_records, config, threshold=best_threshold, device=device, criterion=criterion,
        embedding_cache=embedding_cache,
    )
    test_metrics = compute_metrics(
        test_predictions["label"], test_predictions["probability"], threshold=best_threshold
    ).to_dict()
    test_payload = {
        **test_metrics,
        "experiment_name": config.experiment_name,
        "encoder": config.encoder,
        "training_strategy": config.training_strategy,
        "seed": seed,
        "best_epoch": best_epoch,
    }

    best_dev_predictions.to_csv(run_dir / "dev_predictions_best.csv", index=False)
    final_dev_predictions.to_csv(run_dir / "dev_predictions_final.csv", index=False)
    test_predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    pd.concat([best_dev_attention, test_attention], ignore_index=True).to_csv(
        run_dir / "attention_weights.csv", index=False
    )
    pd.concat([best_dev_instances, test_instances], ignore_index=True).to_csv(
        run_dir / "instance_scores.csv", index=False
    )
    write_json(run_dir / "test_metrics.json", test_payload)

    best_train_predictions, _, _, _ = predict_records(
        best_model, tokenizer, train_records, config, threshold=best_threshold, device=device,
        criterion=criterion, embedding_cache=embedding_cache,
    )
    best_train_metrics = compute_metrics(
        best_train_predictions["label"], best_train_predictions["probability"], threshold=best_threshold,
    )
    best_dev_metrics = compute_metrics(
        best_dev_predictions["label"], best_dev_predictions["probability"], threshold=best_threshold,
    )
    summary_payload: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "best_dev_threshold": best_threshold,
        "best_dev_macro_f1": best_macro_f1,
    }
    for split_name, split_metrics in (
        ("train", best_train_metrics),
        ("dev", best_dev_metrics),
        ("test", compute_metrics(
            test_predictions["label"], test_predictions["probability"], threshold=best_threshold,
        )),
    ):
        for key, value in split_metrics.to_dict().items():
            if key != "confusion_matrix":
                summary_payload[f"best/{split_name}/{key}"] = value
    logger.log_summary(summary_payload)

    try:
        figure_dir = Path(config.output.figures_dir) / ("debug_runs" if debug_mode else "")
        plot_training_curves(metrics_csv, figure_dir / f"{experiment_slug(config)}_seed{seed}_training.png")
        plot_confusion_matrix(run_dir / "test_predictions.csv", figure_dir / f"{experiment_slug(config)}_seed{seed}_confusion.png")
    except ImportError:
        pass

    if config.wandb.log_artifacts:
        logger.log_artifacts(
            [
                *run_dir.glob("*"),
                checkpoint_dir / "best_checkpoint.pt",
                checkpoint_dir / "final_checkpoint.pt",
            ]
        )
    logger.finish()
    return test_payload
