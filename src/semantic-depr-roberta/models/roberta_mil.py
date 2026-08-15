"""RoBERTa encoder with trainable attention MIL pooling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class MILOutput:
    """Outputs from a MIL forward pass."""

    logits: torch.Tensor
    attention_weights: list[torch.Tensor]
    instance_logits: torch.Tensor


class AttentionMILPooling(nn.Module):
    """Trainable linear attention pooling over instance embeddings."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool instance embeddings into one participant embedding."""
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [instances, hidden_size]")
        scores = self.scorer(embeddings).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(weights.unsqueeze(-1) * embeddings, dim=0)
        return pooled, weights


class RoBERTaMILClassifier(nn.Module):
    """Participant-level binary depression classifier with attention MIL."""

    def __init__(
        self,
        encoder_name: str,
        *,
        freeze_encoder: bool,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        pooling: str = "mean",
        mil_pooling: str = "attention",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling: {pooling!r} (expected 'mean' or 'cls')")
        if mil_pooling not in {"attention", "mean"}:
            raise ValueError(f"Unsupported mil_pooling: {mil_pooling!r} (expected 'attention' or 'mean')")
        self.encoder_name = encoder_name
        self.freeze_encoder = freeze_encoder
        self.pooling = pooling
        self.mil_pooling = mil_pooling
        self.encoder = AutoModel.from_pretrained(encoder_name, add_pooling_layer=False)
        hidden_size = int(self.encoder.config.hidden_size)
        if gradient_checkpointing and not freeze_encoder:
            self.encoder.gradient_checkpointing_enable()
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.attention = AttentionMILPooling(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        # Per-feature standardization applied to instance embeddings. Saved as
        # buffers so a reloaded checkpoint is self-contained (identity by default).
        self.register_buffer("feature_mean", torch.zeros(hidden_size))
        self.register_buffer("feature_std", torch.ones(hidden_size))

    def set_feature_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store per-feature standardization stats (computed on the train split)."""
        self.feature_mean.copy_(mean.detach().to(self.feature_mean.device).view(-1))
        self.feature_std.copy_(std.detach().to(self.feature_std.device).view(-1).clamp_min(1e-6))

    def encode_instances(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode flattened instances and return pooled per-instance embeddings.

        Mean pooling masks padding and averages over real tokens (the standard
        sentence representation for a frozen RoBERTa, since its <s>/CLS token is
        not trained as a sequence summary). CLS pooling returns the <s> token.
        """
        encoder_trainable = any(parameter.requires_grad for parameter in self.encoder.parameters())
        with torch.set_grad_enabled(encoder_trainable and torch.is_grad_enabled()):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward_from_embeddings(
        self, embeddings: torch.Tensor, bag_lengths: list[int]
    ) -> MILOutput:
        """Run MIL pooling and classification from instance embeddings."""
        if sum(bag_lengths) != embeddings.shape[0]:
            raise ValueError("bag_lengths do not match number of instance embeddings")
        embeddings = (embeddings - self.feature_mean) / self.feature_std
        pooled_bags: list[torch.Tensor] = []
        attention_weights: list[torch.Tensor] = []
        start = 0
        for length in bag_lengths:
            bag_embeddings = embeddings[start : start + length]
            if self.mil_pooling == "mean":
                pooled = bag_embeddings.mean(dim=0)
                weights = bag_embeddings.new_full((length,), 1.0 / max(1, length))
            else:
                pooled, weights = self.attention(bag_embeddings)
            pooled_bags.append(pooled)
            attention_weights.append(weights)
            start += length
        participant_embeddings = torch.stack(pooled_bags, dim=0)
        logits = self.classifier(self.dropout(participant_embeddings)).squeeze(-1)
        instance_logits = self.classifier(self.dropout(embeddings)).squeeze(-1)
        return MILOutput(
            logits=logits,
            attention_weights=attention_weights,
            instance_logits=instance_logits,
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bag_lengths: list[int],
    ) -> MILOutput:
        """Encode flattened instances and classify participant bags."""
        embeddings = self.encode_instances(input_ids=input_ids, attention_mask=attention_mask)
        return self.forward_from_embeddings(embeddings, bag_lengths)
