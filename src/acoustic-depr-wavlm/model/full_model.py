"""Linear-probe head over cached WavLM features (SUPERB-style, small-data friendly).

Architecture (deliberately tiny to match ~100 training subjects):
    cached feats [B, L, 25, 1024]
      -> learnable softmax layer-weighting       -> [B, L, 1024]
      -> additive attention pooling (masked)      -> [B, 1024]   (subject embedding)
      -> Dropout -> single Linear(1024 -> 1)      -> [B] logit

No transformer encoder, no deep MLP -- capacity is intentionally minimal to avoid
memorising the small training set. WavLM is frozen and used only during the one-time
feature pre-extraction.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention_pool import AttentionPool
from .layer_weighting import LayerWeighting


class AcousticDepressionModel(nn.Module):
    def __init__(
        self,
        num_layers: int = 25,
        feature_dim: int = 1024,
        attn_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.layer_weighting = LayerWeighting(num_layers)
        self.attention_pool = AttentionPool(feature_dim, attn_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor):
        """feats: [B, L, 25, D]; mask: [B, L] (True=valid utterance).

        Returns (logits [B], alpha [B, L]).
        """
        u = self.layer_weighting(feats)                            # [B, L, 1024]
        e_a, alpha = self.attention_pool(u, mask)                  # [B, 1024]
        logits = self.classifier(self.dropout(e_a)).squeeze(-1)    # [B]
        return logits, alpha


def build_model(cfg: dict) -> AcousticDepressionModel:
    """Construct the model from a config dict (shared by train.py and evaluate.py)."""
    return AcousticDepressionModel(
        num_layers=cfg["num_layers"],
        feature_dim=cfg["feature_dim"],
        attn_hidden_dim=cfg["attn_hidden_dim"],
        dropout=cfg["dropout"],
    )
