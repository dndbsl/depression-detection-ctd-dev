"""Learnable softmax-weighted sum over WavLM's 25 hidden-state layers (SUPERB-style).

Input cached features are ``[..., num_utt, 25, 1024]`` (per-layer time-mean-pooled).
A single learnable weight vector of length 25 is softmaxed and used to collapse the
layer axis, yielding one 1024-d embedding per utterance.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerWeighting(nn.Module):
    def __init__(self, num_layers: int = 25):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [..., num_utt, num_layers, D] -> [..., num_utt, D]."""
        weights = torch.softmax(self.layer_logits, dim=0)             # [num_layers]
        # Broadcast over leading dims and the D axis.
        w = weights.view(*([1] * (feats.dim() - 2)), -1, 1)           # [...,1,num_layers,1]
        return (feats * w).sum(dim=-2)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits.detach(), dim=0)
