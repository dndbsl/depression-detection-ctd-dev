"""Attention pooling over a variable-length sequence of utterance embeddings.

Two implementations:
  * AttentionPool          -- additive attention with a single learnable query
                              (score_i = v^T tanh(W u_i + b)); the original head.
  * MultiHeadAttentionPool -- a learnable query vector attends over the sequence
                              via multi-head attention (richer, used by the
                              larger architecture).

Both mask out padded positions before the softmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, u: torch.Tensor, mask: torch.Tensor | None = None):
        """u: [B, L, D]; mask: [B, L] (True = valid). Returns (e_a [B, D], alpha [B, L])."""
        scores = self.v(torch.tanh(self.proj(u))).squeeze(-1)         # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)                          # [B, L]
        e_a = torch.bmm(alpha.unsqueeze(1), u).squeeze(1)             # [B, D]
        return e_a, alpha


class MultiHeadAttentionPool(nn.Module):
    """Learnable-query multi-head attention pooling -> single subject embedding."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: [B, L, d_model]; mask: [B, L] (True = valid). Returns (e_a [B, d], attn)."""
        B = x.size(0)
        q = self.query.expand(B, 1, -1)                              # [B, 1, d]
        key_padding_mask = (~mask) if mask is not None else None     # True = ignore
        out, attn = self.mha(q, x, x, key_padding_mask=key_padding_mask,
                             need_weights=True)
        e_a = self.norm(out.squeeze(1))                             # [B, d]
        return e_a, attn.squeeze(1)                                 # attn: [B, L]
