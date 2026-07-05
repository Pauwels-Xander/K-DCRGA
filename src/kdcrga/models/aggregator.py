"""Hierarchical view aggregation (HAABSA++ Method 4, proposal Sec. 4.7).

Weighs the five readouts (v1..v5) into the pair representation h^k_pair via a
hierarchical attention layer. Filled in stage v4.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalAggregator(nn.Module):
    """HAABSA++ Method 4 (proposal Sec. 4.7, eq:hier).

    Computes a softmax over views via a learned (W, a) projection, then concats
    the beta-weighted view vectors. Output dim = num_views * d_hidden, so it is
    a drop-in replacement for plain `torch.cat(views, dim=-1)`.
    """

    def __init__(self, d_hidden: int, d_attn: int = 64) -> None:
        super().__init__()
        self.W = nn.Linear(d_hidden, d_attn)
        self.a = nn.Parameter(torch.empty(d_attn))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(
        self, views: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # views: list of [E, d]; stack -> [E, V, d]
        stacked = torch.stack(views, dim=1)
        e = torch.tanh(self.W(stacked))              # [E, V, d_attn]
        scores = (e * self.a).sum(-1)                # [E, V]
        betas = torch.softmax(scores, dim=-1)        # [E, V]
        weighted = stacked * betas.unsqueeze(-1)     # [E, V, d]
        return weighted.flatten(1, 2), betas
