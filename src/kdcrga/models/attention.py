"""Doubly-conditional attention coefficient (proposal Sec. 4.4, eq. eq:dcrga).

Extends GAT with two query signals: partner-drug conditioning W_p h_{d_B} (LaGAT)
and side-effect conditioning W_z z_k (new). The central architectural contribution.
Filled in stage v1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as segment_softmax


class DoublyConditionalAttention(nn.Module):
    """Eq. eq:dcrga — attention conditioned on partner drug h_{d_B} AND z_k."""

    def __init__(self, d_hidden: int, d_z: int, d_attn: int = 64) -> None:
        super().__init__()
        self.W_h = nn.Linear(d_hidden, d_attn, bias=False)
        self.W_p = nn.Linear(d_hidden, d_attn, bias=False)
        self.W_z = nn.Linear(d_z, d_attn, bias=False)
        self.b_a = nn.Parameter(torch.zeros(d_attn))
        self.a = nn.Parameter(torch.empty(d_attn))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(
        self,
        h_neighbours: torch.Tensor,
        neighbour_query: torch.Tensor,
        h_partner: torch.Tensor,
        z_k: torch.Tensor,
        num_queries: int,
        return_alpha: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the attention-pooled neighbour readout. When `return_alpha` is
        True, also return the per-neighbour attention weights `alpha` (aligned
        with `h_neighbours` / `neighbour_query`) for RQ4 analysis."""
        if h_neighbours.numel() == 0:
            out = torch.zeros(num_queries, h_partner.size(-1),
                              device=h_partner.device, dtype=h_partner.dtype)
            return (out, torch.empty(0, device=out.device)) if return_alpha else out
        e = self.W_h(h_neighbours)
        e = e + self.W_p(h_partner)[neighbour_query]
        e = e + self.W_z(z_k)[neighbour_query]
        e = torch.tanh(e + self.b_a)
        scores = (e * self.a).sum(-1)
        alpha = segment_softmax(scores, neighbour_query, num_nodes=num_queries)
        weighted = h_neighbours * alpha.unsqueeze(-1)
        out = torch.zeros(num_queries, h_neighbours.size(-1),
                          device=h_neighbours.device, dtype=h_neighbours.dtype)
        out.index_add_(0, neighbour_query, weighted)
        return (out, alpha) if return_alpha else out
