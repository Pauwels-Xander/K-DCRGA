"""Shared-neighbourhood view r_{AB} (proposal Sec. 4.6).

Doubly-conditional readout over the intersection of the two two-hop neighbourhoods,
with a learnable fallback when the intersection is empty. The K-DCRGA-specific
primitive for the shared biological mechanism of a DDI. Filled in stage v3.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kdcrga.models.attention import DoublyConditionalAttention


class SharedView(nn.Module):
    """Shared-neighbourhood readout (proposal Sec. 4.6, eq:shared / eq:shared-readout).

    By default uses the mean of the two raw drug embeddings as partner-conditioning;
    pass an explicit ``partner`` to override this for multi-hop rotatory calls
    (proposal line 621). A learnable fallback vector covers queries whose two-hop
    intersection is empty.
    """

    def __init__(self, d_hidden: int) -> None:
        super().__init__()
        self.fallback = nn.Parameter(torch.zeros(d_hidden))

    def forward(
        self,
        attn: DoublyConditionalAttention,
        h_nb_shared: torch.Tensor,
        q_shared: torch.Tensor,
        h_a_drug: torch.Tensor,
        h_b_drug: torch.Tensor,
        z_k: torch.Tensor,
        num_queries: int,
        partner: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Hop 1 uses the mean of the raw drug embeddings (eq:shared); hops > 1 pass
        # partner = 0.5 * (r_{A|B} + r_{B|A}) from the previous hop (proposal line 621).
        if partner is None:
            partner = 0.5 * (h_a_drug + h_b_drug)
        if h_nb_shared.numel() == 0:
            return self.fallback.expand(num_queries, -1)
        r = attn(h_nb_shared, q_shared, partner, z_k, num_queries=num_queries)
        # Queries with no shared neighbours get the learnable fallback.
        seen = torch.zeros(num_queries, dtype=torch.bool, device=r.device)
        seen[q_shared.unique()] = True
        r = torch.where(seen.unsqueeze(-1), r, self.fallback.unsqueeze(0))
        return r
