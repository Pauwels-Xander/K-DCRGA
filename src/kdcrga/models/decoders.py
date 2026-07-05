"""Reusable link-prediction decoders.

``DedicomDecoder`` is the DEDICOM head from Decagon (Zitnik et al. 2018):
for side effect k it scores a drug pair as

    g(a, k, b) = (u_a ⊙ d_k)ᵀ R (u_b ⊙ d_k)

where ``d_k`` is a per-side-effect diagonal (an embedding row) and ``R`` is a
SINGLE global matrix shared across all side effects. Shared by the Decagon
baseline and the KDCRGADedicom model so their decoders are identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DedicomDecoder(nn.Module):
    """DEDICOM head: shared global ``R`` + per-side-effect diagonal ``d_k``."""

    def __init__(self, hidden_dim: int, num_side_effects: int) -> None:
        super().__init__()
        self.D = nn.Embedding(num_side_effects, hidden_dim)  # per-SE diagonal d_k
        self.R = nn.Parameter(torch.eye(hidden_dim))         # shared global matrix
        nn.init.normal_(self.D.weight, mean=1.0, std=0.1)    # ~identity at start

    def forward(
        self, u_a: torch.Tensor, u_b: torch.Tensor, side_effect: torch.Tensor
    ) -> torch.Tensor:
        d_k = self.D(side_effect)               # [E, d]
        ua = u_a * d_k                          # [E, d]
        ub = u_b * d_k                          # [E, d]
        return ((ua @ self.R) * ub).sum(dim=-1)  # [E] raw logits
