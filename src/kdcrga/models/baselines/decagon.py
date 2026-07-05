"""Decagon baseline (Zitnik et al. 2018).

Multi-relational link prediction with a DEDICOM tensor-factorisation decoder. The
R-GCN encoder is conditioned on neither the partner drug nor the side effect; all
side-effect structure lives in the decoder. For side effect k:

    g(a, k, b) = (h_a ⊙ d_k)ᵀ R (h_b ⊙ d_k)

where ``d_k`` is a per-side-effect diagonal (an embedding row) and ``R`` is a
SINGLE global matrix shared across all side effects. The shared ``R`` plus the
per-relation diagonal is exactly what distinguishes Decagon from the DistMult
decoder of ``dc_rgcn`` (proposal Sec. 4.10.1).
"""
from __future__ import annotations

import torch.nn as nn

from kdcrga.models.decoders import DedicomDecoder
from kdcrga.models.encoder import RGCNEncoder, hetero_to_rgcn_inputs


class Decagon(nn.Module):
    """R-GCN encoder + DEDICOM decoder (shared global ``R``, per-SE diagonal ``d_k``)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_relations: int,
        num_side_effects: int,
        num_layers: int = 3,
        num_bases: int = 30,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = RGCNEncoder(in_dim, hidden_dim, num_relations,
                                   num_layers, num_bases, dropout)
        self.decoder = DedicomDecoder(hidden_dim, num_side_effects)

    def encode(self, data):
        """Run the R-GCN encoder over the full graph -> node embeddings [N, d].
        Exposed so eval can encode the static graph once and reuse the result
        across batched readout chunks (the encoder output is identical across
        chunks when weights are frozen).
        Only reuse a cached ``h`` in eval mode (frozen weights, no dropout); passing a cached ``h`` to ``forward`` during training is unsupported."""
        x, edge_index, edge_type = hetero_to_rgcn_inputs(data)
        return self.encoder(x, edge_index, edge_type)

    def forward(self, data, pair_index, side_effect, h=None):
        if h is None:
            h = self.encode(data)           # [N, d]
        return self.decoder(h[pair_index[0]], h[pair_index[1]], side_effect)
