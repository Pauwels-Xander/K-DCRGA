"""Decoder-conditioned R-GCN baseline (proposal Sec. 4.10.1).

Vanilla R-GCN encoder + a DistMult-style per-side-effect decoder (relation vector
r_k and bias b_k). The key contrast: places side-effect conditioning at the
decoder, isolating whether moving it to the encoder accounts for any gains.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kdcrga.models.encoder import RGCNEncoder, hetero_to_rgcn_inputs


class DecoderConditionedRGCN(nn.Module):
    """Vanilla R-GCN encoder + DistMult-style per-side-effect decoder (Sec. 4.10.1).

    The encoder is NOT conditioned on the side effect; conditioning lives only in
    the decoder relation vector r_k and bias b_k. This isolates the encoder-vs-
    decoder conditioning question that K-DCRGA's contribution rests on.
    """

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
        self.rel = nn.Embedding(num_side_effects, hidden_dim)  # r_k
        self.bias = nn.Embedding(num_side_effects, 1)          # b_k
        nn.init.normal_(self.rel.weight, std=0.1)
        nn.init.zeros_(self.bias.weight)

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
            h = self.encode(data)            # [N, d]
        h_a = h[pair_index[0]]                                # [E, d]
        h_b = h[pair_index[1]]                                # [E, d]
        r_k = self.rel(side_effect)                           # [E, d]
        b_k = self.bias(side_effect).squeeze(-1)              # [E]
        return (h_a * r_k * h_b).sum(dim=-1) + b_k            # [E] raw logits
