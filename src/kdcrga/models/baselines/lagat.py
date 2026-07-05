"""LaGAT baseline (Hong et al. 2022).

Partner-drug-conditional attention over the other drug's KG neighbourhood; no
side-effect conditioning in the attention. The canonical contrast for the
"drop z_k" ablation (proposal Sec. 4.10.2): per the implementation-design spec,
zeroing K-DCRGA's W_z (i.e. driving the doubly-conditional attention with a zero
z_k) recovers LaGAT, so this reuses ``DoublyConditionalAttention`` with z_k = 0.

Side-effect conditioning lives only in the decoder (a DistMult relation vector
``rel_k`` and bias ``b_k``), mirroring ``dc_rgcn`` so the K-DCRGA-vs-LaGAT gap
isolates the value of conditioning the *attention* on z_k.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kdcrga.data.graph import drug_neighbours
from kdcrga.models.attention import DoublyConditionalAttention
from kdcrga.models.encoder import RGCNEncoder, hetero_to_rgcn_inputs


class LaGAT(nn.Module):
    """R-GCN encoder + partner-conditional attention + DistMult per-SE decoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_relations: int,
        num_side_effects: int,
        num_layers: int = 3,
        num_bases: int = 30,
        dropout: float = 0.3,
        d_attn: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = RGCNEncoder(in_dim, hidden_dim, num_relations,
                                   num_layers, num_bases, dropout)
        self.attn = DoublyConditionalAttention(hidden_dim, hidden_dim, d_attn)
        self.rel = nn.Embedding(num_side_effects, hidden_dim)  # r_k
        self.bias = nn.Embedding(num_side_effects, 1)          # b_k
        nn.init.normal_(self.rel.weight, std=0.1)
        nn.init.zeros_(self.bias.weight)
        self.hidden_dim = hidden_dim

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
        e = pair_index.size(1)
        # z_k = 0 -> W_z(z_k) contributes nothing -> partner-only conditioning.
        zero = h.new_zeros(e, self.hidden_dim)
        nb_a, qa = drug_neighbours(data, pair_index[0])
        nb_b, qb = drug_neighbours(data, pair_index[1])
        r_a = self.attn(h[nb_a], qa, h_b, zero, num_queries=e)   # N(A) | partner B
        r_b = self.attn(h[nb_b], qb, h_a, zero, num_queries=e)   # N(B) | partner A
        drug_a = h_a + r_a                                       # attention-enhanced
        drug_b = h_b + r_b
        r_k = self.rel(side_effect)                             # [E, d]
        b_k = self.bias(side_effect).squeeze(-1)               # [E]
        return (drug_a * r_k * drug_b).sum(dim=-1) + b_k       # [E] raw logits
