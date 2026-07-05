"""K-DCRGA with a DEDICOM decoder (the "beat-Decagon" model variant).

Reuses the K-DCRGA v5 encoder stack (R-GCN -> doubly-conditional attention ->
rotatory hops -> optional shared view) but replaces the shared-MLP scorer with
Decagon's DEDICOM head. Per-drug vectors are formed by a parameter-free sum of
the drug's own embedding and its side of the rotatory readouts:

    u_a = h_a + r_A + r_{A|B} (+ r_{AB})
    u_b = h_b + r_B + r_{B|A} (+ r_{AB})
    score = (u_a ⊙ d_k)ᵀ R (u_b ⊙ d_k)

so the ONLY difference from Decagon is the attention readouts added to h, and the
ONLY difference from K-DCRGA v5 is the decoder. ``z_k`` still conditions the
attention (and keeps its ontology init); the decoder owns a separate diagonal d_k.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kdcrga.data.graph import drug_neighbours
from kdcrga.models.attention import DoublyConditionalAttention
from kdcrga.models.decoders import DedicomDecoder
from kdcrga.models.encoder import RGCNEncoder, hetero_to_rgcn_inputs
from kdcrga.models.rotatory import rotatory_readouts
from kdcrga.models.shared_view import SharedView


class KDCRGADedicom(nn.Module):
    """K-DCRGA encoder + DEDICOM decoder."""

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
        num_hops: int = 1,
        use_shared_view: bool = False,
        z_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = RGCNEncoder(in_dim, hidden_dim, num_relations,
                                   num_layers, num_bases, dropout)
        self.attn = DoublyConditionalAttention(hidden_dim, hidden_dim, d_attn)
        if use_shared_view:
            self.shared = SharedView(hidden_dim)
        self.z_k = nn.Embedding(num_side_effects, hidden_dim)
        nn.init.normal_(self.z_k.weight, std=0.1)
        if z_init is not None:
            assert z_init.shape == (num_side_effects, hidden_dim), (
                f"z_init shape {tuple(z_init.shape)} != ({num_side_effects}, {hidden_dim})"
            )
            with torch.no_grad():
                self.z_k.weight.copy_(z_init)
        self.decoder = DedicomDecoder(hidden_dim, num_side_effects)
        self.num_hops = num_hops
        self.use_shared_view = use_shared_view
        self.hidden_dim = hidden_dim

    def encode(self, data):
        """Run the R-GCN encoder over the full graph -> node embeddings [N, d].
        Only reuse a cached ``h`` in eval mode (frozen weights, no dropout)."""
        x, edge_index, edge_type = hetero_to_rgcn_inputs(data)
        return self.encoder(x, edge_index, edge_type)

    def forward(self, data, pair_index, side_effect, h=None):
        if h is None:
            h = self.encode(data)
        h_a = h[pair_index[0]]
        h_b = h[pair_index[1]]
        zk = self.z_k(side_effect)
        nb_a, qa = drug_neighbours(data, pair_index[0])
        nb_b, qb = drug_neighbours(data, pair_index[1])
        nq = pair_index.size(1)
        if self.use_shared_view:
            from kdcrga.data.graph import drug_two_hop_shared
            nb_s, qs = drug_two_hop_shared(data, pair_index)
            r_a, r_b, r_ab, r_ba, r_sh = rotatory_readouts(
                self.attn, h[nb_a], qa, h[nb_b], qb, h_a, h_b, zk,
                num_queries=nq, num_hops=self.num_hops,
                shared=self.shared, h_nb_shared=h[nb_s], q_shared=qs)
            u_a = h_a + r_a + r_ab + r_sh
            u_b = h_b + r_b + r_ba + r_sh
        else:
            r_a, r_b, r_ab, r_ba = rotatory_readouts(
                self.attn, h[nb_a], qa, h[nb_b], qb, h_a, h_b, zk,
                num_queries=nq, num_hops=self.num_hops)
            u_a = h_a + r_a + r_ab
            u_b = h_b + r_b + r_ba
        return self.decoder(u_a, u_b, side_effect)
