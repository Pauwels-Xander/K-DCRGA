"""Top-level K-DCRGA model assembling the staged components.

Wires encoder -> doubly-conditional attention -> rotatory hops -> shared view ->
hierarchical aggregator -> 2-layer MLP (proposal Sec. 4.7). Component toggles drive
the ablation grid (proposal Sec. 4.10.2); each stage v1..v5 lights up one toggle.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kdcrga.data.graph import drug_neighbours
from kdcrga.models.aggregator import HierarchicalAggregator
from kdcrga.models.attention import DoublyConditionalAttention
from kdcrga.models.encoder import RGCNEncoder, hetero_to_rgcn_inputs
from kdcrga.models.rotatory import rotatory_readouts
from kdcrga.models.shared_view import SharedView


class KDCRGA(nn.Module):
    """K-DCRGA model with component toggles for the v1..v5 staged build."""

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
        use_hierarchical: bool = False,
        use_attention: bool = True,
        z_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = RGCNEncoder(in_dim, hidden_dim, num_relations,
                                   num_layers, num_bases, dropout)
        self.attn = DoublyConditionalAttention(hidden_dim, hidden_dim, d_attn)
        if use_shared_view:
            self.shared = SharedView(hidden_dim)
        if use_hierarchical:
            self.aggregator = HierarchicalAggregator(hidden_dim, d_attn)
        self.z_k = nn.Embedding(num_side_effects, hidden_dim)
        nn.init.normal_(self.z_k.weight, std=0.1)
        if z_init is not None:
            assert z_init.shape == (num_side_effects, hidden_dim), (
                f"z_init shape {tuple(z_init.shape)} != ({num_side_effects}, {hidden_dim})"
            )
            with torch.no_grad():
                self.z_k.weight.copy_(z_init)
        self.num_hops = num_hops
        self.use_shared_view = use_shared_view
        self.use_hierarchical = use_hierarchical
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        # v1..v4 are the rotatory readouts (r_A, r_B, r_{A|B}, r_{B|A}); v5 the
        # shared-neighbourhood view when enabled (proposal eqs. 626-643).
        n_views = 4 + (1 if use_shared_view else 0)
        self.n_views = n_views
        # With attention OFF (the 2x2 "plain R-GCN + MLP" cell) the side effect can
        # no longer enter via the attention, so z_k is appended to the pair feats:
        # [h_a, h_b, h_a*h_b, |h_a-h_b|, z_k] -> 5 blocks of hidden_dim.
        mlp_in = (n_views if use_attention else 5) * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

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
            h = self.encode(data)
        h_a_drug = h[pair_index[0]]
        h_b_drug = h[pair_index[1]]
        zk = self.z_k(side_effect)
        if not self.use_attention:
            # plain R-GCN + shared-MLP cell: no neighbour attention/rotatory;
            # side effect enters via z_k concatenated to the pair features.
            h_pair = torch.cat(
                [h_a_drug, h_b_drug, h_a_drug * h_b_drug,
                 (h_a_drug - h_b_drug).abs(), zk], dim=-1)
            return self.mlp(h_pair).squeeze(-1)
        nb_a, qa = drug_neighbours(data, pair_index[0])
        nb_b, qb = drug_neighbours(data, pair_index[1])
        nq = pair_index.size(1)
        if self.use_shared_view:
            from kdcrga.data.graph import drug_two_hop_shared
            nb_s, qs = drug_two_hop_shared(data, pair_index)
            views = list(rotatory_readouts(
                self.attn, h[nb_a], qa, h[nb_b], qb, h_a_drug, h_b_drug, zk,
                num_queries=nq, num_hops=self.num_hops,
                shared=self.shared, h_nb_shared=h[nb_s], q_shared=qs))
        else:
            views = list(rotatory_readouts(
                self.attn, h[nb_a], qa, h[nb_b], qb, h_a_drug, h_b_drug, zk,
                num_queries=nq, num_hops=self.num_hops))
        if self.use_hierarchical:
            h_pair, _ = self.aggregator(views)
        else:
            h_pair = torch.cat(views, dim=-1)
        return self.mlp(h_pair).squeeze(-1)

    @torch.no_grad()
    def attention_for_query(
        self, data, pair_index, side_effect,
    ) -> dict[str, torch.Tensor]:
        """Return the (pre-rotatory) hop-1 attention distributions for the A-side
        and B-side neighbourhoods plus their neighbour query indices, for RQ4
        analysis and case studies.

        For each side, neighbours of the drug attend conditioned on the *partner*
        drug and the side-effect embedding z_k, matching `rotatory_readouts`' hop-1
        step. Returns alpha_a/alpha_b (per-neighbour weights) with the neighbour
        ids (nb_a/nb_b) and query indices (qa/qb).
        """
        h = self.encode(data)
        h_a = h[pair_index[0]]
        h_b = h[pair_index[1]]
        zk = self.z_k(side_effect)
        nb_a, qa = drug_neighbours(data, pair_index[0])
        nb_b, qb = drug_neighbours(data, pair_index[1])
        num_queries = pair_index.size(1)
        _, alpha_a = self.attn(h[nb_a], qa, h_b, zk,
                               num_queries=num_queries, return_alpha=True)
        _, alpha_b = self.attn(h[nb_b], qb, h_a, zk,
                               num_queries=num_queries, return_alpha=True)
        return {"alpha_a": alpha_a, "nb_a": nb_a, "qa": qa,
                "alpha_b": alpha_b, "nb_b": nb_b, "qb": qb}
