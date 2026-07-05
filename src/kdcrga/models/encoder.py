"""R-GCN heterogeneous encoder with basis decomposition (proposal Sec. 4.2).

Produces final-layer node representations h_j reused across all 200 side effects
(amortised inference, proposal Sec. 4.8).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv


def hetero_to_rgcn_inputs(
    data: HeteroData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten the single-'entity' HeteroData into RGCNConv inputs.

    Returns (x, edge_index, edge_type). A numeric relation name "rel_{r}" keeps
    its parsed id r (so the BKG relations rel_0..rel_K map to 0..K, unchanged); a
    non-numeric name such as "rel_atc" (injected ontology relation) is assigned
    the next id after the maximum numeric one, deterministically by name. This
    keeps relation ids contiguous and < num_relations. Concatenates all rel_*
    edge types."""
    x = data["entity"].x
    rel_etypes = [et for et in data.edge_types if et[1].startswith("rel_")]
    max_num = -1
    for et in rel_etypes:
        suffix = et[1].split("_", 1)[1]
        if suffix.isdigit():
            max_num = max(max_num, int(suffix))
    rel_id_of: dict[str, int] = {}
    next_id = max_num + 1
    for et in sorted((e for e in rel_etypes if not e[1].split("_", 1)[1].isdigit()),
                     key=lambda e: e[1]):
        rel_id_of[et[1]] = next_id
        next_id += 1

    edge_index_parts: list[torch.Tensor] = []
    edge_type_parts: list[torch.Tensor] = []
    for etype in rel_etypes:
        suffix = etype[1].split("_", 1)[1]
        rel_id = int(suffix) if suffix.isdigit() else rel_id_of[etype[1]]
        ei = data[etype].edge_index
        edge_index_parts.append(ei)
        edge_type_parts.append(torch.full((ei.size(1),), rel_id, dtype=torch.long))
    edge_index = torch.cat(edge_index_parts, dim=1)
    edge_type = torch.cat(edge_type_parts, dim=0)
    return x, edge_index, edge_type


class RGCNEncoder(nn.Module):
    """Relation-typed R-GCN with basis decomposition (proposal Sec. 4.2, eq:rgcn).

    Maps node features to representations reused across all side effects
    (amortised inference, proposal Sec. 4.8).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_relations: int,
        num_layers: int = 3,
        num_bases: int = 30,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for layer in range(num_layers):
            self.convs.append(
                RGCNConv(dims[layer], dims[layer + 1],
                         num_relations=num_relations, num_bases=num_bases)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = self.dropout(torch.relu(x))
        return x
