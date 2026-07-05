"""Project HeteroData schema for the TWOSIDES+Hetionet graph.

DDI edges are stored as a single side-effect-labelled container (not 200 edge
types), matching how the encoder consumes them and keeping memory bounded.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData

NodeFeatures = dict[str, torch.Tensor]
EdgeIndices = dict[tuple[str, str, str], torch.Tensor]


@dataclass
class DDIEdges:
    pair_index: torch.Tensor          # [2, E] drug-pair endpoints (local drug ids)
    side_effect: torch.Tensor         # [E] side-effect id per pair
    split: torch.Tensor | None = None  # [E] 0=train 1=val 2=test, filled by splits.py
    # [E] source BioSNAP file per triple (0=train.txt 1=valid.txt 2=test.txt). Set by
    # the loader; consumed by attach_file_splits for the pair-disjoint canonical split.
    source_file: torch.Tensor | None = None


def build_hetero_data(
    node_features: NodeFeatures,
    edges: EdgeIndices,
    ddi: DDIEdges,
) -> HeteroData:
    data = HeteroData()
    for ntype, x in node_features.items():
        data[ntype].x = x
    for etype, edge_index in edges.items():
        data[etype].edge_index = edge_index
    data.ddi = ddi
    data.num_side_effects = int(ddi.side_effect.max().item()) + 1
    return data
