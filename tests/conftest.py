"""Shared pytest fixtures.

`synthetic_graph` is a small heterogeneous graph used by every plan's tests:
4 drugs, 3 genes, 2 diseases, a few typed edges, and DDI edges over 3 side effects.
"""
from __future__ import annotations

import pytest
import torch

from kdcrga.data.schema import DDIEdges, build_hetero_data


@pytest.fixture
def synthetic_graph():
    torch.manual_seed(0)
    node_features = {
        "drug": torch.randn(4, 8),
        "gene": torch.randn(3, 8),
        "disease": torch.randn(2, 8),
    }
    edges = {
        ("drug", "binds", "gene"): torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]]),
        ("drug", "treats", "disease"): torch.tensor([[0, 2], [0, 1]]),
        ("gene", "interacts", "gene"): torch.tensor([[0, 1], [1, 2]]),
    }
    # side effect 0 is frequent (3 pairs), 1 medium (2), 2 rare (1)
    ddi = DDIEdges(
        pair_index=torch.tensor([[0, 1, 2, 0, 3, 1], [1, 2, 3, 2, 0, 3]]),
        side_effect=torch.tensor([0, 0, 0, 1, 1, 2]),
    )
    data = build_hetero_data(node_features, edges, ddi)
    data.num_drugs = 4
    return data


@pytest.fixture
def synthetic_rgcn_graph():
    """Mirrors the real loader output: single 'entity' node type, rel_* edges,
    DDIEdges over a few side effects. 6 entities; drugs are entities 0..3."""
    torch.manual_seed(0)
    node_features = {"entity": torch.randn(6, 8)}
    edges = {
        ("entity", "rel_0", "entity"): torch.tensor([[0, 1, 4], [4, 5, 2]]),
        ("entity", "rel_1", "entity"): torch.tensor([[2, 3], [5, 4]]),
    }
    ddi = DDIEdges(
        pair_index=torch.tensor([[0, 1, 2, 0, 3, 1], [1, 2, 3, 2, 0, 3]]),
        side_effect=torch.tensor([0, 0, 0, 1, 1, 2]),
    )
    data = build_hetero_data(node_features, edges, ddi)
    data.num_drugs = 4
    return data
