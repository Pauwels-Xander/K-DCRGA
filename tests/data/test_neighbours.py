import torch

from kdcrga.data.graph import drug_neighbours


def test_drug_neighbours_returns_flat_neighbours_and_query_index(synthetic_rgcn_graph):
    pair_index = torch.tensor([[0, 1], [2, 3]])
    flat, query = drug_neighbours(synthetic_rgcn_graph, pair_index[0])
    assert flat.dtype == torch.long
    assert query.dtype == torch.long
    assert flat.numel() == query.numel()
    assert set(query.tolist()) <= {0, 1}
    n_d0 = flat[query == 0].tolist()
    assert 4 in n_d0
