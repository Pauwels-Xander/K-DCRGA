import torch

from kdcrga.data.graph import drug_two_hop_shared


def test_two_hop_shared_returns_intersection(synthetic_rgcn_graph):
    pair_index = torch.tensor([[0], [1]])
    flat, query = drug_two_hop_shared(synthetic_rgcn_graph, pair_index)
    assert flat.dtype == torch.long and query.dtype == torch.long
    assert flat.numel() == query.numel()
    assert query.tolist() == [0] * flat.numel()
    # all returned nodes must be entities in the graph
    assert (flat < 6).all() and (flat >= 0).all()
    # excludes the drug pair endpoints themselves
    assert 0 not in flat.tolist() and 1 not in flat.tolist()


def test_two_hop_shared_empty_for_disconnected_pair():
    # Build a tiny graph with disconnected components
    from kdcrga.data.schema import DDIEdges, build_hetero_data
    nf = {"entity": torch.randn(4, 4)}
    edges = {("entity", "rel_0", "entity"): torch.tensor([[0], [2]])}  # only 0--2
    ddi = DDIEdges(pair_index=torch.tensor([[0],[1]]),
                   side_effect=torch.tensor([0]))
    data = build_hetero_data(nf, edges, ddi)
    data.num_drugs = 4
    flat, query = drug_two_hop_shared(data, torch.tensor([[0],[1]]))
    # drug 1 has no edges -> intersection is empty
    assert flat.numel() == 0
