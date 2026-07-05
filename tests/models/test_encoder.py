import torch

from kdcrga.models.encoder import hetero_to_rgcn_inputs


def test_hetero_to_rgcn_inputs_flattens_relations(synthetic_rgcn_graph):
    x, edge_index, edge_type = hetero_to_rgcn_inputs(synthetic_rgcn_graph)
    assert x.shape == (6, 8)
    assert edge_index.shape == (2, 5)   # 3 (rel_0) + 2 (rel_1)
    assert edge_type.shape == (5,)
    assert set(edge_type.tolist()) == {0, 1}
    rel0_cols = (edge_type == 0)
    assert edge_index[:, rel0_cols].shape == (2, 3)


def test_hetero_to_rgcn_inputs_assigns_nonnumeric_relation_next_id():
    # rel_atc (injected ontology relation) is not rel_<int>; it must get the id
    # after the max numeric relation, leaving numeric ids untouched.
    from kdcrga.data.schema import DDIEdges, build_hetero_data
    nf = {"entity": torch.randn(5, 4)}
    edges = {
        ("entity", "rel_0", "entity"): torch.tensor([[0, 1], [2, 3]]),
        ("entity", "rel_1", "entity"): torch.tensor([[0], [4]]),
        ("entity", "rel_atc", "entity"): torch.tensor([[0, 1], [4, 4]]),
    }
    ddi = DDIEdges(pair_index=torch.tensor([[0], [1]]), side_effect=torch.tensor([0]))
    data = build_hetero_data(nf, edges, ddi)
    _, _, edge_type = hetero_to_rgcn_inputs(data)
    assert set(edge_type.tolist()) == {0, 1, 2}     # rel_atc -> 2 (max numeric + 1)
    assert int((edge_type == 2).sum()) == 2          # both rel_atc edges mapped to 2


from kdcrga.models.encoder import RGCNEncoder


def test_rgcn_encoder_output_shape(synthetic_rgcn_graph):
    x, edge_index, edge_type = hetero_to_rgcn_inputs(synthetic_rgcn_graph)
    enc = RGCNEncoder(in_dim=8, hidden_dim=16, num_relations=2, num_layers=2,
                      num_bases=2, dropout=0.0)
    h = enc(x, edge_index, edge_type)
    assert h.shape == (6, 16)
    assert torch.isfinite(h).all()
