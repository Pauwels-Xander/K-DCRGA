import torch

from kdcrga.data.schema import DDIEdges, build_hetero_data


def test_build_hetero_data_has_expected_shapes():
    drug_x = torch.randn(4, 8)
    gene_x = torch.randn(3, 8)
    edges = {("drug", "binds", "gene"): torch.tensor([[0, 1, 2], [0, 1, 2]])}
    ddi = DDIEdges(
        pair_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        side_effect=torch.tensor([0, 0, 1]),
    )
    data = build_hetero_data({"drug": drug_x, "gene": gene_x}, edges, ddi)

    assert data["drug"].num_nodes == 4
    assert data["gene"].num_nodes == 3
    assert data["drug", "binds", "gene"].edge_index.shape == (2, 3)
    assert data.ddi.pair_index.shape == (2, 3)
    assert data.ddi.side_effect.tolist() == [0, 0, 1]
    assert data.num_side_effects == 2


def test_synthetic_graph_fixture(synthetic_graph):
    assert synthetic_graph["drug"].num_nodes == 4
    assert synthetic_graph.num_side_effects == 3
    assert synthetic_graph.ddi.pair_index.shape == (2, 6)
