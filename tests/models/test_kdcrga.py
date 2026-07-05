import torch

from kdcrga.models.kdcrga import KDCRGA


def test_kdcrga_v3_with_shared_view(synthetic_rgcn_graph):
    pair_index = synthetic_rgcn_graph.ddi.pair_index
    side_effect = synthetic_rgcn_graph.ddi.side_effect
    model = KDCRGA(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
        num_hops=2, use_shared_view=True, use_hierarchical=False,
    )
    logits = model(synthetic_rgcn_graph, pair_index, side_effect)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()
    assert model.n_views == 5   # 4 rotatory views (r_A, r_B, r_{A|B}, r_{B|A}) + shared


def test_kdcrga_v4_with_hierarchical_aggregator(synthetic_rgcn_graph):
    pair_index = synthetic_rgcn_graph.ddi.pair_index
    side_effect = synthetic_rgcn_graph.ddi.side_effect
    model = KDCRGA(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
        num_hops=2, use_shared_view=True, use_hierarchical=True,
    )
    logits = model(synthetic_rgcn_graph, pair_index, side_effect)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()
    assert hasattr(model, "aggregator")


def test_kdcrga_v1_forward_shape(synthetic_rgcn_graph):
    model = KDCRGA(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
        num_hops=1, use_shared_view=False, use_hierarchical=False,
    )
    pair_index = synthetic_rgcn_graph.ddi.pair_index
    side_effect = synthetic_rgcn_graph.ddi.side_effect
    logits = model(synthetic_rgcn_graph, pair_index, side_effect)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()


def test_kdcrga_z_init_copied_into_embedding(synthetic_rgcn_graph):
    torch.manual_seed(0)
    z_init = torch.full((3, 16), 0.42)
    model = KDCRGA(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0, num_hops=1,
        z_init=z_init,
    )
    assert torch.allclose(model.z_k.weight, z_init)
