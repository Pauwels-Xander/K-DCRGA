import torch

from kdcrga.models.baselines.dc_rgcn import DecoderConditionedRGCN


def test_dc_rgcn_forward_returns_logit_per_query(synthetic_rgcn_graph):
    model = DecoderConditionedRGCN(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
    )
    pair_index = synthetic_rgcn_graph.ddi.pair_index      # [2, 6]
    side_effect = synthetic_rgcn_graph.ddi.side_effect    # [6]
    logits = model(synthetic_rgcn_graph, pair_index, side_effect)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()
    alt = model(synthetic_rgcn_graph, pair_index[:, :1], torch.tensor([2]))
    assert alt.shape == (1,)
