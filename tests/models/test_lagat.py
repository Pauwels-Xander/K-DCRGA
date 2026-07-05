import torch

from kdcrga.models.baselines.lagat import LaGAT


def _model():
    return LaGAT(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                 num_layers=2, num_bases=2, dropout=0.0)


def test_lagat_forward_returns_logit_per_query(synthetic_rgcn_graph):
    model = _model()
    pair_index = synthetic_rgcn_graph.ddi.pair_index      # [2, 6]
    side_effect = synthetic_rgcn_graph.ddi.side_effect    # [6]
    logits = model(synthetic_rgcn_graph, pair_index, side_effect)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()
    alt = model(synthetic_rgcn_graph, pair_index[:, :1], torch.tensor([2]))
    assert alt.shape == (1,)


def test_lagat_score_depends_on_side_effect(synthetic_rgcn_graph):
    # side-effect conditioning lives in the decoder rel_k -> different scores
    torch.manual_seed(0)
    model = _model()
    pair = torch.tensor([[0], [1]])
    s0 = model(synthetic_rgcn_graph, pair, torch.tensor([0]))
    s1 = model(synthetic_rgcn_graph, pair, torch.tensor([1]))
    assert not torch.allclose(s0, s1)


def test_lagat_attention_is_partner_conditional(synthetic_rgcn_graph):
    # The attention readout for drug 0 is conditioned on the PARTNER drug; pairing
    # drug 0 with two different partners (that have neighbours) should change its
    # readout and hence the score for a fixed side effect.
    torch.manual_seed(0)
    model = _model()
    s_with_b1 = model(synthetic_rgcn_graph,
                      torch.tensor([[0], [1]]), torch.tensor([0]))
    s_with_b2 = model(synthetic_rgcn_graph,
                      torch.tensor([[0], [2]]), torch.tensor([0]))
    assert not torch.allclose(s_with_b1, s_with_b2)
