import torch

from kdcrga.models.baselines.decagon import Decagon


def _model():
    return Decagon(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                   num_layers=2, num_bases=2, dropout=0.0)


def test_decagon_forward_returns_logit_per_query(synthetic_rgcn_graph):
    model = _model()
    pair_index = synthetic_rgcn_graph.ddi.pair_index      # [2, 6]
    side_effect = synthetic_rgcn_graph.ddi.side_effect    # [6]
    logits = model(synthetic_rgcn_graph, pair_index, side_effect)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()
    alt = model(synthetic_rgcn_graph, pair_index[:, :1], torch.tensor([2]))
    assert alt.shape == (1,)


def test_decagon_score_depends_on_side_effect(synthetic_rgcn_graph):
    # same drug pair, different side effect -> different score (per-SE diagonal d_k)
    torch.manual_seed(0)
    model = _model()
    pair = torch.tensor([[0], [1]])
    s0 = model(synthetic_rgcn_graph, pair, torch.tensor([0]))
    s1 = model(synthetic_rgcn_graph, pair, torch.tensor([1]))
    assert not torch.allclose(s0, s1)


def test_decagon_R_is_single_shared_matrix():
    model = _model()
    # The global DEDICOM matrix R is one (d, d) parameter shared across all SEs,
    # now owned by the shared DedicomDecoder.
    assert model.decoder.R.shape == (16, 16)
    # per-SE structure lives only in the diagonal embedding D
    assert model.decoder.D.weight.shape == (3, 16)


def test_decagon_forward_matches_explicit_dedicom(synthetic_rgcn_graph):
    torch.manual_seed(0)
    model = _model()
    model.eval()
    pair = synthetic_rgcn_graph.ddi.pair_index
    se = synthetic_rgcn_graph.ddi.side_effect
    with torch.no_grad():
        h = model.encode(synthetic_rgcn_graph)
        out = model(synthetic_rgcn_graph, pair, se)
        d_k = model.decoder.D(se)
        ua = h[pair[0]] * d_k
        ub = h[pair[1]] * d_k
        expected = ((ua @ model.decoder.R) * ub).sum(dim=-1)
    assert torch.allclose(out, expected, atol=1e-5)
