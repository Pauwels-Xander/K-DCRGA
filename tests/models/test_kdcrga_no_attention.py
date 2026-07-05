"""The 4th cell of the encoder x decoder 2x2: KDCRGA with use_attention=False is a
plain R-GCN encoder + shared-MLP scorer. With attention off, the side effect must
still enter the score, so z_k is concatenated to the pair features
[h_a, h_b, h_a*h_b, |h_a-h_b|, z_k] -> MLP (no neighbour attention/rotatory)."""
import torch

from kdcrga.models.kdcrga import KDCRGA


def _model(use_attention):
    return KDCRGA(in_dim=8, hidden_dim=4, num_relations=2, num_side_effects=6,
                  num_layers=2, num_bases=2, dropout=0.0,
                  use_shared_view=False, use_hierarchical=False,
                  use_attention=use_attention)


def test_default_is_attention_with_n_views_mlp():
    m = _model(use_attention=True)
    assert m.use_attention is True
    assert m.mlp[0].in_features == 4 * 4   # 4 rotatory views x hidden_dim


def test_no_attention_mlp_takes_five_views():
    m = _model(use_attention=False)
    assert m.use_attention is False
    assert m.mlp[0].in_features == 5 * 4   # [h_a, h_b, h_a*h_b, |h_a-h_b|, z_k]


def test_no_attention_forward_shape_and_se_conditioning():
    # use_attention=False returns before touching the graph, so a precomputed h
    # lets us exercise the pair-MLP path without building a full graph fixture.
    m = _model(use_attention=False)
    m.eval()
    h = torch.randn(10, 4)
    pair = torch.tensor([[0, 1, 2], [3, 4, 5]])
    se = torch.tensor([0, 1, 2])
    out = m(None, pair, se, h=h)
    assert out.shape == (3,)
    # deterministic in eval mode
    assert torch.allclose(out, m(None, pair, se, h=h))
    # the score depends on the queried side effect (z_k routed into the MLP)
    se_other = torch.tensor([3, 4, 5])
    assert not torch.allclose(out, m(None, pair, se_other, h=h))
