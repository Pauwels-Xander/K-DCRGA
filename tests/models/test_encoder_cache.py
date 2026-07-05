"""Equivalence tests for the encode() / optional-h encoder-cache refactor.

For each of the four models, in EVAL mode with dropout=0.0, passing a
precomputed h to forward() must produce identical output to the default path
(which calls encode() internally). With dropout off and weights frozen the
encoder is deterministic, so allclose(atol=default=1e-8) should hold exactly.
"""
import torch

from kdcrga.models.kdcrga import KDCRGA
from kdcrga.models.baselines.dc_rgcn import DecoderConditionedRGCN
from kdcrga.models.baselines.decagon import Decagon
from kdcrga.models.baselines.lagat import LaGAT


def _common(graph):
    return dict(pair=graph.ddi.pair_index, se=graph.ddi.side_effect)


def test_kdcrga_encode_cache_matches_full_forward(synthetic_rgcn_graph):
    torch.manual_seed(0)
    m = KDCRGA(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
               num_layers=2, num_bases=2, dropout=0.0, num_hops=2,
               use_shared_view=True, use_hierarchical=True)
    m.eval()
    c = _common(synthetic_rgcn_graph)
    with torch.no_grad():
        h = m.encode(synthetic_rgcn_graph)
        out_cached = m(synthetic_rgcn_graph, c["pair"], c["se"], h=h)
        out_full = m(synthetic_rgcn_graph, c["pair"], c["se"])
    assert out_cached.shape == out_full.shape
    assert torch.allclose(out_cached, out_full)


def test_dc_rgcn_encode_cache_matches_full_forward(synthetic_rgcn_graph):
    torch.manual_seed(0)
    m = DecoderConditionedRGCN(in_dim=8, hidden_dim=16, num_relations=2,
                               num_side_effects=3, num_layers=2, num_bases=2, dropout=0.0)
    m.eval()
    c = _common(synthetic_rgcn_graph)
    with torch.no_grad():
        h = m.encode(synthetic_rgcn_graph)
        assert torch.allclose(m(synthetic_rgcn_graph, c["pair"], c["se"], h=h),
                              m(synthetic_rgcn_graph, c["pair"], c["se"]))


def test_decagon_encode_cache_matches_full_forward(synthetic_rgcn_graph):
    torch.manual_seed(0)
    m = Decagon(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                num_layers=2, num_bases=2, dropout=0.0)
    m.eval()
    c = _common(synthetic_rgcn_graph)
    with torch.no_grad():
        h = m.encode(synthetic_rgcn_graph)
        assert torch.allclose(m(synthetic_rgcn_graph, c["pair"], c["se"], h=h),
                              m(synthetic_rgcn_graph, c["pair"], c["se"]))


def test_lagat_encode_cache_matches_full_forward(synthetic_rgcn_graph):
    torch.manual_seed(0)
    m = LaGAT(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
              num_layers=2, num_bases=2, dropout=0.0)
    m.eval()
    c = _common(synthetic_rgcn_graph)
    with torch.no_grad():
        h = m.encode(synthetic_rgcn_graph)
        assert torch.allclose(m(synthetic_rgcn_graph, c["pair"], c["se"], h=h),
                              m(synthetic_rgcn_graph, c["pair"], c["se"]))
