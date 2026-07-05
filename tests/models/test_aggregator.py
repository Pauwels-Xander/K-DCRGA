import torch

from kdcrga.models.aggregator import HierarchicalAggregator


def test_aggregator_betas_sum_to_one():
    torch.manual_seed(0)
    agg = HierarchicalAggregator(d_hidden=8, d_attn=4)
    views = [torch.randn(5, 8) for _ in range(3)]
    h_pair, betas = agg(views)
    assert h_pair.shape == (5, 3 * 8)
    assert betas.shape == (5, 3)
    assert torch.allclose(betas.sum(-1), torch.ones(5), atol=1e-6)


def test_aggregator_single_view_works():
    torch.manual_seed(0)
    agg = HierarchicalAggregator(d_hidden=4, d_attn=3)
    views = [torch.randn(2, 4)]
    h_pair, betas = agg(views)
    assert h_pair.shape == (2, 4)
    assert betas.shape == (2, 1)
    # single view -> beta == 1
    assert torch.allclose(betas, torch.ones(2, 1))
