import torch
from kdcrga.fusion.tables import PredTable
from experiments.fusion_complementarity import error_correlation


def _t(scores, kind):
    a = torch.zeros(4, dtype=torch.int64)
    b = torch.arange(4)
    se = torch.zeros(4, dtype=torch.int64)
    y = torch.tensor([1., 1., 0., 0.])
    return PredTable(a, b, se, y, torch.tensor(scores), kind)


def test_error_correlation_identical_streams_is_one():
    f = _t([0.9, 0.8, 0.1, 0.2], "prob")
    s = _t([0.9, 0.8, 0.1, 0.2], "prob")
    assert abs(error_correlation(f, s) - 1.0) < 1e-6
