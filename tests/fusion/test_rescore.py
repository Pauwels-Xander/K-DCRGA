import torch
from kdcrga.fusion.tables import PredTable
from kdcrga.fusion.rescore import score_table, per_se_metric


def _table(scores):
    # 2 side effects, 4 triples each, perfectly separable by score
    a = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    b = torch.tensor([2, 3, 4, 5, 2, 3, 4, 5])
    se = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    y = torch.tensor([1., 1., 0., 0., 1., 1., 0., 0.])
    return PredTable(a, b, se, y, torch.tensor(scores), "prob")


def test_score_table_perfect_separation():
    t = _table([.9, .8, .1, .2, .9, .8, .1, .2])
    agg, per_se = score_table(t, {0: 0, 1: 0})
    assert agg["auroc"] == 1.0
    assert set(per_se) == {0, 1}
    assert per_se_metric(per_se, "auroc") == {0: 1.0, 1: 1.0}
