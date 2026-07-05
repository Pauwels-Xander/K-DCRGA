import torch

from kdcrga.training.loss import class_weights, weighted_bce_loss


def test_class_weights_rarer_class_gets_higher_weight():
    side_effect = torch.cat([torch.zeros(90, dtype=torch.long),
                             torch.ones(10, dtype=torch.long)])
    w = class_weights(side_effect, num_side_effects=2, gamma=0.5)
    assert w.shape == (2,)
    assert w[1] > w[0]
    w0 = class_weights(side_effect, num_side_effects=2, gamma=0.0)
    assert torch.allclose(w0, torch.ones(2))


def test_weighted_bce_runs_and_is_finite():
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    side_effect = torch.tensor([0, 1, 0, 1])
    w = class_weights(torch.tensor([0, 0, 0, 1]), num_side_effects=2, gamma=0.5)
    loss = weighted_bce_loss(logits, labels, side_effect, w)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
