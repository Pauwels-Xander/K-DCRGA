"""Gradient-accumulation control-flow tests for train().

Accumulating gradients over `accum_steps` micro-batches before stepping lets a
VRAM-bound model (small micro-batch) match a larger effective batch. These tests
pin the optimizer-step count: ceil(n_microbatches / accum_steps) per epoch,
including the final partial-window tail.
"""
import torch

from kdcrga.models.baselines.decagon import Decagon
from kdcrga.training.loop import train


def _graph_with_splits():
    from kdcrga.data.schema import DDIEdges, build_hetero_data
    torch.manual_seed(0)
    nf = {"entity": torch.randn(6, 8)}
    edges = {
        ("entity", "rel_0", "entity"): torch.tensor([[0, 1, 4], [4, 5, 2]]),
        ("entity", "rel_1", "entity"): torch.tensor([[2, 3], [5, 4]]),
    }
    pair = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 0],
                         [1, 2, 3, 0, 2, 3, 0, 1]])
    se = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    ddi = DDIEdges(pair_index=pair, side_effect=se)
    data = build_hetero_data(nf, edges, ddi)
    data.num_drugs = 4
    data.ddi.split = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])  # 6 train, 2 val
    return data


def _count_adam_steps(monkeypatch):
    counter = {"n": 0}
    real = torch.optim.Adam.step

    def counting(self, *a, **k):
        counter["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", counting)
    return counter


def _model():
    return Decagon(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                   num_layers=2, num_bases=2, dropout=0.0)


def _train(accum_steps, max_batches):
    train(_model(), _graph_with_splits(), batch_size=1, max_epochs=1,
          patience=5, max_batches=max_batches, max_val_pairs=2,
          accum_steps=accum_steps, show_progress=False)


def test_accum_steps_1_steps_every_microbatch(monkeypatch):
    counter = _count_adam_steps(monkeypatch)
    _train(accum_steps=1, max_batches=4)
    assert counter["n"] == 4   # 4 micro-batches, step each


def test_accum_steps_2_one_step_per_window(monkeypatch):
    counter = _count_adam_steps(monkeypatch)
    _train(accum_steps=2, max_batches=4)
    assert counter["n"] == 2   # 4 / 2 = 2 steps


def test_accum_steps_tail_does_final_partial_step(monkeypatch):
    counter = _count_adam_steps(monkeypatch)
    _train(accum_steps=2, max_batches=3)
    assert counter["n"] == 2   # ceil(3 / 2): one full window + tail
