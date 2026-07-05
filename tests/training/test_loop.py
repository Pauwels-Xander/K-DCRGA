import pytest
import torch

from kdcrga.models.baselines.dc_rgcn import DecoderConditionedRGCN
from kdcrga.training.loop import train


def test_train_overfits_tiny_graph(synthetic_rgcn_graph):
    from kdcrga.data.graph import attach_splits
    data = attach_splits(synthetic_rgcn_graph, ratios=(0.6, 0.2, 0.2), seed=0)
    model = DecoderConditionedRGCN(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
    )
    history = train(
        model, data,
        lr=1e-2, weight_decay=0.0, gamma=0.5, batch_size=8,
        max_epochs=50, patience=50, seed=0, show_progress=False,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))


def test_train_respects_max_val_pairs(synthetic_rgcn_graph):
    from kdcrga.data.graph import attach_splits
    data = attach_splits(synthetic_rgcn_graph, ratios=(0.6, 0.2, 0.2), seed=0)
    model = DecoderConditionedRGCN(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
    )
    # capping val to 1 pair must not error and still produces a val_auprc series
    history = train(
        model, data, lr=1e-2, weight_decay=0.0, gamma=0.5, batch_size=8,
        max_epochs=2, patience=2, seed=0, show_progress=False, max_val_pairs=1,
    )
    assert len(history["val_auprc"]) >= 1
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_train_runs_on_cuda(synthetic_rgcn_graph):
    from kdcrga.data.graph import attach_splits
    from kdcrga.models.baselines.dc_rgcn import DecoderConditionedRGCN
    data = attach_splits(synthetic_rgcn_graph, ratios=(0.6, 0.2, 0.2), seed=0)
    model = DecoderConditionedRGCN(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
    )
    history = train(model, data, lr=1e-2, batch_size=8, max_epochs=3,
                    patience=3, seed=0, device="cuda", show_progress=False)
    assert len(history["train_loss"]) >= 1
    assert next(model.parameters()).is_cuda
