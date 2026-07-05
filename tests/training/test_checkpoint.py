import json

import torch

from kdcrga.models.baselines.dc_rgcn import DecoderConditionedRGCN
from kdcrga.training.checkpoint import load_checkpoint, save_checkpoint, save_history
from kdcrga.training.loop import train


def test_checkpoint_roundtrip(tmp_path):
    model = DecoderConditionedRGCN(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [0.5], "val_auprc": [0.3]}
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, epoch=2, model=model, optimizer=opt,
        best_val=0.3, waited=1, history=history, seed=0,
    )
    payload = load_checkpoint(path, model, opt)
    assert payload["epoch"] == 2
    assert payload["best_val"] == 0.3
    assert payload["history"] == history


def test_save_history_writes_json(tmp_path):
    history = {"train_loss": [1.0, 0.8], "val_auprc": [0.2, 0.25]}
    path = tmp_path / "history.json"
    save_history(path, history)
    assert json.loads(path.read_text(encoding="utf-8")) == history


def test_train_writes_epoch_checkpoints(synthetic_rgcn_graph, tmp_path):
    from kdcrga.data.graph import attach_splits

    data = attach_splits(synthetic_rgcn_graph, ratios=(0.6, 0.2, 0.2), seed=0)
    model = DecoderConditionedRGCN(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
    )
    ckpt_dir = tmp_path / "run"
    train(
        model, data, lr=1e-2, batch_size=4, max_epochs=2, patience=2,
        seed=0, max_batches=2, checkpoint_dir=ckpt_dir, show_progress=False,
    )
    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "history.json").exists()
    history = json.loads((ckpt_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history["train_loss"]) == 2
    assert len(history["val_auprc"]) == 2


def test_train_resume_continues_history(synthetic_rgcn_graph, tmp_path):
    from kdcrga.data.graph import attach_splits

    data = attach_splits(synthetic_rgcn_graph, ratios=(0.6, 0.2, 0.2), seed=0)
    model = DecoderConditionedRGCN(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
    )
    ckpt_dir = tmp_path / "run"
    h1 = train(
        model, data, lr=1e-2, batch_size=4, max_epochs=4, patience=4,
        seed=0, max_batches=2, checkpoint_dir=ckpt_dir, show_progress=False,
    )
    model2 = DecoderConditionedRGCN(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
    )
    h2 = train(
        model2, data, lr=1e-2, batch_size=4, max_epochs=4, patience=4,
        seed=0, max_batches=2, checkpoint_dir=ckpt_dir, resume=True,
        show_progress=False,
    )
    assert len(h2["train_loss"]) >= len(h1["train_loss"])
