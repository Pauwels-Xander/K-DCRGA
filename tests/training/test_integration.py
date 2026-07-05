from pathlib import Path

import pytest
import torch

DATA = Path("third_party/knowddi/data/BioSNAP")


@pytest.mark.skipif(not DATA.exists(), reason="KnowDDI data not downloaded")
def test_dc_rgcn_loss_decreases_on_real_data():
    from kdcrga.data.graph import attach_splits, load_knowddi_graph
    from kdcrga.models.baselines.dc_rgcn import DecoderConditionedRGCN
    from kdcrga.training.loop import train

    data = attach_splits(load_knowddi_graph(DATA), seed=0)
    num_relations = sum(1 for et in data.edge_types if et[1].startswith("rel_"))
    model = DecoderConditionedRGCN(
        in_dim=data["entity"].x.size(-1), hidden_dim=32,
        num_relations=num_relations, num_side_effects=int(data.num_side_effects),
        num_layers=2, num_bases=10, dropout=0.1,
    )
    history = train(model, data, lr=1e-3, batch_size=2048, max_epochs=2,
                    patience=2, seed=0, max_batches=5, show_progress=False)
    assert len(history["train_loss"]) >= 1
    assert torch.isfinite(torch.tensor(history["train_loss"])).all()
    if len(history["train_loss"]) >= 2:
        assert history["train_loss"][-1] <= history["train_loss"][0]
