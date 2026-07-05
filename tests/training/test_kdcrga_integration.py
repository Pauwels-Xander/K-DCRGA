from pathlib import Path

import pytest
import torch

DATA = Path("third_party/knowddi/data/BioSNAP")


@pytest.mark.skipif(not DATA.exists(), reason="KnowDDI data not downloaded")
def test_kdcrga_v3_runs_on_real_data():
    from kdcrga.data.graph import attach_splits, load_knowddi_graph
    from kdcrga.models.kdcrga import KDCRGA
    from kdcrga.training.loop import train

    data = attach_splits(load_knowddi_graph(DATA), seed=0)
    num_relations = sum(1 for et in data.edge_types if et[1].startswith("rel_"))
    model = KDCRGA(
        in_dim=data["entity"].x.size(-1), hidden_dim=32,
        num_relations=num_relations, num_side_effects=int(data.num_side_effects),
        num_layers=2, num_bases=10, dropout=0.1,
        num_hops=2, use_shared_view=True, use_hierarchical=True,
    )
    history = train(model, data, lr=1e-3, batch_size=512, max_epochs=1,
                    patience=1, seed=0, max_batches=3)
    assert len(history["train_loss"]) >= 1
    assert torch.isfinite(torch.tensor(history["train_loss"])).all()
