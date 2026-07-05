import json

import torch

from kdcrga.eval.run import evaluate_test_set, save_run_outputs


def test_evaluate_test_set_returns_agg_and_per_se(synthetic_rgcn_graph):
    from kdcrga.data.graph import attach_splits
    from kdcrga.models.kdcrga import KDCRGA

    data = attach_splits(synthetic_rgcn_graph, seed=0)
    model = KDCRGA(in_dim=8, hidden_dim=8, num_relations=2,
                   num_side_effects=int(data.num_side_effects),
                   num_layers=2, num_bases=2, dropout=0.0, num_hops=1)
    agg, per_se = evaluate_test_set(model, data, batch_size=4)
    assert "auroc" in agg and "auprc" in agg and "ap50" in agg
    assert isinstance(per_se, dict)
    # per_se keys are ints, each a metric dict (may be empty if a SE lacks both
    # classes in the tiny test split, but the structure must hold)
    for k, v in per_se.items():
        assert isinstance(k, int)
        assert set(v.keys()) == {"auroc", "auprc", "ap50"}


def test_save_run_outputs_writes_three_json_files(tmp_path):
    history = {"train_loss": [0.5, 0.4], "val_auprc": [0.6, 0.65]}
    agg = {"auroc": 0.8, "auprc": 0.7}
    per_se = {0: {"auroc": 0.9, "auprc": 0.8, "ap50": 0.7}}
    save_run_outputs(tmp_path, history, agg, per_se)

    assert json.loads((tmp_path / "history.json").read_text()) == history
    assert json.loads((tmp_path / "test_metrics.json").read_text()) == agg
    # per_se keys are stringified in JSON; reload and cast back
    loaded = json.loads((tmp_path / "per_se_metrics.json").read_text())
    assert loaded == {"0": {"auroc": 0.9, "auprc": 0.8, "ap50": 0.7}}
