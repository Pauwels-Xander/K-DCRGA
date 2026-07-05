"""Train and evaluate K-DCRGA on BioSNAP-200.

Usage: python experiments/train_kdcrga.py [config.yaml]
The config should set training.* and model.* keys, plus optional
model.num_hops, model.use_shared_view, model.use_hierarchical for v1..v4
selection. (v5 -- ontology injection -- requires real MedDRA / ATC data and
is wired via a future ontology-loading helper.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from kdcrga.config import load_config
from kdcrga.data.graph import attach_splits, load_knowddi_graph
from kdcrga.eval.metrics import evaluate
from kdcrga.models.kdcrga import KDCRGA
from kdcrga.training.loop import train
from kdcrga.training.negatives import sample_negatives

DATA = Path("third_party/knowddi/data/BioSNAP")


def main(config_path: str = "configs/base.yaml") -> None:
    cfg = load_config(config_path)
    torch.manual_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cpu")
    device = device if torch.cuda.is_available() else "cpu"
    data = attach_splits(load_knowddi_graph(DATA),
                         ratios=tuple(cfg["data"]["split"].values()),
                         seed=cfg.get("seed", 0))
    m = cfg["model"]
    in_dim = data["entity"].x.size(-1)
    num_relations = sum(1 for et in data.edge_types if et[1].startswith("rel_"))
    model = KDCRGA(
        in_dim=in_dim, hidden_dim=m["hidden_dim"], num_relations=num_relations,
        num_side_effects=int(data.num_side_effects),
        num_layers=m["encoder_depth"], num_bases=m["num_bases"],
        dropout=m["dropout"],
        num_hops=m.get("num_hops", 1),
        use_shared_view=m.get("use_shared_view", False),
        use_hierarchical=m.get("use_hierarchical", False),
    )
    t = cfg["training"]
    train(model, data, lr=t["lr"], weight_decay=t["weight_decay"],
          gamma=t["class_weight_gamma"], batch_size=t["batch_size"],
          max_epochs=t["max_epochs"], patience=t["early_stop_patience"],
          seed=cfg.get("seed", 0), device=device)

    test_mask = data.ddi.split == 2
    test_pairs = data.ddi.pair_index[:, test_mask]
    test_se = data.ddi.side_effect[test_mask]
    positives = {
        (int(data.ddi.pair_index[0, i]), int(data.ddi.pair_index[1, i]),
         int(data.ddi.side_effect[i]))
        for i in range(data.ddi.side_effect.numel())
    }
    model.eval()
    with torch.no_grad():
        neg_pair, neg_se = sample_negatives(test_pairs, test_se,
                                            int(data.num_drugs), positives, seed=1)
        pair = torch.cat([test_pairs, neg_pair], dim=1)
        se = torch.cat([test_se, neg_se], dim=0)
        labels = torch.cat([torch.ones(test_se.numel()),
                            torch.zeros(neg_se.numel())]).to(device)
        res = evaluate(labels, torch.sigmoid(model(data, pair, se)), se,
                       data.quartiles)
    print("TEST:", {k: round(v, 4) for k, v in res.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml")
