"""Train and evaluate the decoder-conditioned R-GCN baseline on BioSNAP-200.

Usage:
    python experiments/train_dc_rgcn.py [config.yaml] [--resume]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from kdcrga.config import load_config
from kdcrga.data.graph import attach_splits, load_knowddi_graph
from kdcrga.eval.metrics import evaluate
from kdcrga.models.baselines.dc_rgcn import DecoderConditionedRGCN
from kdcrga.training.loop import train
from kdcrga.training.negatives import sample_negatives

DATA = Path("third_party/knowddi/data/BioSNAP")


def _log(msg: str) -> None:
    print(msg, flush=True)


def main(config_path: str = "configs/base.yaml", *, resume: bool = False) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    cfg = load_config(config_path)
    device = cfg.get("device", "cpu")
    device = device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 0))

    _log(f"Loading BioSNAP graph from {DATA}…")
    data = attach_splits(load_knowddi_graph(DATA),
                         ratios=tuple(cfg["data"]["split"].values()),
                         seed=cfg.get("seed", 0))
    _log(
        f"Graph loaded: {int(data.num_drugs)} drugs, "
        f"{int(data.num_side_effects)} side effects, "
        f"{data.ddi.side_effect.numel():,} DDI triples"
    )

    m = cfg["model"]
    in_dim = data["entity"].x.size(-1)
    num_relations = sum(1 for et in data.edge_types if et[1].startswith("rel_"))
    model = DecoderConditionedRGCN(
        in_dim=in_dim, hidden_dim=m["hidden_dim"], num_relations=num_relations,
        num_side_effects=int(data.num_side_effects),
        num_layers=m["encoder_depth"], num_bases=m["num_bases"],
        dropout=m["dropout"],
    )
    t = cfg["training"]
    out = cfg.get("output", {})
    checkpoint_dir = out.get("checkpoint_dir")
    if checkpoint_dir:
        _log(f"Checkpoints -> {checkpoint_dir}" + (" (resume)" if resume else ""))
    _log(f"Device: {device}")
    train(model, data, lr=t["lr"], weight_decay=t["weight_decay"],
          gamma=t["class_weight_gamma"], batch_size=t["batch_size"],
          max_epochs=t["max_epochs"], patience=t["early_stop_patience"],
          seed=cfg.get("seed", 0), device=device,
          checkpoint_dir=checkpoint_dir, resume=resume)

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
    print("TEST:", {k: round(v, 4) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = set(sys.argv[1:])
    main(args[0] if args else "configs/base.yaml", resume="--resume" in flags)
