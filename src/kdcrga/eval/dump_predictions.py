"""Score an arbitrary (drug_a, drug_b, side_effect) triple list with a trained
K-DCRGA model and persist the per-triple logits as a PredTable (Stream F)."""
from __future__ import annotations

import torch

from kdcrga.data.biosnap_eval import load_eval_triples
from kdcrga.fusion.tables import PredTable, save_table


def move_graph_to_device(data, device: str):
    """Move the encoder/forward graph tensors onto ``device`` in place.

    Mirrors the in-place moves ``training.loop.train`` performs (loop.py:82-89)
    before any forward pass: the entity features, every ``rel_*`` edge_index, and
    the DDI supervision tensors. The train-then-eval paths inherit that move from
    ``train()``; but ``dump_stream_f`` builds a fresh CPU graph via ``load_data``
    and never runs ``train()``, so without this the encoder matmuls CPU graph
    features against CUDA weights -> ``RGCNConv: mat2 is on cuda:0, other tensors
    on cpu``. Since ``train()`` runs a full forward on CUDA after exactly these
    moves, they are provably sufficient for encode + decoder + neighbour readouts.
    """
    dev = torch.device(device)
    data["entity"].x = data["entity"].x.to(dev)
    for et in list(data.edge_types):
        if et[1].startswith("rel_"):
            data[et].edge_index = data[et].edge_index.to(dev)
    ddi = getattr(data, "ddi", None)
    if ddi is not None:
        if getattr(ddi, "pair_index", None) is not None:
            ddi.pair_index = ddi.pair_index.to(dev)
        if getattr(ddi, "side_effect", None) is not None:
            ddi.side_effect = ddi.side_effect.to(dev)
        if getattr(ddi, "split", None) is not None:
            ddi.split = ddi.split.to(dev)
    return data


def score_triples(model, data, drug_a, drug_b, side_effect, *,
                  batch_size: int = 1024, device: str = "cpu") -> torch.Tensor:
    """Logits [T] for the given triples. Encodes the graph once, scores in
    chunks (mirrors kdcrga/eval/run.py:56-62 to avoid OOM on neighbour readouts)."""
    dev = torch.device(device)
    model.eval()
    pair = torch.stack([drug_a, drug_b]).to(dev)
    se = side_effect.to(dev)
    with torch.no_grad():
        h = model.encode(data)
        chunks: list[torch.Tensor] = []
        for s in range(0, pair.size(1), batch_size):
            chunks.append(model(data, pair[:, s:s + batch_size],
                                se[s:s + batch_size], h=h))
        return torch.cat(chunks).detach().cpu() if chunks else torch.empty(0)


def dump_stream_f(*, config_path: str, checkpoint_path: str, eval_txt: str,
                  out_path: str, device: str = "cpu",
                  model_kind: str = "kdcrga_dedicom") -> PredTable:
    from experiments._sweep_common import build_model, load_data
    from kdcrga.config import load_config

    cfg = load_config(config_path)
    data = load_data(seed=0, cfg=cfg)           # ATC-injected iff cfg sets it
    # model_kind defaults to kdcrga_dedicom (Stream F); pass "decagon"/"lagat" to
    # dump a baseline's predictions for the same-evaluator pair-disjoint control.
    model = build_model(model_kind, cfg, data).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])     # best.pt's "model" == best-AUPRC weights
    # load_data returns a CPU graph and we never ran train() here, so the encoder
    # would matmul CPU graph features against the on-`device` weights. Co-locate.
    move_graph_to_device(data, device)
    triples = load_eval_triples(eval_txt)
    logits = score_triples(model, data, triples.drug_a, triples.drug_b,
                           triples.side_effect,
                           batch_size=cfg["training"]["batch_size"], device=device)
    table = PredTable(triples.drug_a, triples.drug_b, triples.side_effect,
                      triples.label, logits, "logit")
    save_table(table, out_path)
    return table
