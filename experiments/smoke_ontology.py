"""End-to-end smoke for the ontology pipeline (no training).

Loads the real BioSNAP graph, injects ATC ancestor nodes + MedDRA z_k init,
builds the full K-DCRGA, and runs a forward over a handful of validation pairs.
Verifies the augmented graph is consistent (extra nodes, rel_atc relation, drug
ATC neighbours) and that a forward produces finite logits. Runs on CPU with a
tiny batch so it is safe on any machine.

Run: python experiments/smoke_ontology.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _sweep_common import build_model, load_zk_init, maybe_inject_atc  # noqa: E402

from kdcrga.data.graph import attach_splits, load_knowddi_graph  # noqa: E402

DATA = Path("third_party/knowddi/data/BioSNAP")

CFG = {"device": "cpu",
       "model": {"hidden_dim": 128, "encoder_depth": 3, "num_bases": 30,
                 "dropout": 0.3, "num_hops": 3, "use_shared_view": True,
                 "use_hierarchical": True,
                 "use_ontology_zk": True, "use_ontology_atc": True}}


def main() -> None:
    data = attach_splits(load_knowddi_graph(DATA), seed=0)
    n0 = data["entity"].x.size(0)
    rels0 = sum(1 for et in data.edge_types if et[1].startswith("rel_"))

    a2e = maybe_inject_atc(data, CFG)
    n1 = data["entity"].x.size(0)
    rels1 = sum(1 for et in data.edge_types if et[1].startswith("rel_"))
    print(f"entities {n0} -> {n1} (+{n1 - n0} ATC nodes)")
    print(f"rel_* relations {rels0} -> {rels1} (rel_atc added: {rels1 > rels0})")
    assert a2e and n1 > n0 and rels1 == rels0 + 1

    # z_k init actually loaded into the model?
    zk = load_zk_init(CFG, int(data.num_side_effects), CFG["model"]["hidden_dim"])
    model = build_model("kdcrga", CFG, data).eval()
    if zk is not None:
        assert torch.allclose(model.z_k.weight.detach(), zk), "z_k init not applied"
        print(f"z_k init applied: {tuple(zk.shape)}")

    # forward a few validation pairs
    mask = data.ddi.split == 1
    pairs = data.ddi.pair_index[:, mask][:, :32]
    se = data.ddi.side_effect[mask][:32]
    with torch.no_grad():
        logits = model(data, pairs, se)
    finite = bool(torch.isfinite(logits).all())
    print(f"forward OK: logits {tuple(logits.shape)}, all finite: {finite}")
    assert logits.shape == (pairs.size(1),) and finite

    # mapped drugs should now have ATC ancestor nodes among their neighbours
    atc_ids = set(a2e.values())
    mapped = [d for d in range(int(data.num_drugs))
              if atc_ids & set(data.neighbour_index.get(d, []))]
    assert mapped, "no drug gained ATC neighbours after injection"
    ex = mapped[0]
    print(f"{len(mapped)} drugs gained ATC neighbours; e.g. drug {ex} has "
          f"{len(atc_ids & set(data.neighbour_index[ex]))}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
