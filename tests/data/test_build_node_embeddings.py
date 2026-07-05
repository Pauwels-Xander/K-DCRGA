# tests/data/test_build_node_embeddings.py
import json
from pathlib import Path

import torch

from kdcrga.data.embeddings import (
    build_dual_node_feature_cache,
    build_node_feature_cache,
)

FIX = Path(__file__).parent / "fixtures"


def _fake_embed(names):
    return torch.tensor([[float(len(n))] * 4 for n in names])


def _fake_drug_embed(smiles):
    return torch.tensor([[-float(len(s))] * 4 for s in smiles])


def test_build_cache_writes_tensor_and_report(tmp_path):
    out = tmp_path / "node_features.pt"
    report = tmp_path / "report.md"
    stats = build_node_feature_cache(
        bkg_entity2id_path=FIX / "bkg_entity2id.json",
        hetionet_nodes_path=FIX / "hetionet_nodes.tsv",
        out_path=out, report_path=report, embed_fn=_fake_embed)
    feats = torch.load(out, map_location="cpu")
    # total_rows from the fixture is 4 (max id 3 + 1); 3 resolved, row 3 zeros.
    assert feats.shape == (4, 4)
    assert torch.allclose(feats[3], torch.zeros(4))
    assert stats["embedded"] == 3 and stats["zero"] == 1
    text = report.read_text(encoding="utf-8")
    assert "Gene" in text and "collisions" in text.lower()


def test_build_dual_cache_uses_molecular_emb_for_drug_rows(tmp_path):
    # Row 2 (Compound::DB00813 / Fentanyl) is a drug given a SMILES; it must be
    # embedded molecularly, NOT from its "Fentanyl" Hetionet name.
    id2drug = tmp_path / "id2drug.json"
    id2drug.write_text(json.dumps(
        {"2": {"cid": "CID1", "db": "DB00813", "smiles": "CCO"}}))
    out = tmp_path / "node_features.pt"
    report = tmp_path / "report.md"
    stats = build_dual_node_feature_cache(
        bkg_entity2id_path=FIX / "bkg_entity2id.json",
        hetionet_nodes_path=FIX / "hetionet_nodes.tsv",
        id2drug_path=id2drug, out_path=out, report_path=report,
        text_embed_fn=_fake_embed, drug_embed_fn=_fake_drug_embed)
    feats = torch.load(out, map_location="cpu")
    assert feats.shape == (4, 4)
    assert torch.allclose(feats[2], torch.full((4,), -3.0))  # drug: len("CCO")
    assert torch.allclose(feats[0], torch.full((4,), 5.0))   # text: len("ANXA8")
    assert torch.allclose(feats[3], torch.zeros(4))          # unresolved
    assert stats["drugs_embedded"] == 1
    assert report.exists() and "drug" in report.read_text(encoding="utf-8").lower()
