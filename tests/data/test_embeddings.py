import json

import pytest
import torch

from kdcrga.data.embeddings import (
    build_dual_node_features,
    build_node_features,
    load_drug_smiles,
    _canonicalize_smiles,
    _masked_mean_pool,
)


def _fake_embed(names):
    # Deterministic 4-dim vector per name (by length), for testing only.
    return torch.tensor([[float(len(n))] * 4 for n in names])


def _fake_drug_embed(smiles):
    # Negative of length, to distinguish drug embeddings from text ones.
    return torch.tensor([[-float(len(s))] * 4 for s in smiles])


def test_build_node_features_scatters_and_zeros():
    names_by_row = {0: "ANXA8", 2: "Fentanyl", 3: "ANXA8"}
    feats, stats = build_node_features(names_by_row, num_entities=5,
                                       embed_fn=_fake_embed)
    assert feats.shape == (5, 4)
    assert torch.allclose(feats[0], torch.full((4,), 5.0))   # len("ANXA8")
    assert torch.allclose(feats[2], torch.full((4,), 8.0))   # len("Fentanyl")
    assert torch.allclose(feats[3], feats[0])                # same name -> same vec (dedup)
    assert torch.allclose(feats[1], torch.zeros(4))          # unresolved -> zeros
    assert torch.allclose(feats[4], torch.zeros(4))
    assert stats["embedded"] == 3 and stats["zero"] == 2 and stats["unique_names"] == 2


def test_build_dual_node_features_uses_drug_emb_for_smiles_rows():
    # Rows 0,1 are drugs (have a SMILES); row 5 is a text entity (gene).
    # Row 0 also carries a name, which must be IGNORED in favour of its SMILES.
    names_by_row = {0: "Penicillin", 1: "Fentanyl", 5: "ANXA8"}
    smiles_by_row = {0: "CCO", 1: "CCCC"}
    feats, stats = build_dual_node_features(
        names_by_row, smiles_by_row, num_entities=6,
        text_embed_fn=_fake_embed, drug_embed_fn=_fake_drug_embed)
    assert feats.shape == (6, 4)
    # Drug rows use the molecular embedder (negative), NOT their text name.
    assert torch.allclose(feats[0], torch.full((4,), -3.0))   # len("CCO")
    assert torch.allclose(feats[1], torch.full((4,), -4.0))   # len("CCCC")
    # Text row uses the text embedder.
    assert torch.allclose(feats[5], torch.full((4,), 5.0))    # len("ANXA8")
    # Rows with neither SMILES nor name stay zero.
    assert torch.allclose(feats[2], torch.zeros(4))
    assert torch.allclose(feats[3], torch.zeros(4))
    assert torch.allclose(feats[4], torch.zeros(4))
    assert stats["drugs_embedded"] == 2
    assert stats["text_embedded"] == 1
    assert stats["zero"] == 3


def test_build_dual_node_features_rejects_dim_mismatch():
    def text3(names):
        return torch.zeros(len(names), 3)

    def drug5(smiles):
        return torch.zeros(len(smiles), 5)

    with pytest.raises(ValueError, match="dim"):
        build_dual_node_features(
            {5: "ANXA8"}, {0: "CCO"}, num_entities=6,
            text_embed_fn=text3, drug_embed_fn=drug5)


def test_load_drug_smiles_parses_row_to_smiles(tmp_path):
    p = tmp_path / "id2drug.json"
    p.write_text(json.dumps({
        "0": {"cid": "CID000002173", "db": None, "smiles": "CCO"},
        "1": {"cid": "CID000003345", "db": "DB00813", "smiles": "CCCC"},
        "2": {"cid": "CID000000000", "db": None, "smiles": ""},  # empty -> skipped
    }))
    smiles = load_drug_smiles(p)
    assert smiles == {0: "CCO", 1: "CCCC"}


def test_canonicalize_smiles_normalizes_and_passes_through_invalid():
    out = _canonicalize_smiles(["OCC", "CCO", "not_a_smiles"])
    assert out[0] == out[1]             # equivalent SMILES -> same canonical form
    assert out[2] == "not_a_smiles"     # unparseable -> passthrough unchanged


def test_masked_mean_pool_ignores_padding():
    # two tokens, second is padding (mask 0) -> mean == first token only.
    last_hidden = torch.tensor([[[1.0, 1.0], [9.0, 9.0]]])   # [batch=1, tokens=2, dim=2]
    mask = torch.tensor([[1, 0]])
    pooled = _masked_mean_pool(last_hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[1.0, 1.0]]))


def test_build_node_features_fallback_dim_when_nothing_resolves():
    feats, stats = build_node_features({}, num_entities=3, embed_fn=_fake_embed,
                                       dim=7)
    assert feats.shape == (3, 7)
    assert torch.allclose(feats, torch.zeros(3, 7))
    assert stats["embedded"] == 0
