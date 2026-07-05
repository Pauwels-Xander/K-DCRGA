# tests/data/test_graph_features.py
import torch

from kdcrga.data.graph import load_or_placeholder_features


def test_loads_cache_on_row_match(tmp_path):
    cache = tmp_path / "node_features.pt"
    torch.save(torch.ones(5, 4), cache)
    feats = load_or_placeholder_features(cache, num_entities=5, placeholder_dim=16)
    assert feats.shape == (5, 4)
    assert torch.allclose(feats, torch.ones(5, 4))


def test_placeholder_when_absent(tmp_path):
    feats = load_or_placeholder_features(tmp_path / "missing.pt",
                                         num_entities=5, placeholder_dim=16)
    assert feats.shape == (5, 16)


def test_placeholder_on_row_mismatch(tmp_path):
    cache = tmp_path / "node_features.pt"
    torch.save(torch.ones(5, 4), cache)
    feats = load_or_placeholder_features(cache, num_entities=6, placeholder_dim=16)
    assert feats.shape == (6, 16)         # mismatch -> placeholder, not the cache
