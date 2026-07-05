import pytest
import torch
import torch.nn as nn

from kdcrga.eval.zero_shot import (
    global_mean_init,
    overwrite_held_out_rows,
    overwrite_held_out_zk,
    random_init,
    stratified_se_split,
)
from kdcrga.data.meddra.zk_init import meddra_zk_init


def test_stratified_se_split_partitions_side_effects():
    quartiles = {k: k % 4 for k in range(40)}     # 10 per quartile
    train_ks, test_ks = stratified_se_split(quartiles, train_frac=0.8, seed=0)
    assert set(train_ks).isdisjoint(set(test_ks))
    assert set(train_ks) | set(test_ks) == set(quartiles)
    # Each quartile contributes ~80% to train
    for q in range(4):
        in_train = sum(1 for k in train_ks if quartiles[k] == q)
        assert 7 <= in_train <= 9


def test_stratified_se_split_is_deterministic():
    quartiles = {k: k % 4 for k in range(40)}
    a = stratified_se_split(quartiles, seed=3)
    b = stratified_se_split(quartiles, seed=3)
    c = stratified_se_split(quartiles, seed=4)
    assert a == b
    assert a != c            # different seed -> different partition


def test_overwrite_held_out_zk_copies_rows_only():
    from kdcrga.models.kdcrga import KDCRGA
    model = KDCRGA(in_dim=8, hidden_dim=4, num_relations=2, num_side_effects=6,
                   num_layers=2, num_bases=2, dropout=0.0, num_hops=1)
    held_out = [1, 3, 5]
    before = model.z_k.weight.detach().clone()
    zk_init = torch.ones(3, 4) * 7.0
    overwrite_held_out_zk(model, held_out, zk_init)
    w = model.z_k.weight.detach()
    for i, k in enumerate(held_out):
        assert torch.allclose(w[k], zk_init[i])
    # untouched rows unchanged
    for k in (0, 2, 4):
        assert torch.allclose(w[k], before[k])


def test_overwrite_held_out_zk_rejects_shape_mismatch():
    from kdcrga.models.kdcrga import KDCRGA
    model = KDCRGA(in_dim=8, hidden_dim=4, num_relations=2, num_side_effects=6,
                   num_layers=2, num_bases=2, dropout=0.0, num_hops=1)
    with pytest.raises(AssertionError):
        overwrite_held_out_zk(model, [1, 2], torch.ones(3, 4))   # 3 != 2


# --- generic held-out helpers (DEDICOM has two per-SE tables: z_k AND d_k) ---

def test_overwrite_held_out_rows_writes_only_named_rows():
    emb = nn.Embedding(6, 4)
    before = emb.weight.detach().clone()
    held_out = [1, 3, 5]
    fill = torch.arange(12.0).reshape(3, 4)
    overwrite_held_out_rows(emb, held_out, fill)
    w = emb.weight.detach()
    for i, k in enumerate(held_out):
        assert torch.allclose(w[k], fill[i])
    for k in (0, 2, 4):
        assert torch.allclose(w[k], before[k])


def test_overwrite_held_out_rows_rejects_shape_mismatch():
    emb = nn.Embedding(6, 4)
    with pytest.raises(AssertionError):
        overwrite_held_out_rows(emb, [1, 2], torch.ones(3, 4))   # 3 != 2


def test_overwrite_held_out_zk_still_targets_z_k():
    # back-compat: the thin wrapper must keep writing model.z_k rows
    from kdcrga.models.kdcrga import KDCRGA
    model = KDCRGA(in_dim=8, hidden_dim=4, num_relations=2, num_side_effects=6,
                   num_layers=2, num_bases=2, dropout=0.0, num_hops=1)
    overwrite_held_out_zk(model, [2, 4], torch.full((2, 4), 9.0))
    w = model.z_k.weight.detach()
    assert torch.allclose(w[2], torch.full((4,), 9.0))
    assert torch.allclose(w[4], torch.full((4,), 9.0))


def test_global_mean_init_is_mean_of_trained_train_rows():
    table = torch.arange(24.0).reshape(6, 4)        # rows 0..5
    train_ks = [0, 2, 4]
    held_out_ks = [1, 3]
    out = global_mean_init(table, train_ks, held_out_ks)
    assert out.shape == (2, 4)
    expected = table[torch.tensor(train_ks)].mean(0)
    for i in range(len(held_out_ks)):
        assert torch.allclose(out[i], expected)


def test_random_init_shape_scale_and_determinism():
    a = random_init([1, 3, 5], dim=8, mean=1.0, std=0.1, seed=7)
    b = random_init([1, 3, 5], dim=8, mean=1.0, std=0.1, seed=7)
    c = random_init([1, 3, 5], dim=8, mean=1.0, std=0.1, seed=8)
    assert a.shape == (3, 8)
    assert torch.allclose(a, b)        # seed-deterministic
    assert not torch.allclose(a, c)    # different seed -> different draw
    # centred near `mean` (d_k starts ~identity at 1.0)
    assert 0.5 < a.mean().item() < 1.5


def test_meddra_zk_init_works_on_a_d_k_shaped_table():
    # meddra_zk_init must be table-agnostic so it serves decoder.D (d_k) too.
    from types import SimpleNamespace
    dim = 4
    d_k = torch.randn(6, dim)
    mapping = {k: SimpleNamespace(hlt=f"h{k % 2}", hlgt="g", soc="s") for k in range(6)}
    train_ks, held_out_ks = [0, 2, 4], [1, 3, 5]
    out = meddra_zk_init(d_k, train_ks, held_out_ks, mapping)
    assert out.shape == (3, dim)
    # held-out k=1 shares hlt 'h1' with train k odd? mapping: hlt=h{k%2}.
    # k=1 -> h1; train sharing h1: k in {} since train are even (h0). Falls back.
    # k=5 -> h1 likewise even-only train -> fallback to global mean. Just assert finite.
    assert torch.isfinite(out).all()
