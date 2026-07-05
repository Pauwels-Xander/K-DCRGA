import torch

from kdcrga.data.splits import stratified_split
from kdcrga.data.splits import frequency_quartiles


def test_stratified_split_is_per_side_effect_and_deterministic():
    # 100 pairs of side effect 0, 20 of side effect 1
    side_effect = torch.cat([torch.zeros(100, dtype=torch.long),
                             torch.ones(20, dtype=torch.long)])
    split_a = stratified_split(side_effect, ratios=(0.7, 0.15, 0.15), seed=42)
    split_b = stratified_split(side_effect, ratios=(0.7, 0.15, 0.15), seed=42)

    assert torch.equal(split_a, split_b)                      # deterministic
    assert split_a.shape == side_effect.shape

    for k in (0, 1):
        mask = side_effect == k
        n = int(mask.sum())
        train = int((split_a[mask] == 0).sum())
        # ~70% of each side effect's pairs land in train
        assert abs(train / n - 0.70) < 0.06
        # every split value is one of {0,1,2}
        assert set(split_a[mask].tolist()) <= {0, 1, 2}


def test_frequency_quartiles_order_and_size():
    # 8 side effects with strictly decreasing frequency
    side_effect = torch.cat([torch.full((c,), k, dtype=torch.long)
                             for k, c in enumerate([80, 70, 60, 50, 40, 30, 20, 10])])
    q = frequency_quartiles(side_effect, n_bins=4)  # dict: side_effect_id -> quartile

    assert q[0] == 0 and q[1] == 0      # most frequent -> Q1 (bin 0)
    assert q[6] == 3 and q[7] == 3      # least frequent -> Q4 (bin 3)
    # 8 side effects into 4 bins -> 2 each
    counts = {b: sum(v == b for v in q.values()) for b in range(4)}
    assert counts == {0: 2, 1: 2, 2: 2, 3: 2}
