"""1:1 negative sampling by drug corruption (spec Sec. 9).

For each positive (drug_a, drug_b, side_effect), corrupt one endpoint (chosen
uniformly) to a random drug so the resulting triple is not a known positive.
Mirrors Decagon's "random non-edge of the same side-effect type".
"""
from __future__ import annotations

import torch


def sample_negatives(
    pos_pair_index: torch.Tensor,
    pos_side_effect: torch.Tensor,
    num_drugs: int,
    positives: set[tuple[int, int, int]],
    seed: int = 0,
    max_tries: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (neg_pair_index [2,E], neg_side_effect [E]) — one negative per
    positive, side effect preserved, filtered against `positives`.

    All random draws are generated vectorised up front and materialised to Python
    lists, so the rejection loop runs in pure Python with no per-element GPU<->CPU
    syncs (the old `int(...)`/`.item()` calls dominated batch time).

    Random draws are pre-generated vectorised, so for a given ``seed`` the output
    is deterministic but differs from the earlier per-element implementation (the
    generator is consumed in a different order); no sweep has been run against the
    old stream, so nothing downstream depends on it.  If ``max_tries`` is exhausted
    for an entry (no valid negative found), that entry falls back to the positive's
    own endpoints — a rare false negative; with the real ~604-drug graph and
    max_tries=20 this is effectively never hit."""
    g = torch.Generator().manual_seed(seed)
    e = pos_pair_index.size(1)
    if e == 0:
        return pos_pair_index.clone(), pos_side_effect.clone()
    # pre-generate every draw once (~80 MB Python heap at e=75k, max_tries=20) to
    # eliminate the ~e*max_tries GPU->CPU syncs the old per-element loop caused
    corrupt_a = (torch.rand(e, max_tries, generator=g) < 0.5).tolist()
    rand_drug = torch.randint(num_drugs, (e, max_tries), generator=g).tolist()
    a_list = pos_pair_index[0].tolist()
    b_list = pos_pair_index[1].tolist()
    k_list = pos_side_effect.tolist()
    # exhaustion fallback: entries with no valid negative keep the positive's own endpoints
    neg_a = list(a_list)
    neg_b = list(b_list)
    for i in range(e):
        a, b, k = a_list[i], b_list[i], k_list[i]
        for t in range(max_tries):
            if corrupt_a[i][t]:
                a2, b2 = rand_drug[i][t], b
            else:
                a2, b2 = a, rand_drug[i][t]
            if a2 != b2 and (a2, b2, k) not in positives:
                neg_a[i], neg_b[i] = a2, b2
                break
    neg = torch.tensor([neg_a, neg_b], dtype=pos_pair_index.dtype,
                       device=pos_pair_index.device)
    return neg, pos_side_effect.clone()
