"""Train/val/test splits and frequency-quartile binning (proposal Sec. 3.5).

70/15/15 split stratified by side-effect type; the background KG is held fixed
across splits. All methods (including KnowDDI) are re-run on this split for
comparability. The 200 side effects are binned into frequency quartiles Q1-Q4
for the long-tail evaluation (RQ2). Filled in the data-pipeline stage.
"""
from __future__ import annotations

import torch


def stratified_split(
    side_effect: torch.Tensor,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> torch.Tensor:
    """Return a [E] tensor of split ids (0=train, 1=val, 2=test), stratified so
    each side effect's pairs are split by `ratios` independently."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    generator = torch.Generator().manual_seed(seed)
    split = torch.empty_like(side_effect)
    for k in side_effect.unique():
        idx = (side_effect == k).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.numel(), generator=generator)]
        n_train = int(round(ratios[0] * perm.numel()))
        n_val = int(round(ratios[1] * perm.numel()))
        split[perm[:n_train]] = 0
        split[perm[n_train:n_train + n_val]] = 1
        split[perm[n_train + n_val:]] = 2
    return split


def frequency_quartiles(
    side_effect: torch.Tensor,
    n_bins: int = 4,
) -> dict[int, int]:
    """Map each side-effect id to a frequency bin (0 = most frequent ... n_bins-1 =
    least frequent), with side effects partitioned into n_bins equal-size groups."""
    ids, counts = side_effect.unique(return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)
    ids_sorted = ids[order].tolist()
    bins: dict[int, int] = {}
    per_bin = len(ids_sorted) / n_bins
    for rank, se_id in enumerate(ids_sorted):
        bins[int(se_id)] = min(int(rank // per_bin), n_bins - 1)
    return bins
