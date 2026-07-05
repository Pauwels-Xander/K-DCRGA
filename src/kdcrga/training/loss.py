"""Class-weighted multi-label BCE loss (proposal Sec. 4.9, spec Sec. 9).

Per-side-effect weight w_k = ((N - N^+_k) / N^+_k)^gamma, gamma default 0.5 (the
proposal's sqrt dampening). L2 regularisation is applied via the optimiser's
weight_decay. Negative sampling lives in ``negatives.py``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def class_weights(
    side_effect: torch.Tensor,
    num_side_effects: int,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Per-side-effect weight w_k = ((N - N^+_k) / N^+_k)^gamma (spec Sec. 9).

    Computed from positive counts in `side_effect`. gamma=0 -> all ones;
    gamma=0.5 -> the proposal's sqrt dampening.
    """
    counts = torch.bincount(side_effect, minlength=num_side_effects).float()
    n_total = counts.sum()
    pos = counts.clamp(min=1.0)
    w = ((n_total - pos) / pos).pow(gamma)
    return w


def weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    side_effect: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Mean class-weighted BCE-with-logits. `weights` is the [K] per-side-effect
    weight tensor; each example is weighted by weights[its side effect]."""
    per_example = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return (per_example * weights[side_effect]).mean()
