"""Zero-shot generalisation to held-out side effects (proposal Sec. 4.10.3, RQ5).

`stratified_se_split` partitions the 200 side effects 80/20 stratified by
frequency quartile. `overwrite_held_out_zk` writes MedDRA-derived embeddings into
the held-out rows of `model.z_k.weight`; the training loop is expected to mask
gradient updates to those rows (see `experiments/sweep_rq5_zero_shot.py`).
"""
from __future__ import annotations

import random

import torch


def stratified_se_split(
    quartiles: dict[int, int],
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Partition side-effect ids into (train, test), stratified by quartile."""
    rng = random.Random(seed)
    by_q: dict[int, list[int]] = {}
    for k, q in quartiles.items():
        by_q.setdefault(q, []).append(k)
    train: list[int] = []
    test: list[int] = []
    for q in sorted(by_q):
        ks = sorted(by_q[q])
        rng.shuffle(ks)
        n_train = int(round(train_frac * len(ks)))
        train.extend(ks[:n_train])
        test.extend(ks[n_train:])
    return sorted(train), sorted(test)


def overwrite_held_out_rows(
    embedding,
    held_out_ks: list[int],
    fill: torch.Tensor,
) -> None:
    """Copy ``fill[i]`` into ``embedding.weight[held_out_ks[i]]``, leaving all other
    rows untouched. Generic over any ``nn.Embedding`` so it serves both per-side-
    effect tables of the DEDICOM model: ``z_k`` (attention) and ``decoder.D`` (d_k).
    """
    assert fill.shape == (len(held_out_ks), embedding.weight.size(1)), (
        f"fill shape {tuple(fill.shape)} != "
        f"({len(held_out_ks)}, {embedding.weight.size(1)})"
    )
    fill = fill.to(embedding.weight.device, embedding.weight.dtype)
    with torch.no_grad():
        for i, k in enumerate(held_out_ks):
            embedding.weight[k].copy_(fill[i])


def overwrite_held_out_zk(
    model,
    held_out_ks: list[int],
    zk_init: torch.Tensor,
) -> None:
    """Back-compat wrapper: overwrite the held-out rows of ``model.z_k``."""
    overwrite_held_out_rows(model.z_k, held_out_ks, zk_init)


def global_mean_init(
    trained_table: torch.Tensor,
    train_ks: list[int],
    held_out_ks: list[int],
) -> torch.Tensor:
    """Fill every held-out row with the mean of the *trained* train-SE rows.

    The "no ontology structure" null: the best uniform guess for an unseen side
    effect when ancestry is ignored. Returns ``Tensor[len(held_out_ks), dim]``.
    """
    dim = trained_table.size(1)
    if train_ks:
        mean = trained_table[torch.tensor(train_ks)].mean(0)
    else:
        mean = torch.zeros(dim)
    return mean.unsqueeze(0).expand(len(held_out_ks), dim).contiguous()


def random_init(
    held_out_ks: list[int],
    dim: int,
    mean: float = 0.0,
    std: float = 0.1,
    seed: int = 0,
) -> torch.Tensor:
    """Seeded Gaussian fill at a table's init scale (z_k∼N(0,0.1); d_k∼N(1,0.1)).

    The "no ontology at all" floor (the proposal's random-frozen control).
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(len(held_out_ks), dim, generator=g) * std + mean
