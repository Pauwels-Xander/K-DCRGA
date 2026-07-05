"""Attention analysis for RQ4 (proposal Sec. 4.10.3).

Jensen-Shannon divergence between attention distributions across side-effect pairs,
and Spearman correlation with MedDRA hierarchical distance, against a frozen-random
z_k control. SOC-stratified. Filled in the evaluation stage.
"""
from __future__ import annotations

from typing import Iterable

import torch
from scipy.stats import spearmanr


def _normalise(p: torch.Tensor) -> torch.Tensor:
    """Renormalise a non-negative score vector into a probability distribution."""
    return p / p.sum().clamp(min=1e-12)


def attention_jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence between two attention distributions (same support).

    Uses natural log, so the value lives in [0, ln 2 ~ 0.693]. Inputs are
    renormalised to distributions first, so unnormalised attention scores are
    accepted directly.
    """
    p = _normalise(p)
    q = _normalise(q)
    m = 0.5 * (p + q)

    def _kl(a: torch.Tensor, b: torch.Tensor) -> float:
        mask = a > 0
        return float((a[mask] * (torch.log(a[mask])
                                 - torch.log(b[mask].clamp(min=1e-12)))).sum())

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def spearman_vs_meddra_distance(
    jsd_pairs: Iterable[tuple[int, int, float]],
    distances: dict[tuple[int, int], float],
) -> tuple[float, float]:
    """Correlate per-side-effect-pair JSDs with their MedDRA tree distance.

    `jsd_pairs`: iterable of (k1, k2, jsd_value).
    `distances`: maps (k1, k2) -> tree distance (PT-PT path length); a reversed
    (k2, k1) key is also matched.
    Side-effect pairs missing from `distances` are dropped.
    Returns (Spearman rho, two-sided p-value); NaN if fewer than 3 usable pairs.
    """
    js: list[float] = []
    ds: list[float] = []
    for k1, k2, j in jsd_pairs:
        d = distances.get((k1, k2), distances.get((k2, k1)))
        if d is None:
            continue
        js.append(j)
        ds.append(d)
    if len(js) < 3:
        return float("nan"), float("nan")
    rho, p = spearmanr(js, ds)
    return float(rho), float(p)


def spearman_per_soc(
    jsd_pairs: Iterable[tuple[int, int, float]],
    distances: dict[tuple[int, int], float],
    soc_of: dict[int, str],
) -> dict[str, tuple[float, float]]:
    """SOC-stratified Spearman: restricts pairs to those whose k1 and k2 share
    the same System Organ Class. Returns {soc: (rho, p)}."""
    by_soc: dict[str, list[tuple[int, int, float]]] = {}
    for k1, k2, j in jsd_pairs:
        s1, s2 = soc_of.get(k1), soc_of.get(k2)
        if s1 is None or s1 != s2:
            continue
        by_soc.setdefault(s1, []).append((k1, k2, j))
    return {s: spearman_vs_meddra_distance(p, distances) for s, p in by_soc.items()}


def jsd_matrix(dists: torch.Tensor) -> torch.Tensor:
    """Pairwise Jensen-Shannon divergence between the rows of ``dists``.

    ``dists`` is a ``[K, n]`` tensor of K attention distributions over a shared
    n-neighbour support (rows are renormalised first, so raw attention scores are
    accepted). Returns a symmetric ``[K, K]`` matrix of JSDs in nats (zero
    diagonal, off-diagonal in ``[0, ln 2]``), consistent with :func:`attention_jsd`.
    Uses the entropy identity ``JSD = H(M) - 1/2 H(P) - 1/2 H(Q)`` row by row to
    avoid materialising the ``[K, K, n]`` mixture tensor.
    """
    p = dists / dists.sum(dim=1, keepdim=True).clamp(min=1e-12)

    def _entropy(x: torch.Tensor) -> torch.Tensor:
        logx = torch.where(x > 0, x.log(), torch.zeros_like(x))
        return -(x * logx).sum(dim=-1)

    h_rows = _entropy(p)                                  # [K]
    k = p.size(0)
    out = torch.zeros(k, k, dtype=p.dtype, device=p.device)
    for i in range(k):
        m = 0.5 * (p[i].unsqueeze(0) + p)                 # [K, n]
        out[i] = (_entropy(m) - 0.5 * (h_rows[i] + h_rows)).clamp(min=0.0)
    return out


def pair_attention_distributions(model, data, h, drug_a, drug_b, side_effects):
    """A-side attention over ``drug_a``'s neighbours, conditioned on the partner
    ``drug_b`` and each queried side effect (the model's hop-1 A-side attention,
    matching ``KDCRGA.attention_for_query``).

    Encodes nothing: caller passes the cached node embeddings ``h``. Works on any
    model exposing ``.attn`` (a ``DoublyConditionalAttention``) and ``.z_k``, so
    both the readout-MLP and DEDICOM K-DCRGA variants are supported. Returns a
    ``[K, n_a]`` tensor (``K = len(side_effects)``, ``n_a`` = #neighbours of
    ``drug_a``) whose rows are attention distributions over the SAME support, or
    ``None`` if ``drug_a`` has no neighbours.
    """
    from kdcrga.data.graph import drug_neighbours

    device = h.device
    side_effects = side_effects.to(device)
    k = int(side_effects.numel())
    a_rep = torch.full((k,), int(drug_a), dtype=torch.long, device=device)
    nb_a, qa = drug_neighbours(data, a_rep)
    if nb_a.numel() == 0:
        return None
    assert nb_a.numel() % k == 0, "ragged neighbour blocks; expected k equal blocks"
    n_a = nb_a.numel() // k
    h_b = h[torch.full((k,), int(drug_b), dtype=torch.long, device=device)]
    zk = model.z_k(side_effects)
    _, alpha = model.attn(h[nb_a], qa, h_b, zk, num_queries=k, return_alpha=True)
    # drug_neighbours emits drug_a's neighbours in one contiguous block per query,
    # so a row-major reshape recovers [SE, neighbour] with a shared support.
    return alpha.reshape(k, n_a)
