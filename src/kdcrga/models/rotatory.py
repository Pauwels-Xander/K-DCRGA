"""Rotatory mechanism: n-hop drug-to-context / context-to-drug alternation.

Implements the conditioning suffix r_{X|Y} (proposal Sec. 4.5), with shared
parameters across hops. Hop count n in {1, 2, 3}. Filled in stage v2.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kdcrga.models.shared_view import SharedView

from kdcrga.models.attention import DoublyConditionalAttention


def rotatory_readouts(
    attn: DoublyConditionalAttention,
    h_nb_a: torch.Tensor, qa: torch.Tensor,
    h_nb_b: torch.Tensor, qb: torch.Tensor,
    h_a_drug: torch.Tensor, h_b_drug: torch.Tensor,
    z_k: torch.Tensor,
    num_queries: int,
    num_hops: int,
    shared: "SharedView | None" = None,
    h_nb_shared: torch.Tensor | None = None,
    q_shared: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Two-step-per-hop rotatory readouts (proposal Sec. 4.5, eqs. 481-509).

    Each hop has Step 1 (drug-to-context -> r_A, r_B) and Step 2 (context-to-drug
    -> r_{A|B}, r_{B|A}). Hop 1 Step 1 conditions on the raw partner-drug
    embedding; hops > 1 condition on the previous hop's refined cross readout.
    Returns the four final-hop readouts (r_A, r_B, r_{A|B}, r_{B|A}) = v1..v4.
    When ``shared`` (a SharedView) and its neighbourhood are given, the shared
    view runs in parallel at each hop and is appended as v5 = r_{AB}; the partner
    conditioning is the raw drug mean at hop 1 and 0.5*(r_{A|B}+r_{B|A}) after.
    Attention parameters are shared across hops and steps.
    """
    assert (shared is None) == (h_nb_shared is None) == (q_shared is None), \
        "pass shared, h_nb_shared and q_shared together, or none of them"
    nq = num_queries
    use_shared = shared is not None
    # --- Hop 1, Step 1 (eq:hop1-step1): raw drug embeddings as partner ---
    r_a = attn(h_nb_a, qa, h_b_drug, z_k, num_queries=nq)
    r_b = attn(h_nb_b, qb, h_a_drug, z_k, num_queries=nq)
    # --- Hop 1, Step 2 (eq:hop1-step2): partner = other side's hop-1 readout ---
    r_ab = attn(h_nb_a, qa, r_b, z_k, num_queries=nq)
    r_ba = attn(h_nb_b, qb, r_a, z_k, num_queries=nq)
    # --- Shared view, hop 1 (eq:shared): partner = mean of raw drug embeddings ---
    r_sh = None
    if use_shared:
        r_sh = shared(attn, h_nb_shared, q_shared, h_a_drug, h_b_drug, z_k,
                      num_queries=nq)
    # --- Hops t > 1 ---
    for _ in range(num_hops - 1):
        # Shared view at hop t uses the PREVIOUS hop's cross readouts (line 621);
        # r_ab / r_ba still hold the hop-(t-1) values here.
        if use_shared:
            r_sh = shared(attn, h_nb_shared, q_shared, h_a_drug, h_b_drug, z_k,
                          num_queries=nq, partner=0.5 * (r_ab + r_ba))
        # Step 1 (eq:hop-t-step1): partner = previous hop's refined cross readout
        r_a = attn(h_nb_a, qa, r_ba, z_k, num_queries=nq)
        r_b = attn(h_nb_b, qb, r_ab, z_k, num_queries=nq)
        # Step 2: partner = current hop's Step-1 readout
        r_ab = attn(h_nb_a, qa, r_b, z_k, num_queries=nq)
        r_ba = attn(h_nb_b, qb, r_a, z_k, num_queries=nq)
    if use_shared:
        return r_a, r_b, r_ab, r_ba, r_sh
    return r_a, r_b, r_ab, r_ba
