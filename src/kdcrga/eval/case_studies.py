"""Interpretability case studies (proposal Sec. 4.10.3).

Extracts the highest-attention path between a drug pair for a selected side effect
and inspects alignment with known biology (e.g. hERG for cardiotoxicity, CYP450 for
hepatotoxicity). Filled in the evaluation stage.
"""
from __future__ import annotations

import torch


def top_attention_path(
    model,
    data,
    drug_a: int,
    drug_b: int,
    side_effect: int,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Return the top-k attention-weighted neighbours of drug_a for this query.

    Uses `model.attention_for_query`. The returned list is sorted by descending
    attention weight. Caller can run this twice (a-side, b-side) and concatenate
    if both sides are wanted, or render to markdown for the thesis case study.
    """
    pair_index = torch.tensor([[drug_a], [drug_b]], dtype=torch.long)
    se = torch.tensor([side_effect], dtype=torch.long)
    res = model.attention_for_query(data, pair_index, se)
    mask = res["qa"] == 0
    if mask.numel() == 0 or not mask.any():
        return []
    nbrs = res["nb_a"][mask].tolist()
    weights = res["alpha_a"][mask].tolist()
    pairs = sorted(zip(nbrs, weights), key=lambda x: -x[1])
    return [(int(n), float(w)) for n, w in pairs[:top_k]]


def render_case_study(
    drug_a_name: str, drug_b_name: str, side_effect_name: str,
    a_path: list[tuple[int, float]],
    b_path: list[tuple[int, float]],
    entity_name: dict[int, str],
) -> str:
    """Markdown rendering of a single case study for the thesis Results section."""
    def _fmt(path: list[tuple[int, float]]) -> str:
        return "\n".join(f"- {entity_name.get(e, str(e))} (weight {w:.3f})"
                         for e, w in path)
    return (
        f"### {drug_a_name} + {drug_b_name} -> {side_effect_name}\n\n"
        f"Top attention neighbours of {drug_a_name}:\n{_fmt(a_path)}\n\n"
        f"Top attention neighbours of {drug_b_name}:\n{_fmt(b_path)}\n"
    )
