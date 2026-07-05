"""Score single-stream and fused PredTables through the project evaluator."""
from __future__ import annotations

import torch

from kdcrga.eval.metrics import evaluate
from kdcrga.fusion.tables import PredTable


def _as_prob(table: PredTable) -> torch.Tensor:
    if table.score_kind == "prob":
        return table.score
    return torch.sigmoid(table.score)


def score_table(table: PredTable, quartiles: dict[int, int]) -> tuple[dict, dict]:
    """evaluate(...) over (label, prob, side_effect); returns (aggregate, per_se)."""
    return evaluate(table.label, _as_prob(table), table.side_effect,
                    quartiles, return_per_se=True)


def per_se_metric(per_se: dict[int, dict], metric: str) -> dict[int, float]:
    return {k: v[metric] for k, v in per_se.items()}
