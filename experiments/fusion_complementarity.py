"""Show the two streams are complementary (not redundant): error correlation
and per-quartile win maps. Operates on aligned PredTables."""
from __future__ import annotations

import numpy as np
import torch

from kdcrga.fusion.rescore import per_se_metric, score_table
from kdcrga.fusion.tables import PredTable, to_logit


def _prob(t: PredTable) -> torch.Tensor:
    return t.score if t.score_kind == "prob" else torch.sigmoid(to_logit(t.score, "logit"))


def error_correlation(f: PredTable, s: PredTable) -> float:
    ef = (f.label - _prob(f)).abs().numpy()
    es = (s.label - _prob(s)).abs().numpy()
    if ef.std() == 0 or es.std() == 0:
        return float("nan")
    return float(np.corrcoef(ef, es)[0, 1])


def per_quartile_wins(f: PredTable, s: PredTable,
                      quartiles: dict[int, int]) -> dict[int, dict]:
    _, f_per = score_table(f, quartiles)
    _, s_per = score_table(s, quartiles)
    fa, sa = per_se_metric(f_per, "auroc"), per_se_metric(s_per, "auroc")
    out: dict[int, dict] = {}
    for q in sorted(set(quartiles.values())):
        ks = [k for k in fa if k in sa and quartiles.get(k) == q]
        if not ks:
            continue
        f_wins = sum(fa[k] > sa[k] for k in ks)
        out[q] = {"n": len(ks), "f_win_frac": f_wins / len(ks),
                  "knowddi_win_frac": 1 - f_wins / len(ks)}
    return out
