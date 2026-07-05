"""Tier 0 average + Tier 1 learned stacker for two-stream fusion.

All functions assume f and s are ALREADY aligned (same (drug_a, drug_b,
side_effect) rows in the same order); call kdcrga.fusion.tables.align first.
The stacker is fit on validation tables and applied to test tables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from kdcrga.fusion.tables import PredTable, to_logit


def _features(f: PredTable, s: PredTable) -> np.ndarray:
    lf = to_logit(f.score, f.score_kind).numpy()
    ls = to_logit(s.score, s.score_kind).numpy()
    return np.stack([lf, ls], axis=1)


def fuse_average(f: PredTable, s: PredTable, mode: str = "logit") -> torch.Tensor:
    if mode == "prob":
        pf = torch.sigmoid(to_logit(f.score, f.score_kind))
        ps = torch.sigmoid(to_logit(s.score, s.score_kind))
        return (pf + ps) / 2
    avg = (to_logit(f.score, f.score_kind) + to_logit(s.score, s.score_kind)) / 2
    return torch.sigmoid(avg)


@dataclass
class Stacker:
    gating: str
    global_model: LogisticRegression | None
    per_q: dict[int, LogisticRegression]
    quartiles: dict[int, int]

    def apply(self, f: PredTable, s: PredTable) -> torch.Tensor:
        x = _features(f, s)
        out = np.zeros(x.shape[0], dtype=np.float32)
        if self.gating == "global":
            out = self.global_model.predict_proba(x)[:, 1]
        else:
            se = f.side_effect.tolist()
            for i, row in enumerate(x):
                q = self.quartiles.get(int(se[i]), -1)
                model = self.per_q.get(q, self.global_model)
                out[i] = model.predict_proba(row.reshape(1, -1))[0, 1]
        return torch.from_numpy(out.astype(np.float32))


def fit_stacker(f_val: PredTable, s_val: PredTable, *, gating: str = "global",
                quartiles: dict[int, int] | None = None) -> Stacker:
    quartiles = quartiles or {}
    x = _features(f_val, s_val)
    y = f_val.label.numpy().astype(int)
    g = LogisticRegression(max_iter=1000).fit(x, y)
    per_q: dict[int, LogisticRegression] = {}
    if gating == "quartile":
        se = f_val.side_effect.tolist()
        q_of = np.array([quartiles.get(int(k), -1) for k in se])
        for q in sorted(set(q_of.tolist())):
            mask = q_of == q
            if mask.sum() >= 10 and len(set(y[mask].tolist())) == 2:
                per_q[int(q)] = LogisticRegression(max_iter=1000).fit(x[mask], y[mask])
    return Stacker(gating, g, per_q, quartiles)
