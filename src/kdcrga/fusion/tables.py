"""Shared per-triple prediction table + alignment for two-stream fusion."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

_log = logging.getLogger(__name__)


@dataclass
class PredTable:
    drug_a: torch.Tensor       # [T] int64
    drug_b: torch.Tensor       # [T] int64
    side_effect: torch.Tensor  # [T] int64
    label: torch.Tensor        # [T] float32
    score: torch.Tensor        # [T] float32
    score_kind: str            # "logit" | "prob"


def save_table(table: PredTable, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(table) | {"score_kind": table.score_kind}, path)


def load_table(path: str | Path) -> PredTable:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return PredTable(d["drug_a"], d["drug_b"], d["side_effect"],
                     d["label"], d["score"], d["score_kind"])


def _keys(t: PredTable) -> list[tuple[int, int, int]]:
    return list(zip(t.drug_a.tolist(), t.drug_b.tolist(), t.side_effect.tolist()))


def _reindex(t: PredTable, order: list[int]) -> PredTable:
    idx = torch.tensor(order, dtype=torch.long)
    return PredTable(t.drug_a[idx], t.drug_b[idx], t.side_effect[idx],
                     t.label[idx], t.score[idx], t.score_kind)


def align(a: PredTable, b: PredTable) -> tuple[PredTable, PredTable]:
    """Inner-join two tables on (drug_a, drug_b, side_effect); return both sorted
    identically. Raises if a shared key carries disagreeing labels."""
    pos_a = {k: i for i, k in enumerate(_keys(a))}
    pos_b = {k: i for i, k in enumerate(_keys(b))}
    common = sorted(set(pos_a) & set(pos_b))
    dropped = (len(pos_a) - len(common)) + (len(pos_b) - len(common))
    if dropped:
        _log.warning("align dropped %d non-shared triples (a=%d b=%d common=%d)",
                     dropped, len(pos_a), len(pos_b), len(common))
    oa = _reindex(a, [pos_a[k] for k in common])
    ob = _reindex(b, [pos_b[k] for k in common])
    if not torch.equal(oa.label, ob.label):
        raise ValueError("labels disagree on shared (drug_a, drug_b, side_effect) keys")
    return oa, ob


def to_logit(score: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "logit":
        return score
    p = score.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))
