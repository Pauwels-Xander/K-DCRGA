"""Parse a BioSNAP split file into the canonical fusion evaluation triple set.

Each row is ``drug_a \\t drug_b \\t label_vector \\t flag`` where ``label_vector``
is 200 comma-separated bits and ``flag`` is the row polarity (1 positive, 0 the
1:1-sampled negative). For EVERY row, each set bit ``i`` becomes one
``(drug_a, drug_b, side_effect=i, label=flag)`` triple. This reproduces exactly
the (pred, polarity) pairs KnowDDI's multilabel evaluator aggregates per side
effect (third_party/knowddi/pytorch/manager/evaluator.py:98-106).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class EvalTriples:
    drug_a: torch.Tensor       # [T] int64
    drug_b: torch.Tensor       # [T] int64
    side_effect: torch.Tensor  # [T] int64
    label: torch.Tensor        # [T] float32 (1.0 / 0.0)


def load_eval_triples(path: str | Path) -> EvalTriples:
    a: list[int] = []
    b: list[int] = []
    se: list[int] = []
    y: list[float] = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            da, db = int(parts[0]), int(parts[1])
            flag = float(int(parts[3]))
            for idx, token in enumerate(parts[2].split(",")):
                if token == "1":
                    a.append(da)
                    b.append(db)
                    se.append(idx)
                    y.append(flag)
    return EvalTriples(
        drug_a=torch.from_numpy(np.array(a, dtype=np.int64)),
        drug_b=torch.from_numpy(np.array(b, dtype=np.int64)),
        side_effect=torch.from_numpy(np.array(se, dtype=np.int64)),
        label=torch.from_numpy(np.array(y, dtype=np.float32)),
    )
