"""Expand a KnowDDI prediction dump (.npz) into the shared PredTable.

The dump holds, per scored (drug_a, drug_b) example in dataloader order:
``preds [N,200]`` (sigmoid scores), ``labels [N,200]`` (the row multi-hot),
``polarity [N]`` (row flag), ``pair_ids [N,2]`` (nodes[0], nodes[1] = the pair in
the original id space). This reproduces exactly the (pred, polarity) pairs
KnowDDI's evaluator aggregates per side effect."""
from __future__ import annotations

import numpy as np
import torch

from kdcrga.fusion.tables import PredTable


def knowddi_npz_to_table(npz_path: str) -> PredTable:
    z = np.load(npz_path)
    preds, labels = z["preds"], z["labels"]
    polarity, pair_ids = z["polarity"], z["pair_ids"]
    a: list[int] = []
    b: list[int] = []
    se: list[int] = []
    y: list[float] = []
    sc: list[float] = []
    for j in range(preds.shape[0]):
        set_bits = np.nonzero(labels[j])[0]
        for i in set_bits:
            a.append(int(pair_ids[j, 0]))
            b.append(int(pair_ids[j, 1]))
            se.append(int(i))
            y.append(float(polarity[j]))
            sc.append(float(preds[j, i]))
    return PredTable(
        torch.tensor(a, dtype=torch.int64), torch.tensor(b, dtype=torch.int64),
        torch.tensor(se, dtype=torch.int64), torch.tensor(y, dtype=torch.float32),
        torch.tensor(sc, dtype=torch.float32), "prob")
