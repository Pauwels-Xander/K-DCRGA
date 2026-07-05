"""Primary metrics: AUROC, AUPRC, AP@50, averaged over the 200 side effects.

Reported overall and stratified by frequency quartile (RQ2), following
Zitnik et al. (2018) and Wang et al. (2024); proposal Sec. 4.10.1.

AP@50 implementation note: KnowDDI's evaluator.py does not implement AP@50 at
all — its BioSNAP multilabel evaluator uses accuracy over thresholded
predictions. Following Decagon's convention, AP@50 is implemented here as sklearn
average_precision_score over the 50 highest-scored predictions. Because KnowDDI
reports no AP@50, AUROC and AUPRC are the metrics used for head-to-head comparison
with KnowDDI; AP@50 is reported for continuity with the Decagon line.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def auroc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    return float(roc_auc_score(_np(y_true), _np(y_score)))


def auprc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    return float(average_precision_score(_np(y_true), _np(y_score)))


def ap_at_50(y_true: torch.Tensor, y_score: torch.Tensor, k: int = 50) -> float:
    """Average precision restricted to the k highest-scored predictions.

    KnowDDI does not implement AP@50 at all; its BioSNAP evaluator uses
    accuracy_score over thresholded outputs. AP@50 is therefore implemented
    here as sklearn average_precision_score computed over the top-k candidates
    (sorted by descending score), which equals precision@k when all positives
    outrank all negatives and gracefully degrades otherwise.
    """
    k = min(k, y_score.numel())
    top = torch.topk(y_score, k).indices
    return float(average_precision_score(_np(y_true[top]), _np(y_score[top])))


def best_f1_threshold(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Score threshold that maximises F1 on ``(y_true, y_score)``.

    Scans the operating points of the precision-recall curve (the distinct score
    values), which is exact for a single global threshold. Used to pick the
    operating point on the validation set; the chosen threshold is then applied to
    the test set (proposal Sec. 4.10.1)."""
    p, r, thr = precision_recall_curve(_np(y_true), _np(y_score))
    if thr.size == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))])


def prf_at_threshold(
    y_true: torch.Tensor, y_score: torch.Tensor, threshold: float,
) -> dict[str, float]:
    """Precision, recall, F1 for ``y_score >= threshold``. F1 is recomputed from
    the (precision, recall) pair so it is exact even when a class is empty."""
    yt = _np(y_true).astype(int)
    pred = (_np(y_score) >= threshold).astype(int)
    p = float(precision_score(yt, pred, zero_division=0))
    r = float(recall_score(yt, pred, zero_division=0))
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def threshold_metrics(
    val_true: torch.Tensor,
    val_score: torch.Tensor,
    test_true: torch.Tensor,
    test_score: torch.Tensor,
    test_side_effect: torch.Tensor,
    quartiles: dict[int, int],
) -> dict[str, float]:
    """Operating-point metrics: pick the F1-maximising threshold on the validation
    predictions, then report precision/recall/F1 on the test predictions, overall
    and per frequency quartile (pooled within each quartile). Returns a flat dict
    with ``threshold`` and ``{precision,recall,f1}[_q{0..}]`` (proposal Sec. 4.10.1)."""
    thr = best_f1_threshold(val_true, val_score)
    out: dict[str, float] = {"threshold": thr}
    out.update(prf_at_threshold(test_true, test_score, thr))
    n_bins = (max(quartiles.values()) + 1) if quartiles else 0
    se = test_side_effect.tolist()
    for q in range(n_bins):
        mask = torch.tensor([quartiles.get(int(s), -1) == q for s in se])
        if not bool(mask.any()):
            continue
        prf = prf_at_threshold(test_true[mask], test_score[mask], thr)
        for m, v in prf.items():
            out[f"{m}_q{q}"] = v
    return out


def evaluate(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    side_effect: torch.Tensor,
    quartiles: dict[int, int],
    return_per_se: bool = False,
):
    """Per-side-effect AUROC/AUPRC/AP@50 averaged overall and per frequency
    quartile. A side effect is skipped if it lacks both classes in this set.

    Returns the aggregate dict. When ``return_per_se`` is True, returns a tuple
    ``(aggregate_dict, per_se_dict)`` where ``per_se_dict[k]`` holds the three
    metrics for side effect ``k`` (used for paired Wilcoxon tests in RQ1)."""
    per_se: dict[int, dict[str, float]] = {}
    for k in side_effect.unique().tolist():
        mask = side_effect == k
        yt, ys = y_true[mask], y_score[mask]
        if yt.min() == yt.max():
            continue
        per_se[k] = {"auroc": auroc(yt, ys), "auprc": auprc(yt, ys),
                     "ap50": ap_at_50(yt, ys)}

    def _avg(metric: str, keys: list[int]) -> float:
        vals = [per_se[k][metric] for k in keys if k in per_se]
        return float(np.mean(vals)) if vals else float("nan")

    all_keys = list(per_se.keys())
    out = {m: _avg(m, all_keys) for m in ("auroc", "auprc", "ap50")}
    n_bins = (max(quartiles.values()) + 1) if quartiles else 0
    for q in range(n_bins):
        q_keys = [k for k in all_keys if quartiles.get(k) == q]
        out[f"auroc_q{q}"] = _avg("auroc", q_keys)
        out[f"auprc_q{q}"] = _avg("auprc", q_keys)
        out[f"ap50_q{q}"] = _avg("ap50", q_keys)
    if return_per_se:
        return out, per_se
    return out
