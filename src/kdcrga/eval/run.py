"""Reusable test-set evaluation + per-run output persistence for the sweeps.

Factors the test-set scoring previously duplicated across the experiment scripts
into one place, and writes the on-disk layout the aggregator/compiler expect:

    runs/<config>/seed_<N>/history.json        (train/val history)
    runs/<config>/seed_<N>/test_metrics.json   (aggregate evaluate(...) dict)
    runs/<config>/seed_<N>/per_se_metrics.json (per-side-effect breakdown)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from kdcrga.eval.metrics import evaluate
from kdcrga.training.negatives import sample_negatives


def evaluate_test_set(
    model,
    data,
    *,
    batch_size: int = 1024,
    device: str = "cpu",
    neg_seed: int = 1,
    max_test_pairs: int | None = None,
) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    """Score the test split (split id 2) with 1:1 sampled negatives.

    The forward is chunked at ``batch_size`` so K-DCRGA's neighbour-materialising
    readouts do not OOM on the full test set. ``max_test_pairs`` truncates the
    positive test set (smoke runs only). Returns ``(aggregate, per_se)``.
    """
    dev = torch.device(device)
    test_mask = data.ddi.split == 2
    test_pairs = data.ddi.pair_index[:, test_mask]
    test_se = data.ddi.side_effect[test_mask]
    if max_test_pairs is not None:
        test_pairs = test_pairs[:, :max_test_pairs]
        test_se = test_se[:max_test_pairs]
    positives = {
        (int(data.ddi.pair_index[0, i]), int(data.ddi.pair_index[1, i]),
         int(data.ddi.side_effect[i]))
        for i in range(data.ddi.side_effect.numel())
    }
    model.eval()
    with torch.no_grad():
        neg_pair, neg_se = sample_negatives(
            test_pairs, test_se, int(data.num_drugs), positives, seed=neg_seed)
        pair = torch.cat([test_pairs, neg_pair], dim=1)
        se = torch.cat([test_se, neg_se], dim=0)
        labels = torch.cat([torch.ones(test_se.numel()),
                            torch.zeros(neg_se.numel())]).to(dev)
        # Encode the static graph ONCE; the encoder is identical across chunks
        # in eval mode (weights frozen, no dropout).
        h = model.encode(data)
        chunks: list[torch.Tensor] = []
        for s in range(0, pair.size(1), batch_size):
            chunks.append(model(data, pair[:, s:s + batch_size], se[s:s + batch_size], h=h))
        logits = torch.cat(chunks) if chunks else torch.empty(0, device=dev)
        agg, per_se = evaluate(labels, torch.sigmoid(logits), se,
                               getattr(data, "quartiles", {}), return_per_se=True)
    return agg, per_se


def save_run_outputs(
    out_dir: str | Path,
    history: dict[str, list[float]],
    aggregate: dict[str, float],
    per_se: dict[int, dict[str, float]],
) -> None:
    """Write history.json, test_metrics.json, per_se_metrics.json to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps(aggregate, indent=2))
    # JSON object keys must be strings; the compiler casts them back to int.
    (out_dir / "per_se_metrics.json").write_text(
        json.dumps({str(k): v for k, v in per_se.items()}, indent=2))
