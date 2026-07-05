"""Zero-shot combined story: the fused model retains F's unseen-SE capability;
KnowDDI structurally cannot (its W_final has a fixed 200-relation head, so an
unseen relation index has no column). On held-out SEs the fusion is F-only.

Reuses the RQ5 held-out protocol (experiments/sweep_rq5_zero_shot.py): 40 SEs
stratified by quartile, the model never sees a held-out triple in training, and
z_k / d_k for held-out SEs are filled by the global-mean control (the RQ5 winner).
"""
from __future__ import annotations

import json
from pathlib import Path


def knowddi_can_score_unseen_se() -> bool:
    """KnowDDI's classifier head is fixed at num_rels=200 columns; an unseen
    side effect has no head column, so it cannot be scored. Always False."""
    return False


def zero_shot_table(out_path: str = "runs/fusion/zero_shot.json") -> dict:
    """Pull the headline RQ5 held-out numbers for the fused (= F-only) model and
    record KnowDDI as not-applicable.

    Reads from runs/rq5/dedicom_v5/seed_0/global_mean/test_metrics.json (the RQ5
    winner arm) and extracts auroc, auprc, ap50. Each key is a top-level field."""
    rq5_path = Path("runs/rq5/dedicom_v5/seed_0/global_mean/test_metrics.json")
    test_metrics = json.loads(rq5_path.read_text())

    result = {
        "fused_zero_shot": {
            "auroc": test_metrics["auroc"],
            "auprc": test_metrics["auprc"],
            "ap50": test_metrics["ap50"],
            "note": "fused == F-only on unseen SEs"
        },
        "knowddi_zero_shot": "N/A (no unseen-relation head)",
        "knowddi_can_score_unseen_se": knowddi_can_score_unseen_se(),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result
