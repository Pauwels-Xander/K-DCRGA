import torch

from kdcrga.eval.metrics import (
    ap_at_50,
    auprc,
    auroc,
    best_f1_threshold,
    evaluate,
    prf_at_threshold,
    threshold_metrics,
)


def test_perfect_scores_give_perfect_metrics():
    y_true = torch.tensor([1, 1, 0, 0])
    y_score = torch.tensor([0.9, 0.8, 0.2, 0.1])
    assert auroc(y_true, y_score) == 1.0
    assert auprc(y_true, y_score) == 1.0


def test_ap_at_50_precision_of_top_k():
    y_true = torch.tensor([1, 1, 1, 0, 0])
    y_score = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1])
    assert ap_at_50(y_true, y_score) == 1.0


def test_evaluate_returns_overall_and_per_quartile():
    y_true = torch.tensor([1, 0, 1, 0])
    y_score = torch.tensor([0.9, 0.1, 0.8, 0.2])
    side_effect = torch.tensor([0, 0, 1, 1])
    quartiles = {0: 0, 1: 1}
    out = evaluate(y_true, y_score, side_effect, quartiles)
    assert set(out.keys()) >= {"auroc", "auprc", "ap50"}
    assert "auroc_q0" in out and "auroc_q1" in out
    assert 0.0 <= out["auroc"] <= 1.0


def test_evaluate_return_per_se_gives_breakdown():
    y_true = torch.tensor([1, 0, 1, 0])
    y_score = torch.tensor([0.9, 0.1, 0.8, 0.2])
    side_effect = torch.tensor([0, 0, 1, 1])
    quartiles = {0: 0, 1: 1}
    out, per_se = evaluate(y_true, y_score, side_effect, quartiles,
                           return_per_se=True)
    # aggregate dict unchanged in shape
    assert "auroc" in out
    # per-se dict keyed by side-effect id, each with the three metrics
    assert set(per_se.keys()) == {0, 1}
    assert set(per_se[0].keys()) == {"auroc", "auprc", "ap50"}


def test_evaluate_default_return_is_single_dict():
    # backward-compatibility: default call still returns just the aggregate dict
    out = evaluate(torch.tensor([1, 0]), torch.tensor([0.9, 0.1]),
                   torch.tensor([0, 0]), {0: 0})
    assert isinstance(out, dict)
    assert "auroc" in out


# --- threshold (operating-point) metrics ---------------------------------------

def test_prf_at_threshold_perfect_separation():
    y_true = torch.tensor([1, 1, 0, 0])
    y_score = torch.tensor([0.9, 0.8, 0.2, 0.1])
    out = prf_at_threshold(y_true, y_score, 0.5)
    assert out["precision"] == 1.0 and out["recall"] == 1.0 and out["f1"] == 1.0


def test_prf_at_threshold_all_predicted_negative_is_zero():
    y_true = torch.tensor([1, 0])
    y_score = torch.tensor([0.3, 0.1])
    out = prf_at_threshold(y_true, y_score, 0.9)  # nothing passes -> no positives
    assert out["precision"] == 0.0 and out["recall"] == 0.0 and out["f1"] == 0.0


def test_best_f1_threshold_separates_classes():
    y_true = torch.tensor([0, 0, 1, 1])
    y_score = torch.tensor([0.1, 0.2, 0.8, 0.9])
    thr = best_f1_threshold(y_true, y_score)
    # any threshold in (0.2, 0.8] gives perfect F1; the chosen one must too
    assert prf_at_threshold(y_true, y_score, thr)["f1"] == 1.0


def test_threshold_metrics_uses_val_threshold_and_splits_quartiles():
    # threshold picked on val (perfectly separable at ~0.5)
    val_true = torch.tensor([1, 1, 0, 0])
    val_score = torch.tensor([0.9, 0.7, 0.3, 0.1])
    test_true = torch.tensor([1, 0, 1, 0])
    test_score = torch.tensor([0.8, 0.2, 0.6, 0.4])
    test_se = torch.tensor([0, 0, 1, 1])
    out = threshold_metrics(val_true, val_score, test_true, test_score,
                            test_se, {0: 0, 1: 1})
    assert "threshold" in out
    assert {"precision", "recall", "f1"} <= set(out)
    assert "f1_q0" in out and "f1_q1" in out
