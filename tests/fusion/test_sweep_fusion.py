import pytest
import torch
from kdcrga.fusion.tables import PredTable
from experiments.sweep_fusion import fuse_seed

def _mk(score_f, score_s, kind_s="prob"):
    n = len(score_f)
    a = torch.zeros(n, dtype=torch.int64)
    b = torch.arange(n, dtype=torch.int64)
    se = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    y = torch.tensor([1., 1., 0., 0., 1., 1., 0., 0.])
    f = PredTable(a, b, se, y, torch.tensor(score_f), "logit")
    s = PredTable(a.clone(), b.clone(), se.clone(), y.clone(), torch.tensor(score_s), kind_s)
    return f, s

def test_fuse_seed_reports_all_streams():
    f_test, s_test = _mk([2., 1., -1., -2., 2., 1., -1., -2.],
                         [.9, .8, .1, .2, .9, .8, .1, .2])
    f_val, s_val = _mk([2., 1., -1., -2., 2., 1., -1., -2.],
                       [.9, .8, .1, .2, .9, .8, .1, .2])
    out = fuse_seed(f_test, s_test, f_val, s_val, {0: 0, 1: 1})
    for key in ("f_auroc", "knowddi_auroc", "fused_auroc",
                "fused_variant", "wilcoxon_auroc_p"):
        assert key in out
    assert out["fused_auroc"] >= max(out["f_auroc"], out["knowddi_auroc"]) - 1e-9


def test_fuse_seed_reports_f_vs_knowddi_wilcoxon():
    """The headline claim is K-DCRGA (F) vs KnowDDI, so fuse_seed must report that
    paired per-SE test directly (not only fused-vs-KnowDDI / fused-vs-F)."""
    f_test, s_test = _mk([2., 1., -1., -2., 2., 1., -1., -2.],
                         [.9, .8, .1, .2, .9, .8, .1, .2])
    out = fuse_seed(f_test, s_test, f_test, s_test, {0: 0, 1: 1})
    # Presence + float type (matches test_fuse_seed_reports_all_streams): a 2-SE
    # fixture makes the paired Wilcoxon degenerate, so the p can be NaN here; its
    # numeric validity is covered by the summarize() tests and the 200-SE real run.
    for key in ("wilcoxon_f_vs_knowddi_auroc_p", "wilcoxon_f_vs_knowddi_auprc_p"):
        assert key in out
        assert isinstance(out[key], float)


def _row(f_au, k_au, f_ap, k_ap, p_fk, p_vf=0.4):
    """A minimal fuse_seed-style result row for summarize() tests."""
    return {
        "f_auroc": f_au, "knowddi_auroc": k_au, "fused_auroc": max(f_au, k_au) + 1e-3,
        "f_auprc": f_ap, "knowddi_auprc": k_ap, "fused_auprc": max(f_ap, k_ap) + 1e-3,
        "wilcoxon_f_vs_knowddi_auroc_p": p_fk, "wilcoxon_f_vs_knowddi_auprc_p": p_fk,
        "wilcoxon_vs_f_auroc_p": p_vf, "wilcoxon_vs_f_auprc_p": p_vf,
    }


def test_summarize_verdict_beat_when_consistent_and_significant():
    from experiments.sweep_fusion import summarize
    per_seed = [_row(0.970, 0.951, 0.80, 0.73, 1e-10),
                _row(0.969, 0.950, 0.79, 0.72, 1e-9),
                _row(0.971, 0.952, 0.81, 0.74, 1e-11)]
    s = summarize(per_seed)
    assert s["f_vs_knowddi"]["auroc"]["verdict"] == "beat"
    assert s["f_vs_knowddi"]["auroc"]["margin_mean"] > 0


def test_summarize_verdict_tie_when_sign_inconsistent_across_seeds():
    from experiments.sweep_fusion import summarize
    per_seed = [_row(0.960, 0.951, 0.74, 0.73, 1e-3),
                _row(0.948, 0.952, 0.71, 0.73, 1e-3),   # F loses this seed
                _row(0.951, 0.950, 0.72, 0.72, 0.20)]
    s = summarize(per_seed)
    assert s["f_vs_knowddi"]["auroc"]["verdict"] == "tie"


def test_summarize_verdict_tie_when_not_significant():
    from experiments.sweep_fusion import summarize
    per_seed = [_row(0.970, 0.951, 0.80, 0.73, 0.30),   # positive but p>0.05
                _row(0.969, 0.950, 0.79, 0.72, 0.40),
                _row(0.971, 0.952, 0.81, 0.74, 0.22)]
    s = summarize(per_seed)
    assert s["f_vs_knowddi"]["auroc"]["verdict"] == "tie"


def _mk_keyed(drug_b_vals, labels, score_f, score_s, kind_s="prob"):
    """Build two PredTables with explicit per-row keys; labels must be identical."""
    n = len(drug_b_vals)
    a = torch.zeros(n, dtype=torch.int64)
    b = torch.tensor(drug_b_vals, dtype=torch.int64)
    se = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1][:n])
    y = torch.tensor(labels, dtype=torch.float32)
    f = PredTable(a, b, se, y, torch.tensor(score_f, dtype=torch.float32), "logit")
    s = PredTable(a.clone(), b.clone(), se.clone(), y.clone(),
                  torch.tensor(score_s, dtype=torch.float32), kind_s)
    return f, s


def test_fuse_seed_selection_is_val_driven_not_test_driven():
    """Prove that variant selection is driven by VALIDATION AUPRC, not test AUPRC.

    Construction
    ------------
    Both streams share the same (drug_a, drug_b, side_effect) keys on val and
    test (so align keeps all 8 rows).  Val scores are engineered so both
    streams agree perfectly (labels rank-preserved), making avg_logit have
    perfect AUPRC on val.  Test scores are engineered so stream-f is
    strongly predictive but stream-s is *anti*-correlated — in isolation
    avg_logit would suffer on test.  Stackers on these noisy test scores
    would fit noise; but they cannot see test during selection anyway.

    Assertions
    ----------
    (a) fused_variant == argmax(val_auprc_by_variant)  — selection is val-driven.
    (b) n_test_triples and n_val_triples are present and equal 8 (row count).
    """
    # Shared keys: drug_b 0..7, side_effects 0,0,0,0,1,1,1,1
    drug_b_vals = list(range(8))
    labels = [1., 1., 0., 0., 1., 1., 0., 0.]

    # --- val: both streams agree perfectly with labels ---
    # f_val logits: positives high, negatives low (rank-perfect)
    val_score_f = [4., 3., -3., -4., 4., 3., -3., -4.]
    # s_val probs: positives near 1, negatives near 0 (rank-perfect)
    val_score_s = [0.95, 0.85, 0.15, 0.05, 0.95, 0.85, 0.15, 0.05]

    # --- test: f is perfect, s is anti-correlated (but test not used for selection) ---
    test_score_f = [4., 3., -3., -4., 4., 3., -3., -4.]
    test_score_s = [0.05, 0.15, 0.85, 0.95, 0.05, 0.15, 0.85, 0.95]  # inverted

    f_val, s_val = _mk_keyed(drug_b_vals, labels, val_score_f, val_score_s)
    f_test, s_test = _mk_keyed(drug_b_vals, labels, test_score_f, test_score_s)

    quartiles = {0: 0, 1: 1}
    out = fuse_seed(f_test, s_test, f_val, s_val, quartiles)

    # (a) selection must match val argmax — not driven by test
    best_by_val = max(out["val_auprc_by_variant"], key=out["val_auprc_by_variant"].get)
    assert out["fused_variant"] == best_by_val, (
        f"fused_variant={out['fused_variant']!r} differs from "
        f"val argmax={best_by_val!r}; selection leaked test data."
    )

    # (b) coverage counts are present and correct
    assert "n_test_triples" in out, "n_test_triples missing from output"
    assert "n_val_triples" in out, "n_val_triples missing from output"
    assert out["n_test_triples"] == 8, f"Expected 8 test triples, got {out['n_test_triples']}"
    assert out["n_val_triples"] == 8, f"Expected 8 val triples, got {out['n_val_triples']}"


def _mk_explicit(b_vals, se_vals, labels, score_f, score_s, kind_s="prob"):
    """Build two identically-keyed PredTables from explicit (b, se, label) rows."""
    a = torch.zeros(len(b_vals), dtype=torch.int64)
    b = torch.tensor(b_vals, dtype=torch.int64)
    se = torch.tensor(se_vals, dtype=torch.int64)
    y = torch.tensor(labels, dtype=torch.float32)
    f = PredTable(a, b, se, y, torch.tensor(score_f, dtype=torch.float32), "logit")
    s = PredTable(a.clone(), b.clone(), se.clone(), y.clone(),
                  torch.tensor(score_s, dtype=torch.float32), kind_s)
    return f, s


def test_fuse_seed_allows_duplicate_keys_in_split():
    """BioSNAP split files contain a few duplicate (a, b, side_effect) triples;
    both streams reproduce them identically. Coverage is compared on key SETS, so
    fuse_seed must run (not raise on a raw-count diff) and report the collapse."""
    # rows: (0,0,0) appears twice -> 1 duplicate row collapsed; key set is identical
    b_vals = [0, 1, 2, 3, 0, 4, 5, 6, 7]
    se_vals = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    labels = [1., 1., 0., 0., 1., 1., 1., 0., 0.]
    sf = [3., 2., -2., -3., 3., 3., 2., -2., -3.]
    ss = [.9, .8, .2, .1, .9, .9, .8, .2, .1]
    f_test, s_test = _mk_explicit(b_vals, se_vals, labels, sf, ss)
    f_val, s_val = _mk_explicit(b_vals, se_vals, labels, sf, ss)
    out = fuse_seed(f_test, s_test, f_val, s_val, {0: 0, 1: 1})
    assert out["dup_rows_collapsed_test"] == 1
    assert out["n_test_triples"] == 8  # 9 rows, 8 unique keys


def test_fuse_seed_raises_on_true_coverage_mismatch():
    """A key present in one stream but not the other is a real coverage error."""
    f_test, s_test = _mk_explicit([0, 1], [0, 0], [1., 0.], [2., -2.], [.9, .1])
    # break coverage: change one of s_test's keys (drug_b 1 -> 9)
    s_bad = PredTable(s_test.drug_a, torch.tensor([0, 9], dtype=torch.int64),
                      s_test.side_effect, s_test.label, s_test.score, s_test.score_kind)
    with pytest.raises(ValueError, match="coverage mismatch"):
        fuse_seed(f_test, s_bad, f_test, s_bad, {0: 0})
