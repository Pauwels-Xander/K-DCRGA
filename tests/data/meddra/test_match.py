# tests/data/meddra/test_match.py
from pathlib import Path

from kdcrga.data.meddra.ascii import load_llt, load_pt
from kdcrga.data.meddra.match import normalize, match_name

FIX = Path(__file__).parent / "fixtures"


def _terms():
    return load_llt(FIX / "llt.asc"), load_pt(FIX / "pt.asc")


def test_normalize_strips_punctuation_and_case():
    assert normalize('  "Ventilation", pneumonitis ') == "ventilation pneumonitis"


def test_exact_llt_match():
    llts, pts = _terms()
    m = match_name("enteritis", llts, pts, threshold=90)
    assert m.pt_code == "10014866"
    assert m.match_type == "exact_llt"
    assert m.score == 100.0


def test_prefers_current_llt_on_duplicate_name():
    # 'enteritis' exists as current (10014860) and non-current (10014861),
    # both -> PT 10014866; either way the PT is the same and current is chosen.
    llts, pts = _terms()
    m = match_name("enteritis", llts, pts, threshold=90)
    assert m.current is True


def test_fuzzy_match_on_reordered_tokens():
    llts, pts = _terms()
    m = match_name("folate deficiency anaemia", llts, pts, threshold=90)
    assert m.pt_code == "10002043"
    assert m.match_type == "fuzzy_llt"
    assert m.score >= 90.0


def test_unmatched_returns_none_pt():
    llts, pts = _terms()
    m = match_name("totally unmatched gibberish term", llts, pts, threshold=90)
    assert m.pt_code is None
    assert m.match_type == "unmatched"
