"""Name-match cascade: side-effect name -> MedDRA PT.

Order: normalized exact vs LLT names, exact vs PT names, fuzzy vs LLT names
(rapidfuzz token_set_ratio, difflib fallback). Below threshold -> unmatched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from kdcrga.data.meddra.ascii import LltRow, PtRow

try:
    from rapidfuzz import fuzz

    def _ratio(a: str, b: str) -> float:
        return float(fuzz.token_set_ratio(a, b))
except ImportError:  # stdlib fallback (weaker, no token reordering)
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return 100.0 * SequenceMatcher(None, a, b).ratio()


_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Match:
    pt_code: str | None
    match_type: str          # exact_llt | exact_pt | fuzzy_llt | unmatched
    score: float
    current: bool            # currency of the matched LLT (True for PT/unmatched)


@lru_cache(maxsize=None)
def normalize(name: str) -> str:
    # Cached: the offline build normalizes the same ~91k LLT names once per SE
    # query (~200x); memoising collapses ~36M calls to ~91k unique computations.
    s = name.strip().strip('"').lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def match_name(name: str, llts: list[LltRow], pts: list[PtRow],
               threshold: float = 90.0) -> Match:
    q = normalize(name)
    if not q:
        return Match(None, "unmatched", 0.0, True)

    # 1. exact LLT (prefer current when the same normalized name repeats).
    exact_llt = [e for e in llts if normalize(e.name) == q]
    if exact_llt:
        best = max(exact_llt, key=lambda e: e.current)
        return Match(best.pt_code, "exact_llt", 100.0, best.current)

    # 2. exact PT.
    for p in pts:
        if normalize(p.name) == q:
            return Match(p.pt_code, "exact_pt", 100.0, True)

    # 3. fuzzy vs LLT names.
    best_e, best_s = None, -1.0
    for e in llts:
        s = _ratio(q, normalize(e.name))
        if s > best_s:
            best_e, best_s = e, s
    if best_e is not None and best_s >= threshold:
        return Match(best_e.pt_code, "fuzzy_llt", best_s, best_e.current)

    return Match(None, "unmatched", max(best_s, 0.0), True)
