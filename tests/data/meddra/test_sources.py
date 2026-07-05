# tests/data/meddra/test_sources.py
from pathlib import Path

from kdcrga.data.meddra.sources import load_se_names

FIX = Path(__file__).parent / "fixtures"


def test_load_se_names_chains_index_to_cui_to_name():
    names = load_se_names(FIX / "id2relation.json", FIX / "bio-decagon-combo.csv")
    # SE index -> (cui, name)
    assert names[0] == ("C0014863", "enteritis")
    assert names[1] == ("C0015695", "folate deficiency anaemia")
    assert names[2] == ("C9999999", "totally unmatched gibberish term")
