# tests/data/test_entity_names.py
from pathlib import Path

from kdcrga.data.entity_names import resolve_entity_names

FIX = Path(__file__).parent / "fixtures"


def test_resolves_rows_to_hetionet_names():
    names, stats = resolve_entity_names(
        FIX / "bkg_entity2id.json", FIX / "hetionet_nodes.tsv")
    assert names[0] == "ANXA8"
    assert names[2] == "Fentanyl"
    # id 1 is aliased by Disease::DOID:9 and Symptom::D007383; sorted pick wins.
    assert names[1] == "some disease"
    assert 3 not in names                      # Gene::ZZZNONAME has no Hetionet name


def test_resolution_stats():
    _, stats = resolve_entity_names(
        FIX / "bkg_entity2id.json", FIX / "hetionet_nodes.tsv")
    assert stats.total_rows == 4               # max id 3 -> 4 rows
    assert stats.resolved == 3
    assert stats.collisions == 1               # one id had >1 string
    assert stats.per_kind == {"Gene": 1, "Disease": 1, "Compound": 1}
