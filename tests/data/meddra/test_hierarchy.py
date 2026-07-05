# tests/data/meddra/test_hierarchy.py
from pathlib import Path

from kdcrga.data.meddra.ascii import load_mdhier, load_soc
from kdcrga.data.meddra.hierarchy import HierarchyIndex

FIX = Path(__file__).parent / "fixtures"


def _index() -> HierarchyIndex:
    return HierarchyIndex.from_rows(
        load_mdhier(FIX / "mdhier.asc"), load_soc(FIX / "soc.asc"))


def test_pt_lookup_returns_ancestors():
    idx = _index()
    node = idx.node("10002043")
    assert (node.hlt, node.hlgt, node.soc) == ("10002042", "10002086", "10005329")
    assert idx.soc_name("10002043") == "Blood and lymphatic system disorders"


def test_distance_same_hlt_is_two():
    idx = _index()
    assert idx.distance("10002043", "10002080") == 2.0


def test_distance_different_soc_is_eight():
    idx = _index()
    assert idx.distance("10002043", "10014866") == 8.0


def test_distance_same_pt_is_zero():
    idx = _index()
    assert idx.distance("10014866", "10014866") == 0.0
