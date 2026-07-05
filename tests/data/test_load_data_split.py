"""load_data must expose the pair-disjoint (canonical BioSNAP) split as an opt-in,
leaving the default random stratified split (RQ1) unchanged."""
from pathlib import Path

import pytest

DATA = Path("third_party/knowddi/data/BioSNAP")


def _pairs_spanning_splits(data) -> int:
    a = data.ddi.pair_index[0].tolist()
    b = data.ddi.pair_index[1].tolist()
    sp = data.ddi.split.tolist()
    by_pair: dict[tuple[int, int], set[int]] = {}
    for ai, bi, s in zip(a, b, sp):
        by_pair.setdefault((ai, bi), set()).add(s)
    return sum(1 for ss in by_pair.values() if len(ss) > 1)


@pytest.mark.skipif(not DATA.exists(), reason="KnowDDI data not downloaded")
def test_load_data_pair_disjoint_flag_controls_split():
    from experiments._sweep_common import load_data
    pd = load_data(seed=0, cfg=None, pair_disjoint=True)
    assert _pairs_spanning_splits(pd) == 0, \
        "pair_disjoint=True must yield a pair-disjoint (inductive) split"
    strat = load_data(seed=0, cfg=None, pair_disjoint=False)
    assert _pairs_spanning_splits(strat) > 0, \
        "default stratified split is expected to scatter a pair's SEs across splits"
