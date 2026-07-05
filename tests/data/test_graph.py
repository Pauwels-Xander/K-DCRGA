from pathlib import Path

import pytest

from kdcrga.data.graph import (
    attach_splits,
    audit_graph,
    ddi_side_effect_counts,
    load_knowddi_graph,
)

DATA = Path("third_party/knowddi/data/BioSNAP")


def _biosnap_row(a, b, active, flag, n_se=200):
    vec = ",".join("1" if i in active else "0" for i in range(n_se))
    return f"{a}\t{b}\t{vec}\t{flag}\n"


def test_ddi_side_effect_counts_expands_positives_only(tmp_path):
    (tmp_path / "train.txt").write_text(
        _biosnap_row(0, 1, {0, 2}, 1, n_se=4)   # +1 to se 0 and se 2
        + _biosnap_row(2, 3, {1, 2}, 0, n_se=4))  # flag==0 -> ignored
    (tmp_path / "valid.txt").write_text(_biosnap_row(1, 2, {2}, 1, n_se=4))
    (tmp_path / "test.txt").write_text(_biosnap_row(0, 3, {0, 2}, 1, n_se=4))
    counts = ddi_side_effect_counts(tmp_path, num_side_effects=4)
    assert counts.tolist() == [2, 0, 3, 0]   # se2 in 3 positive rows; neg row skipped


@pytest.mark.skipif(not DATA.exists(), reason="KnowDDI data not downloaded")
def test_ddi_side_effect_counts_match_loaded_graph():
    counts = ddi_side_effect_counts(DATA)
    assert counts.shape == (200,)
    assert int(counts.sum()) == 252111
    # counts must equal the loaded graph's own side-effect distribution
    import numpy as np
    se = load_knowddi_graph(DATA).ddi.side_effect.numpy()
    assert np.array_equal(counts, np.bincount(se, minlength=200))


def test_audit_counts_on_synthetic(synthetic_graph):
    stats = audit_graph(synthetic_graph)
    assert stats["num_drugs"] == 4
    assert stats["num_side_effects"] == 3
    assert stats["num_ddi"] == 6


@pytest.mark.skipif(not DATA.exists(), reason="KnowDDI data not downloaded")
def test_loaded_graph_matches_biosnap_statistics():
    data = load_knowddi_graph(DATA)
    stats = audit_graph(data)
    assert stats["num_drugs"] == 604
    assert stats["num_side_effects"] == 200
    assert stats["num_ddi"] == 252111          # flag==1 positives, exact


@pytest.mark.skipif(not DATA.exists(), reason="KnowDDI data not downloaded")
def test_loaded_graph_has_relation_typed_bkg_edges():
    data = load_knowddi_graph(DATA)
    # single entity node type, relation-typed BKG edges, drugs are entities 0..603
    assert "entity" in data.node_types
    rel_edge_types = [et for et in data.edge_types if et[1].startswith("rel_")]
    assert len(rel_edge_types) == 23
    # at least some BKG edges touch a drug id (entity 0..603) -> drugs connect to KG
    touches_drug = any(
        (data[et].edge_index < 604).any() for et in rel_edge_types
    )
    assert touches_drug


def test_attach_splits_populates_ddi_split_and_quartiles(synthetic_graph):
    data = attach_splits(synthetic_graph, ratios=(0.7, 0.15, 0.15), seed=0)
    assert data.ddi.split is not None
    assert data.ddi.split.numel() == data.ddi.side_effect.numel()
    assert set(data.ddi.split.tolist()) <= {0, 1, 2}
    assert set(data.quartiles.keys()) == {0, 1, 2}      # one entry per side effect


def test_attach_file_splits_uses_source_file_verbatim(synthetic_graph):
    """The pair-disjoint (canonical BioSNAP) split assigns each DDI triple to the
    file it came from (train=0, valid=1, test=2), NOT a random re-split."""
    import torch

    from kdcrga.data.graph import attach_file_splits
    synthetic_graph.ddi.source_file = torch.tensor([0, 0, 1, 2, 0, 2])
    data = attach_file_splits(synthetic_graph)
    assert torch.equal(data.ddi.split, torch.tensor([0, 0, 1, 2, 0, 2]))
    assert set(data.quartiles.keys()) == {0, 1, 2}      # one entry per side effect


def test_attach_file_splits_requires_source_file(synthetic_graph):
    from kdcrga.data.graph import attach_file_splits
    synthetic_graph.ddi.source_file = None
    with pytest.raises(ValueError, match="source_file"):
        attach_file_splits(synthetic_graph)


@pytest.mark.skipif(not DATA.exists(), reason="KnowDDI data not downloaded")
def test_load_knowddi_graph_records_source_file():
    data = load_knowddi_graph(DATA)
    sf = data.ddi.source_file
    assert sf is not None
    assert sf.numel() == data.ddi.side_effect.numel()
    assert set(sf.tolist()) == {0, 1, 2}


@pytest.mark.skipif(not DATA.exists(), reason="KnowDDI data not downloaded")
def test_file_boundary_split_is_pair_disjoint():
    """No drug-pair may appear in more than one split -- this is exactly the
    property the random stratified split violates and the leak depended on."""
    from kdcrga.data.graph import attach_file_splits
    data = attach_file_splits(load_knowddi_graph(DATA))
    a = data.ddi.pair_index[0].tolist()
    b = data.ddi.pair_index[1].tolist()
    sp = data.ddi.split.tolist()
    pair_splits: dict[tuple[int, int], set[int]] = {}
    for ai, bi, s in zip(a, b, sp):
        pair_splits.setdefault((ai, bi), set()).add(s)
    spanning = [p for p, ss in pair_splits.items() if len(ss) > 1]
    assert not spanning, f"{len(spanning)} pairs span >1 split (not pair-disjoint)"


@pytest.mark.skipif(not DATA.exists(), reason="KnowDDI data not downloaded")
def test_file_boundary_test_split_equals_biosnap_test_positives():
    from kdcrga.data.biosnap_eval import load_eval_triples
    from kdcrga.data.graph import attach_file_splits
    data = attach_file_splits(load_knowddi_graph(DATA))
    a = data.ddi.pair_index[0].tolist()
    b = data.ddi.pair_index[1].tolist()
    se = data.ddi.side_effect.tolist()
    sp = data.ddi.split.tolist()
    got = {(ai, bi, si) for ai, bi, si, s in zip(a, b, se, sp) if s == 2}
    ev = load_eval_triples(DATA / "test.txt")
    pos = ev.label == 1.0
    expected = set(zip(ev.drug_a[pos].tolist(), ev.drug_b[pos].tolist(),
                       ev.side_effect[pos].tolist()))
    assert got == expected
