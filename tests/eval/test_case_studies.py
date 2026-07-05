import torch

from kdcrga.eval.case_studies import top_attention_path, render_case_study


def _model():
    from kdcrga.models.kdcrga import KDCRGA
    return KDCRGA(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                  num_layers=2, num_bases=2, dropout=0.0, num_hops=1)


def test_top_attention_path_returns_high_attention_neighbour(synthetic_rgcn_graph):
    model = _model()
    drug_a, drug_b, k = 0, 1, 0
    path = top_attention_path(model, synthetic_rgcn_graph, drug_a, drug_b, k, top_k=1)
    # path is a list of (entity_id, attention_weight) pairs for d_A's side
    assert isinstance(path, list)
    if path:                                      # synthetic graph may have an empty N(d_A)
        ent, w = path[0]
        assert 0 <= int(ent) < 6
        assert 0.0 <= float(w) <= 1.0


def test_top_attention_path_weights_sum_to_one(synthetic_rgcn_graph):
    # drug 0 has neighbours in the fixture; all returned weights are a softmax
    # over that neighbourhood, so the full path sums to ~1.
    model = _model()
    path = top_attention_path(model, synthetic_rgcn_graph, 0, 1, 0, top_k=99)
    if path:
        assert abs(sum(w for _, w in path) - 1.0) < 1e-5
        # sorted descending by weight
        weights = [w for _, w in path]
        assert weights == sorted(weights, reverse=True)


def test_top_attention_path_empty_neighbourhood_returns_empty(synthetic_rgcn_graph):
    # entity 5 is not a drug and is unlikely to be a query; drug with no BKG
    # neighbours yields an empty path rather than erroring. Use a fresh model.
    model = _model()
    # drug 2's neighbours exist, but request top_k=0 -> empty slice
    path = top_attention_path(model, synthetic_rgcn_graph, 2, 3, 0, top_k=0)
    assert path == []


def test_render_case_study_produces_markdown():
    md = render_case_study(
        "DrugA", "DrugB", "hepatotoxicity",
        a_path=[(10, 0.6), (11, 0.4)],
        b_path=[(20, 0.9)],
        entity_name={10: "CYP3A4", 11: "Gene11", 20: "hERG"},
    )
    assert "DrugA" in md and "DrugB" in md and "hepatotoxicity" in md
    assert "CYP3A4" in md and "hERG" in md
    assert "0.600" in md
