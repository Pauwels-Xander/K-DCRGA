import torch

from kdcrga.eval.attention_analysis import (
    attention_jsd, jsd_matrix, pair_attention_distributions,
    spearman_vs_meddra_distance, spearman_per_soc,
)
from kdcrga.models.kdcrga import KDCRGA


def test_attention_jsd_equal_distributions_is_zero():
    p = torch.tensor([0.5, 0.3, 0.2])
    assert attention_jsd(p, p) < 1e-12


def test_attention_jsd_disjoint_distributions_is_max():
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])
    # JSD with natural log has max ln(2) ~ 0.693
    assert abs(attention_jsd(p, q) - 0.693) < 1e-2


def test_attention_jsd_is_symmetric_and_renormalises():
    # unnormalised inputs should be renormalised to distributions
    p = torch.tensor([2.0, 1.0, 1.0])
    q = torch.tensor([1.0, 1.0, 2.0])
    assert abs(attention_jsd(p, q) - attention_jsd(q, p)) < 1e-9
    assert attention_jsd(p, q) > 0


def test_spearman_perfect_positive_correlation():
    # JSD rises with MedDRA distance -> perfect monotonic -> rho == 1.0.
    jsd_pairs = [(0, 1, 0.0), (0, 2, 0.5), (0, 3, 1.0)]
    distances = {(0, 1): 0, (0, 2): 2, (0, 3): 4}
    rho, p = spearman_vs_meddra_distance(jsd_pairs, distances)
    assert abs(rho - 1.0) < 1e-9


def test_spearman_handles_reversed_distance_keys():
    # distances stored as (k2, k1) should still match a (k1, k2) jsd pair
    jsd_pairs = [(1, 0, 0.0), (2, 0, 0.5), (3, 0, 1.0)]
    distances = {(0, 1): 0, (0, 2): 2, (0, 3): 4}
    rho, p = spearman_vs_meddra_distance(jsd_pairs, distances)
    assert abs(rho - 1.0) < 1e-9


def test_spearman_too_few_pairs_returns_nan():
    import math
    rho, p = spearman_vs_meddra_distance([(0, 1, 0.5)], {(0, 1): 1})
    assert math.isnan(rho)


def test_spearman_per_soc_restricts_to_same_soc():
    jsd_pairs = [(0, 1, 0.0), (0, 2, 0.5), (0, 3, 1.0),
                 (0, 4, 0.0), (4, 5, 1.0)]
    distances = {(0, 1): 0, (0, 2): 2, (0, 3): 4, (0, 4): 0, (4, 5): 4}
    soc_of = {0: "cardiac", 1: "cardiac", 2: "cardiac", 3: "cardiac",
              4: "hepatic", 5: "hepatic"}
    out = spearman_per_soc(jsd_pairs, distances, soc_of)
    # cardiac has 3 within-SOC pairs (0-1,0-2,0-3); hepatic only 1 -> NaN
    assert "cardiac" in out
    assert abs(out["cardiac"][0] - 1.0) < 1e-9


# --- jsd_matrix ----------------------------------------------------------------

def test_jsd_matrix_matches_pairwise_attention_jsd():
    torch.manual_seed(0)
    p = torch.rand(5, 7)
    m = jsd_matrix(p)
    for i in range(5):
        for j in range(5):
            assert abs(m[i, j].item() - attention_jsd(p[i], p[j])) < 1e-6


def test_jsd_matrix_symmetric_zero_diagonal():
    torch.manual_seed(1)
    m = jsd_matrix(torch.rand(6, 4))
    assert torch.allclose(m, m.T, atol=1e-6)
    assert torch.allclose(m.diagonal(), torch.zeros(6), atol=1e-6)


def test_jsd_matrix_identical_rows_are_zero():
    row = torch.tensor([0.5, 0.3, 0.2])
    m = jsd_matrix(torch.stack([row, row, row]))
    assert torch.allclose(m, torch.zeros(3, 3), atol=1e-9)


def test_jsd_matrix_renormalises_unnormalised_rows():
    # a scaled row is the same distribution -> zero divergence to its normalised twin
    m = jsd_matrix(torch.tensor([[2.0, 1.0, 1.0], [4.0, 2.0, 2.0]]))
    assert m[0, 1].item() < 1e-9


# --- pair_attention_distributions ----------------------------------------------

def test_pair_attention_distributions_shape_and_normalised(synthetic_rgcn_graph):
    torch.manual_seed(0)
    model = KDCRGA(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                   num_layers=2, num_bases=2, dropout=0.0, num_hops=2,
                   use_shared_view=True, use_hierarchical=True)
    model.eval()
    side_effects = torch.arange(3)
    with torch.no_grad():
        h = model.encode(synthetic_rgcn_graph)
        # entity 2 has two neighbours (4 and 5) in the fixture -> non-trivial support
        dist = pair_attention_distributions(model, synthetic_rgcn_graph, h,
                                             drug_a=2, drug_b=0,
                                             side_effects=side_effects)
    assert dist.shape == (3, 2)
    assert torch.allclose(dist.sum(dim=1), torch.ones(3), atol=1e-6)


def test_pair_attention_distributions_none_when_no_neighbours(synthetic_rgcn_graph):
    torch.manual_seed(0)
    model = KDCRGA(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                   num_layers=2, num_bases=2, dropout=0.0)
    model.eval()
    # entity 5 is not a drug and has no cached drug-neighbour entry -> empty
    with torch.no_grad():
        h = model.encode(synthetic_rgcn_graph)
        dist = pair_attention_distributions(model, synthetic_rgcn_graph, h,
                                            drug_a=5, drug_b=0,
                                            side_effects=torch.arange(3))
    assert dist is None
