import pytest
import torch

from kdcrga.models.attention import DoublyConditionalAttention
from kdcrga.models.rotatory import rotatory_readouts
from kdcrga.models.shared_view import SharedView

from kdcrga.models.kdcrga import KDCRGA


def _rotatory_setup():
    torch.manual_seed(0)
    d, nq = 4, 3
    attn = DoublyConditionalAttention(d_hidden=d, d_z=d, d_attn=3)
    h_nb_a = torch.randn(7, d)
    qa = torch.tensor([0, 0, 1, 2, 2, 2, 1])
    h_nb_b = torch.randn(5, d)
    qb = torch.tensor([0, 1, 1, 2, 0])
    h_a, h_b, zk = torch.randn(nq, d), torch.randn(nq, d), torch.randn(nq, d)
    return attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk, nq, d


def test_rotatory_returns_four_views_when_no_shared():
    attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk, nq, d = _rotatory_setup()
    out = rotatory_readouts(attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk,
                            num_queries=nq, num_hops=1)
    assert len(out) == 4
    for v in out:
        assert v.shape == (nq, d)
        assert torch.isfinite(v).all()


def test_rotatory_hop1_matches_equations():
    attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk, nq, d = _rotatory_setup()
    rA, rB, rAB, rBA = rotatory_readouts(attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk,
                                         num_queries=nq, num_hops=1)
    # eq:hop1-step1 — Step 1 partner is the RAW partner-drug embedding
    exp_rA = attn(h_nb_a, qa, h_b, zk, num_queries=nq)
    exp_rB = attn(h_nb_b, qb, h_a, zk, num_queries=nq)
    assert torch.allclose(rA, exp_rA)
    assert torch.allclose(rB, exp_rB)
    # eq:hop1-step2 — Step 2 partner is the OTHER side's hop-1 readout
    exp_rAB = attn(h_nb_a, qa, exp_rB, zk, num_queries=nq)
    exp_rBA = attn(h_nb_b, qb, exp_rA, zk, num_queries=nq)
    assert torch.allclose(rAB, exp_rAB)
    assert torch.allclose(rBA, exp_rBA)


def test_rotatory_hop2_uses_previous_cross_readout():
    attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk, nq, d = _rotatory_setup()
    _, _, rAB1, rBA1 = rotatory_readouts(attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk,
                                         num_queries=nq, num_hops=1)
    rA2, rB2, rAB2, rBA2 = rotatory_readouts(attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk,
                                             num_queries=nq, num_hops=2)
    # eq:hop-t-step1 — Step 1 partner = previous hop's refined cross readout
    exp_rA2 = attn(h_nb_a, qa, rBA1, zk, num_queries=nq)
    exp_rB2 = attn(h_nb_b, qb, rAB1, zk, num_queries=nq)
    assert torch.allclose(rA2, exp_rA2)
    assert torch.allclose(rB2, exp_rB2)
    # Step 2 partner = current hop's Step-1 readout
    assert torch.allclose(rAB2, attn(h_nb_a, qa, exp_rB2, zk, num_queries=nq))
    assert torch.allclose(rBA2, attn(h_nb_b, qb, exp_rA2, zk, num_queries=nq))


def test_rotatory_shared_view_runs_per_hop():
    attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk, nq, d = _rotatory_setup()
    shared = SharedView(d_hidden=d)
    # re-seed only to vary the shared neighbourhood tensors
    torch.manual_seed(1)
    h_nb_s = torch.randn(4, d)
    qs = torch.tensor([0, 1, 1, 2])
    # num_hops=1 -> shared partner is the mean of the raw drug embeddings
    out1 = rotatory_readouts(attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk,
                             num_queries=nq, num_hops=1,
                             shared=shared, h_nb_shared=h_nb_s, q_shared=qs)
    assert len(out1) == 5
    exp_sh1 = shared(attn, h_nb_s, qs, h_a, h_b, zk, num_queries=nq)
    assert torch.allclose(out1[4], exp_sh1)
    # num_hops=2 -> shared partner is the mean of the hop-1 cross readouts
    # re-run hop 1 without the shared view to recover the hop-1 cross readouts
    _, _, rAB1, rBA1 = rotatory_readouts(attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk,
                                         num_queries=nq, num_hops=1)
    out2 = rotatory_readouts(attn, h_nb_a, qa, h_nb_b, qb, h_a, h_b, zk,
                             num_queries=nq, num_hops=2,
                             shared=shared, h_nb_shared=h_nb_s, q_shared=qs)
    exp_sh2 = shared(attn, h_nb_s, qs, h_a, h_b, zk, num_queries=nq,
                     partner=0.5 * (rAB1 + rBA1))
    assert torch.allclose(out2[4], exp_sh2)





def test_kdcrga_view_count_and_mlp_dim():
    torch.manual_seed(0)
    m4 = KDCRGA(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                num_layers=2, num_bases=2, dropout=0.0, num_hops=2,
                use_shared_view=False)
    assert m4.n_views == 4
    assert m4.mlp[0].in_features == 4 * 16
    m5 = KDCRGA(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                num_layers=2, num_bases=2, dropout=0.0, num_hops=2,
                use_shared_view=True, use_hierarchical=True)
    assert m5.n_views == 5
    assert m5.mlp[0].in_features == 5 * 16


def test_kdcrga_full_five_view_forward(synthetic_rgcn_graph):
    pair_index = synthetic_rgcn_graph.ddi.pair_index
    side_effect = synthetic_rgcn_graph.ddi.side_effect
    torch.manual_seed(0)
    m = KDCRGA(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
               num_layers=2, num_bases=2, dropout=0.0, num_hops=3,
               use_shared_view=True, use_hierarchical=True)
    out = m(synthetic_rgcn_graph, pair_index, side_effect)
    assert out.shape == (6,)
    assert torch.isfinite(out).all()


def test_kdcrga_rotatory_changes_with_hops(synthetic_rgcn_graph):
    pair_index = synthetic_rgcn_graph.ddi.pair_index
    side_effect = synthetic_rgcn_graph.ddi.side_effect
    torch.manual_seed(0)
    m1 = KDCRGA(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                num_layers=2, num_bases=2, dropout=0.0, num_hops=1)
    torch.manual_seed(0)
    m3 = KDCRGA(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
                num_layers=2, num_bases=2, dropout=0.0, num_hops=3)
    o1 = m1(synthetic_rgcn_graph, pair_index, side_effect)
    o3 = m3(synthetic_rgcn_graph, pair_index, side_effect)
    assert o1.shape == o3.shape == (6,)
    assert torch.isfinite(o1).all() and torch.isfinite(o3).all()
    assert not torch.allclose(o1, o3)   # rotatory must change the output


@pytest.mark.parametrize("use_shared_view,use_hierarchical", [
    (False, False), (False, True), (True, False), (True, True),
])
def test_kdcrga_ablation_view_configs_forward(synthetic_rgcn_graph,
                                              use_shared_view, use_hierarchical):
    pair_index = synthetic_rgcn_graph.ddi.pair_index
    side_effect = synthetic_rgcn_graph.ddi.side_effect
    torch.manual_seed(0)
    m = KDCRGA(in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
               num_layers=2, num_bases=2, dropout=0.0, num_hops=2,
               use_shared_view=use_shared_view, use_hierarchical=use_hierarchical)
    expected_views = 5 if use_shared_view else 4
    assert m.n_views == expected_views
    assert m.mlp[0].in_features == expected_views * 16
    out = m(synthetic_rgcn_graph, pair_index, side_effect)
    assert out.shape == (6,)
    assert torch.isfinite(out).all()
