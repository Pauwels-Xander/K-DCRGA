import torch

from kdcrga.data.meddra.artifact import MeddraEntry
from kdcrga.data.ontology import (
    build_meddra_zk_init,
    inject_atc_into_graph,
    inject_atc_nodes,
    meddra_zk_init,
)


def _fake_text_embed(names):
    # Deterministic 2-d vector per name: [len(name), 1.0].
    return torch.tensor([[float(len(n)), 1.0] for n in names])


def _entry(pt_code, pt_name, hlt, hlgt, soc, soc_name):
    return MeddraEntry(cui="C", name="x", pt_code=pt_code, pt_name=pt_name,
                       hlt=hlt, hlgt=hlgt, soc=soc, soc_name=soc_name,
                       match_type="exact_llt", match_score=100.0)


def test_build_meddra_zk_init_averages_pt_and_ancestors_then_scatters():
    mapping = {1: _entry("P0", "aaa", "H0", "G0", "S0", "ssss")}
    term_names = {"P0": "aaa", "H0": "bb", "G0": "ccccc", "S0": "ssss"}
    z = build_meddra_zk_init(mapping, term_names, num_side_effects=3, d_z=2,
                             text_embed_fn=_fake_text_embed, project=lambda x: x)
    assert z.shape == (3, 2)
    # Matched SE row = mean of embeddings of [own PT + 3 ancestor names].
    expected = _fake_text_embed(["aaa", "bb", "ccccc", "ssss"]).mean(0)
    assert torch.allclose(z[1], expected)
    # Unmatched side effects stay zero (their z_k remains trainable from there).
    assert torch.allclose(z[0], torch.zeros(2))
    assert torch.allclose(z[2], torch.zeros(2))


def test_build_meddra_zk_init_seeded_projection_is_deterministic():
    mapping = {0: _entry("P0", "aaa", "H0", "G0", "S0", "ssss")}
    term_names = {"P0": "aaa", "H0": "bb", "G0": "ccccc", "S0": "ssss"}
    kw = dict(num_side_effects=2, d_z=4, text_embed_fn=_fake_text_embed, seed=7)
    z1 = build_meddra_zk_init(mapping, term_names, **kw)
    z2 = build_meddra_zk_init(mapping, term_names, **kw)
    assert z1.shape == (2, 4)             # fixed P projects 2-d embeddings -> d_z=4
    assert torch.allclose(z1, z2)         # seeded P -> reproducible
    assert not torch.allclose(z1[0], torch.zeros(4))  # matched row non-zero


def test_meddra_zk_init_averages_own_plus_ancestors():
    # 2 side effects; embeddings are a small deterministic table.
    raw = torch.tensor([
        [1.0, 0.0],   # row 0: side effect 0
        [0.0, 1.0],   # row 1: side effect 1
        [3.0, 3.0],   # row 2: ancestor 10
        [5.0, 5.0],   # row 3: ancestor 11
    ])
    ancestors = {0: [10], 1: [11]}
    se_to_row = {0: 0, 1: 1}
    anc_to_row = {10: 2, 11: 3}
    z = meddra_zk_init([0, 1], ancestors, raw, se_to_row, anc_to_row,
                       d_z=2, project=None)
    # eq:zk-init: ((own + sum(ancestors)) / (1 + |ancestors|))
    assert torch.allclose(z[0], torch.tensor([2.0, 1.5]))
    assert torch.allclose(z[1], torch.tensor([2.5, 3.0]))
    assert z.shape == (2, 2)


def test_meddra_zk_init_projects_when_project_supplied():
    raw = torch.tensor([[1.0, 0.0, 0.0]])
    ancestors = {0: []}
    z = meddra_zk_init(
        [0], ancestors, raw, {0: 0}, {}, d_z=2,
        project=torch.nn.Linear(3, 2, bias=False),
    )
    assert z.shape == (1, 2)


def test_inject_atc_into_graph_extends_features_edges_and_clears_cache():
    from kdcrga.data.graph import build_neighbour_cache
    from kdcrga.data.schema import DDIEdges, build_hetero_data
    nf = {"entity": torch.randn(6, 4)}
    edges = {("entity", "rel_0", "entity"): torch.tensor([[0], [3]])}
    ddi = DDIEdges(pair_index=torch.tensor([[0], [1]]), side_effect=torch.tensor([0]))
    data = build_hetero_data(nf, edges, ddi)
    data.num_drugs = 2
    build_neighbour_cache(data, 2)            # pre-existing cache to be invalidated
    assert hasattr(data, "neighbour_index")

    drug_to_atc = {0: ["N05", "N05B"], 1: ["N05"]}
    atc_codes = ["N05", "N05B"]               # -> entity ids 6, 7
    atc_features = torch.full((2, 4), 5.0)
    a2e = inject_atc_into_graph(data, drug_to_atc, atc_codes, atc_features)

    assert a2e == {"N05": 6, "N05B": 7}
    assert data["entity"].x.shape == (8, 4)                    # 6 + 2 ATC nodes
    assert torch.allclose(data["entity"].x[6], torch.full((4,), 5.0))
    et = ("entity", "rel_atc", "entity")
    assert et in data.edge_types
    assert {0, 1, 6, 7} <= set(data[et].edge_index.flatten().tolist())
    # Neighbour cache rebuilt to include rel_atc: drugs gain their ATC nodes...
    assert 6 in data.neighbour_index[0] and 7 in data.neighbour_index[0]
    assert 6 in data.neighbour_index[1]
    # ...and drugs sharing an ATC ancestor (N05=node 6) become 2-hop neighbours.
    assert 1 in data.two_hop_index[0]


def test_inject_atc_nodes_adds_drug_to_atc_edges():
    from kdcrga.data.schema import DDIEdges, build_hetero_data
    nf = {"entity": torch.randn(8, 4)}
    edges = {("entity", "rel_0", "entity"): torch.tensor([[0], [4]])}
    ddi = DDIEdges(pair_index=torch.tensor([[0],[1]]),
                   side_effect=torch.tensor([0]))
    data = build_hetero_data(nf, edges, ddi)
    data.num_drugs = 4
    # drug 0 maps to ATC code 100 -> entity 5; drug 1 maps to ATC 101 -> entity 6
    inject_atc_nodes(data,
                     drug_to_atc_ancestors={0: [100], 1: [101]},
                     atc_to_entity_id={100: 5, 101: 6})
    et = ("entity", "rel_atc", "entity")
    assert et in data.edge_types
    ei = data[et].edge_index
    # Each pair is added bidirectionally -> 4 edges total
    assert ei.shape == (2, 4)
    # Endpoints contain both the drugs and the ATC entity ids
    endpoints = set(ei.flatten().tolist())
    assert {0, 1, 5, 6} <= endpoints
