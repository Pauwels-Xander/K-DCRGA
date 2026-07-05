import torch

from kdcrga.models.kdcrga_dedicom import KDCRGADedicom


def _model(use_shared_view=True, num_hops=2, z_init=None):
    return KDCRGADedicom(
        in_dim=8, hidden_dim=16, num_relations=2, num_side_effects=3,
        num_layers=2, num_bases=2, dropout=0.0,
        num_hops=num_hops, use_shared_view=use_shared_view, z_init=z_init,
    )


def test_forward_shape_with_shared_view(synthetic_rgcn_graph):
    model = _model(use_shared_view=True)
    logits = model(synthetic_rgcn_graph,
                   synthetic_rgcn_graph.ddi.pair_index,
                   synthetic_rgcn_graph.ddi.side_effect)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()


def test_forward_shape_without_shared_view(synthetic_rgcn_graph):
    model = _model(use_shared_view=False)
    logits = model(synthetic_rgcn_graph,
                   synthetic_rgcn_graph.ddi.pair_index,
                   synthetic_rgcn_graph.ddi.side_effect)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()


def test_encode_cache_matches_full_forward(synthetic_rgcn_graph):
    torch.manual_seed(0)
    model = _model(use_shared_view=True)
    model.eval()
    pair = synthetic_rgcn_graph.ddi.pair_index
    se = synthetic_rgcn_graph.ddi.side_effect
    with torch.no_grad():
        h = model.encode(synthetic_rgcn_graph)
        out_cached = model(synthetic_rgcn_graph, pair, se, h=h)
        out_full = model(synthetic_rgcn_graph, pair, se)
    assert torch.allclose(out_cached, out_full)


def test_symmetric_under_pair_swap_at_init(synthetic_rgcn_graph):
    # At init the decoder's R = I, and the encoder/readouts relabel symmetrically,
    # so swapping the drug pair (a,b)->(b,a) must not change the score.
    torch.manual_seed(0)
    model = _model(use_shared_view=True)
    model.eval()
    pair = synthetic_rgcn_graph.ddi.pair_index
    se = synthetic_rgcn_graph.ddi.side_effect
    flipped = pair.flip(0)
    with torch.no_grad():
        out = model(synthetic_rgcn_graph, pair, se)
        out_swapped = model(synthetic_rgcn_graph, flipped, se)
    assert torch.allclose(out, out_swapped, atol=1e-5)


def test_gradients_reach_encoder_zk_and_decoder(synthetic_rgcn_graph):
    model = _model(use_shared_view=True)
    logits = model(synthetic_rgcn_graph,
                   synthetic_rgcn_graph.ddi.pair_index,
                   synthetic_rgcn_graph.ddi.side_effect)
    logits.sum().backward()
    assert model.z_k.weight.grad is not None
    assert model.decoder.D.weight.grad is not None
    assert model.decoder.R.grad is not None
    enc_grads = [p.grad is not None for p in model.encoder.parameters()]
    assert any(enc_grads)


def test_z_init_copied_into_embedding(synthetic_rgcn_graph):
    z_init = torch.full((3, 16), 0.42)
    model = _model(z_init=z_init)
    assert torch.allclose(model.z_k.weight, z_init)
