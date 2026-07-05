import torch

from kdcrga.models.decoders import DedicomDecoder


def _decoder():
    torch.manual_seed(0)
    return DedicomDecoder(hidden_dim=16, num_side_effects=3)


def test_forward_returns_logit_per_triple():
    dec = _decoder()
    u_a = torch.randn(5, 16)
    u_b = torch.randn(5, 16)
    se = torch.tensor([0, 1, 2, 0, 1])
    out = dec(u_a, u_b, se)
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


def test_R_is_single_shared_matrix_D_per_se():
    dec = _decoder()
    assert dec.R.shape == (16, 16)
    assert dec.D.weight.shape == (3, 16)


def test_symmetric_at_init_and_matches_explicit_formula():
    # At init R = I, so the head is a symmetric weighted dot product.
    dec = _decoder()
    u_a = torch.randn(4, 16)
    u_b = torch.randn(4, 16)
    se = torch.tensor([0, 1, 2, 0])
    out = dec(u_a, u_b, se)
    assert torch.allclose(out, dec(u_b, u_a, se), atol=1e-5)   # symmetry at R=I
    d_k = dec.D(se)
    expected = ((u_a * d_k) * (u_b * d_k)).sum(dim=-1)         # R=I explicit form
    assert torch.allclose(out, expected, atol=1e-5)


def test_gradients_reach_D_and_R():
    dec = _decoder()
    u_a = torch.randn(3, 16, requires_grad=True)
    u_b = torch.randn(3, 16, requires_grad=True)
    se = torch.tensor([0, 1, 2])
    dec(u_a, u_b, se).sum().backward()
    assert dec.D.weight.grad is not None
    assert dec.R.grad is not None
