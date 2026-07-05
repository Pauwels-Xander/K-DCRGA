import torch

from kdcrga.models.attention import DoublyConditionalAttention
from kdcrga.models.shared_view import SharedView


def test_shared_view_uses_fallback_when_empty():
    d = 8
    attn = DoublyConditionalAttention(d_hidden=d, d_z=d, d_attn=4)
    sv = SharedView(d_hidden=d)
    empty_nb = torch.empty(0, d)
    empty_q = torch.empty(0, dtype=torch.long)
    h_a = torch.randn(3, d); h_b = torch.randn(3, d); z_k = torch.randn(3, d)
    out = sv(attn, empty_nb, empty_q, h_a, h_b, z_k, num_queries=3)
    assert out.shape == (3, d)
    # all rows equal the fallback
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])


def test_shared_view_uses_attention_when_nonempty():
    d = 8
    torch.manual_seed(0)
    attn = DoublyConditionalAttention(d_hidden=d, d_z=d, d_attn=4)
    sv = SharedView(d_hidden=d)
    h_nb = torch.randn(5, d)
    q = torch.tensor([0, 0, 1, 1, 2])
    h_a = torch.randn(3, d); h_b = torch.randn(3, d); z_k = torch.randn(3, d)
    out = sv(attn, h_nb, q, h_a, h_b, z_k, num_queries=3)
    assert out.shape == (3, d)
    assert torch.isfinite(out).all()


def test_shared_view_partner_override():
    torch.manual_seed(0)
    d, nq = 4, 3
    attn = DoublyConditionalAttention(d_hidden=d, d_z=d, d_attn=3)
    shared = SharedView(d_hidden=d)
    h_nb = torch.randn(4, d)
    q = torch.tensor([0, 1, 1, 2])
    h_a, h_b, zk = torch.randn(nq, d), torch.randn(nq, d), torch.randn(nq, d)
    custom = torch.randn(nq, d)

    # default partner == mean of the two raw drug embeddings
    default_out = shared(attn, h_nb, q, h_a, h_b, zk, num_queries=nq)
    via_mean = shared(attn, h_nb, q, h_a, h_b, zk, num_queries=nq,
                      partner=0.5 * (h_a + h_b))
    assert torch.allclose(default_out, via_mean)

    # an explicit partner overrides the raw-mean default
    overridden = shared(attn, h_nb, q, h_a, h_b, zk, num_queries=nq, partner=custom)
    assert not torch.allclose(overridden, default_out)
