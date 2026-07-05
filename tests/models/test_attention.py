import torch

from kdcrga.models.attention import DoublyConditionalAttention


def test_attention_changes_with_side_effect():
    torch.manual_seed(0)
    M, E, d, dz, da = 7, 3, 8, 8, 4
    mod = DoublyConditionalAttention(d_hidden=d, d_z=dz, d_attn=da)
    h_nb = torch.randn(M, d)
    nb_query = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    h_partner = torch.randn(E, d)
    z_k = torch.randn(E, dz)
    z_k2 = torch.randn(E, dz)
    r1 = mod(h_nb, nb_query, h_partner, z_k, num_queries=E)
    r2 = mod(h_nb, nb_query, h_partner, z_k2, num_queries=E)
    assert r1.shape == (E, d)
    assert torch.isfinite(r1).all()
    assert not torch.allclose(r1, r2)
