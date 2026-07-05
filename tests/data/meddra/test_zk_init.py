# tests/data/meddra/test_zk_init.py
import torch

from kdcrga.data.meddra.artifact import MeddraEntry
from kdcrga.data.meddra.zk_init import meddra_zk_init


def _mapping():
    return {
        1: MeddraEntry("c", "fda", "10002043", "x", "10002042", "10002086",
                       "10005329", "Blood", "exact_llt", 100.0),
        3: MeddraEntry("c", "b12", "10002080", "x", "10002042", "10002086",
                       "10005329", "Blood", "exact_llt", 100.0),
        5: MeddraEntry("c", "ent", "10014866", "x", "10017977", "10017969",
                       "10017947", "Gastro", "exact_llt", 100.0),
    }


def test_held_out_inherits_same_hlt_sibling():
    # z_k rows: index 1 = ones, others = zeros. Held-out 3 shares HLT with 1,
    # so its init equals row 1.
    z = torch.zeros(6, 4)
    z[1] = 1.0
    init = meddra_zk_init(z, train_ks=[1], held_out_ks=[3], mapping=_mapping())
    assert torch.allclose(init[0], torch.ones(4))


def test_no_shared_ancestor_falls_back_to_global_mean():
    # Held-out 5 (Gastro) shares no SOC with train {1,3} (Blood) -> global mean.
    z = torch.zeros(6, 4)
    z[1] = 2.0
    z[3] = 4.0
    init = meddra_zk_init(z, train_ks=[1, 3], held_out_ks=[5], mapping=_mapping())
    assert torch.allclose(init[0], torch.full((4,), 3.0))   # mean(2,4)
