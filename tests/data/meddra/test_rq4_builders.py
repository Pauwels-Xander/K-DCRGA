# tests/data/meddra/test_rq4_builders.py
from kdcrga.data.meddra.artifact import MeddraEntry
from kdcrga.data.meddra.hierarchy import build_se_distance_matrix, build_soc_of


def _mapping():
    # SE0 = Enteritis (Gastro), SE1 = folate-def anaemia (Blood, HLT ...042),
    # SE3 = B12-def anaemia (Blood, same HLT ...042 as SE1).
    return {
        0: MeddraEntry("c", "enteritis", "10014866", "Enteritis", "10017977",
                       "10017969", "10017947", "Gastrointestinal disorders",
                       "exact_llt", 100.0),
        1: MeddraEntry("c", "fda", "10002043", "Anaemia folate deficiency",
                       "10002042", "10002086", "10005329",
                       "Blood and lymphatic system disorders", "fuzzy_llt", 95.0),
        3: MeddraEntry("c", "b12", "10002080", "Anaemia vitamin B12 deficiency",
                       "10002042", "10002086", "10005329",
                       "Blood and lymphatic system disorders", "exact_llt", 100.0),
    }


def test_distance_matrix_same_hlt_and_cross_soc():
    d = build_se_distance_matrix(_mapping())
    assert d[(1, 3)] == 2.0          # same HLT
    assert d[(0, 1)] == 8.0          # different SOC
    assert (1, 1) not in d            # no self-pairs


def test_soc_of():
    soc = build_soc_of(_mapping())
    assert soc[0] == "Gastrointestinal disorders"
    assert soc[1] == soc[3] == "Blood and lymphatic system disorders"
