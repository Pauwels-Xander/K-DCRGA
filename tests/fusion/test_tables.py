import torch
from kdcrga.fusion.tables import PredTable, align, to_logit, save_table, load_table

def _t(a, b, se, y, s, kind):
    return PredTable(torch.tensor(a), torch.tensor(b), torch.tensor(se),
                     torch.tensor(y, dtype=torch.float32),
                     torch.tensor(s, dtype=torch.float32), kind)

def test_align_inner_join_sorted_and_matched():
    f = _t([0, 0, 9], [1, 1, 9], [2, 0, 5], [1., 1., 0.], [0.1, 0.2, 0.3], "logit")
    s = _t([0, 0], [1, 1], [0, 2], [1., 1.], [0.7, 0.8], "prob")
    fa, sa = align(f, s)
    key = list(zip(fa.drug_a.tolist(), fa.drug_b.tolist(), fa.side_effect.tolist()))
    assert key == [(0, 1, 0), (0, 1, 2)]                  # sorted, intersection only
    assert key == list(zip(sa.drug_a.tolist(), sa.drug_b.tolist(), sa.side_effect.tolist()))
    assert fa.label.tolist() == sa.label.tolist()

def test_to_logit_prob_roundtrip():
    p = torch.tensor([0.5, 0.7311], dtype=torch.float32)
    assert torch.allclose(to_logit(p, "prob"), torch.tensor([0.0, 1.0]), atol=1e-3)

def test_save_load_roundtrip(tmp_path):
    f = _t([0], [1], [2], [1.], [0.4], "logit")
    save_table(f, tmp_path / "f.pt")
    g = load_table(tmp_path / "f.pt")
    assert g.score_kind == "logit" and g.drug_a.tolist() == [0]
