from pathlib import Path
import torch
from kdcrga.data.biosnap_eval import load_eval_triples

def test_expands_set_bits_with_row_polarity(tmp_path: Path):
    # 4-bit label vectors for readability; real files use 200.
    # row 1: pair (0,1) positive for SEs 0 and 2
    # row 2: pair (3,4) negative for SE 1
    p = tmp_path / "mini.txt"
    p.write_text("0\t1\t1,0,1,0\t1\n3\t4\t0,1,0,0\t0\n")
    t = load_eval_triples(p)
    got = sorted(zip(t.drug_a.tolist(), t.drug_b.tolist(),
                     t.side_effect.tolist(), t.label.tolist()))
    assert got == [(0, 1, 0, 1.0), (0, 1, 2, 1.0), (3, 4, 1, 0.0)]
    assert t.drug_a.dtype == torch.int64 and t.label.dtype == torch.float32
