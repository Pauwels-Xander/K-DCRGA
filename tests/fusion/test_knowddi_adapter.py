# tests/fusion/test_knowddi_adapter.py
import numpy as np
from kdcrga.fusion.knowddi_adapter import knowddi_npz_to_table


def test_expands_npz_to_pred_table(tmp_path):
    # 2 examples, 4 "relations" for brevity.
    np.savez(
        tmp_path / "k.npz",
        preds=np.array([[0.9, 0.1, 0.8, 0.0], [0.2, 0.7, 0.0, 0.0]], dtype=np.float32),
        labels=np.array([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=np.int64),
        polarity=np.array([1, 0], dtype=np.int64),
        pair_ids=np.array([[0, 1], [3, 4]], dtype=np.int64),
    )
    t = knowddi_npz_to_table(str(tmp_path / "k.npz"))
    rows = sorted(zip(t.drug_a.tolist(), t.drug_b.tolist(), t.side_effect.tolist(),
                      t.label.tolist(), [round(s, 3) for s in t.score.tolist()]))
    assert rows == [(0, 1, 0, 1.0, 0.9), (0, 1, 2, 1.0, 0.8), (3, 4, 1, 0.0, 0.7)]
    assert t.score_kind == "prob"
