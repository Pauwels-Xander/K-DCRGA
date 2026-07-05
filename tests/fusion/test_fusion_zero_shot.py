from experiments.fusion_zero_shot import knowddi_can_score_unseen_se


def test_knowddi_cannot_score_unseen_side_effect():
    """KnowDDI's classifier head is W_final: Linear(3*d, num_rels=200); an unseen
    relation index has no column, so zero-shot scoring is structurally impossible.
    """
    assert knowddi_can_score_unseen_se() is False
