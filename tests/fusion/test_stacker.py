import torch
from kdcrga.fusion.tables import PredTable
from kdcrga.fusion.stacker import fuse_average, fit_stacker

def _pair(scores_f_logit, scores_s_prob, y, se):
    a = torch.zeros(len(y), dtype=torch.int64)
    b = torch.arange(len(y), dtype=torch.int64)
    se = torch.tensor(se)
    f = PredTable(a, b, se, torch.tensor(y, dtype=torch.float32),
                  torch.tensor(scores_f_logit), "logit")
    s = PredTable(a.clone(), b.clone(), se.clone(), torch.tensor(y, dtype=torch.float32),
                  torch.tensor(scores_s_prob), "prob")
    return f, s

def test_fuse_average_logit_midpoint():
    f, s = _pair([0.0, 0.0], [0.5, 0.5], [1., 0.], [0, 0])
    out = fuse_average(f, s, mode="logit")          # logit_F=0, logit_S=0 -> sigmoid(0)=0.5
    assert torch.allclose(out, torch.tensor([0.5, 0.5]), atol=1e-4)

def test_fit_stacker_recovers_better_stream():
    # S is perfectly informative, F is noise -> stacker should track S.
    y = [1., 1., 0., 0.] * 5
    se = [0] * 20
    s_prob = [0.95, 0.9, 0.05, 0.1] * 5
    f_logit = [0.0] * 20
    f, s = _pair(f_logit, s_prob, y, se)
    st = fit_stacker(f, s, gating="global")
    fused = st.apply(f, s)
    # fused ranks positives above negatives like S does
    assert fused[0] > fused[2] and fused[1] > fused[3]
