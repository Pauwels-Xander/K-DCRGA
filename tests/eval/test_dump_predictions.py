import torch
from torch_geometric.data import HeteroData

from kdcrga.eval.dump_predictions import move_graph_to_device, score_triples


class _StubModel(torch.nn.Module):
    """Returns a deterministic logit = drug_a + drug_b + side_effect."""
    def encode(self, data):
        return data["entity"].x            # any cached tensor; unused by forward
    def forward(self, data, pair_index, side_effect, h=None):
        return (pair_index[0] + pair_index[1] + side_effect).float()


class _StubData:
    """Stub HeteroData that returns a constant tensor when indexed."""
    def __getitem__(self, k):
        class _N:
            x = torch.zeros(10, 4)
        return _N()


def test_score_triples_chunks_and_preserves_order():
    data = _StubData()
    a = torch.tensor([0, 1, 2, 3, 4])
    b = torch.tensor([5, 5, 5, 5, 5])
    se = torch.tensor([1, 1, 1, 1, 1])
    out = score_triples(_StubModel(), data, a, b, se, batch_size=2, device="cpu")
    assert torch.equal(out, torch.tensor([6., 7., 8., 9., 10.]))


def test_dump_stream_f_forwards_model_kind(tmp_path, monkeypatch):
    """dump_stream_f must be able to dump ANY model kind (Decagon/LaGAT), not only
    kdcrga_dedicom, so baselines can be scored through the same common evaluator
    for the pair-disjoint control. The kind must reach build_model."""
    import experiments._sweep_common as sc
    import kdcrga.config as cfgmod
    import kdcrga.eval.dump_predictions as dp
    from kdcrga.data.biosnap_eval import EvalTriples

    captured = {}

    class _FakeModel(torch.nn.Module):
        def encode(self, data):
            return None
        def forward(self, data, pair, se, h=None):
            return torch.zeros(pair.size(1))

    def _fake_build_model(kind, cfg, data):
        captured["kind"] = kind
        return _FakeModel()

    monkeypatch.setattr(sc, "build_model", _fake_build_model)
    monkeypatch.setattr(sc, "load_data", lambda seed=0, cfg=None: object())
    monkeypatch.setattr(cfgmod, "load_config", lambda p: {"training": {"batch_size": 4}})
    monkeypatch.setattr(dp, "move_graph_to_device", lambda data, device: data)
    monkeypatch.setattr(dp, "load_eval_triples", lambda p: EvalTriples(
        torch.tensor([0, 1]), torch.tensor([1, 2]),
        torch.tensor([0, 0]), torch.tensor([1.0, 0.0])))
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"model": {}})

    out = tmp_path / "f.pt"
    dp.dump_stream_f(config_path="x", checkpoint_path="y", eval_txt="z",
                     out_path=str(out), device="cpu", model_kind="decagon")
    assert captured["kind"] == "decagon"
    assert out.exists()


def test_move_graph_to_device_moves_features_and_rel_edges():
    """dump_stream_f builds a fresh CPU graph and never runs train(), so the
    encoder/forward graph tensors must be moved onto the compute device before
    the forward, or RGCNConv matmuls CPU features against CUDA weights
    ('mat2 is on cuda:0, other tensors on cpu'). Uses the 'meta' device so the
    move is a real cross-device move that needs no GPU."""
    data = HeteroData()
    data["entity"].x = torch.zeros(3, 4)
    data["entity", "rel_0", "entity"].edge_index = torch.tensor([[0, 1], [1, 2]])
    data["entity", "rel_atc", "entity"].edge_index = torch.tensor([[0], [2]])

    move_graph_to_device(data, "meta")

    assert data["entity"].x.device.type == "meta"
    assert data["entity", "rel_0", "entity"].edge_index.device.type == "meta"
    assert data["entity", "rel_atc", "entity"].edge_index.device.type == "meta"
