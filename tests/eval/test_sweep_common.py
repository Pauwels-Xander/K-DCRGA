"""Orchestration tests for the sweep driver (checkpoint wiring + resume).

`train`/`evaluate_test_set`/`build_model`/`load_config` are monkeypatched so the
test exercises run_config's control flow without real training or graph data.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import _sweep_common as sc  # noqa: E402


def test_load_zk_init_returns_none_when_flag_off(tmp_path):
    p = tmp_path / "zk.pt"
    torch.save(torch.zeros(3, 4), p)
    cfg = {"model": {"use_ontology_zk": False}}
    assert sc.load_zk_init(cfg, 3, 4, path=p) is None


def test_load_zk_init_loads_cache_when_flag_on_and_shape_matches(tmp_path):
    p = tmp_path / "zk.pt"
    z = torch.arange(12.0).reshape(3, 4)
    torch.save(z, p)
    cfg = {"model": {"use_ontology_zk": True}}
    out = sc.load_zk_init(cfg, 3, 4, path=p)
    assert out is not None and torch.allclose(out, z)


def test_load_zk_init_none_when_file_missing(tmp_path):
    cfg = {"model": {"use_ontology_zk": True}}
    assert sc.load_zk_init(cfg, 3, 4, path=tmp_path / "nope.pt") is None


def test_load_zk_init_none_on_shape_mismatch(tmp_path):
    p = tmp_path / "zk.pt"
    torch.save(torch.zeros(2, 4), p)   # wrong num_side_effects
    cfg = {"model": {"use_ontology_zk": True}}
    assert sc.load_zk_init(cfg, 3, 4, path=p) is None


def _tiny_graph():
    from kdcrga.data.schema import DDIEdges, build_hetero_data
    nf = {"entity": torch.randn(6, 4)}
    edges = {("entity", "rel_0", "entity"): torch.tensor([[0], [3]])}
    ddi = DDIEdges(pair_index=torch.tensor([[0], [1]]), side_effect=torch.tensor([0]))
    data = build_hetero_data(nf, edges, ddi)
    data.num_drugs = 2
    return data


def test_maybe_inject_atc_returns_none_when_flag_off(tmp_path):
    data = _tiny_graph()
    cfg = {"model": {"use_ontology_atc": False}}
    assert sc.maybe_inject_atc(data, cfg, path=tmp_path / "atc.pt") is None
    assert ("entity", "rel_atc", "entity") not in data.edge_types


def test_maybe_inject_atc_none_when_cache_missing(tmp_path):
    data = _tiny_graph()
    cfg = {"model": {"use_ontology_atc": True}}
    assert sc.maybe_inject_atc(data, cfg, path=tmp_path / "missing.pt") is None


def test_maybe_inject_atc_injects_when_flag_on_and_cache_present(tmp_path):
    data = _tiny_graph()
    art = {"drug_to_atc_ancestors": {0: ["N05"], 1: ["N05"]},
           "atc_codes": ["N05"], "atc_features": torch.full((1, 4), 2.0)}
    p = tmp_path / "atc.pt"
    torch.save(art, p)
    cfg = {"model": {"use_ontology_atc": True}}
    a2e = sc.maybe_inject_atc(data, cfg, path=p)
    assert a2e == {"N05": 6}                                   # appended after entity 5
    assert ("entity", "rel_atc", "entity") in data.edge_types
    assert data["entity"].x.shape == (7, 4)


def test_progress_bar_renders_fraction():
    assert sc._progress_bar(0, 4, width=4) == "[----] 0%"
    assert sc._progress_bar(2, 4, width=4) == "[##--] 50%"
    assert sc._progress_bar(4, 4, width=4) == "[####] 100%"


def test_fmt_duration():
    assert sc._fmt_duration(45) == "45s"
    assert sc._fmt_duration(125) == "2m05s"
    assert sc._fmt_duration(3725) == "1h02m"


def test_run_sweep_iterates_full_grid_config_major(monkeypatch, tmp_path):
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(sc, "DATA", tmp_path)             # exists() -> True
    monkeypatch.setattr(sc, "load_config", lambda p: {"model": {}})
    monkeypatch.setattr(sc, "load_data", lambda seed=0, cfg=None: object())
    monkeypatch.setattr(sc, "run_config",
                        lambda name, path, kind, seed, data, **kw:
                        calls.append((name, seed)))
    cfgs = [("a", "a.yaml", "dc_rgcn"), ("b", "b.yaml", "kdcrga")]
    sc.run_sweep(cfgs, seeds=[0, 1], runs_root=tmp_path)
    assert calls == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]


def test_run_sweep_loads_separate_graphs_per_ontology_variant(monkeypatch, tmp_path):
    # v5 needs an ATC-injected graph; v4/baselines need the plain graph. The
    # sweep must load each variant once (not per config/seed) and hand each run
    # the right one.
    monkeypatch.setattr(sc, "DATA", tmp_path)
    cfg_by_path = {"plain.yaml": {"model": {}},
                   "atc.yaml": {"model": {"use_ontology_atc": True}}}
    monkeypatch.setattr(sc, "load_config", lambda p: cfg_by_path[p])

    loads: list[bool] = []

    def fake_load_data(seed=0, cfg=None):
        use_atc = bool((cfg or {}).get("model", {}).get("use_ontology_atc", False))
        loads.append(use_atc)
        return "ATC" if use_atc else "PLAIN"

    monkeypatch.setattr(sc, "load_data", fake_load_data)
    seen: list[tuple[str, object]] = []
    monkeypatch.setattr(sc, "run_config",
                        lambda name, path, kind, seed, data, **kw:
                        seen.append((name, data)))

    cfgs = [("p", "plain.yaml", "decagon"), ("v5", "atc.yaml", "kdcrga")]
    sc.run_sweep(cfgs, seeds=[0, 1], runs_root=tmp_path)

    # one load per variant, not per (config, seed)
    assert loads.count(False) == 1 and loads.count(True) == 1
    # each run received the graph matching its config's ontology setting
    assert all(d == "PLAIN" for n, d in seen if n == "p")
    assert all(d == "ATC" for n, d in seen if n == "v5")


@pytest.fixture
def patched(monkeypatch):
    calls: dict = {}
    cfg = {"device": "cpu",
           "training": {"lr": 1e-3, "weight_decay": 0.0, "class_weight_gamma": 0.0,
                        "batch_size": 8, "max_epochs": 5, "early_stop_patience": 3}}
    monkeypatch.setattr(sc, "load_config", lambda p: cfg)
    monkeypatch.setattr(sc, "build_model", lambda kind, cfg, data: object())

    def fake_train(model, data, **kw):
        calls["train_kwargs"] = kw
        return {"train_loss": [0.5], "val_auprc": [0.3]}

    monkeypatch.setattr(sc, "train", fake_train)
    monkeypatch.setattr(
        sc, "evaluate_test_set",
        lambda *a, **k: ({"auroc": 0.8, "auprc": 0.7}, {0: {"auroc": 0.8}}))
    return calls


def test_run_config_wires_checkpoint_dir_and_resume(patched, tmp_path):
    sc.run_config("cfgA", "x.yaml", "dc_rgcn", seed=1, data=None,
                  runs_root=tmp_path, resume=True)
    kw = patched["train_kwargs"]
    assert kw["checkpoint_dir"] == tmp_path / "cfgA" / "seed_1"   # per-seed dir
    assert kw["resume"] is True
    assert (tmp_path / "cfgA" / "seed_1" / "test_metrics.json").exists()


def test_run_config_skips_completed_run_on_resume(patched, tmp_path):
    # First run writes test_metrics.json.
    sc.run_config("cfgA", "x.yaml", "dc_rgcn", seed=0, data=None,
                  runs_root=tmp_path, resume=True)
    patched.clear()
    # Re-running with resume must NOT retrain a finished (config, seed).
    agg = sc.run_config("cfgA", "x.yaml", "dc_rgcn", seed=0, data=None,
                        runs_root=tmp_path, resume=True)
    assert "train_kwargs" not in patched          # train was not called
    assert agg["auroc"] == 0.8                     # cached metrics returned


def test_run_config_reruns_when_not_resuming(patched, tmp_path):
    sc.run_config("cfgA", "x.yaml", "dc_rgcn", seed=0, data=None,
                  runs_root=tmp_path, resume=True)
    patched.clear()
    sc.run_config("cfgA", "x.yaml", "dc_rgcn", seed=0, data=None,
                  runs_root=tmp_path, resume=False)
    assert "train_kwargs" in patched               # fresh run retrains


def test_smoke_disables_checkpointing(patched, tmp_path):
    sc.run_config("cfgA", "x.yaml", "dc_rgcn", seed=0, data=None,
                  runs_root=tmp_path, smoke=True)
    assert patched["train_kwargs"]["checkpoint_dir"] is None
    assert patched["train_kwargs"]["resume"] is False


def test_run_config_accum_steps_defaults_to_one(patched, tmp_path):
    sc.run_config("cfgA", "x.yaml", "decagon", seed=0, data=None,
                  runs_root=tmp_path)
    assert patched["train_kwargs"]["accum_steps"] == 1


def test_run_config_accum_steps_from_effective_batch(monkeypatch, tmp_path):
    cfg = {"device": "cpu",
           "training": {"lr": 1e-3, "weight_decay": 0.0, "class_weight_gamma": 0.0,
                        "batch_size": 128, "effective_batch_size": 1024,
                        "max_epochs": 5, "early_stop_patience": 3}}
    captured: dict = {}
    monkeypatch.setattr(sc, "load_config", lambda p: cfg)
    monkeypatch.setattr(sc, "build_model", lambda kind, cfg, data: object())
    monkeypatch.setattr(
        sc, "train",
        lambda model, data, **kw: (captured.update(kw),
                                   {"train_loss": [0.1], "val_auprc": [0.2]})[1])
    monkeypatch.setattr(
        sc, "evaluate_test_set",
        lambda *a, **k: ({"auroc": 0.8, "auprc": 0.7}, {0: {"auroc": 0.8}}))
    sc.run_config("c", "x.yaml", "decagon", seed=0, data=None, runs_root=tmp_path)
    assert captured["accum_steps"] == 8   # 1024 // 128


def test_build_model_kdcrga_dedicom_dispatch(synthetic_rgcn_graph):
    from kdcrga.models.kdcrga_dedicom import KDCRGADedicom
    cfg = {"model": {"hidden_dim": 16, "encoder_depth": 2, "num_bases": 2,
                     "dropout": 0.0, "num_hops": 2, "use_shared_view": True,
                     "use_ontology_zk": False}}
    model = sc.build_model("kdcrga_dedicom", cfg, synthetic_rgcn_graph)
    assert isinstance(model, KDCRGADedicom)
