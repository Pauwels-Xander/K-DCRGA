"""Shared driver for the RQ1 / RQ3 sweeps.

`run_config` trains one (config, seed) on a shared, pre-loaded graph and writes
``runs/<name>/seed_<N>/{history,test_metrics,per_se_metrics}.json`` via
``kdcrga.eval.run.save_run_outputs``. The data split is FIXED across all runs
(proposal Sec. 3.5 — every method is re-run on the same split); only the model
init and minibatch/negative sampling vary with ``seed``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from kdcrga.config import load_config
from kdcrga.eval.run import evaluate_test_set, save_run_outputs
from kdcrga.training.loop import train

DATA = Path("third_party/knowddi/data/BioSNAP")
ZK_INIT_PATH = Path("data/processed/zk_init.pt")
ATC_INJECT_PATH = Path("data/processed/atc_inject.pt")


def maybe_inject_atc(data, cfg, path=ATC_INJECT_PATH):
    """If ``model.use_ontology_atc`` is set and the ATC artifact exists, inject
    ATC ancestor nodes/edges into ``data`` in place (artifact built by
    ``experiments/build_atc_inject.py``). Returns ``atc_to_entity_id`` or None
    when no injection happened. Must run before ``build_model`` so the encoder's
    relation count includes ``rel_atc``."""
    if not cfg["model"].get("use_ontology_atc", False):
        return None
    p = Path(path)
    if not p.exists():
        print(f"use_ontology_atc set but {p} missing; skipping ATC injection",
              flush=True)
        return None
    from kdcrga.data.ontology import inject_atc_into_graph
    art = torch.load(p, map_location="cpu", weights_only=False)
    a2e = inject_atc_into_graph(data, art["drug_to_atc_ancestors"],
                                art["atc_codes"], art["atc_features"])
    print(f"injected {len(art['atc_codes'])} ATC nodes; "
          f"{len(art['drug_to_atc_ancestors'])} drugs linked to ATC", flush=True)
    return a2e


def load_zk_init(cfg, num_side_effects: int, hidden_dim: int, path=ZK_INIT_PATH):
    """Return the cached MedDRA z_k init tensor when ``model.use_ontology_zk`` is
    set and the cache exists with shape ``(num_side_effects, hidden_dim)``; else
    None (the model falls back to random z_k). The cache is built once by
    ``experiments/build_zk_init.py`` (Eq. zk-init, SapBERT-embedded terms)."""
    if not cfg["model"].get("use_ontology_zk", False):
        return None
    p = Path(path)
    if not p.exists():
        print(f"use_ontology_zk set but {p} missing; using random z_k", flush=True)
        return None
    z = torch.load(p, map_location="cpu")
    if tuple(z.shape) != (num_side_effects, hidden_dim):
        print(f"zk_init shape {tuple(z.shape)} != ({num_side_effects}, "
              f"{hidden_dim}); using random z_k", flush=True)
        return None
    print(f"loaded MedDRA z_k init from {p}", flush=True)
    return z


def _resolve_device(cfg) -> str:
    device = cfg.get("device", "cpu")
    return device if torch.cuda.is_available() else "cpu"


def build_model(kind: str, cfg, data):
    """Construct the model named by ``kind`` (kdcrga / dc_rgcn / decagon / lagat)."""
    m = cfg["model"]
    in_dim = data["entity"].x.size(-1)
    num_relations = sum(1 for et in data.edge_types if et[1].startswith("rel_"))
    num_se = int(data.num_side_effects)
    common = dict(
        in_dim=in_dim, hidden_dim=m["hidden_dim"], num_relations=num_relations,
        num_side_effects=num_se, num_layers=m["encoder_depth"],
        num_bases=m["num_bases"], dropout=m["dropout"])
    if kind == "dc_rgcn":
        from kdcrga.models.baselines.dc_rgcn import DecoderConditionedRGCN
        return DecoderConditionedRGCN(**common)
    if kind == "decagon":
        from kdcrga.models.baselines.decagon import Decagon
        return Decagon(**common)
    if kind == "lagat":
        from kdcrga.models.baselines.lagat import LaGAT
        return LaGAT(**common)
    if kind == "kdcrga":
        from kdcrga.models.kdcrga import KDCRGA
        z_init = load_zk_init(cfg, num_se, m["hidden_dim"])
        return KDCRGA(
            in_dim=in_dim, hidden_dim=m["hidden_dim"], num_relations=num_relations,
            num_side_effects=num_se, num_layers=m["encoder_depth"],
            num_bases=m["num_bases"], dropout=m["dropout"],
            num_hops=m.get("num_hops", 1),
            use_shared_view=m.get("use_shared_view", False),
            use_hierarchical=m.get("use_hierarchical", False),
            use_attention=m.get("use_attention", True),
            z_init=z_init)
    if kind == "kdcrga_dedicom":
        from kdcrga.models.kdcrga_dedicom import KDCRGADedicom
        z_init = load_zk_init(cfg, num_se, m["hidden_dim"])
        return KDCRGADedicom(
            in_dim=in_dim, hidden_dim=m["hidden_dim"], num_relations=num_relations,
            num_side_effects=num_se, num_layers=m["encoder_depth"],
            num_bases=m["num_bases"], dropout=m["dropout"],
            num_hops=m.get("num_hops", 1),
            use_shared_view=m.get("use_shared_view", False),
            z_init=z_init)
    raise ValueError(f"unknown model kind: {kind!r}")


def run_config(
    name: str,
    config_path: str,
    kind: str,
    seed: int,
    data,
    runs_root: str | Path = "runs",
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, float]:
    """Train + evaluate one (config, seed); persist outputs; return aggregate.

    With ``resume=True`` the run is crash-tolerant in two ways:
      * config-level: if ``test_metrics.json`` already exists the (config, seed)
        is complete and is skipped (its cached metrics are returned), so a
        relaunched sweep does not re-train finished runs;
      * epoch-level: ``train`` is given ``checkpoint_dir``/``resume`` so a run
        that died mid-training continues from its last per-epoch checkpoint.
    """
    out_dir = Path(runs_root) / name / f"seed_{seed}"
    metrics_path = out_dir / "test_metrics.json"
    if resume and not smoke and metrics_path.exists():
        agg = json.loads(metrics_path.read_text())
        print(f"[{name} seed {seed}] already complete -> skip "
              f"(auroc={agg.get('auroc', float('nan')):.4f})")
        return agg

    cfg = load_config(config_path)
    # Smoke runs force CPU: the full-scale 128-dim model over the ~1.6M-edge BKG
    # OOMs small GPUs (Plan 4 "OOM at scale"); the smoke only checks plumbing.
    device = "cpu" if smoke else _resolve_device(cfg)
    torch.manual_seed(seed)
    model = build_model(kind, cfg, data)

    t = cfg["training"]
    # Effective-batch matching: accumulate gradients over (eff // batch_size)
    # micro-batches so VRAM-bound models train at the same effective batch as the
    # large-batch baselines. Absent the key, accum_steps stays 1 (no change).
    eff = t.get("effective_batch_size")
    accum_steps = max(1, eff // t["batch_size"]) if eff else 1
    max_epochs = 1 if smoke else t["max_epochs"]
    max_batches = 2 if smoke else None
    max_val_pairs = 64 if smoke else None
    # Checkpoint into the per-seed run dir (NOT cfg["output"]["checkpoint_dir"],
    # which lacks a seed subdir and would let 3 seeds clobber each other). Smoke
    # runs skip checkpointing — they are throwaway plumbing checks.
    history = train(
        model, data, lr=t["lr"], weight_decay=t["weight_decay"],
        gamma=t["class_weight_gamma"], batch_size=t["batch_size"],
        accum_steps=accum_steps,
        max_epochs=max_epochs, patience=t["early_stop_patience"], seed=seed,
        device=device, max_batches=max_batches, max_val_pairs=max_val_pairs,
        show_progress=not smoke,
        checkpoint_dir=None if smoke else out_dir,
        resume=resume and not smoke)

    agg, per_se = evaluate_test_set(
        model, data, batch_size=t["batch_size"], device=device,
        max_test_pairs=64 if smoke else None)
    save_run_outputs(out_dir, history, agg, per_se)
    print(f"[{name} seed {seed}] auroc={agg['auroc']:.4f} auprc={agg['auprc']:.4f}"
          f" -> {out_dir}")
    return agg


def load_data(seed: int = 0, cfg=None, pair_disjoint: bool = False):
    """Load the BioSNAP graph with a FIXED split (seed 0 by default). When ``cfg``
    is given and ``model.use_ontology_atc`` is set, ATC ancestor nodes are injected
    so the returned graph is the ontology-augmented one for that config.

    ``pair_disjoint=True`` uses the canonical BioSNAP file-boundary split (the
    standard inductive benchmark KnowDDI is compared on) instead of the random
    per-triple stratified split. ``seed`` is then ignored (the split is fixed by the
    files). Default stays the stratified split so RQ1 is unaffected."""
    from kdcrga.data.graph import (
        attach_file_splits,
        attach_splits,
        load_knowddi_graph,
    )
    graph = load_knowddi_graph(DATA)
    data = attach_file_splits(graph) if pair_disjoint else attach_splits(graph, seed=seed)
    if cfg is not None:
        maybe_inject_atc(data, cfg)
    return data


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration: '45s', '2m05s', '1h02m'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    """ASCII progress bar, e.g. '[####------] 40%'."""
    frac = done / total if total > 0 else 1.0
    filled = int(width * frac)
    return f"[{'#' * filled}{'-' * (width - filled)}] {int(100 * frac)}%"


def run_sweep(configs: list[tuple[str, str, str]], seeds: list[int],
              smoke: bool = False, runs_root: str | Path = "runs",
              resume: bool = False) -> None:
    """configs: list of (name, config_path, kind). Loads data once, fixed split.

    ``resume=True`` makes a relaunched sweep skip already-finished (config, seed)
    runs and continue any that died mid-training from their last checkpoint.

    A sweep-level progress line (ASCII bar, runs done/total, elapsed, rough ETA)
    is printed before each run via ``tqdm.write`` so it does not corrupt the
    per-epoch training bar and stays readable in a redirected log file.
    """
    if not DATA.exists():
        print(f"SKIP: {DATA} not found.")
        return
    # Configs that inject ATC need the ontology-augmented graph; the rest need
    # the plain graph. The split mode (stratified vs pair-disjoint) also changes the
    # graph, so cache each (use_atc, split_mode) variant once and reuse it across that
    # variant's runs — the split is fixed (seed 0 / file boundaries) for both.
    graph_cache: dict[tuple[bool, str], object] = {}

    def _graph_for(config_path):
        cfg = load_config(config_path)
        use_atc = bool(cfg["model"].get("use_ontology_atc", False))
        split_mode = cfg.get("data", {}).get("split_mode", "stratified")
        key = (use_atc, split_mode)
        if key not in graph_cache:
            graph_cache[key] = load_data(
                seed=0, cfg=cfg, pair_disjoint=(split_mode == "pair_disjoint"))
        return graph_cache[key]

    grid = [(name, path, kind, seed)
            for (name, path, kind) in configs for seed in seeds]
    total = len(grid)
    start = time.time()
    for done, (name, path, kind, seed) in enumerate(grid):
        elapsed = time.time() - start
        eta = (elapsed / done * (total - done)) if done else None
        line = (f"{_progress_bar(done, total)} {done}/{total} runs "
                f"| elapsed {_fmt_duration(elapsed)}"
                + (f" | ETA ~{_fmt_duration(eta)}" if eta is not None else "")
                + f" | -> {name} seed {seed}")
        tqdm.write(line)
        run_config(name, path, kind, seed, _graph_for(path), runs_root=runs_root,
                   smoke=smoke, resume=resume)
    tqdm.write(f"{_progress_bar(total, total)} {total}/{total} runs DONE in "
               f"{_fmt_duration(time.time() - start)}; runs in {runs_root}/")
