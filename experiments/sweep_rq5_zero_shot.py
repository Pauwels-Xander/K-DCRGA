"""RQ5 zero-shot generalisation to held-out side effects (proposal Sec. 4.10.3).

Headline model = ``kdcrga_dedicom_v5`` (attention + ATC graph + MedDRA z_k init).
The DEDICOM model has TWO per-side-effect tables — ``z_k`` (attention) and
``decoder.D`` = d_k (scorer) — and BOTH must be initialised for held-out side
effects, or zero-shot is broken. ``meddra_zk_init`` is generic over any table.

Pipeline:
  1. Split the 200 side effects 80/20, stratified by frequency quartile.
  2. Re-label DDI triples so every held-out-SE triple is test (split id 2). The
     model therefore sees NO held-out-SE triple (positive or negative) in
     training -> held-out z_k / d_k receive no gradient by construction.
  3. Train on the 160 train SEs (standard loop, early stop on val AUPRC).
  4. Post-hoc, on the SAME trained weights, evaluate three held-out fills of BOTH
     z_k and d_k:
        random       — seeded draw at each table's init scale (the floor)
        global_mean  — mean of the trained train-SE rows (no-ontology null)
        meddra       — MedDRA ancestor-average of the trained train-SE rows
     meddra >> global_mean >> random  =>  the ontology (not the architecture)
     enables zero-shot. Results -> runs/rq5/dedicom_v5/seed_<n>/<arm>/.

Run: python experiments/sweep_rq5_zero_shot.py [--config ...] [--seed N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from kdcrga.config import load_config
from kdcrga.data.meddra.artifact import load_mapping
from kdcrga.data.meddra.zk_init import meddra_zk_init
from kdcrga.eval.run import evaluate_test_set, save_run_outputs
from kdcrga.eval.zero_shot import (
    global_mean_init,
    overwrite_held_out_rows,
    random_init,
    stratified_se_split,
)
from kdcrga.training.loop import train

sys.path.insert(0, str(Path(__file__).parent))
from _sweep_common import build_model, load_data  # noqa: E402

_MAPPING = Path("data/processed/se_to_meddra.json")
DEFAULT_CONFIG = "configs/rq5_dedicom_v5.yaml"
ARMS = ("random", "global_mean", "meddra")


def build_zero_shot_splits(data, held_out_ks: list[int], seed: int = 0):
    """Overwrite ``data.ddi.split`` so every held-out-SE triple is test (2) and the
    remaining triples are split train (0) / val (1) ~85/15. Returns ``data``."""
    held = set(held_out_ks)
    se = data.ddi.side_effect
    g = torch.Generator().manual_seed(seed)
    split = torch.empty(se.numel(), dtype=torch.long)
    is_test = torch.tensor([int(k) in held for k in se.tolist()])
    split[is_test] = 2
    train_idx = (~is_test).nonzero(as_tuple=True)[0]
    perm = train_idx[torch.randperm(train_idx.numel(), generator=g)]
    n_val = int(round(0.15 * perm.numel()))
    split[perm[:n_val]] = 1
    split[perm[n_val:]] = 0
    data.ddi.split = split
    return data


def freeze_rows(weight: torch.Tensor, held_out_ks: list[int]) -> None:
    """Mask the gradient of held-out rows to zero. Belt-and-suspenders only: the
    split already guarantees held-out SEs appear in no training triple, and the
    post-hoc fills below are what make the arms well-defined."""
    mask = torch.ones(weight.size(0), 1)
    for k in held_out_ks:
        mask[k] = 0.0

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * mask.to(grad.device)

    weight.register_hook(_hook)


def _held_out_fill(arm, z_tab, d_tab, train_ks, held_out_ks, mapping, dim, seed):
    """Return ``(z_fill, d_fill)`` for ``arm``, or ``None`` if unavailable."""
    if arm == "random":
        return (random_init(held_out_ks, dim, mean=0.0, std=0.1, seed=seed),
                random_init(held_out_ks, dim, mean=1.0, std=0.1, seed=seed + 1))
    if arm == "global_mean":
        return (global_mean_init(z_tab, train_ks, held_out_ks),
                global_mean_init(d_tab, train_ks, held_out_ks))
    if arm == "meddra":
        if mapping is None:
            return None
        return (meddra_zk_init(z_tab, train_ks, held_out_ks, mapping),
                meddra_zk_init(d_tab, train_ks, held_out_ks, mapping))
    raise ValueError(f"unknown arm {arm!r}")


def main(config_path: str = DEFAULT_CONFIG, seed: int = 0,
         runs_root: str = "runs", max_batches: int | None = None) -> None:
    cfg = load_config(config_path)
    device = cfg.get("device", "cpu")
    device = device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    # Graph (ATC-injected when the config sets use_ontology_atc) + fixed split.
    data = load_data(seed=seed, cfg=cfg)
    train_ks, held_out_ks = stratified_se_split(data.quartiles, train_frac=0.8,
                                                seed=seed)
    print(f"Zero-shot: {len(train_ks)} train SEs, {len(held_out_ks)} held-out SEs",
          flush=True)
    build_zero_shot_splits(data, held_out_ks, seed=seed)

    model = build_model("kdcrga_dedicom", cfg, data)  # z_k MedDRA-init for train SEs
    freeze_rows(model.z_k.weight, held_out_ks)
    freeze_rows(model.decoder.D.weight, held_out_ks)

    t = cfg["training"]
    eff = t.get("effective_batch_size")
    accum_steps = max(1, eff // t["batch_size"]) if eff else 1
    out_root = Path(runs_root) / "rq5" / "dedicom_v5" / f"seed_{seed}"

    history = train(
        model, data, lr=t["lr"], weight_decay=t["weight_decay"],
        gamma=t["class_weight_gamma"], batch_size=t["batch_size"],
        accum_steps=accum_steps, max_epochs=t["max_epochs"],
        patience=t["early_stop_patience"], seed=seed, device=device,
        max_batches=max_batches, checkpoint_dir=out_root, resume=True)

    # Snapshot the TRAINED tables once; fills read these, overwrites touch only
    # held-out rows, so the arms are independent and order-free.
    dim = model.hidden_dim
    z_tab = model.z_k.weight.detach().cpu()
    d_tab = model.decoder.D.weight.detach().cpu()
    mapping = load_mapping(_MAPPING) if _MAPPING.exists() else None
    if mapping is None:
        print(f"WARNING: {_MAPPING} missing -> 'meddra' arm skipped", flush=True)

    summary: dict[str, dict] = {}
    for arm in ARMS:
        fills = _held_out_fill(arm, z_tab, d_tab, train_ks, held_out_ks,
                               mapping, dim, seed)
        if fills is None:
            print(f"[{arm}] SKIP", flush=True)
            continue
        z_fill, d_fill = fills
        overwrite_held_out_rows(model.z_k, held_out_ks, z_fill)
        overwrite_held_out_rows(model.decoder.D, held_out_ks, d_fill)
        agg, per_se = evaluate_test_set(model, data, batch_size=t["batch_size"],
                                        device=device)
        save_run_outputs(out_root / arm, history, agg, per_se)
        summary[arm] = agg
        print(f"ZERO-SHOT [{arm}] auroc={agg['auroc']:.4f} "
              f"auprc={agg['auprc']:.4f} ap50={agg['ap50']:.4f} -> {out_root / arm}",
              flush=True)

    print("\n=== RQ5 zero-shot summary (held-out side effects) ===", flush=True)
    for arm in ARMS:
        if arm in summary:
            a = summary[arm]
            print(f"  {arm:12s} AUROC={a['auroc']:.4f}  AUPRC={a['auprc']:.4f}  "
                  f"AP@50={a['ap50']:.4f}", flush=True)
    if "meddra" in summary and "global_mean" in summary:
        d = summary["meddra"]["auroc"] - summary["global_mean"]["auroc"]
        print(f"  ontology effect (meddra - global_mean) AUROC: {d:+.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap batches/epoch (fast smoke); None = full data")
    args = ap.parse_args()
    main(args.config, seed=args.seed, runs_root=args.runs_root,
         max_batches=args.max_batches)
