"""Operating-point metrics for the benchmark arms: precision, recall, F1 at the
validation-F1 threshold (proposal Sec. 4.10.1), overall and per frequency quartile.

Loads each trained benchmark checkpoint (no retraining), scores the val and test
splits with 1:1 sampled negatives, picks the F1-maximising threshold on validation,
and reports precision/recall/F1 on test. Writes
``runs/bench/<arm>/seed_<n>/prf_metrics.json`` next to each checkpoint.

Run: python experiments/eval_prf.py [--arms decagon kdcrga_dedicom_v5 ...] [--device cuda]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

from kdcrga.config import load_config
from kdcrga.eval.metrics import threshold_metrics
from kdcrga.training.checkpoint import load_checkpoint
from kdcrga.training.negatives import sample_negatives

sys.path.insert(0, str(Path(__file__).parent))
from _sweep_common import build_model, load_data  # noqa: E402

# arm name -> (bench config, model kind). rgcn_mlp and kdcrga_v5 are both the
# KDCRGA model (attention off / readout-MLP on, resp.); decagon/lagat are baselines.
ARMS: dict[str, tuple[str, str]] = {
    "kdcrga_dedicom_v5": ("configs/bench/kdcrga_dedicom_v5.yaml", "kdcrga_dedicom"),
    "decagon": ("configs/bench/decagon.yaml", "decagon"),
    "rgcn_mlp": ("configs/bench/rgcn_mlp.yaml", "kdcrga"),
    "kdcrga_v5": ("configs/bench/kdcrga_v5.yaml", "kdcrga"),
    "kdcrga_dedicom_v4": ("configs/bench/kdcrga_dedicom_v4.yaml", "kdcrga_dedicom"),
    "lagat": ("configs/bench/lagat.yaml", "lagat"),
}


def _to_device(data, dev):
    """Mirror train()'s surgical device move (loop.py): node features, every
    relation's edge_index, and the DDI tensors."""
    data["entity"].x = data["entity"].x.to(dev)
    for et in data.edge_types:
        if et[1].startswith("rel_"):
            data[et].edge_index = data[et].edge_index.to(dev)
    data.ddi.pair_index = data.ddi.pair_index.to(dev)
    data.ddi.side_effect = data.ddi.side_effect.to(dev)
    data.ddi.split = data.ddi.split.to(dev)
    return data


def _all_positives(data) -> set[tuple[int, int, int]]:
    """Positive (drug_a, drug_b, side_effect) triples, from CPU copies (one host
    transfer, not 252k element syncs)."""
    a = data.ddi.pair_index[0].cpu().tolist()
    b = data.ddi.pair_index[1].cpu().tolist()
    se = data.ddi.side_effect.cpu().tolist()
    return {(a[i], b[i], se[i]) for i in range(len(se))}


def _split_scores(model, data, split_id, batch_size, neg_seed, h, positives):
    """(labels, sigmoid-scores, side_effect) for a split with 1:1 sampled negatives."""
    mask = data.ddi.split == split_id
    pairs = data.ddi.pair_index[:, mask]
    se = data.ddi.side_effect[mask]
    neg_pair, neg_se = sample_negatives(pairs, se, int(data.num_drugs), positives,
                                        seed=neg_seed)
    allpair = torch.cat([pairs, neg_pair], dim=1)
    allse = torch.cat([se, neg_se], dim=0)
    labels = torch.cat([torch.ones(se.numel()), torch.zeros(neg_se.numel())])
    chunks = []
    for s in range(0, allpair.size(1), batch_size):
        out = model(data, allpair[:, s:s + batch_size], allse[s:s + batch_size], h=h)
        chunks.append(out.detach().cpu())
    scores = torch.sigmoid(torch.cat(chunks)) if chunks else torch.empty(0)
    return labels, scores, allse.cpu()


def eval_arm(name, config_path, kind, data, positives, quartiles, device):
    cfg = load_config(config_path)
    batch_size = cfg["training"]["batch_size"]
    out: dict[int, dict] = {}
    for ckpt in sorted(glob.glob(f"runs/bench/{name}/seed_*/best.pt")):
        seed = int(Path(ckpt).parent.name.split("_")[-1])
        model = build_model(kind, cfg, data)
        load_checkpoint(ckpt, model)
        model.to(device).eval()
        with torch.no_grad():
            h = model.encode(data)
            vl, vs, _ = _split_scores(model, data, 1, batch_size, 2, h, positives)
            tl, ts, tse = _split_scores(model, data, 2, batch_size, 1, h, positives)
        m = threshold_metrics(vl, vs, tl, ts, tse, quartiles)
        (Path(ckpt).parent / "prf_metrics.json").write_text(json.dumps(m, indent=2))
        out[seed] = m
        print(f"[{name} seed {seed}] thr={m['threshold']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  F1={m['f1']:.3f}", flush=True)
    if not out:
        print(f"[{name}] no checkpoints found under runs/bench/{name}/seed_*/", flush=True)
    return out


def main(arms=None, device="cuda") -> None:
    device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    arms = arms or list(ARMS)
    print(f"device={device}; arms={arms}", flush=True)
    # Cache the (graph, positives, quartiles) by whether ATC is injected: the four
    # ontology arms share one graph, decagon/lagat share the other. Split is fixed
    # (seed 0), so the graph + positives are identical across an arm's seeds.
    cache: dict[bool, tuple] = {}
    for name in arms:
        config_path, kind = ARMS[name]
        cfg = load_config(config_path)
        use_atc = bool(cfg["model"].get("use_ontology_atc", False))
        if use_atc not in cache:
            data = _to_device(load_data(seed=0, cfg=cfg), torch.device(device))
            cache[use_atc] = (data, _all_positives(data),
                              getattr(data, "quartiles", {}))
        data, positives, quartiles = cache[use_atc]
        eval_arm(name, config_path, kind, data, positives, quartiles, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                    help=f"subset of {list(ARMS)}; default all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(arms=args.arms, device=args.device)
