"""RQ4 attention analysis (proposal Sec. 4.10.3).

Does the doubly-conditional attention vary across side effects in a way that
tracks the MedDRA hierarchy? We take a TRAINED K-DCRGA checkpoint (no retraining)
and, for a sample of test drug pairs, read the hop-1 A-side attention distribution
over each drug's neighbours under every one of the 200 side effects. The pairwise
Jensen-Shannon divergence between two side effects' attention distributions
(averaged over the sampled pairs) is then correlated (Spearman) with the MedDRA
tree distance between those side effects, overall and stratified by SOC.

The side effect enters the attention ONLY through ``W_z z_k`` (attention.py), so we
isolate where any structure comes from by swapping the ``z_k`` table post-hoc on the
same frozen weights -- exactly the RQ5 trick:
  trained      -- the learned z_k (MedDRA-initialised, then trained)
  meddra_init  -- the SapBERT MedDRA ancestor-average init, frozen (no training)
  random       -- a frozen-random z_k at the init scale (the floor)
If trained > meddra_init > random, training adds SE-structure beyond what the
ontology init already bakes in; if random ~ 0, the structure is real, not an
artefact of the neighbourhoods.

Run: python experiments/sweep_rq4.py [--ckpt ...] [--max-pairs N] [--verify]
Inference-only; results -> runs/rq4/<model>/seed_<n>/rq4_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from kdcrga.config import load_config
from kdcrga.data.meddra.artifact import load_mapping
from kdcrga.data.meddra.hierarchy import build_se_distance_matrix, build_soc_of
from kdcrga.eval.attention_analysis import (
    jsd_matrix,
    pair_attention_distributions,
    spearman_per_soc,
    spearman_vs_meddra_distance,
)
from kdcrga.training.checkpoint import load_checkpoint

sys.path.insert(0, str(Path(__file__).parent))
from _sweep_common import build_model, load_data  # noqa: E402

_MAPPING = Path("data/processed/se_to_meddra.json")
_ZK_INIT = Path("data/processed/zk_init.pt")
DEFAULT_CONFIG = "configs/bench/kdcrga_dedicom_v5.yaml"
DEFAULT_CKPT = "runs/bench/kdcrga_dedicom_v5/seed_1/best.pt"


def sample_pairs(data, min_neighbours: int, max_pairs: int, seed: int
                 ) -> list[tuple[int, int]]:
    """Unique test-split drug pairs (both orderings, so each drug gets read as the
    focal A-side) whose focal drug has >= ``min_neighbours`` neighbours. Sampled
    deterministically down to ``max_pairs``."""
    from kdcrga.data.graph import build_neighbour_cache
    if not hasattr(data, "neighbour_index"):
        build_neighbour_cache(data, int(data.num_drugs))
    nbr = data.neighbour_index
    test_mask = data.ddi.split == 2
    pi = data.ddi.pair_index[:, test_mask]
    seen: set[tuple[int, int]] = set()
    cand: list[tuple[int, int]] = []
    for i in range(pi.size(1)):
        a, b = int(pi[0, i]), int(pi[1, i])
        for x, y in ((a, b), (b, a)):
            if (x, y) in seen:
                continue
            seen.add((x, y))
            if len(nbr.get(x, [])) >= min_neighbours:
                cand.append((x, y))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(cand), generator=g).tolist()
    return [cand[i] for i in perm[:max_pairs]]


def mean_jsd_matrix(model, data, h, pairs, side_effects):
    """Average the per-pair [K, K] attention-JSD matrix over the sampled pairs.
    Pairs whose focal drug has fewer than two neighbours give a degenerate
    distribution and are skipped. Returns (mean_jsd [K, K], n_pairs_used)."""
    k = int(side_effects.numel())
    total = torch.zeros(k, k)
    used = 0
    for a, b in pairs:
        dist = pair_attention_distributions(model, data, h, a, b, side_effects)
        if dist is None or dist.size(1) < 2:
            continue
        total += jsd_matrix(dist).cpu()
        used += 1
    return (total / max(used, 1)), used


def jsd_pair_list(mean_jsd):
    """Upper-triangle (k1<k2, jsd) tuples for the Spearman helpers."""
    k = mean_jsd.size(0)
    return [(i, j, float(mean_jsd[i, j]))
            for i in range(k) for j in range(i + 1, k)]


def _zk_tables(model, num_se, dim, seed):
    """The three z_k conditions, each as a [num_se, dim] tensor."""
    tables = {"trained": model.z_k.weight.detach().clone()}
    g = torch.Generator().manual_seed(seed)
    tables["random"] = torch.normal(0.0, 0.1, size=(num_se, dim), generator=g)
    if _ZK_INIT.exists():
        z = torch.load(_ZK_INIT, map_location="cpu")
        if tuple(z.shape) == (num_se, dim):
            tables["meddra_init"] = z
        else:
            print(f"WARNING: {_ZK_INIT} shape {tuple(z.shape)} != "
                  f"({num_se}, {dim}); skipping meddra_init arm", flush=True)
    else:
        print(f"WARNING: {_ZK_INIT} missing -> skipping meddra_init arm", flush=True)
    return tables


def main(config_path=DEFAULT_CONFIG, ckpt=DEFAULT_CKPT, seed=1, runs_root="runs",
         max_pairs=200, min_neighbours=10, device="cpu", verify=False,
         sample_seed=0) -> None:
    cfg = load_config(config_path)
    device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    # ``seed`` only labels the output dir / identifies the trained checkpoint;
    # ``sample_seed`` drives the (drug-pair sample, random-control z_k) so they can
    # be held FIXED across checkpoint seeds -> a clean cross-seed comparison where
    # only the trained weights vary.
    torch.manual_seed(sample_seed)

    # Split is fixed at seed 0 across the whole benchmark (only model init varies
    # with the run seed), so the checkpoint matches the seed-0 split graph.
    data = load_data(seed=0, cfg=cfg)
    model = build_model("kdcrga_dedicom", cfg, data)
    load_checkpoint(ckpt, model)
    model.to(device).eval()
    print(f"loaded {ckpt}", flush=True)

    num_se = int(data.num_side_effects)
    dim = model.hidden_dim
    side_effects = torch.arange(num_se)

    if verify:
        from kdcrga.eval.run import evaluate_test_set
        agg, _ = evaluate_test_set(model, data, batch_size=1024, device=device)
        print(f"[verify] reloaded checkpoint test AUROC={agg['auroc']:.4f} "
              f"AUPRC={agg['auprc']:.4f} (expect ~ the recorded run value)",
              flush=True)

    mapping = load_mapping(_MAPPING)
    distances = build_se_distance_matrix(mapping)
    soc_of = build_soc_of(mapping)
    print(f"MedDRA: {len(mapping)}/{num_se} side effects matched; "
          f"{len(distances)} SE-pair distances", flush=True)

    pairs = sample_pairs(data, min_neighbours, max_pairs, sample_seed)
    print(f"sampled {len(pairs)} test drug pairs "
          f"(min_neighbours={min_neighbours}, sample_seed={sample_seed})", flush=True)

    z_orig = model.z_k.weight.detach().clone()
    tables = _zk_tables(model, num_se, dim, sample_seed)

    with torch.no_grad():
        h = model.encode(data).to(device)
        results: dict[str, dict] = {}
        for arm in ("trained", "meddra_init", "random"):
            if arm not in tables:
                continue
            model.z_k.weight.copy_(tables[arm].to(device))
            mean_jsd, used = mean_jsd_matrix(model, data, h, pairs, side_effects)
            pair_list = jsd_pair_list(mean_jsd)
            rho, p = spearman_vs_meddra_distance(pair_list, distances)
            per_soc = {s: {"rho": r, "p": pv, }
                       for s, (r, pv) in
                       spearman_per_soc(pair_list, distances, soc_of).items()
                       if r == r}  # drop NaN (too few within-SOC pairs)
            results[arm] = {
                "rho": rho, "p": p, "n_pairs_used": used,
                "mean_jsd_overall": float(mean_jsd[mean_jsd > 0].mean())
                if (mean_jsd > 0).any() else 0.0,
                "per_soc": per_soc,
            }
            print(f"[{arm:12s}] Spearman(JSD, MedDRA dist) rho={rho:+.4f} "
                  f"p={p:.2e}  (pairs={used}, within-SOC SOCs={len(per_soc)})",
                  flush=True)
        model.z_k.weight.copy_(z_orig.to(device))

    model_name = Path(config_path).stem
    out_dir = Path(runs_root) / "rq4" / model_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(ckpt),
        "config": config_path,
        "seed": seed,
        "sample_seed": sample_seed,
        "max_pairs": max_pairs,
        "min_neighbours": min_neighbours,
        "n_side_effects": num_se,
        "n_matched_meddra": len(mapping),
        "arms": results,
    }
    (out_dir / "rq4_results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'rq4_results.json'}", flush=True)

    if "trained" in results and "random" in results:
        d = results["trained"]["rho"] - results["random"]["rho"]
        print(f"\n=== RQ4 summary ===\n  trained rho   = {results['trained']['rho']:+.4f}"
              f"\n  random  rho   = {results['random']['rho']:+.4f}"
              f"\n  trained-random = {d:+.4f}", flush=True)
        if "meddra_init" in results:
            print(f"  meddra_init rho = {results['meddra_init']['rho']:+.4f}",
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--seed", type=int, default=1,
                    help="checkpoint seed; labels the output dir runs/rq4/.../seed_<seed>")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="seed for the drug-pair sample + random-control z_k; hold "
                         "fixed across checkpoint seeds for a clean comparison")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--min-neighbours", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verify", action="store_true",
                    help="re-evaluate the test set to confirm the checkpoint loaded")
    args = ap.parse_args()
    main(args.config, ckpt=args.ckpt, seed=args.seed, runs_root=args.runs_root,
         max_pairs=args.max_pairs, min_neighbours=args.min_neighbours,
         device=args.device, verify=args.verify, sample_seed=args.sample_seed)
