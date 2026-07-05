"""Per-seed two-stream fusion + multi-seed aggregation.

Reads the four per-seed dumps, fuses (Tier 0 average + Tier 1 stackers fit on the
aligned validation tables), re-scores F / KnowDDI / fused through the common
evaluator, and tests the fused-vs-KnowDDI margin with a paired Wilcoxon over the
per-side-effect scores. Writes runs/fusion/seed_<s>/fusion_metrics.json and an
aggregate.json (mean +/- s.d. across seeds). Test set is scored once here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from kdcrga.eval.significance import wilcoxon_paired
from kdcrga.fusion.knowddi_adapter import knowddi_npz_to_table
from kdcrga.fusion.rescore import per_se_metric, score_table
from kdcrga.fusion.stacker import fit_stacker, fuse_average
from kdcrga.fusion.tables import PredTable, align, load_table


def _fused_table(ref: PredTable, prob: torch.Tensor) -> PredTable:
    return PredTable(ref.drug_a, ref.drug_b, ref.side_effect, ref.label, prob, "prob")


def _keyset(t: PredTable) -> set[tuple[int, int, int]]:
    return set(zip(t.drug_a.tolist(), t.drug_b.tolist(), t.side_effect.tolist()))


def _assert_identical_coverage(f: PredTable, s: PredTable, what: str) -> int:
    """Assert both streams cover an identical eval set, comparing the SET of
    (drug_a, drug_b, side_effect) keys -- NOT raw row counts.

    BioSNAP split files legitimately contain a handful of duplicate triples (a
    (pair, side_effect) that appears as both a sampled positive and a sampled
    negative row). Both streams reproduce them identically, and ``align`` collapses
    them consistently, so equal key SETS -- not equal row counts -- is the correct
    "identical coverage" invariant. Returns the number of duplicate rows ``align``
    will collapse (0 when the split has no duplicate keys)."""
    kf, ks = _keyset(f), _keyset(s)
    only_f, only_s = kf - ks, ks - kf
    if only_f or only_s:
        raise ValueError(
            f"{what}-set coverage mismatch: {len(only_f)} keys only in Stream F, "
            f"{len(only_s)} only in Stream S. The two streams must cover an "
            f"identical eval set."
        )
    return len(f.label) - len(kf)


def fuse_seed(f_test: PredTable, s_test: PredTable, f_val: PredTable,
              s_val: PredTable, quartiles: dict[int, int]) -> dict:
    dup_test = _assert_identical_coverage(f_test, s_test, "Test")
    ft, st = align(f_test, s_test)
    dup_val = _assert_identical_coverage(f_val, s_val, "Val")
    fv, sv = align(f_val, s_val)

    f_agg, f_per = score_table(ft, quartiles)
    k_agg, k_per = score_table(st, quartiles)

    candidates: dict[str, torch.Tensor] = {
        "avg_logit": fuse_average(fv, sv, mode="logit"),
        "stack_global": fit_stacker(fv, sv, gating="global").apply(fv, sv),
        "stack_quartile": fit_stacker(fv, sv, gating="quartile",
                                      quartiles=quartiles).apply(fv, sv),
    }
    # Select the fusion variant on VALIDATION AUPRC, then apply it to test.
    val_auprc = {}
    fitted: dict[str, object] = {
        "avg_logit": None,
        "stack_global": fit_stacker(fv, sv, gating="global"),
        "stack_quartile": fit_stacker(fv, sv, gating="quartile", quartiles=quartiles),
    }
    for name, prob in candidates.items():
        agg, _ = score_table(_fused_table(fv, prob), quartiles)
        val_auprc[name] = agg["auprc"]
    best = max(val_auprc, key=val_auprc.get)

    if best == "avg_logit":
        test_prob = fuse_average(ft, st, mode="logit")
    else:
        test_prob = fitted[best].apply(ft, st)   # type: ignore[union-attr]
    fused_agg, fused_per = score_table(_fused_table(ft, test_prob), quartiles)

    # fused vs KnowDDI (weaker stream) -- sanity; fused contains it so expect tiny p
    _, p_auroc = wilcoxon_paired(per_se_metric(fused_per, "auroc"),
                                 per_se_metric(k_per, "auroc"))
    _, p_auprc = wilcoxon_paired(per_se_metric(fused_per, "auprc"),
                                 per_se_metric(k_per, "auprc"))
    # fused vs K-DCRGA (stronger stream) -- the test that answers "does fusion add
    # anything beyond the best single model?"
    _, p_auroc_f = wilcoxon_paired(per_se_metric(fused_per, "auroc"),
                                   per_se_metric(f_per, "auroc"))
    _, p_auprc_f = wilcoxon_paired(per_se_metric(fused_per, "auprc"),
                                   per_se_metric(f_per, "auprc"))
    # K-DCRGA vs KnowDDI -- THE headline comparison (the SOTA claim). Per-SE paired
    # test of the two single models, independent of any fusion.
    _, p_fk_auroc = wilcoxon_paired(per_se_metric(f_per, "auroc"),
                                    per_se_metric(k_per, "auroc"))
    _, p_fk_auprc = wilcoxon_paired(per_se_metric(f_per, "auprc"),
                                    per_se_metric(k_per, "auprc"))
    return {
        "f_auroc": f_agg["auroc"], "f_auprc": f_agg["auprc"],
        "knowddi_auroc": k_agg["auroc"], "knowddi_auprc": k_agg["auprc"],
        "fused_auroc": fused_agg["auroc"], "fused_auprc": fused_agg["auprc"],
        "fused_variant": best, "val_auprc_by_variant": val_auprc,
        "wilcoxon_auroc_p": p_auroc, "wilcoxon_auprc_p": p_auprc,
        "wilcoxon_vs_f_auroc_p": p_auroc_f, "wilcoxon_vs_f_auprc_p": p_auprc_f,
        "wilcoxon_f_vs_knowddi_auroc_p": p_fk_auroc,
        "wilcoxon_f_vs_knowddi_auprc_p": p_fk_auprc,
        "n_test_triples": int(len(ft.label)), "n_val_triples": int(len(fv.label)),
        "dup_rows_collapsed_test": int(dup_test),
        "dup_rows_collapsed_val": int(dup_val),
    }


def summarize(per_seed: list[dict]) -> dict:
    """Aggregate per-seed fusion rows into means, margins, and beat/tie/lose
    verdicts. The headline ``f_vs_knowddi`` compares the two single models
    (K-DCRGA vs KnowDDI, the SOTA claim); ``fused_vs_f`` asks whether the ensemble
    adds anything over K-DCRGA alone. Works for any seed count (n=1 -> go/no-go:
    margin_std is 0, so ``robust`` reduces to "margin positive").

    Verdict gate -- BEAT requires the margin to be positive in EVERY seed, the
    per-SE Wilcoxon significant (p<0.05) in EVERY seed, AND the mean margin to
    exceed its own across-seed s.d. (so a margin within seed noise -> TIE)."""
    keys = ("f_auroc", "f_auprc", "knowddi_auroc", "knowddi_auprc",
            "fused_auroc", "fused_auprc")
    metrics = {k: {"mean": float(np.mean([r[k] for r in per_seed])),
                   "std": float(np.std([r[k] for r in per_seed]))} for k in keys}

    def _cmp(pa: str, pb: str, p_tmpl: str) -> dict:
        out: dict[str, dict] = {}
        for m in ("auroc", "auprc"):
            margins = [r[f"{pa}_{m}"] - r[f"{pb}_{m}"] for r in per_seed]
            ps = [r[p_tmpl.format(m=m)] for r in per_seed]
            mean, std = float(np.mean(margins)), float(np.std(margins))
            out[m] = {
                "margin_mean": mean, "margin_std": std,
                "per_seed_margin": [float(x) for x in margins],
                "wilcoxon_p_per_seed": [float(p) for p in ps],
                "max_p": float(max(ps)),
                "all_positive": all(x > 0 for x in margins),
                "all_negative": all(x < 0 for x in margins),
                "significant": all(p < 0.05 for p in ps),
                "robust": mean > std,
            }
        return out

    fk = _cmp("f", "knowddi", "wilcoxon_f_vs_knowddi_{m}_p")
    for m in ("auroc", "auprc"):
        d = fk[m]
        d["verdict"] = ("beat" if d["all_positive"] and d["significant"] and d["robust"]
                        else "lose" if d["all_negative"] and d["significant"]
                        else "tie")
    ff = _cmp("fused", "f", "wilcoxon_vs_f_{m}_p")
    for m in ("auroc", "auprc"):
        d = ff[m]
        d["verdict"] = ("adds" if d["all_positive"] and d["significant"] and d["robust"]
                        else "neutral")
    return {"n_seeds": len(per_seed), "metrics": metrics,
            "f_vs_knowddi": fk, "fused_vs_f": ff}


def run(seeds: list[int], runs_root: str | Path = "runs/fusion") -> None:
    from experiments._sweep_common import load_data
    from kdcrga.config import load_config
    cfg = load_config("configs/bench/kdcrga_dedicom_v5.yaml")
    quartiles = load_data(seed=0, cfg=cfg).quartiles

    root = Path(runs_root)
    per_seed: list[dict] = []
    for s in seeds:
        d = root / f"seed_{s}"
        res = fuse_seed(
            load_table(d / "f_test.pt"),
            knowddi_npz_to_table(str(d / "knowddi_test.npz")),
            load_table(d / "f_val.pt"),
            knowddi_npz_to_table(str(d / "knowddi_valid.npz")),
            quartiles)
        (d / "fusion_metrics.json").write_text(json.dumps(res, indent=2))
        per_seed.append(res)
        print(f"[seed {s}] fused AUROC={res['fused_auroc']:.4f} "
              f"(K-DCRGA {res['f_auroc']:.4f}, KnowDDI {res['knowddi_auroc']:.4f}) "
              f"variant={res['fused_variant']} "
              f"p(fused vs K-DCRGA)={res['wilcoxon_vs_f_auroc_p']:.2e}")

    summary = summarize(per_seed)
    (root / "aggregate.json").write_text(json.dumps(summary, indent=2))
    mt = summary["metrics"]
    fk, ff = summary["f_vs_knowddi"], summary["fused_vs_f"]
    print(f"--- {summary['n_seeds']} seed(s) ---")
    print(f"K-DCRGA  AUROC {mt['f_auroc']['mean']:.4f}+/-{mt['f_auroc']['std']:.4f}  "
          f"AUPRC {mt['f_auprc']['mean']:.4f}+/-{mt['f_auprc']['std']:.4f}")
    print(f"KnowDDI  AUROC {mt['knowddi_auroc']['mean']:.4f}+/-{mt['knowddi_auroc']['std']:.4f}"
          f"  AUPRC {mt['knowddi_auprc']['mean']:.4f}+/-{mt['knowddi_auprc']['std']:.4f}")
    print(f"Fused    AUROC {mt['fused_auroc']['mean']:.4f}+/-{mt['fused_auroc']['std']:.4f}  "
          f"AUPRC {mt['fused_auprc']['mean']:.4f}+/-{mt['fused_auprc']['std']:.4f}")
    print(f"VERDICT  K-DCRGA vs KnowDDI: "
          f"AUROC={fk['auroc']['verdict'].upper()} "
          f"(margin {fk['auroc']['margin_mean']:+.4f}+/-{fk['auroc']['margin_std']:.4f}, "
          f"max p={fk['auroc']['max_p']:.1e}) | "
          f"AUPRC={fk['auprc']['verdict'].upper()} "
          f"(margin {fk['auprc']['margin_mean']:+.4f}+/-{fk['auprc']['margin_std']:.4f}, "
          f"max p={fk['auprc']['max_p']:.1e})")
    print(f"         Fusion over K-DCRGA alone: "
          f"AUROC={ff['auroc']['verdict']} | AUPRC={ff['auprc']['verdict']}")


if __name__ == "__main__":
    import sys
    run([int(x) for x in (sys.argv[1:] or ["0", "1", "2"])])
