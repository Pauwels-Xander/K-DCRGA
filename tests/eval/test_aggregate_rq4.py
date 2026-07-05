"""Unit tests for the RQ4 multi-seed aggregator (experiments/aggregate_rq4.py)."""
import importlib.util
from math import isnan, nan
from pathlib import Path

_PATH = Path(__file__).parents[2] / "experiments" / "aggregate_rq4.py"
_spec = importlib.util.spec_from_file_location("aggregate_rq4", _PATH)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def test_within_soc_median_ignores_nan():
    per_soc = {"a": {"rho": 0.5, "p": 0.0}, "b": {"rho": 0.1, "p": 0.0},
               "c": {"rho": nan, "p": 1.0}}
    assert abs(agg.within_soc_median(per_soc) - 0.3) < 1e-9


def test_within_soc_median_all_nan_is_nan():
    assert isnan(agg.within_soc_median({"a": {"rho": nan, "p": 1.0}}))


def _arm(rho, jsd):
    return {"rho": rho, "mean_jsd_overall": jsd,
            "per_soc": {"s1": {"rho": rho, "p": 0.0}}}


def test_summarise_means_min_max_across_seeds():
    per_seed = {
        1: {"trained": _arm(0.08, 0.011), "random": _arm(0.03, 0.004)},
        2: {"trained": _arm(0.13, 0.012), "random": _arm(-0.01, 0.004)},
    }
    out = agg.summarise(per_seed)
    assert out["trained"]["n_seeds"] == 2
    assert abs(out["trained"]["rho_mean"] - 0.105) < 1e-9
    assert out["trained"]["rho_min"] == 0.08
    assert out["trained"]["rho_max"] == 0.13
    assert abs(out["random"]["rho_mean"] - 0.01) < 1e-9


def test_summarise_skips_absent_arms():
    # meddra_init missing entirely -> not in the summary
    out = agg.summarise({1: {"trained": _arm(0.1, 0.01)}})
    assert "trained" in out and "meddra_init" not in out
