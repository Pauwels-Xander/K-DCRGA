"""Build the TMS reproducibility archive: kdcrga_submission.zip.

Bundles everything needed to reproduce the thesis results:
  * the code (src, experiments, configs, tests) and docs,
  * the derived data artifacts (node features, z_k init, SE->MedDRA map),
  * the benchmark splits (KnowDDI's public preprocessed BioSNAP/Hetionet),
  * the recorded run metrics (the JSON files the table/figure generators read).

With these, a reviewer can regenerate every thesis table and figure on CPU in
seconds (compile_results.py / make_figures.py / aggregate_rq4.py / sweep_fusion.py),
and, given a GPU, re-run the full training pipeline.

Deliberately EXCLUDED (large and regenerable, or licensed):
  * trained checkpoints (runs/**/best.pt, last.pt) and prediction dumps
    (*.pt, *.npz) -- about 2 GB; regenerate by training, or fetch from the repo;
  * the virtualenv, git history, __pycache__, staging tarballs;
  * the licensed raw MedDRA distribution (the derived artifacts are included instead).

Run: python experiments/package_submission.py
"""
from __future__ import annotations

import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kdcrga_submission.zip"

INCLUDE_DIRS = ["src", "experiments", "configs", "tests"]
INCLUDE_FILES = ["pyproject.toml", "README.md", "REPRODUCING.md", "LICENSE",
                 "docs/thesis.tex"]
INCLUDE_DATA = ["data/processed/node_features.pt",
                "data/processed/zk_init.pt",
                "data/processed/se_to_meddra.json"]
INCLUDE_GLOBS = ["third_party/knowddi/data/BioSNAP/*"]
# Recorded run outputs: metrics only. Tables and figures regenerate from these.
RUN_METRIC_NAMES = {"test_metrics.json", "per_se_metrics.json", "history.json",
                    "aggregate.json", "rq4_results.json", "fusion_metrics.json"}
EXCLUDE = ["*.pyc", "*__pycache__*"]


def _excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in EXCLUDE)


def _add(zf: zipfile.ZipFile, path: Path) -> int:
    rel = path.relative_to(ROOT).as_posix()
    if _excluded(rel):
        return 0
    zf.write(path, rel)
    return 1


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in INCLUDE_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file():
                    n += _add(zf, p)
        for f in INCLUDE_FILES + INCLUDE_DATA:
            p = ROOT / f
            if p.is_file():
                n += _add(zf, p)
        for g in INCLUDE_GLOBS:
            for p in sorted(ROOT.glob(g)):
                if p.is_file():
                    n += _add(zf, p)
        runs = ROOT / "runs"
        if runs.exists():
            for p in sorted(runs.rglob("*")):
                if p.is_file() and p.name in RUN_METRIC_NAMES:
                    n += _add(zf, p)
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.name}: {n} files, {size_mb:.1f} MB")
    print("Excluded by design: trained checkpoints / prediction dumps (~2 GB),"
          " virtualenv, git history, raw MedDRA. See the module docstring.")


if __name__ == "__main__":
    main()
