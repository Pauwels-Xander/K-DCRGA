# K-DCRGA

**Knowledge-Injected Doubly-Conditional Rotatory Graph Attention** for polypharmacy
adverse drug event (ADE) prediction. BSc Econometrics & Operations Research thesis,
Erasmus School of Economics (Xander Pauwels, 2026).

K-DCRGA transfers the HAABSA++ family of aspect-based sentiment models (rotatory
aspect-conditional attention plus medical-ontology knowledge injection) to
heterogeneous biomedical knowledge graphs, and evaluates it against Decagon, LaGAT, and
the subgraph-learning state of the art KnowDDI on the TWOSIDES + Hetionet benchmark.

The thesis is a controlled account of *where* polypharmacy-prediction performance comes
from. The headline finding is that the decoder (a bilinear DEDICOM scorer) and the input
features decide performance, while the side-effect-conditional attention and the ontology
injection add nothing once those are controlled for. This README explains how to set up
the code and reproduce every number, table, and figure in the thesis. The exact mapping
from each thesis result to the command that produced it is in
[`REPRODUCING.md`](REPRODUCING.md).

---

## 1. Requirements

- Python >= 3.10
- A CUDA GPU is required to *train* the full-graph models (a 24 GB card such as an
  RTX 4090 trains one seed in roughly an hour for the baselines and longer for the full
  model). Evaluation, scoring, figures, and the fusion analysis run on CPU.
- The KnowDDI baseline runs in its own pinned environment (PyTorch 1.6 / DGL 0.6,
  CUDA 10.2) and therefore only on Turing/Volta-class GPUs (T4, V100, RTX 2080 Ti); see
  Section 5.4. Everything else uses the modern environment below.

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -e ".[dev,embeddings,viz]"
```

Install a PyTorch build that matches your CUDA version first, then
`pip install torch-geometric`. The optional `torch-scatter`/`torch-sparse` C++
extensions are not required for PyTorch Geometric >= 2.5.

Run the test suite to confirm the install:

```bash
pytest
```

## 3. Data

The benchmark data are not redistributed in this repository; they are obtained as
follows. (The pre-built artifacts below are bundled in the TMS submission zip, so a
reviewer can skip Steps 3.2–3.3 and go straight to Section 4.)

### 3.0 Data sources at a glance

| Dataset | Source | Notes |
| --- | --- | --- |
| TWOSIDES + Hetionet graph, splits & KnowDDI baseline | `github.com/LARS-research/KnowDDI` (pinned SHA in `docs/knowddi_format.md`) | cloned into `third_party/knowddi/` (§3.1) |
| Node-feature models: MoLFormer-XL, SapBERT | HuggingFace; repo IDs + revisions pinned in `src/kdcrga/data/embeddings.py` | auto-downloaded by `transformers` (§3.2) |
| `bio-decagon-combo.csv` (side-effect labels) | Stanford SNAP: `snap.stanford.edu/decagon` | public; place in `data/raw/` (§3.3) |
| MedDRA MedAscii distribution | `meddra.org` (MSSO licence) | licensed, **not** redistributed; only needed to rebuild the ontology artifacts (§3.3) |
| DrugCentral dump (`drugcentral.dump.11012023.sql`, ~4.7 GB) | `drugcentral.org/download` | only needed for the ATC injection arm; place in `data/` |

All inputs sit under the gitignored `data/` and `third_party/` trees, so nothing above is
committed to this repository. The derived artifacts they produce
(`data/processed/node_features.pt`, `zk_init.pt`, `se_to_meddra.json`) are bundled in the
TMS submission archive so the build steps can be skipped; the two MedDRA-derived artifacts
cannot be redistributed here because they embed licensed MedDRA term names.

### 3.1 Vendored KnowDDI graph and splits
K-DCRGA reuses KnowDDI's preprocessed TWOSIDES + Hetionet graph, its train/validation/
test split, and the KnowDDI baseline itself. Clone KnowDDI at its pinned commit into
`third_party/knowddi/`:

```bash
git clone https://github.com/LARS-research/KnowDDI third_party/knowddi
# checkout the pinned commit recorded in docs/knowddi_format.md
```

This provides `third_party/knowddi/data/BioSNAP/{train,valid,test}.txt` (the 604-drug,
200-side-effect TWOSIDES slice) and `third_party/knowddi/raw_data/` (Hetionet nodes,
entity-id maps).

### 3.2 Node features (built once)
```bash
python experiments/build_node_embeddings.py     # -> data/processed/node_features.pt
```
Drug nodes are embedded with MoLFormer-XL over canonical SMILES; all other nodes with
SapBERT over their Hetionet names (768-dimensional).

### 3.3 Ontology artifacts (built once; needed only for the ontology arms)
```bash
python experiments/build_meddra_mapping.py       # -> data/processed/se_to_meddra.json
python experiments/build_zk_init.py              # -> data/processed/zk_init.pt
python experiments/build_atc_inject.py           # -> ATC injection data
```
`build_meddra_mapping.py` requires a licensed MedDRA MedAscii distribution and the public
`bio-decagon-combo.csv`. **MedDRA is licensed and is not redistributed.** The two derived
artifacts (`se_to_meddra.json`, `zk_init.pt`) are bundled in the submission zip so that
these steps can be skipped.

## 4. Reproducing the thesis results

Each command below writes to `runs/<name>/seed_<N>/`. After the sweeps, the table and
figure generators read those run directories. The full result-to-command map is in
[`REPRODUCING.md`](REPRODUCING.md); the summary is:

```bash
# RQ1 + RQ3: matched in-distribution benchmark and ablation (3 seeds)
python experiments/sweep_bench.py --resume --all-seeds

# RQ4: attention vs MedDRA-hierarchy analysis (inference only, no retraining)
python experiments/sweep_rq4.py
python experiments/aggregate_rq4.py

# RQ5: zero-shot to held-out side effects (seed 0)
python experiments/sweep_rq5_zero_shot.py --seed 0

# Tables (-> docs/results_*.tex) and figures (-> docs/figures/)
python experiments/compile_results.py
python experiments/aggregate_prf.py
python experiments/make_figures.py
```

The state-of-the-art comparison and the K-DCRGA + KnowDDI fusion (thesis Table on the
pair-disjoint benchmark) use a separate, leak-free pipeline; see Section 5 of
[`REPRODUCING.md`](REPRODUCING.md).

`--resume` skips any `(config, seed)` whose `test_metrics.json` already exists and
continues a crashed run from its last checkpoint, so a sweep can be relaunched safely.
Use `--smoke` for a one-epoch plumbing check.

## 5. Repository layout

```text
src/kdcrga/          model, data, training, and evaluation code (the installed package)
  data/              graph construction, BioSNAP loading, MedDRA/ATC ontologies, features
  models/            R-GCN encoder, doubly-conditional attention, decoders, baselines
  training/          training loop, loss, negative sampling, checkpointing
  eval/              metrics, prediction dumping, aggregation, figures, attention analysis
  fusion/            two-stream K-DCRGA + KnowDDI stacking ensemble
configs/             YAML configs: base.yaml + per-experiment overrides; bench/ and ablations/
experiments/         entry-point scripts that bind configs to runs (Section 4)
tests/               pytest unit and sanity tests
docs/                thesis source (thesis.tex), proposal, design specs, generated tables
third_party/knowddi/ vendored KnowDDI graph, splits, and baseline (cloned at setup)
data/                raw / processed inputs (gitignored; built in Section 3)
runs/                sweep outputs: runs/<name>/seed_<N>/ (gitignored)
```

Configuration is layered: every config sets `base:` to a parent and overrides only what
changes, so the single-variable contrasts in the thesis differ by exactly one field.

## 6. Tests

```bash
pytest                      # full suite
pytest tests/eval           # a subset
```

## 7. Citation

If you use this code, please cite the thesis (`docs/thesis.tex`). The work builds on
KnowDDI (Wang et al., 2024), Decagon (Zitnik et al., 2018), LaGAT (Hong et al., 2022),
and the HAABSA++ line (Truşcă et al., 2020).
