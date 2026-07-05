"""Build the ATC injection artifact from the DrugCentral dump (run once).

Joins KnowDDI's 604 drugs (id2drug.json CID / DrugBank ids) through DrugCentral's
identifier -> struct2atc -> atc tables to each drug's ATC L2/L3/L4 ancestor codes,
embeds the ATC level names with SapBERT, and saves
``data/processed/atc_inject.pt`` = {drug_to_atc_ancestors, atc_codes,
atc_features, atc_names}. ``maybe_inject_atc`` loads it (when model.use_ontology_atc
is set) and calls ``inject_atc_into_graph``.

Prerequisites:
  - third_party/knowddi/raw_data/BioSNAP/id2drug.json   (vendored)
  - data/drugcentral.dump.11012023.sql                  (DrugCentral, gitignored)
  - transformers + SapBERT

Run: python experiments/build_atc_inject.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from kdcrga.data.drugcentral import build_drug_atc_ancestors, iter_copy_rows
from kdcrga.data.embeddings import sapbert_embed

ID2DRUG = Path("third_party/knowddi/raw_data/BioSNAP/id2drug.json")
DUMP = Path("data/drugcentral.dump.11012023.sql")
OUT = Path("data/processed/atc_inject.pt")
LEVELS = ("l2", "l3", "l4")   # ancestor levels to add as nodes (matches proposal fig.)


def main() -> None:
    for p in (ID2DRUG, DUMP):
        if not p.exists():
            print(f"MISSING: {p} — cannot build ATC artifact.")
            return
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("transformers not installed. Run: pip install 'transformers>=4.40'")
        return

    id2drug = {int(k): v for k, v in json.loads(ID2DRUG.read_text()).items()}
    print("streaming DrugCentral tables from the dump (identifier, struct2atc, atc)...",
          flush=True)
    identifier = list(iter_copy_rows(DUMP, "identifier"))
    struct2atc = list(iter_copy_rows(DUMP, "struct2atc"))
    atc = list(iter_copy_rows(DUMP, "atc"))
    d2a, names, stats = build_drug_atc_ancestors(
        id2drug, identifier, struct2atc, atc, levels=LEVELS)

    atc_codes = sorted({c for codes in d2a.values() for c in codes})
    print(f"embedding {len(atc_codes)} ATC node names with SapBERT...", flush=True)
    atc_features = (sapbert_embed([names.get(c, c) for c in atc_codes])
                    if atc_codes else torch.empty(0, 768))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"drug_to_atc_ancestors": d2a, "atc_codes": atc_codes,
                "atc_features": atc_features,
                "atc_names": {c: names.get(c, c) for c in atc_codes}}, OUT)

    cov = 100 * stats["drugs_with_atc"] / max(stats["total_drugs"], 1)
    print(f"ATC artifact: {stats['drugs_with_atc']}/{stats['total_drugs']} drugs "
          f"({cov:.1f}%) mapped -> {len(atc_codes)} ATC nodes; "
          f"features {tuple(atc_features.shape)}")
    print(f"Cache -> {OUT}")


if __name__ == "__main__":
    main()
