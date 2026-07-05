"""Resolve BioSNAP side-effect index -> (UMLS CUI, side-effect name).

index -> CUI from KnowDDI's id2relation.json; CUI -> name from the public
Decagon bio-decagon-combo.csv (columns include 'Polypharmacy Side Effect' = CUI
and 'Side Effect Name').
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_se_names(id2relation_path: str | Path, decagon_csv_path: str | Path
                  ) -> dict[int, tuple[str, str]]:
    """Return {se_index: (cui, name)}. SEs whose CUI is absent from the Decagon
    CSV get name '' (still recorded so the report shows them as unmatched)."""
    with open(id2relation_path, encoding="utf-8") as fh:
        idx_to_cui = {int(k): v for k, v in json.load(fh).items()}

    df = pd.read_csv(decagon_csv_path)
    cui_to_name: dict[str, str] = {}
    for cui, name in zip(df["Polypharmacy Side Effect"], df["Side Effect Name"]):
        cui_to_name.setdefault(str(cui), str(name))

    return {idx: (cui, cui_to_name.get(cui, ""))
            for idx, cui in idx_to_cui.items()}
