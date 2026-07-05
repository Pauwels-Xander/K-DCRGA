"""Build the committed SE -> MedDRA mapping artifact + match report.

Prerequisites:
  - data/MedDRA_29_0_English/MedAscii/*.asc            (MSSO MedAscii)
  - data/raw/bio-decagon-combo.csv                     (public Decagon dataset)
  - data/meddra_overrides.csv  (optional hand-fixes: se_index,pt_code)

Run: python experiments/build_meddra_mapping.py
"""
from __future__ import annotations

from pathlib import Path

from kdcrga.data.meddra.build import build_mapping

ID2REL = Path("third_party/knowddi/raw_data/BioSNAP/id2relation.json")
DECAGON = Path("data/raw/bio-decagon-combo.csv")
MEDASCII = Path("data/MedDRA_29_0_English/MedAscii")
ARTIFACT = Path("data/processed/se_to_meddra.json")
REPORT = Path("docs/meddra_match_report.md")
OVERRIDES = Path("data/meddra_overrides.csv")


def main() -> None:
    for p in (ID2REL, DECAGON, MEDASCII):
        if not p.exists():
            print(f"MISSING: {p} — cannot build mapping.")
            return
    summary = build_mapping(
        id2relation=ID2REL, decagon_csv=DECAGON, medascii_dir=MEDASCII,
        artifact_path=ARTIFACT, report_path=REPORT,
        overrides_path=OVERRIDES if OVERRIDES.exists() else None, threshold=90)
    print(f"Mapping built: {summary}")
    print(f"Artifact -> {ARTIFACT}\nReport   -> {REPORT}")


if __name__ == "__main__":
    main()
