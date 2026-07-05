"""Build the MedDRA-derived z_k init cache (Eq. zk-init) — run once.

Embeds each side effect's MedDRA PT term + its HLT/HLGT/SOC ancestor names with
SapBERT, averages in embedding space, and projects to the model z_k width with a
fixed seeded projection P. Saves data/processed/zk_init.pt
[num_side_effects, d_z], which build_model loads when model.use_ontology_zk is set.

Prerequisites:
  - data/processed/se_to_meddra.json              (build_meddra_mapping.py)
  - data/MedDRA_29_0_English/MedAscii/mdhier.asc  (licensed MedDRA distribution)
  - transformers installed + SapBERT download

Run: python experiments/build_zk_init.py
"""
from __future__ import annotations

from pathlib import Path

import torch

from kdcrga.data.embeddings import sapbert_embed
from kdcrga.data.meddra.artifact import load_mapping
from kdcrga.data.meddra.ascii import load_meddra_term_names
from kdcrga.data.ontology import build_meddra_zk_init

MAPPING = Path("data/processed/se_to_meddra.json")
MDHIER = Path("data/MedDRA_29_0_English/MedAscii/mdhier.asc")
OUT = Path("data/processed/zk_init.pt")

D_Z = 128       # model hidden_dim (configs/base.yaml) == z_k width
NUM_SE = 200    # BioSNAP side effects
SEED = 0        # fixes the projection P for reproducibility


def main() -> None:
    for p in (MAPPING, MDHIER):
        if not p.exists():
            print(f"MISSING: {p} — cannot build z_k init.")
            return
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("transformers not installed. Run: pip install 'transformers>=4.40'")
        return
    mapping = load_mapping(MAPPING)
    term_names = load_meddra_term_names(MDHIER)
    z = build_meddra_zk_init(mapping, term_names, num_side_effects=NUM_SE,
                             d_z=D_Z, text_embed_fn=sapbert_embed, seed=SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z, OUT)
    nonzero = int((z.abs().sum(1) > 0).sum())
    print(f"z_k init built: shape {tuple(z.shape)}, "
          f"{nonzero}/{NUM_SE} side effects initialised from MedDRA")
    print(f"Cache -> {OUT}")


if __name__ == "__main__":
    main()
