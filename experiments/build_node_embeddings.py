"""Build the node-feature cache + coverage report (run once).

Dual-encoder build (decided 2026-06-08):
  - drug rows (entities 0..603)  -> MoLFormer-XL over canonicalized SMILES
  - all other named rows         -> SapBERT over the Hetionet entity name
Both produce 768-d vectors, so the matrix stays uniform. This replaces the
earlier BioBERT-only build and fills the 388 previously-zero drug rows.

Prerequisites:
  - third_party/knowddi/raw_data/BioSNAP/BKG_entity2Id.json        (vendored)
  - third_party/knowddi/raw_data/hetionet/hetionet-v1.0-nodes.tsv  (vendored)
  - third_party/knowddi/raw_data/BioSNAP/id2drug.json              (vendored)
  - transformers + rdkit installed; MoLFormer/SapBERT downloads (pinned revs)

Run: python experiments/build_node_embeddings.py
"""
from __future__ import annotations

from pathlib import Path

from kdcrga.data.embeddings import (
    build_dual_node_feature_cache,
    molformer_embed,
    sapbert_embed,
)

BKG = Path("third_party/knowddi/raw_data/BioSNAP/BKG_entity2Id.json")
HET = Path("third_party/knowddi/raw_data/hetionet/hetionet-v1.0-nodes.tsv")
ID2DRUG = Path("third_party/knowddi/raw_data/BioSNAP/id2drug.json")
OUT = Path("data/processed/node_features.pt")
REPORT = Path("docs/node_embeddings_report.md")


def main() -> None:
    for p in (BKG, HET, ID2DRUG):
        if not p.exists():
            print(f"MISSING: {p} — cannot build node features.")
            return
    try:
        import transformers  # noqa: F401
        from rdkit import Chem  # noqa: F401
    except ImportError as e:
        print(f"missing dependency ({e.name}). Run: pip install 'transformers>=4.40' rdkit")
        return
    stats = build_dual_node_feature_cache(
        bkg_entity2id_path=BKG, hetionet_nodes_path=HET, id2drug_path=ID2DRUG,
        out_path=OUT, report_path=REPORT,
        text_embed_fn=sapbert_embed, drug_embed_fn=molformer_embed)
    print(f"Node features built: {stats}")
    print(f"Cache  -> {OUT}\nReport -> {REPORT}")


if __name__ == "__main__":
    main()
