"""Pretrained biomedical node-embedding initialisation (proposal Sec. 3.4).

BioBERT embeddings of entity Hetionet names. Mol2Vec for drugs is a separate
task; drug rows that resolve to no name stay zero here. The feature builder is
encoder-agnostic (inject `embed_fn`) so tests run without downloading a model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch

from kdcrga.data.entity_names import resolve_entity_names

DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.1"

# Pinned model revisions for reproducible thesis runs (resolved 2026-06-08).
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SAPBERT_REVISION = "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"
MOLFORMER_MODEL = "ibm/MoLFormer-XL-both-10pct"
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"


def load_drug_smiles(id2drug_path: str | Path) -> dict[int, str]:
    """Parse KnowDDI's ``id2drug.json`` into ``{drug_row_id: SMILES}``.

    Drug ids are entity row ids (drugs are entities 0..603). Rows with no SMILES
    string are skipped, so the result keys exactly the drugs we can embed."""
    raw = json.loads(Path(id2drug_path).read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for k, v in raw.items():
        smiles = (v or {}).get("smiles")
        if smiles:
            out[int(k)] = smiles
    return out


def _masked_mean_pool(last_hidden: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-padding tokens. last_hidden [B,T,D], mask [B,T] -> [B,D]."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)   # [B,T,1]
    summed = (last_hidden * mask).sum(dim=1)                     # [B,D]
    counts = mask.sum(dim=1).clamp(min=1.0)                     # [B,1]
    return summed / counts


def biobert_embed(names: list[str], encode_fn: Callable | None = None,
                  batch_size: int = 64, model_name: str = DEFAULT_MODEL
                  ) -> torch.Tensor:
    """Embed names -> [len(names), 768]. With `encode_fn` given, delegate to it
    (used by tests). Otherwise lazy-load BioBERT, eval/no-grad, masked mean-pool.
    """
    if encode_fn is not None:
        return encode_fn(names)

    from transformers import AutoModel, AutoTokenizer  # lazy: only the build needs it

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(names), batch_size):
            batch = names[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=32,
                      return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            pooled = _masked_mean_pool(hidden, enc["attention_mask"])
            out.append(pooled.cpu())
    return torch.cat(out, dim=0) if out else torch.empty(0, model.config.hidden_size)


def _canonicalize_smiles(smiles: list[str]) -> list[str]:
    """RDKit-canonicalize each SMILES; pass an unparseable string through
    unchanged. MoLFormer was trained on canonical SMILES, so this matters."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # silence parse warnings for invalid inputs
    out: list[str] = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(Chem.MolToSmiles(m) if m is not None else s)
    return out


def sapbert_embed(names: list[str], encode_fn: Callable | None = None,
                  batch_size: int = 64, model_name: str = SAPBERT_MODEL,
                  revision: str = SAPBERT_REVISION) -> torch.Tensor:
    """Embed entity-name strings -> [len(names), 768] via SapBERT.

    SapBERT is trained (UMLS synonym alignment) so its **[CLS]** vector is the
    intended entity representation — we use [CLS], not mean-pooling. With
    `encode_fn` given, delegate to it (tests). Otherwise lazy-load SapBERT."""
    if encode_fn is not None:
        return encode_fn(names)

    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(names), batch_size):
            enc = tok(names[i:i + batch_size], padding=True, truncation=True,
                      max_length=32, return_tensors="pt").to(device)
            cls = model(**enc).last_hidden_state[:, 0]  # [CLS]
            out.append(cls.cpu())
    return torch.cat(out, dim=0) if out else torch.empty(0, model.config.hidden_size)


def molformer_embed(smiles: list[str], encode_fn: Callable | None = None,
                    batch_size: int = 64, model_name: str = MOLFORMER_MODEL,
                    revision: str = MOLFORMER_REVISION,
                    canonicalize: bool = True) -> torch.Tensor:
    """Embed SMILES -> [len(smiles), 768] via MoLFormer-XL masked mean-pooling.

    SMILES are RDKit-canonicalized first. `trust_remote_code` is required (the
    model ships its own SMILES tokenizer); the revision is pinned. With
    `encode_fn` given, delegate to it (tests)."""
    if encode_fn is not None:
        return encode_fn(smiles)
    if canonicalize:
        smiles = _canonicalize_smiles(smiles)

    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision,
                                        trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, revision=revision,
                                      deterministic_eval=True,
                                      trust_remote_code=True).to(device).eval()
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            enc = tok(smiles[i:i + batch_size], padding=True,
                      return_tensors="pt").to(device)
            pooled = _masked_mean_pool(model(**enc).last_hidden_state,
                                       enc["attention_mask"])
            out.append(pooled.cpu())
    return torch.cat(out, dim=0) if out else torch.empty(0, model.config.hidden_size)


def build_node_features(names_by_row: dict[int, str], num_entities: int,
                        embed_fn: Callable[[list[str]], torch.Tensor],
                        dim: int = 768) -> tuple[torch.Tensor, dict]:
    """Embed unique names once and scatter to rows; unresolved rows stay zeros.

    Feature width is inferred from `embed_fn`'s output; `dim` is only the
    fallback width when no rows resolve. Returns (features[num_entities, D], stats).
    """
    unique = sorted(set(names_by_row.values()))
    if unique:
        vecs = embed_fn(unique)
        width = vecs.size(1)
        name_to_vec = {n: vecs[i] for i, n in enumerate(unique)}
    else:
        width = dim
        name_to_vec = {}
    feats = torch.zeros(num_entities, width)
    for rid, name in names_by_row.items():
        feats[rid] = name_to_vec[name]
    return feats, {"embedded": len(names_by_row),
                   "zero": num_entities - len(names_by_row),
                   "unique_names": len(unique)}


def build_dual_node_features(
    names_by_row: dict[int, str],
    smiles_by_row: dict[int, str],
    num_entities: int,
    text_embed_fn: Callable[[list[str]], torch.Tensor],
    drug_embed_fn: Callable[[list[str]], torch.Tensor],
    dim: int = 768,
) -> tuple[torch.Tensor, dict]:
    """Embed drug rows from SMILES and the remaining named rows from text into
    one ``[num_entities, D]`` matrix.

    A row present in ``smiles_by_row`` is a drug: it is embedded molecularly via
    ``drug_embed_fn`` and its text name (if any) is ignored. Remaining rows in
    ``names_by_row`` are embedded via ``text_embed_fn``. Rows with neither stay
    zero. Unique strings are embedded once then scattered. Both embedders must
    produce the same width (asserted in production via matching 768-d models)."""
    uniq_smiles = sorted(set(smiles_by_row.values()))
    text_rows = {r: n for r, n in names_by_row.items() if r not in smiles_by_row}
    uniq_names = sorted(set(text_rows.values()))

    drug_width: int | None = None
    text_width: int | None = None
    drug_vecs: dict[str, torch.Tensor] = {}
    text_vecs: dict[str, torch.Tensor] = {}
    if uniq_smiles:
        dv = drug_embed_fn(uniq_smiles)
        drug_width = dv.size(1)
        drug_vecs = {s: dv[i] for i, s in enumerate(uniq_smiles)}
    if uniq_names:
        tv = text_embed_fn(uniq_names)
        text_width = tv.size(1)
        text_vecs = {n: tv[i] for i, n in enumerate(uniq_names)}
    if drug_width is not None and text_width is not None and drug_width != text_width:
        raise ValueError(
            f"drug embedding dim {drug_width} != text embedding dim {text_width}; "
            "both encoders must produce the same width")
    width = drug_width or text_width or dim

    feats = torch.zeros(num_entities, width)
    for r, s in smiles_by_row.items():
        feats[r] = drug_vecs[s]
    for r, n in text_rows.items():
        feats[r] = text_vecs[n]
    return feats, {
        "drugs_embedded": len(smiles_by_row),
        "text_embedded": len(text_rows),
        "zero": num_entities - len(smiles_by_row) - len(text_rows),
        "unique_smiles": len(uniq_smiles),
        "unique_names": len(uniq_names),
    }


def build_node_feature_cache(*, bkg_entity2id_path, hetionet_nodes_path,
                             out_path, report_path, embed_fn) -> dict:
    """Resolve row->name, embed, write the [total_rows, D] cache + a markdown
    coverage report. `embed_fn` is `biobert_embed` in production, a fake in tests.
    Returns the coverage stats dict."""
    names_by_row, res = resolve_entity_names(bkg_entity2id_path, hetionet_nodes_path)
    feats, stats = build_node_features(names_by_row, res.total_rows, embed_fn)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, out_path)
    _write_feature_report(report_path, res, stats, tuple(feats.shape))
    return {**stats, "total_rows": res.total_rows, "collisions": res.collisions}


def build_dual_node_feature_cache(*, bkg_entity2id_path, hetionet_nodes_path,
                                  id2drug_path, out_path, report_path,
                                  text_embed_fn, drug_embed_fn) -> dict:
    """Build the dual node-feature cache: drug rows from SMILES (MoLFormer) and
    the remaining named rows from text (SapBERT), written as one tensor + a
    coverage report. ``text_embed_fn`` / ``drug_embed_fn`` are injected
    (``sapbert_embed`` / ``molformer_embed`` in production, fakes in tests)."""
    names_by_row, res = resolve_entity_names(bkg_entity2id_path, hetionet_nodes_path)
    smiles_by_row = load_drug_smiles(id2drug_path)
    feats, stats = build_dual_node_features(
        names_by_row, smiles_by_row, res.total_rows, text_embed_fn, drug_embed_fn)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, out_path)
    _write_dual_feature_report(report_path, res, stats, tuple(feats.shape))
    return {**stats, "total_rows": res.total_rows, "collisions": res.collisions}


def _write_feature_report(report_path, res, stats, shape) -> None:
    lines = ["# Node embeddings coverage report", "",
             f"- feature tensor shape: {shape[0]} x {shape[1]}",
             f"- rows embedded: {stats['embedded']}",
             f"- rows zero (unresolved): {stats['zero']}",
             f"- unique names embedded: {stats['unique_names']}",
             f"- alias collisions (ids with >1 string): {res.collisions}",
             "", "## Resolved rows by entity kind", "",
             "| kind | count |", "|------|-------|"]
    for kind, n in sorted(res.per_kind.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {kind} | {n} |")
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dual_feature_report(report_path, res, stats, shape) -> None:
    lines = ["# Node embeddings coverage report (SapBERT text + MoLFormer drugs)", "",
             f"- feature tensor shape: {shape[0]} x {shape[1]}",
             f"- drug rows embedded (MoLFormer): {stats['drugs_embedded']}",
             f"- text rows embedded (SapBERT): {stats['text_embedded']}",
             f"- rows zero (no SMILES and no name): {stats['zero']}",
             f"- unique SMILES: {stats['unique_smiles']}",
             f"- unique names: {stats['unique_names']}",
             f"- alias collisions (ids with >1 string): {res.collisions}",
             "", "## Name-resolved rows by entity kind (drugs re-embedded from SMILES)",
             "", "| kind | count |", "|------|-------|"]
    for kind, n in sorted(res.per_kind.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {kind} | {n} |")
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
