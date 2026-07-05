"""Biomedical ontology injection (proposal Sec. 4.3).

MedDRA ancestor lookup for side-effect query init z_k (eq. eq:zk-init) and
ATC-ancestor node injection into the graph via a new ``drug-belongsTo-ATC``
relation (proposal Fig. fig:ontology). Filled in stage v5.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def meddra_zk_init(
    side_effects: list[int],
    ancestors: dict[int, list[int]],
    raw_embeddings: torch.Tensor,
    se_to_row: dict[int, int],
    anc_to_row: dict[int, int],
    d_z: int,
    project: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Eq. eq:zk-init: average own + ancestor embeddings, optionally projected.

    Data-source agnostic: callers supply the embedding matrix (e.g. BioBERT of
    MedDRA entity descriptions) plus the id->row maps. Ancestors that have no
    row in `anc_to_row` are silently skipped (treated as "unknown").
    """
    rows: list[torch.Tensor] = []
    for k in side_effects:
        own = raw_embeddings[se_to_row[k]]
        anc_rows = [raw_embeddings[anc_to_row[a]]
                    for a in ancestors.get(k, []) if a in anc_to_row]
        if anc_rows:
            avg = (own + torch.stack(anc_rows).sum(0)) / (1 + len(anc_rows))
        else:
            avg = own
        rows.append(avg)
    z = torch.stack(rows)
    if project is not None:
        z = project(z)
    return z


def build_meddra_zk_init(
    mapping,
    term_names: dict[str, str],
    num_side_effects: int,
    d_z: int,
    text_embed_fn,
    project: Optional[object] = None,
    seed: int = 0,
) -> torch.Tensor:
    """Eq. eq:zk-init for every side effect (the headline-model cold-start init).

    For each side effect in ``mapping`` (``{se_index: MeddraEntry}``), embed its
    PT term plus its HLT/HLGT/SOC ancestor names via ``text_embed_fn`` (SapBERT in
    production), average in embedding space, then project to ``d_z`` with a fixed
    projection ``P``. Term strings come from ``term_names`` (``{code: name}``).

    Side effects absent from ``mapping`` get a zero row, leaving their ``z_k``
    trainable from zero. When ``project`` is None a fixed, seeded, frozen
    ``Linear(D, d_z)`` is built (the proposal's fixed ``P``); pass a callable to
    inject one (tests). Returns ``Tensor[num_side_effects, d_z]``."""
    ses = sorted(mapping)
    own_name = {se: (mapping[se].pt_name or term_names.get(mapping[se].pt_code, ""))
                for se in ses}
    anc_codes = {se: [mapping[se].hlt, mapping[se].hlgt, mapping[se].soc]
                 for se in ses}

    uniq = sorted(
        {own_name[se] for se in ses if own_name[se]}
        | {term_names[c] for se in ses for c in anc_codes[se] if term_names.get(c)}
    )
    if not uniq:
        return torch.zeros(num_side_effects, d_z)
    raw = text_embed_fn(uniq)
    name_to_row = {n: i for i, n in enumerate(uniq)}

    se_to_row = {se: name_to_row[own_name[se]] for se in ses
                 if own_name[se] in name_to_row}
    anc_to_row: dict[str, int] = {}
    ancestors: dict[int, list[str]] = {}
    for se in ses:
        ids: list[str] = []
        for c in anc_codes[se]:
            nm = term_names.get(c)
            if nm and nm in name_to_row:
                anc_to_row[c] = name_to_row[nm]
                ids.append(c)
        ancestors[se] = ids

    if project is None:
        g = torch.Generator().manual_seed(seed)
        lin = nn.Linear(raw.size(1), d_z, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(d_z, raw.size(1), generator=g)
                             / (raw.size(1) ** 0.5))
        lin.weight.requires_grad_(False)
        project = lin

    matched = [se for se in ses if se in se_to_row]
    out = torch.zeros(num_side_effects, d_z)
    if matched:
        z = meddra_zk_init(matched, ancestors, raw, se_to_row, anc_to_row,
                           d_z, project=project)
        for i, se in enumerate(matched):
            out[se] = z[i]
    return out


def inject_atc_nodes(
    data,
    drug_to_atc_ancestors: dict[int, list[int]],
    atc_to_entity_id: dict[int, int],
) -> None:
    """Add a ("entity","rel_atc","entity") edge type connecting each drug to its
    ATC ancestor entities (bidirectional). Modifies `data` in place. ATC entities
    must already exist in the entity id space; `atc_to_entity_id` maps an ATC code
    to its entity id."""
    src: list[int] = []
    dst: list[int] = []
    for drug, atcs in drug_to_atc_ancestors.items():
        for atc in atcs:
            ent = atc_to_entity_id.get(atc)
            if ent is None:
                continue
            src.append(drug); dst.append(ent)
            src.append(ent); dst.append(drug)
    if src:
        ei = torch.tensor([src, dst], dtype=torch.long)
        data["entity", "rel_atc", "entity"].edge_index = ei


def inject_atc_into_graph(data, drug_to_atc_ancestors, atc_codes, atc_features):
    """Add ATC ancestor nodes to ``data`` and connect each drug to its ancestors.

    ``atc_codes``: ordered unique ATC codes; their entity ids are assigned
    contiguously after the existing entities. ``atc_features``: ``[len(atc_codes),
    D]`` rows in the same order. Extends ``data['entity'].x``, adds the
    ``('entity','rel_atc','entity')`` edges (via :func:`inject_atc_nodes`), and
    rebuilds the drug neighbour cache so it includes the new edges (drugs gain
    their ATC nodes as neighbours, and drugs sharing an ancestor become two-hop
    neighbours). Returns ``atc_to_entity_id``."""
    from kdcrga.data.graph import build_neighbour_cache

    x = data["entity"].x
    num_entities = x.size(0)
    atc_to_entity_id = {code: num_entities + i for i, code in enumerate(atc_codes)}
    data["entity"].x = torch.cat([x, atc_features.to(x.dtype)], dim=0)
    inject_atc_nodes(data, drug_to_atc_ancestors, atc_to_entity_id)
    build_neighbour_cache(data, int(data.num_drugs))
    return atc_to_entity_id
