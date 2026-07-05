"""Wrap KnowDDI's preprocessed BioSNAP graph into a PyG ``HeteroData``.

Mirrors KnowDDI's authoritative loader ``process_files_decagon`` (see
``third_party/knowddi/pytorch/utils/data_utils.py``): drugs and the background
knowledge graph (BKG) share ONE identity-mapped integer id space (``entity2id[h]=h``
for both), so a single "entity" node type carries both. BKG edges are
relation-typed; DDI edges are side-effect-labelled positives only (``flag==1``).

An audit gate checks the wrapped graph reproduces KnowDDI's BioSNAP statistics
(604 drugs == entities 0..603, 200 side effects, 252,111 positive associations).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from kdcrga.data.schema import DDIEdges, build_hetero_data
from kdcrga.data.splits import frequency_quartiles, stratified_split

# Placeholder feature width; real node features are pretrained embeddings injected
# by a later embeddings plan.
_PLACEHOLDER_DIM = 16

# Number of DDI drugs in the BioSNAP split: drug ids 0..603 ARE entities 0..603.
_NUM_DRUGS = 604


def audit_graph(data: HeteroData) -> dict[str, int]:
    """Report the three statistics the week-1 audit gate checks.

    Format-independent: works on both the synthetic fixture and the real graph.
    """
    return {
        "num_drugs": int(data.num_drugs),
        "num_side_effects": int(data.num_side_effects),
        "num_ddi": int(data.ddi.side_effect.numel()),
    }


def _load_ddi_edges(root: Path) -> tuple[DDIEdges, int]:
    """Pool train/valid/test and expand the 200-dim multi-hot label vectors.

    Each split row is ``drug_a \\t drug_b \\t label_vector \\t flag`` where
    ``label_vector`` is 200 comma-separated bits and ``flag`` is 1 (positive) or 0
    (a 1:1-sampled negative). KnowDDI's ``triplets_train`` collects ``flag==1`` rows
    only; we do the same -- a set bit ``i`` of a ``flag==1`` row is one
    ``(drug_a, drug_b, side_effect=i)`` positive triple. ``flag==0`` rows are
    ignored for the graph (recoverable later from the raw files if needed).
    Returns the DDI edges plus the max entity id observed (drug ids).
    """
    a_list: list[int] = []
    b_list: list[int] = []
    se_list: list[int] = []
    src_list: list[int] = []   # source file per triple: train=0, valid=1, test=2
    max_id = 0

    for src, fname in enumerate(("train.txt", "valid.txt", "test.txt")):
        with open(root / fname) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                drug_a = int(parts[0])
                drug_b = int(parts[1])
                if drug_a > max_id:
                    max_id = drug_a
                if drug_b > max_id:
                    max_id = drug_b
                if int(parts[3]) != 1:  # positives only
                    continue
                for idx, token in enumerate(parts[2].split(",")):
                    if token == "1":
                        a_list.append(drug_a)
                        b_list.append(drug_b)
                        se_list.append(idx)
                        src_list.append(src)

    pair_index = torch.from_numpy(np.array([a_list, b_list], dtype=np.int64))
    side_effect = torch.from_numpy(np.array(se_list, dtype=np.int64))
    source_file = torch.from_numpy(np.array(src_list, dtype=np.int64))
    return DDIEdges(pair_index=pair_index, side_effect=side_effect,
                    source_file=source_file), max_id


def ddi_side_effect_counts(root: str | Path,
                           num_side_effects: int = 200) -> np.ndarray:
    """Per-side-effect positive-triple counts for the loaded BioSNAP slice.

    Reuses ``_load_ddi_edges`` (positives-only, multi-hot expansion), so the
    returned counts match ``data.ddi.side_effect`` exactly -- and hence the
    frequency quartiles ``frequency_quartiles`` derives. The counts sum to the
    252,111 positive associations the audit gate checks. Returns an int64 array
    of length ``num_side_effects``.
    """
    ddi, _ = _load_ddi_edges(Path(root))
    return np.bincount(ddi.side_effect.numpy(),
                       minlength=num_side_effects).astype(np.int64)


def _load_bkg_edges(
    path: Path,
) -> tuple[dict[tuple[str, str, str], np.ndarray], int]:
    """Load ``BKG_file.txt`` (``head_id tail_id relation_id`` per line) as
    relation-typed edge tensors over the shared entity id space.

    Raw ids ``h, t`` are used directly as entity indices. Per-relation src/dst
    lists are grown with plain Python then converted once to int64 arrays (no
    per-edge tensor growth -- ~1.67M edges). Returns the edge dict plus the max
    entity id observed.
    """
    src: dict[int, list[int]] = {}
    dst: dict[int, list[int]] = {}
    max_id = 0

    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                continue
            h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
            if h > max_id:
                max_id = h
            if t > max_id:
                max_id = t
            src.setdefault(r, []).append(h)
            dst.setdefault(r, []).append(t)

    edges: dict[tuple[str, str, str], np.ndarray] = {}
    for r in src:
        etype = ("entity", f"rel_{r}", "entity")
        edges[etype] = np.array([src[r], dst[r]], dtype=np.int64)
    return edges, max_id


def load_or_placeholder_features(path, num_entities: int,
                                 placeholder_dim: int) -> torch.Tensor:
    """Return the cached node-feature tensor when it exists and its row count
    matches `num_entities`; otherwise random placeholder features [num_entities,
    placeholder_dim]. Lets training run with real BioBERT features when built,
    and stay runnable (tests, pre-build) when not."""
    p = Path(path)
    if p.exists():
        feats = torch.load(p, map_location="cpu")
        if feats.size(0) == num_entities:
            print(f"node features: loaded cache {p} (dim {feats.size(1)})", flush=True)
            return feats
        print(f"node features: cache {p} has {feats.size(0)} rows != "
              f"{num_entities}; using placeholders", flush=True)
    else:
        print(f"node features: no cache at {p}; using random placeholders", flush=True)
    return torch.randn(num_entities, placeholder_dim)


def load_knowddi_graph(root: str | Path,
                       node_features_path: str | Path = "data/processed/node_features.pt"
                       ) -> HeteroData:
    """Load the vendored KnowDDI BioSNAP graph as a PyG ``HeteroData``.

    ``root`` is ``third_party/knowddi/data/BioSNAP`` holding ``train/valid/test.txt``
    and ``BKG_file.txt``. Builds a single "entity" node type (drugs are entities
    0..603) with relation-typed BKG edges and side-effect-labelled positive DDI edges.
    """
    root = Path(root)

    # --- DDI edges (positives only) ------------------------------------------
    ddi, max_ddi_id = _load_ddi_edges(root)
    # Boundary guard: drugs must occupy entity ids 0.._NUM_DRUGS-1. If the data
    # ever has out-of-range drug ids, num_drugs would silently disagree with the
    # entity count, so fail loudly here.
    assert max_ddi_id == _NUM_DRUGS - 1, (
        f"expected drug ids 0..{_NUM_DRUGS - 1}, saw max {max_ddi_id}"
    )

    # --- BKG edges (relation-typed, shared id space) -------------------------
    bkg_edges, max_bkg_id = _load_bkg_edges(root / "BKG_file.txt")
    edges: dict[tuple[str, str, str], torch.Tensor] = {
        etype: torch.from_numpy(arr) for etype, arr in bkg_edges.items()
    }

    # --- node features (PLACEHOLDERS) ----------------------------------------
    num_entities = max(max_ddi_id, max_bkg_id) + 1
    node_features = {"entity": load_or_placeholder_features(
        node_features_path, num_entities, _PLACEHOLDER_DIM)}

    data = build_hetero_data(node_features, edges, ddi)
    data.num_drugs = _NUM_DRUGS  # drugs are entities 0..603
    return data


def build_neighbour_cache(data, num_drugs: int) -> None:
    """Precompute 1-hop and 2-hop neighbour sets for drug ids 0..num_drugs-1.

    Stores `data.neighbour_index: dict[int, list[int]]` (sorted, distinct) and
    `data.two_hop_index: dict[int, set[int]]`. Iterates the BKG once; per-query
    helpers become O(1) per drug.

    Note: if `inject_atc_nodes` is called after this cache is built, callers that
    need updated neighbourhoods must call `build_neighbour_cache` again explicitly.
    """
    one_hop: dict[int, set[int]] = {d: set() for d in range(num_drugs)}
    # Build undirected adjacency restricted to edges that touch a drug at one
    # endpoint, AND a full undirected adjacency for the 2-hop expansion.
    full_adj: dict[int, set[int]] = {}
    for etype in data.edge_types:
        if not etype[1].startswith("rel_"):
            continue
        ei = data[etype].edge_index
        src = ei[0].tolist()
        dst = ei[1].tolist()
        for s, t in zip(src, dst):
            full_adj.setdefault(s, set()).add(t)
            full_adj.setdefault(t, set()).add(s)
            if s < num_drugs:
                one_hop[s].add(t)
            if t < num_drugs:
                one_hop[t].add(s)
    # Drop self-loops
    for d in range(num_drugs):
        one_hop[d].discard(d)

    two_hop: dict[int, set[int]] = {}
    for d in range(num_drugs):
        nb = one_hop[d]
        s: set[int] = set(nb)
        for v in nb:
            s.update(full_adj.get(v, ()))
        s.discard(d)
        two_hop[d] = s

    data.neighbour_index = {d: sorted(one_hop[d]) for d in range(num_drugs)}
    data.two_hop_index = two_hop


def drug_neighbours(data, drugs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache-backed 1-hop neighbour lookup (see `build_neighbour_cache`).

    Returns (flat_neighbours [M], query_index [M]) for the batch of drug ids.
    Treats BKG edges as undirected for message passing.
    """
    if not hasattr(data, "neighbour_index"):
        build_neighbour_cache(data, int(getattr(data, "num_drugs", 0)) or
                              int(drugs.max().item()) + 1)
    device = drugs.device
    flat_parts: list[int] = []
    query_parts: list[int] = []
    drug_list = drugs.tolist()
    for i, d in enumerate(drug_list):
        nbrs = data.neighbour_index.get(d, [])
        flat_parts.extend(nbrs)
        query_parts.extend([i] * len(nbrs))
    if not flat_parts:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device))
    return (torch.tensor(flat_parts, dtype=torch.long, device=device),
            torch.tensor(query_parts, dtype=torch.long, device=device))


def drug_two_hop_shared(
    data, pair_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache-backed two-hop intersection (see `build_neighbour_cache`).

    For each (drug_a, drug_b) pair, return the intersection of their 2-hop
    neighbourhoods as (flat_nodes [M], query_index [M]). The drug pair endpoints
    are excluded by construction.
    """
    if not hasattr(data, "two_hop_index"):
        build_neighbour_cache(data, int(getattr(data, "num_drugs", 0)) or
                              int(pair_index.max().item()) + 1)
    device = pair_index.device
    flat_parts: list[int] = []
    query_parts: list[int] = []
    e = pair_index.size(1)
    for i in range(e):
        a = int(pair_index[0, i]); b = int(pair_index[1, i])
        shared = data.two_hop_index.get(a, set()) & data.two_hop_index.get(b, set())
        shared.discard(a); shared.discard(b)
        for v in shared:
            flat_parts.append(v); query_parts.append(i)
    if not flat_parts:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device))
    return (torch.tensor(flat_parts, dtype=torch.long, device=device),
            torch.tensor(query_parts, dtype=torch.long, device=device))


def attach_splits(
    data: HeteroData,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> HeteroData:
    """Populate data.ddi.split (0=train,1=val,2=test) and data.quartiles in place."""
    data.ddi.split = stratified_split(data.ddi.side_effect, ratios=ratios, seed=seed)
    data.quartiles = frequency_quartiles(data.ddi.side_effect)
    return data


def attach_file_splits(data: HeteroData) -> HeteroData:
    """Populate data.ddi.split from the canonical BioSNAP file boundaries.

    Each DDI triple is assigned to the split of the file it was read from
    (train.txt=0, valid.txt=1, test.txt=2; recorded in ``data.ddi.source_file`` by
    ``load_knowddi_graph``). Because the BioSNAP files are pair-disjoint, this yields
    the standard *inductive* split KnowDDI is benchmarked on — test drug-pairs have no
    training edges — unlike ``attach_splits``'s random per-triple re-split, which
    scatters a pair's side effects across train/test and caused the fusion leak."""
    if data.ddi.source_file is None:
        raise ValueError(
            "attach_file_splits needs data.ddi.source_file; load the graph via "
            "load_knowddi_graph (which records it). Use attach_splits for the random "
            "stratified split instead.")
    data.ddi.split = data.ddi.source_file.clone()
    data.quartiles = frequency_quartiles(data.ddi.side_effect)
    return data
