"""Resolve graph entity row ids to Hetionet names.

Row ids come from KnowDDI's BKG_entity2Id.json (string -> row id). That map is
aliased (multiple strings share an id); we pick the lexicographically-first
string per id deterministically and count the collisions. Names come from the
Hetionet nodes TSV (id -> name), which covers 100% of BKG strings.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolutionStats:
    total_rows: int          # max row id + 1 implied by the entity map
    resolved: int            # rows with a Hetionet name
    collisions: int          # ids carrying more than one entity string
    per_kind: dict[str, int]  # resolved-row counts by "Type::" prefix


def _load_hetionet_names(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["id"]] = row["name"]
    return out


def resolve_entity_names(bkg_entity2id_path: str | Path,
                         hetionet_nodes_path: str | Path
                         ) -> tuple[dict[int, str], ResolutionStats]:
    """Return ({row_id: name} for resolved rows, ResolutionStats)."""
    with open(bkg_entity2id_path, encoding="utf-8") as fh:
        ent2id: dict[str, int] = json.load(fh)
    het = _load_hetionet_names(hetionet_nodes_path)

    # Invert to row_id -> list of entity strings.
    by_id: dict[int, list[str]] = {}
    for ent, rid in ent2id.items():
        by_id.setdefault(int(rid), []).append(ent)

    names: dict[int, str] = {}
    per_kind: dict[str, int] = {}
    collisions = 0
    for rid, ents in by_id.items():
        if len(ents) > 1:
            collisions += 1
        chosen = sorted(ents)[0]
        name = het.get(chosen)
        if name is None:
            continue
        names[rid] = name
        kind = chosen.split("::")[0]
        per_kind[kind] = per_kind.get(kind, 0) + 1

    total_rows = (max(by_id) + 1) if by_id else 0
    return names, ResolutionStats(total_rows=total_rows, resolved=len(names),
                                  collisions=collisions, per_kind=per_kind)
