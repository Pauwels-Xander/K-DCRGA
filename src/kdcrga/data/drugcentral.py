"""Read ATC mappings out of a DrugCentral PostgreSQL dump (pg_dump plain text).

We never load DrugCentral into Postgres; instead we stream the relevant
``COPY public.<table> ... FROM stdin;`` blocks straight from the .sql file. The
tables used (see drugcentral.org/download):
  - ``atc``        : ATC L5 code -> L1..L4 ancestor codes + names
  - ``struct2atc`` : DrugCentral structure id -> ATC L5 code
  - ``identifier`` : external id (PUBCHEM_CID / DRUGBANK_ID) -> structure id

Joining KnowDDI's per-drug CID / DrugBank id (id2drug.json) through these yields
each drug's ATC ancestors (L2/L3/L4), the nodes injected by inject_atc_nodes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_copy_rows(dump_path: str | Path, table: str) -> Iterator[dict]:
    """Yield ``{column: value}`` dicts from the ``public.<table>`` COPY block.

    Values are tab-separated; ``\\N`` becomes ``None``; the block ends at a ``\\.``
    line. Matches ``COPY public.<table> (`` exactly so ``atc`` does not also match
    ``atc_ddd``."""
    target = f"COPY public.{table} ("
    cols: list[str] | None = None
    with open(dump_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if cols is None:
                if line.startswith(target):
                    inside = line[line.index("(") + 1: line.index(")")]
                    cols = [c.strip() for c in inside.split(",")]
                continue
            if line.rstrip("\r\n") == "\\.":
                return
            values = line.rstrip("\r\n").split("\t")
            yield {c: (None if v == "\\N" else v)
                   for c, v in zip(cols, values)}


def _norm_cid(cid: str | None) -> int | None:
    """KnowDDI CIDs are zero-padded ('CID000002173'); DrugCentral PUBCHEM_CID
    values are bare integers ('2173'). Normalise both to int for joining."""
    if not cid:
        return None
    digits = "".join(ch for ch in cid if ch.isdigit())
    return int(digits) if digits else None


def build_drug_atc_ancestors(id2drug, identifier_rows, struct2atc_rows, atc_rows,
                             levels=("l2", "l3", "l4")):
    """Map each drug row to its ATC ancestor codes (default L2/L3/L4 — matching
    the proposal figure) by joining KnowDDI drug ids through DrugCentral.

    ``id2drug``: ``{drug_row: {"cid":..., "db":...}}``. The other three are
    iterables of dict rows from ``iter_copy_rows`` (``identifier`` / ``struct2atc``
    / ``atc``). Returns ``(drug_to_atc_ancestors, atc_names, stats)`` where
    ``drug_to_atc_ancestors`` maps a drug row to a sorted list of unique ancestor
    codes, ``atc_names`` maps each ancestor code to its name, and ``stats`` reports
    coverage. Drugs are resolved to a structure id via PUBCHEM_CID first, then
    DRUGBANK_ID."""
    cid_to_struct: dict[int, str] = {}
    db_to_struct: dict[str, str] = {}
    for r in identifier_rows:
        sid, val, kind = r.get("struct_id"), r.get("identifier"), r.get("id_type")
        if sid is None or val is None:
            continue
        if kind == "PUBCHEM_CID":
            c = _norm_cid(val)
            if c is not None:
                cid_to_struct.setdefault(c, sid)
        elif kind == "DRUGBANK_ID":
            db_to_struct.setdefault(val, sid)

    struct_atc: dict[str, list[str]] = {}
    for r in struct2atc_rows:
        struct_atc.setdefault(r["struct_id"], []).append(r["atc_code"])

    code_anc: dict[str, list[str]] = {}
    atc_names: dict[str, str] = {}
    for r in atc_rows:
        anc: list[str] = []
        for lv in levels:
            code, name = r.get(f"{lv}_code"), r.get(f"{lv}_name")
            if code:
                anc.append(code)
                if name:
                    atc_names[code] = name
        code_anc[r["code"]] = anc

    drug_to_atc: dict[int, list[str]] = {}
    for drug, info in id2drug.items():
        sid = None
        c = _norm_cid(info.get("cid"))
        if c is not None and c in cid_to_struct:
            sid = cid_to_struct[c]
        elif info.get("db") and info["db"] in db_to_struct:
            sid = db_to_struct[info["db"]]
        if sid is None:
            continue
        anc = {code for l5 in struct_atc.get(sid, []) for code in code_anc.get(l5, [])}
        if anc:
            drug_to_atc[drug] = sorted(anc)

    stats = {
        "total_drugs": len(id2drug),
        "drugs_with_atc": len(drug_to_atc),
        "atc_nodes": len({c for codes in drug_to_atc.values() for c in codes}),
    }
    return drug_to_atc, atc_names, stats
