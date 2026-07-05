"""Orchestrate the SE -> MedDRA mapping build: source names, match, resolve
hierarchy, apply overrides, write artifact + human-readable report."""
from __future__ import annotations

import csv
from pathlib import Path

from kdcrga.data.meddra.artifact import MeddraEntry, save_mapping
from kdcrga.data.meddra.ascii import load_llt, load_mdhier, load_pt, load_soc
from kdcrga.data.meddra.hierarchy import HierarchyIndex
from kdcrga.data.meddra.match import match_name
from kdcrga.data.meddra.sources import load_se_names


def _load_overrides(path: str | Path | None) -> dict[int, str]:
    if path is None or not Path(path).exists():
        return {}
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[int(row["se_index"])] = str(row["pt_code"]).strip()
    return out


def build_mapping(*, id2relation, decagon_csv, medascii_dir, artifact_path,
                  report_path, overrides_path=None, threshold: float = 90.0
                  ) -> dict[str, int]:
    medascii_dir = Path(medascii_dir)
    llts = load_llt(medascii_dir / "llt.asc")
    pts = load_pt(medascii_dir / "pt.asc")
    idx = HierarchyIndex.from_rows(
        load_mdhier(medascii_dir / "mdhier.asc"),
        load_soc(medascii_dir / "soc.asc"))
    pt_name = {p.pt_code: p.name for p in pts}
    se_names = load_se_names(id2relation, decagon_csv)
    overrides = _load_overrides(overrides_path)

    mapping: dict[int, MeddraEntry] = {}
    report_rows: list[tuple[int, str, str, str, float, str, bool]] = []
    for se in sorted(se_names):
        cui, name = se_names[se]
        if se in overrides:
            pt_code, mtype, score, current = overrides[se], "override", 100.0, True
        else:
            m = match_name(name, llts, pts, threshold=threshold)
            pt_code, mtype, score, current = (m.pt_code, m.match_type, m.score,
                                              m.current)
        node = idx.node(pt_code) if pt_code else None
        report_rows.append((se, cui, name, mtype, score, pt_code or "-", current))
        if pt_code is None or node is None:
            continue
        mapping[se] = MeddraEntry(
            cui=cui, name=name, pt_code=pt_code,
            pt_name=pt_name.get(pt_code, ""), hlt=node.hlt, hlgt=node.hlgt,
            soc=node.soc, soc_name=node.soc_name, match_type=mtype,
            match_score=score)

    save_mapping(artifact_path, mapping)
    _write_report(report_path, report_rows)
    matched = len(mapping)
    return {"total": len(se_names), "matched": matched,
            "unmatched": len(se_names) - matched}


def _write_report(report_path, rows) -> None:
    """rows: (se, cui, name, match_type, score, pt_code, current)."""
    by_type: dict[str, int] = {}
    for row in rows:
        by_type[row[3]] = by_type.get(row[3], 0) + 1
    lines = ["# MedDRA mapping match report", "",
             "| SE | CUI | name | match | score | PT | current |",
             "|----|-----|------|-------|-------|----|---------|"]
    for se, cui, name, mtype, score, pt, current in rows:
        lines.append(f"| {se} | {cui} | {name} | {mtype} | {score:.1f} | {pt} "
                     f"| {'Y' if current else 'N'} |")
    lines += ["", "## Summary"]
    for mtype, n in sorted(by_type.items()):
        lines.append(f"- {mtype}: {n}")
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
