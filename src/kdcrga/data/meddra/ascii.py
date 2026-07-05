"""Parsers for the MedDRA MedAscii distribution ($-delimited .asc files).

Field layouts per dist_file_format_29_0_English.pdf. We read only the columns
the RQ4/RQ5 mapping needs; trailing empty fields are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LltRow:
    code: str
    name: str
    pt_code: str
    current: bool


@dataclass(frozen=True)
class PtRow:
    pt_code: str
    name: str
    primary_soc: str


@dataclass(frozen=True)
class HierRow:
    pt_code: str
    hlt: str
    hlgt: str
    soc: str
    pt_name: str
    soc_name: str
    primary: bool


def _fields(line: str) -> list[str]:
    return line.rstrip("\r\n").split("$")


def load_llt(path: str | Path) -> list[LltRow]:
    """llt.asc: code$name$pt_code$...$currency(Y/N)$ (currency is field index 9)."""
    out: list[LltRow] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = _fields(line)
            if len(f) < 10 or not f[0]:
                continue
            out.append(LltRow(code=f[0], name=f[1], pt_code=f[2],
                              current=(f[9] == "Y")))
    return out


def load_pt(path: str | Path) -> list[PtRow]:
    """pt.asc: pt_code$pt_name$null$primary_soc_code$..."""
    out: list[PtRow] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = _fields(line)
            if len(f) < 4 or not f[0]:
                continue
            out.append(PtRow(pt_code=f[0], name=f[1], primary_soc=f[3]))
    return out


def load_mdhier(path: str | Path) -> list[HierRow]:
    """mdhier.asc: pt$hlt$hlgt$soc$pt_name$hlt_name$hlgt_name$soc_name$
    soc_abbrev$null$pt_soc_code$primary(Y/N)$."""
    out: list[HierRow] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = _fields(line)
            if len(f) < 12 or not f[0]:
                continue
            out.append(HierRow(pt_code=f[0], hlt=f[1], hlgt=f[2], soc=f[3],
                              pt_name=f[4], soc_name=f[7], primary=(f[11] == "Y")))
    return out


def load_soc(path: str | Path) -> dict[str, str]:
    """soc.asc: soc_code$soc_name$... -> {code: name}."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = _fields(line)
            if len(f) < 2 or not f[0]:
                continue
            out[f[0]] = f[1]
    return out


def load_meddra_term_names(path: str | Path) -> dict[str, str]:
    """Map every MedDRA code -> term name from mdhier.asc, covering PT/HLT/HLGT/SOC.

    Columns: pt$hlt$hlgt$soc$pt_name$hlt_name$hlgt_name$soc_name$... So a single
    pass yields names for a PT and all three of its ancestor levels. Used to
    embed each side effect's own term plus its ancestors for the z_k init."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = _fields(line)
            if len(f) < 8 or not f[0]:
                continue
            out[f[0]] = f[4]  # PT
            out[f[1]] = f[5]  # HLT
            out[f[2]] = f[6]  # HLGT
            out[f[3]] = f[7]  # SOC
    return out
