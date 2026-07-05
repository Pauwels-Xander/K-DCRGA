# tests/data/meddra/test_ascii.py
from pathlib import Path

from kdcrga.data.meddra.ascii import (
    load_llt, load_mdhier, load_meddra_term_names, load_pt, load_soc,
)

FIX = Path(__file__).parent / "fixtures"


def test_load_meddra_term_names_covers_pt_hlt_hlgt_soc():
    names = load_meddra_term_names(FIX / "mdhier.asc")
    assert names["10002043"] == "Anaemia folate deficiency"             # PT
    assert names["10002042"] == "Anaemia deficiencies"                  # HLT
    assert names["10002086"] == "Anaemias NEC"                          # HLGT
    assert names["10005329"] == "Blood and lymphatic system disorders"  # SOC
    assert names["10017977"] == "Gastroenteritis NEC"                   # another HLT


def test_load_llt_maps_name_to_pt_and_currency():
    llts = load_llt(FIX / "llt.asc")
    # each entry: (llt_name, pt_code, is_current)
    assert ("enteritis", "10014866", True) in [
        (e.name, e.pt_code, e.current) for e in llts
    ]
    assert ("enteritis old", "10014866", False) in [
        (e.name, e.pt_code, e.current) for e in llts
    ]


def test_load_pt_has_primary_soc():
    pts = load_pt(FIX / "pt.asc")
    by_code = {p.pt_code: p for p in pts}
    assert by_code["10014866"].name == "Enteritis"
    assert by_code["10014866"].primary_soc == "10017947"


def test_load_mdhier_full_path():
    rows = load_mdhier(FIX / "mdhier.asc")
    by_pt = {r.pt_code: r for r in rows if r.primary}
    r = by_pt["10002043"]
    assert (r.hlt, r.hlgt, r.soc) == ("10002042", "10002086", "10005329")
    assert r.soc_name == "Blood and lymphatic system disorders"


def test_load_soc_names():
    socs = load_soc(FIX / "soc.asc")
    assert socs["10017947"] == "Gastrointestinal disorders"


def test_crlf_lines_do_not_leave_carriage_return(tmp_path):
    # Real MedDRA MedAscii files are CRLF; the trailing flag field must not keep \r.
    p = tmp_path / "mdhier.asc"
    p.write_bytes(
        b"10002043$10002042$10002086$10005329$N$N$N$Blood$Bld$$10005329$Y$\r\n"
    )
    rows = load_mdhier(p)
    assert rows[0].primary is True            # would be False if "Y\r" != "Y"
    assert rows[0].soc_name == "Blood"
