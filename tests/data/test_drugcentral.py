from kdcrga.data.drugcentral import build_drug_atc_ancestors, iter_copy_rows

# A tiny synthetic pg_dump fragment covering the four tables we read.
MINI_DUMP = (
    "SET statement_timeout = 0;\n"
    "COPY public.atc (id, code, l1_code, l1_name, l2_code, l2_name, "
    "l3_code, l3_name, l4_code, l4_name) FROM stdin;\n"
    "1\tN05BA01\tN\tNERVOUS SYSTEM\tN05\tPSYCHOLEPTICS\tN05B\tAnxiolytics\tN05BA\tBenzodiazepine derivatives\n"
    "2\tN05BA04\tN\tNERVOUS SYSTEM\tN05\tPSYCHOLEPTICS\tN05B\tAnxiolytics\tN05BA\tBenzodiazepine derivatives\n"
    "3\tC07AB02\tC\tCARDIOVASCULAR SYSTEM\tC07\tBETA BLOCKING AGENTS\tC07A\tBeta blocking agents\tC07AB\tBeta blocking agents, selective\n"
    "\\.\n"
    "COPY public.struct2atc (struct_id, atc_code, id) FROM stdin;\n"
    "100\tN05BA01\t1\n"
    "200\tN05BA04\t2\n"
    "300\tC07AB02\t3\n"
    "\\.\n"
    "COPY public.identifier (id, identifier, id_type, struct_id, parent_match) FROM stdin;\n"
    "1\tCID000002173\tPUBCHEM_CID\t100\t\\N\n"
    "2\tDB00813\tDRUGBANK_ID\t200\t\\N\n"
    "3\tCID000000999\tPUBCHEM_CID\t300\t\\N\n"
    "\\.\n"
)


def test_iter_copy_rows_parses_columns_and_nulls(tmp_path):
    dump = tmp_path / "mini.sql"
    dump.write_text(MINI_DUMP, encoding="utf-8")
    rows = list(iter_copy_rows(dump, "struct2atc"))
    assert rows == [
        {"struct_id": "100", "atc_code": "N05BA01", "id": "1"},
        {"struct_id": "200", "atc_code": "N05BA04", "id": "2"},
        {"struct_id": "300", "atc_code": "C07AB02", "id": "3"},
    ]
    # \N becomes None
    ident = list(iter_copy_rows(dump, "identifier"))
    assert ident[0]["parent_match"] is None
    # the 'atc' matcher must not also swallow a hypothetical 'atc_ddd' block
    assert len(list(iter_copy_rows(dump, "atc"))) == 3


def test_build_drug_atc_ancestors_joins_via_cid_then_drugbank():
    id2drug = {
        0: {"cid": "CID000002173", "db": None},        # joins via CID -> struct 100
        1: {"cid": "CID999999999", "db": "DB00813"},   # CID misses -> DrugBank -> 200
        2: {"cid": "CID000000777", "db": None},        # no ATC at all
    }
    identifier = [
        {"id_type": "PUBCHEM_CID", "identifier": "2173", "struct_id": "100"},
        {"id_type": "DRUGBANK_ID", "identifier": "DB00813", "struct_id": "200"},
    ]
    struct2atc = [
        {"struct_id": "100", "atc_code": "N05BA01"},
        {"struct_id": "200", "atc_code": "C07AB02"},
    ]
    atc = [
        {"code": "N05BA01", "l2_code": "N05", "l2_name": "PSYCHOLEPTICS",
         "l3_code": "N05B", "l3_name": "Anxiolytics",
         "l4_code": "N05BA", "l4_name": "Benzodiazepine derivatives"},
        {"code": "C07AB02", "l2_code": "C07", "l2_name": "BETA BLOCKING AGENTS",
         "l3_code": "C07A", "l3_name": "Beta blocking agents",
         "l4_code": "C07AB", "l4_name": "Beta blocking agents, selective"},
    ]
    d2a, names, stats = build_drug_atc_ancestors(
        id2drug, identifier, struct2atc, atc, levels=("l2", "l3", "l4"))
    assert d2a[0] == ["N05", "N05B", "N05BA"]   # via CID (normalised CID000002173->2173)
    assert d2a[1] == ["C07", "C07A", "C07AB"]   # via DrugBank fallback
    assert 2 not in d2a                          # no ATC mapping
    assert names["N05"] == "PSYCHOLEPTICS"
    assert names["N05BA"] == "Benzodiazepine derivatives"
    assert stats["drugs_with_atc"] == 2 and stats["total_drugs"] == 3
    assert stats["atc_nodes"] == 6               # 3 unique ancestor codes per drug
