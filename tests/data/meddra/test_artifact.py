# tests/data/meddra/test_artifact.py
from kdcrga.data.meddra.artifact import MeddraEntry, save_mapping, load_mapping


def test_roundtrip(tmp_path):
    mapping = {
        0: MeddraEntry(cui="C0014863", name="enteritis", pt_code="10014866",
                       pt_name="Enteritis", hlt="10017977", hlgt="10017969",
                       soc="10017947", soc_name="Gastrointestinal disorders",
                       match_type="exact_llt", match_score=100.0),
    }
    path = tmp_path / "se_to_meddra.json"
    save_mapping(path, mapping)
    loaded = load_mapping(path)
    assert loaded[0] == mapping[0]              # int keys restored
    assert loaded[0].soc == "10017947"
