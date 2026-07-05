# tests/data/meddra/test_build.py
from pathlib import Path

from kdcrga.data.meddra.artifact import load_mapping
from kdcrga.data.meddra.build import build_mapping

FIX = Path(__file__).parent / "fixtures"


def test_build_mapping_writes_artifact_and_report(tmp_path):
    artifact = tmp_path / "se_to_meddra.json"
    report = tmp_path / "report.md"
    summary = build_mapping(
        id2relation=FIX / "id2relation.json",
        decagon_csv=FIX / "bio-decagon-combo.csv",
        medascii_dir=FIX,
        artifact_path=artifact,
        report_path=report,
        overrides_path=None,
        threshold=90,
    )
    assert summary["matched"] == 2 and summary["unmatched"] == 1
    mapping = load_mapping(artifact)
    assert mapping[0].pt_code == "10014866"        # enteritis (exact)
    assert mapping[1].pt_code == "10002043"        # fuzzy reorder
    assert 2 not in mapping                          # unmatched omitted
    assert report.exists()
    assert "unmatched" in report.read_text(encoding="utf-8").lower()


def test_overrides_win(tmp_path):
    overrides = tmp_path / "overrides.csv"
    overrides.write_text("se_index,pt_code\n2,10014866\n", encoding="utf-8")
    artifact = tmp_path / "se_to_meddra.json"
    build_mapping(
        id2relation=FIX / "id2relation.json",
        decagon_csv=FIX / "bio-decagon-combo.csv",
        medascii_dir=FIX, artifact_path=artifact, report_path=tmp_path / "r.md",
        overrides_path=overrides, threshold=90,
    )
    mapping = load_mapping(artifact)
    assert mapping[2].pt_code == "10014866"          # override applied
    assert mapping[2].match_type == "override"
