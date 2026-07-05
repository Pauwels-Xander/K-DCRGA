"""The committed SE-index -> MedDRA mapping artifact (se_to_meddra.json)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeddraEntry:
    cui: str
    name: str
    pt_code: str
    pt_name: str
    hlt: str
    hlgt: str
    soc: str
    soc_name: str
    match_type: str
    match_score: float


def save_mapping(path: str | Path, mapping: dict[int, MeddraEntry]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): asdict(v) for k, v in sorted(mapping.items())}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_mapping(path: str | Path) -> dict[int, MeddraEntry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): MeddraEntry(**v) for k, v in raw.items()}
