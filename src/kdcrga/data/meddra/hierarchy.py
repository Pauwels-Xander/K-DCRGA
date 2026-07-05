"""MedDRA PT hierarchy index + tree distance on the primary SOC path."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from kdcrga.data.meddra.ascii import HierRow


@dataclass(frozen=True)
class PtNode:
    pt_code: str
    hlt: str
    hlgt: str
    soc: str
    soc_name: str


class HierarchyIndex:
    """PT -> (HLT, HLGT, SOC) on the primary axis, with path-length distance."""

    def __init__(self, nodes: dict[str, PtNode]):
        self._nodes = nodes

    @classmethod
    def from_rows(cls, hier_rows: list[HierRow], soc_names: dict[str, str]
                  ) -> "HierarchyIndex":
        nodes: dict[str, PtNode] = {}
        for r in hier_rows:
            # Prefer the primary-axis row for each PT; fall back to first seen.
            if r.pt_code in nodes and not r.primary:
                continue
            nodes[r.pt_code] = PtNode(
                pt_code=r.pt_code, hlt=r.hlt, hlgt=r.hlgt, soc=r.soc,
                soc_name=soc_names.get(r.soc, r.soc_name))
        return cls(nodes)

    def node(self, pt_code: str) -> PtNode | None:
        return self._nodes.get(pt_code)

    def soc_name(self, pt_code: str) -> str | None:
        n = self._nodes.get(pt_code)
        return n.soc_name if n else None

    def distance(self, pt_a: str, pt_b: str) -> float:
        """Tree distance on the primary path: same PT 0, HLT 2, HLGT 4, SOC 6,
        cross-SOC 8. Returns NaN if either PT is unknown."""
        a, b = self._nodes.get(pt_a), self._nodes.get(pt_b)
        if a is None or b is None:
            return float("nan")
        if a.pt_code == b.pt_code:
            return 0.0
        if a.hlt == b.hlt:
            return 2.0
        if a.hlgt == b.hlgt:
            return 4.0
        if a.soc == b.soc:
            return 6.0
        return 8.0


def _distance_from_levels(a, b) -> float:
    if a.pt_code == b.pt_code:
        return 0.0
    if a.hlt == b.hlt:
        return 2.0
    if a.hlgt == b.hlgt:
        return 4.0
    if a.soc == b.soc:
        return 6.0
    return 8.0


def build_se_distance_matrix(mapping) -> dict[tuple[int, int], float]:
    """MedDRA tree distance for every matched SE pair, keyed (k1, k2) with k1<k2.
    `mapping`: {se_index: MeddraEntry}. Entries already carry hlt/hlgt/soc/pt."""
    out: dict[tuple[int, int], float] = {}
    for a, b in combinations(sorted(mapping), 2):
        out[(a, b)] = _distance_from_levels(mapping[a], mapping[b])
    return out


def build_soc_of(mapping) -> dict[int, str]:
    """{se_index: SOC name} for SOC-stratified Spearman."""
    return {k: e.soc_name for k, e in mapping.items()}
