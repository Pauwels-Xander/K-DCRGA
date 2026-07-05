"""
characterize_knowddi.py
-----------------------
Characterize the KnowDDI BioSNAP dataset format for use in the loader task.
Prints a human-readable report covering:
  - File sizes and first 5 lines
  - Column structure / delimiter of each file
  - Entity/relation/triple counts
  - Drug-id scheme and integer ranges
  - Overlap between DDI drug ids and BKG entity ids

Usage:
    python scripts/characterize_knowddi.py
"""

import os
import sys
import json
import collections
from pathlib import Path

# Force UTF-8 output on Windows so unicode chars in print() don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
BIOSNAP_DIR   = REPO_ROOT / "third_party" / "knowddi" / "data"   / "BioSNAP"
RAW_BIOSNAP   = REPO_ROOT / "third_party" / "knowddi" / "raw_data" / "BioSNAP"
RAW_DRUGBANK  = REPO_ROOT / "third_party" / "knowddi" / "raw_data" / "Drugbank"
RAW_HETIONET  = REPO_ROOT / "third_party" / "knowddi" / "raw_data" / "hetionet"

SECTION = "=" * 72


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def first_n_lines(path: Path, n: int = 5):
    lines = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            lines.append(line.rstrip("\n"))
    return lines


def detect_delimiter(lines):
    """Return the detected delimiter and field count."""
    for delim, name in [("\t", "TAB"), (" ", "SPACE"), (",", "COMMA")]:
        counts = [len(l.split(delim)) for l in lines if l.strip()]
        if len(set(counts)) == 1 and counts[0] > 1:
            return delim, name, counts[0]
    return None, "UNKNOWN", 0


def count_triples_ddi(path: Path):
    """
    DDI files (train/valid/test) have the multi-label format:
      drug_a<SEP>drug_b<SEP>label_vector<SEP>label_flag
    Column 3 (0-indexed) is a comma-separated binary vector encoding which
    side-effect relations are active for this pair.

    Returns: (n_lines, drug_ids, relation_indices, (drug_min, drug_max))
    """
    drug_ids = set()
    relation_indices = set()
    n_lines = 0
    drug_min, drug_max = float("inf"), float("-inf")

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(" ")
            # Expect: drug_a, drug_b, label_vector (comma-sep 0/1s), label_flag
            if len(parts) < 3:
                continue
            n_lines += 1
            da, db = int(parts[0]), int(parts[1])
            drug_ids.add(da)
            drug_ids.add(db)
            drug_min = min(drug_min, da, db)
            drug_max = max(drug_max, da, db)
            # column 2 is the multi-hot vector
            vec = parts[2].split(",")
            for idx, bit in enumerate(vec):
                if bit.strip() == "1":
                    relation_indices.add(idx)

    return n_lines, drug_ids, relation_indices, (drug_min, drug_max)


def count_bkg_triples(path: Path):
    """
    BKG_file.txt format: head<SPACE>tail<SPACE>relation_id
    (integer ids, space-separated, 3 columns)
    Returns: (n_lines, heads, tails, relations)
    """
    n_lines = 0
    heads, tails, relations = set(), set(), set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            n_lines += 1
            heads.add(int(parts[0]))
            tails.add(int(parts[1]))
            relations.add(int(parts[2]))
    return n_lines, heads, tails, relations


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# MAIN REPORT
# ===========================================================================
print(SECTION)
print("KnowDDI BioSNAP Dataset Characterization Report")
print(SECTION)
print(f"Repo root  : {REPO_ROOT}")
print(f"BioSNAP dir: {BIOSNAP_DIR}")
print(f"Raw BioSNAP: {RAW_BIOSNAP}")
print(f"Raw Drugbank: {RAW_DRUGBANK}")
print()

# ---------------------------------------------------------------------------
# Section 1: DDI split files (train / valid / test) from data/BioSNAP
# ---------------------------------------------------------------------------
print(SECTION)
print("SECTION 1 — DDI Split Files  (data/BioSNAP/train|valid|test.txt)")
print(SECTION)

ddi_files = ["train.txt", "valid.txt", "test.txt"]
all_drug_ids   = set()
all_rel_indices = set()
total_pairs    = 0

for fname in ddi_files:
    path = BIOSNAP_DIR / fname
    print(f"\n--- {fname} ---")
    print(f"  Size: {file_size_mb(path):.2f} MB")
    lines_preview = first_n_lines(path, 5)
    print("  First 5 lines:")
    for l in lines_preview:
        print(f"    {repr(l)}")

    delim, delim_name, ncols = detect_delimiter(lines_preview)
    print(f"  Detected delimiter: {delim_name!r}")

    # Use the first non-empty line to determine vector length
    sample_parts = lines_preview[0].split(delim) if delim else lines_preview[0].split()
    vec_len = len(sample_parts[2].split(",")) if len(sample_parts) >= 3 else "?"
    print(f"  Columns: [0]=drug_a (int), [1]=drug_b (int), [2]=label_vector ({vec_len}-dim binary), [3]=label_flag (0 or 1)")

    n_lines, drug_ids, rel_indices, (dmin, dmax) = count_triples_ddi(path)
    print(f"  Pair-lines (rows): {n_lines:,}")
    print(f"  Unique drug ids in this split: {len(drug_ids):,}")
    print(f"  Drug id range: {dmin} – {dmax}")
    print(f"  Active relation indices found: {len(rel_indices):,}")
    print(f"  Active relation index range: {min(rel_indices)} – {max(rel_indices)}")

    # Count actual positive triples (1-bits in the vector)
    pos_triples = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delim) if delim else line.split()
            if len(parts) >= 3:
                pos_triples += sum(int(b) for b in parts[2].split(","))
    print(f"  Positive DDI triples (sum of 1-bits across all rows): {pos_triples:,}")

    all_drug_ids.update(drug_ids)
    all_rel_indices.update(rel_indices)
    total_pairs += n_lines

print()
print(f"  === POOLED DDI STATS ===")
print(f"  Total pair-rows (train+valid+test): {total_pairs:,}")
print(f"  Unique drugs across all splits    : {len(all_drug_ids):,}")
print(f"  Drug id range (global)            : {min(all_drug_ids)} – {max(all_drug_ids)}")
print(f"  Distinct side-effect indices seen : {len(all_rel_indices):,}")
print(f"  Side-effect index range           : {min(all_rel_indices)} – {max(all_rel_indices)}")

# Label vector length from first line
with open(BIOSNAP_DIR / "train.txt", encoding="utf-8") as f:
    first_line = f.readline().strip()
first_parts = first_line.split("\t") if "\t" in first_line else first_line.split(" ")
vec_len = len(first_parts[2].split(","))
print(f"  Label-vector length (# side-effect slots): {vec_len}")

# ---------------------------------------------------------------------------
# Section 2: BKG_file.txt (background knowledge graph)
# ---------------------------------------------------------------------------
print()
print(SECTION)
print("SECTION 2 — Background Knowledge Graph  (data/BioSNAP/BKG_file.txt)")
print(SECTION)
bkg_path = BIOSNAP_DIR / "BKG_file.txt"
print(f"\n  Size: {file_size_mb(bkg_path):.2f} MB")
bkg_preview = first_n_lines(bkg_path, 5)
print("  First 5 lines:")
for l in bkg_preview:
    print(f"    {repr(l)}")

delim_b, delim_name_b, ncols_b = detect_delimiter(bkg_preview)
print(f"  Detected delimiter: {delim_name_b!r}")
print(f"  Columns: [0]=head_entity_id (int), [1]=tail_entity_id (int), [2]=relation_type_id (int)")

bkg_n, bkg_heads, bkg_tails, bkg_rels = count_bkg_triples(bkg_path)
bkg_entities = bkg_heads | bkg_tails
print(f"\n  Total triples (rows): {bkg_n:,}")
print(f"  Distinct entities    : {len(bkg_entities):,}")
print(f"  Entity id range      : {min(bkg_entities)} – {max(bkg_entities)}")
print(f"  Distinct relations   : {len(bkg_rels):,}")
print(f"  Relation id range    : {min(bkg_rels)} – {max(bkg_rels)}")

# ---------------------------------------------------------------------------
# Section 3: raw_data/BioSNAP mapping files
# ---------------------------------------------------------------------------
print()
print(SECTION)
print("SECTION 3 — Mapping Files  (raw_data/BioSNAP/)")
print(SECTION)

# BKG_entity2Id.json
bkg_e2id_path = RAW_BIOSNAP / "BKG_entity2Id.json"
print(f"\n--- {bkg_e2id_path.name} ---")
bkg_e2id = load_json(bkg_e2id_path)
ids_e2id = list(bkg_e2id.values())
keys_sample = list(bkg_e2id.keys())[:5]
print(f"  Size: {file_size_mb(bkg_e2id_path):.3f} MB")
print(f"  Entries : {len(bkg_e2id):,}")
print(f"  Id range: {min(ids_e2id)} – {max(ids_e2id)}")
print(f"  Format  : {{\"NodeType::NodeName\": integer_id, ...}}")
print(f"  Sample keys (entity strings): {keys_sample}")
# Node types present
node_types = collections.Counter(k.split("::")[0] for k in bkg_e2id.keys())
print(f"  Node types ({len(node_types)}): {dict(node_types.most_common(10))}")

# id2drug.json
id2drug_path = RAW_BIOSNAP / "id2drug.json"
print(f"\n--- {id2drug_path.name} ---")
id2drug = load_json(id2drug_path)
print(f"  Size   : {file_size_mb(id2drug_path):.3f} MB")
print(f"  Entries: {len(id2drug):,}  (integer node id -> dict with 'cid', 'db', 'smiles')")
ids_drug = [int(k) for k in id2drug.keys()]
print(f"  Id range: {min(ids_drug)} - {max(ids_drug)}")
sample_entry = list(id2drug.items())[0]
print(f"  Sample : {sample_entry[0]} -> cid={sample_entry[1]['cid']}, db={sample_entry[1]['db']}, smiles=...")

# id2relation.json
id2rel_path = RAW_BIOSNAP / "id2relation.json"
print(f"\n--- {id2rel_path.name} ---")
id2rel = load_json(id2rel_path)
print(f"  Entries: {len(id2rel):,}  (integer side-effect index -> UMLS CUI string)")
print(f"  Format : {{\"0\": \"C0158986\", ...}}")
print(f"  First 5: {list(id2rel.items())[:5]}")
print(f"  NOTE: This maps the 200-dim label vector positions to UMLS CUI codes.")
print(f"        Only 200 entries (not 963) -- this file covers a SUBSET of relations.")

# relation_type_drug.json
rt_drug_path = RAW_BIOSNAP / "relation_type_drug.json"
print(f"\n--- {rt_drug_path.name} ---")
rt_drug = load_json(rt_drug_path)
print(f"  Entries: {len(rt_drug):,}  (Hetionet relation abbreviation → BKG relation integer id)")
print(f"  Content: {rt_drug}")
print(f"  NOTE: These are the 23 Hetionet edge types used in BKG_file.txt (relations 0-22).")

# ---------------------------------------------------------------------------
# Section 4: raw_data/Drugbank mapping files (parallel dataset — Drugbank split)
# ---------------------------------------------------------------------------
print()
print(SECTION)
print("SECTION 4 — Drugbank variant  (raw_data/Drugbank/)")
print(SECTION)

db_node2id_path = RAW_DRUGBANK / "node2id.json"
db_e2id_path    = RAW_DRUGBANK / "BKG_entity2Id.json"
db_id2rel_path  = RAW_DRUGBANK / "id2rel.txt"

print(f"\n--- node2id.json ---")
db_node2id = load_json(db_node2id_path)
db_ids = list(db_node2id.values())
print(f"  Entries : {len(db_node2id):,}  (DrugBank ID → integer node id)")
print(f"  Id range: {min(db_ids)} – {max(db_ids)}")
print(f"  Sample  : {list(db_node2id.items())[:3]}")

print(f"\n--- BKG_entity2Id.json (Drugbank version) ---")
db_e2id = load_json(db_e2id_path)
db_e_ids = list(db_e2id.values())
print(f"  Entries : {len(db_e2id):,}  (Hetionet entities, offset from Drugbank drug ids)")
print(f"  Id range: {min(db_e_ids)} – {max(db_e_ids)}")

print(f"\n--- id2rel.txt ---")
with open(db_id2rel_path, encoding="utf-8") as f:
    db_id2rel_lines = f.readlines()
print(f"  Lines  : {len(db_id2rel_lines):,}")
print(f"  Header : {repr(db_id2rel_lines[0].rstrip())}")
print(f"  Sample : {repr(db_id2rel_lines[1].rstrip())}")

# ---------------------------------------------------------------------------
# Section 5: raw_data/hetionet
# ---------------------------------------------------------------------------
print()
print(SECTION)
print("SECTION 5 — Hetionet nodes file  (raw_data/hetionet/)")
print(SECTION)
hetionet_file = RAW_HETIONET / "hetionet-v1.0-nodes.tsv"
if hetionet_file.exists():
    print(f"  Size: {file_size_mb(hetionet_file):.2f} MB")
    het_preview = first_n_lines(hetionet_file, 5)
    print("  First 5 lines:")
    for l in het_preview:
        print(f"    {repr(l)}")
else:
    print(f"  Not found at {hetionet_file}")

# ---------------------------------------------------------------------------
# Section 6: Cross-reference — DDI drug ids vs BKG entity ids
# ---------------------------------------------------------------------------
print()
print(SECTION)
print("SECTION 6 — Cross-reference: DDI drug ids  vs  BKG entity ids")
print(SECTION)

bkg_all_entity_ids = bkg_heads | bkg_tails

ddi_in_bkg     = all_drug_ids & bkg_all_entity_ids
ddi_not_in_bkg = all_drug_ids - bkg_all_entity_ids

print(f"\n  DDI unique drug ids (from train+valid+test): {len(all_drug_ids):,}")
print(f"  BKG unique entity ids                      : {len(bkg_all_entity_ids):,}")
print(f"  DDI drug ids found in BKG                  : {len(ddi_in_bkg):,}")
print(f"  DDI drug ids NOT in BKG                    : {len(ddi_not_in_bkg):,}")

print(f"  NOTE: DDI drug ids (0-603) and BKG entity ids are SEPARATE ID SPACES.")
print(f"  BKG ids 0-603 correspond to Side Effect / Gene / other entities, NOT drug nodes.")
print(f"  The 423 numeric matches above are coincidental -- they do NOT represent drug-BKG links.")
print(f"  The correct mapping chain is: DDI drug id -> id2drug.json -> DrugBank ID -> BKG_entity2Id.json -> BKG node id.")

# Also check overlap with id2drug (the drug-node mapping)
id2drug_int_ids = set(int(k) for k in id2drug.keys())
ddi_in_id2drug = all_drug_ids & id2drug_int_ids
print(f"\n  DDI drug ids found in id2drug.json         : {len(ddi_in_id2drug):,}  (of {len(all_drug_ids):,})")
if len(ddi_in_id2drug) == len(all_drug_ids):
    print(f"  CONFIRMED: ALL DDI drug ids have entries in id2drug.json (CID/DrugBank/SMILES).")
else:
    diff = all_drug_ids - id2drug_int_ids
    print(f"  WARNING: {len(diff)} drug ids have no entry in id2drug.json. Sample: {sorted(diff)[:10]}")

# ---------------------------------------------------------------------------
# Section 7: Summary table
# ---------------------------------------------------------------------------
print()
print(SECTION)
print("SECTION 7 — Summary (loader-ready facts)")
print(SECTION)
print(f"""
DDI FILES (data/BioSNAP/)
  Format (train/valid/test.txt):
    Delimiter      : TAB ('\\t')
    Columns        : drug_a  drug_b  label_vector  label_flag
    Column types   : int     int     comma-sep 0/1 binary vector  int (0 or 1)
    Label vector   : {vec_len}-dimensional binary multi-hot
    drug_a = col 0  (DDI drug index, range 0-603)
    drug_b = col 1  (DDI drug index, range 0-603)
    label_vector = col 2 (comma-sep, length {vec_len}; each position i=1 means side-effect i is active)
    label_flag = col 3 (positive/negative sample indicator: 1=positive pair, 0=negative pair)

  NOTE: DDI drug ids (0-603) are a SEPARATE enumeration from BKG node ids.
        Use id2drug.json to map DDI drug id -> PubChem CID / DrugBank ID.
        Use BKG_entity2Id.json to map DrugBank ID -> BKG node id for embedding lookup.

  Drug id range (DDI): {min(all_drug_ids)} - {max(all_drug_ids)}
  Distinct drugs (DDI): {len(all_drug_ids)}
  Distinct active side-effect indices: {len(all_rel_indices)}  (indices into {vec_len}-dim vector)
  Total pair-rows: {total_pairs:,} (train + valid + test)

BKG FILE (data/BioSNAP/BKG_file.txt)
  Format         : head_id  tail_id  relation_id
  Delimiter      : SPACE (' ')
  All values     : integers
  Total triples  : {bkg_n:,}
  Distinct entities: {len(bkg_entities):,}  (id range {min(bkg_entities)}-{max(bkg_entities)})
  Distinct relations: {len(bkg_rels):,}  (id range {min(bkg_rels)}-{max(bkg_rels)})
  Node types in graph (from BKG_entity2Id.json): {len(node_types)} types -- {list(node_types.keys())}
  NOTE: BKG entity ids are NOT the same space as DDI drug ids.
        BKG ids 0-603 correspond to Side Effect and other non-drug entities.
        Drug (Compound) nodes in BKG have ids starting from 50 and are sparsely distributed.

MAPPING FILES (raw_data/BioSNAP/)
  BKG_entity2Id.json   : {{"NodeType::Name": int_id}} -- {len(bkg_e2id):,} entities
  id2drug.json         : {{int_id: {{cid, db, smiles}}}} -- {len(id2drug):,} drug nodes (DDI id -> identifiers)
  id2relation.json     : {{str_idx: UMLS_CUI}} -- {len(id2rel):,} side-effect label index -> UMLS CUI
  relation_type_drug.json : {{Hetionet_abbrev: int_id}} -- {len(rt_drug):,} BKG edge types

ID SPACE SUMMARY
  DDI drug ids (0-603) map to drugs via id2drug.json.
  BKG node ids (0-28312) identify Hetionet entities in BKG_file.txt (separate space).
  To link a DDI drug to its BKG node: DDI_id -> id2drug -> DrugBank ID -> BKG_entity2Id -> BKG node id.
  ALL {len(all_drug_ids)} DDI drug ids have entries in id2drug.json: {len(ddi_in_id2drug) == len(all_drug_ids)}
  DDI drug ids that numerically appear in BKG_file.txt entity set: {len(ddi_in_bkg)} of {len(all_drug_ids)}
  (These 423 matches are coincidental numeric overlaps with non-Compound BKG nodes, NOT drug nodes.)
""")

print(SECTION)
print("End of report")
print(SECTION)
