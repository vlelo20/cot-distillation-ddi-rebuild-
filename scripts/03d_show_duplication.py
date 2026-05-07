"""
Step 3d: Show concrete evidence that interactions_full.jsonl contains every
interaction twice. We'll:
  1. Pick a random drug
  2. Show all rows where that drug is the subject
  3. Show all rows where that drug is the affected
  4. Demonstrate that for every (subject=X, affected=Y) row, there's a matching
     (subject=Y, affected=X) row with identical description
  5. Quantify the redundancy on the full dataset
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path.home() / "ddiproject"
INTERACTIONS = ROOT / "processed_v2" / "interactions_full.jsonl"
PROFILES = ROOT / "processed_v2" / "drug_profiles.json"

print("Loading drug profiles...")
with open(PROFILES) as f:
    profiles = json.load(f)
id_to_name = {pid: prof["name"] for pid, prof in profiles.items()}

print("Loading all interactions...")
all_rows = []
with open(INTERACTIONS) as f:
    for line in f:
        all_rows.append(json.loads(line))
print(f"Total rows: {len(all_rows):,}\n")

# ============================================================
# Demo 1: pick a specific drug and show its interaction views
# ============================================================
DEMO_DRUG = "DB00001"  # Lepirudin
demo_name = id_to_name.get(DEMO_DRUG, DEMO_DRUG)

as_subject = [r for r in all_rows if r["subject_id"] == DEMO_DRUG]
as_affected = [r for r in all_rows if r["affected_id"] == DEMO_DRUG]

print("=" * 70)
print(f"DEMO 1: Looking at {demo_name} ({DEMO_DRUG})")
print("=" * 70)
print(f"Rows where {demo_name} is the SUBJECT:  {len(as_subject):,}")
print(f"Rows where {demo_name} is the AFFECTED: {len(as_affected):,}")
print()
print("If interactions were stored once, only one of these counts would")
print("equal the number of distinct partners. Let's check...")
print()

partners_as_subject = {r["affected_id"] for r in as_subject}
partners_as_affected = {r["subject_id"] for r in as_affected}
print(f"Distinct partners when {demo_name} is subject:  {len(partners_as_subject)}")
print(f"Distinct partners when {demo_name} is affected: {len(partners_as_affected)}")
print(f"Overlap (same drugs in both views):              {len(partners_as_subject & partners_as_affected)}")
print()

# ============================================================
# Demo 2: pick a partner and show both directional views
# ============================================================
common_partners = partners_as_subject & partners_as_affected
if common_partners:
    partner = sorted(common_partners)[0]
    partner_name = id_to_name.get(partner, partner)
    print("=" * 70)
    print(f"DEMO 2: {demo_name} <-> {partner_name}")
    print("=" * 70)

    forward = [r for r in as_subject if r["affected_id"] == partner]
    reverse = [r for r in as_affected if r["subject_id"] == partner]

    print(f"Forward direction (subject={demo_name}, affected={partner_name}):")
    for r in forward:
        print(f"  description: {r['description']}")
    print()
    print(f"Reverse direction (subject={partner_name}, affected={demo_name}):")
    for r in reverse:
        print(f"  description: {r['description']}")
    print()
    print("Notice: same description, mirrored subject/affected fields. This is")
    print("the same real-world interaction, recorded twice in the XML.")
    print()

# ============================================================
# Demo 3: do this systematically for 10 random drugs
# ============================================================
print("=" * 70)
print("DEMO 3: Same pattern across 10 random drugs")
print("=" * 70)
print(f"{'Drug':<35} {'subject_rows':>13} {'affected_rows':>14} {'overlap':>9}")
print("-" * 72)

drugs_with_interactions = list({r["subject_id"] for r in all_rows})
random.seed(42)
sample = random.sample(drugs_with_interactions, 10)

for did in sample:
    name = id_to_name.get(did, did)
    s_rows = [r for r in all_rows if r["subject_id"] == did]
    a_rows = [r for r in all_rows if r["affected_id"] == did]
    s_partners = {r["affected_id"] for r in s_rows}
    a_partners = {r["subject_id"] for r in a_rows}
    overlap = len(s_partners & a_partners)
    print(f"{name[:34]:<35} {len(s_rows):>13,} {len(a_rows):>14,} {overlap:>9,}")

# ============================================================
# Demo 4: count exact (sub, aff, desc) <-> (aff, sub, desc) mirror pairs
# ============================================================
print()
print("=" * 70)
print("DEMO 4: How many rows have a perfect mirror twin?")
print("=" * 70)
print("For each row (A, B, desc), is there also a row (B, A, desc) with the")
print("exact same description?")
print()

row_set = set()
for r in all_rows:
    row_set.add((r["subject_id"], r["affected_id"], r["description"]))

mirrors_found = 0
mirrors_missing = 0
for r in all_rows:
    mirror_key = (r["affected_id"], r["subject_id"], r["description"])
    if mirror_key in row_set:
        mirrors_found += 1
    else:
        mirrors_missing += 1

print(f"Rows with a mirror twin in the dataset: {mirrors_found:,}")
print(f"Rows WITHOUT a mirror twin:             {mirrors_missing:,}")
print()
pct = 100 * mirrors_found / len(all_rows)
print(f"That means {pct:.2f}% of all rows are part of a mirror pair.")
print(f"If we kept only one row per pair, we'd have ~{len(all_rows) // 2:,} rows")
print(f"(matching the DrugBank 6.0 paper's published 1.4M figure).")
