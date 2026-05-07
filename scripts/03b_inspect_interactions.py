"""
Step 3b: Inspect interaction descriptions to determine the dedup strategy.

We need to answer two questions:
  Q1: When the same drug pair has interactions in both directions
      (A->B and B->A), are the descriptions identical or different?
  Q2: Does DrugBank consistently put the "actor" (subject) first
      in the description, before the recipient (affected)?

Outputs to console; nothing written to disk.
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "ddiproject"
INTERACTIONS = ROOT / "processed_v2" / "interactions_full.jsonl"
PROFILES = ROOT / "processed_v2" / "drug_profiles.json"

# Load drug names for ID -> name lookup
print("Loading drug profiles...")
with open(PROFILES) as f:
    profiles = json.load(f)
id_to_name = {pid: prof["name"] for pid, prof in profiles.items()}

# Load all interactions; group by unordered pair
print("Loading interactions and grouping by unordered pair...")
pair_to_rows = defaultdict(list)
with open(INTERACTIONS) as f:
    for line in f:
        row = json.loads(line)
        pair = frozenset({row["subject_id"], row["affected_id"]})
        pair_to_rows[pair].append(row)

print(f"Total interactions: {sum(len(v) for v in pair_to_rows.values()):,}")
print(f"Unique unordered pairs: {len(pair_to_rows):,}")

# How many pairs appear with 1 row vs 2 rows vs more?
size_distribution = defaultdict(int)
for rows in pair_to_rows.values():
    size_distribution[len(rows)] += 1

print("\nDistribution of rows-per-pair:")
for size, count in sorted(size_distribution.items()):
    print(f"  {size} row(s): {count:,} pairs")

# ----- Q1: For pairs with 2 rows, are the descriptions identical? -----
print("\n" + "=" * 60)
print("Q1: For pairs with 2 rows, do the descriptions match?")
print("=" * 60)

identical = 0
different = 0
sample_identical = []
sample_different = []

for pair, rows in pair_to_rows.items():
    if len(rows) != 2:
        continue
    desc_a = rows[0]["description"]
    desc_b = rows[1]["description"]
    if desc_a == desc_b:
        identical += 1
        if len(sample_identical) < 5:
            sample_identical.append(rows)
    else:
        different += 1
        if len(sample_different) < 5:
            sample_different.append(rows)

print(f"Pairs with identical descriptions on both sides: {identical:,}")
print(f"Pairs with DIFFERENT descriptions on both sides: {different:,}")

print("\n--- Sample: identical descriptions (5 examples) ---")
for rows in sample_identical:
    sub_a = id_to_name.get(rows[0]["subject_id"], rows[0]["subject_id"])
    aff_a = id_to_name.get(rows[0]["affected_id"], rows[0]["affected_id"])
    sub_b = id_to_name.get(rows[1]["subject_id"], rows[1]["subject_id"])
    aff_b = id_to_name.get(rows[1]["affected_id"], rows[1]["affected_id"])
    print(f"\n  Pair: {sub_a} <-> {aff_a}")
    print(f"    Row1 [subject={sub_a}, affected={aff_a}]: {rows[0]['description']}")
    print(f"    Row2 [subject={sub_b}, affected={aff_b}]: {rows[1]['description']}")

if sample_different:
    print("\n--- Sample: DIFFERENT descriptions (5 examples) ---")
    for rows in sample_different:
        sub_a = id_to_name.get(rows[0]["subject_id"], rows[0]["subject_id"])
        aff_a = id_to_name.get(rows[0]["affected_id"], rows[0]["affected_id"])
        sub_b = id_to_name.get(rows[1]["subject_id"], rows[1]["subject_id"])
        aff_b = id_to_name.get(rows[1]["affected_id"], rows[1]["affected_id"])
        print(f"\n  Pair: {sub_a} <-> {aff_a}")
        print(f"    Row1 [subject={sub_a}, affected={aff_a}]: {rows[0]['description']}")
        print(f"    Row2 [subject={sub_b}, affected={aff_b}]: {rows[1]['description']}")

# ----- Q2: For each row, does subject's name appear before affected's name in description? -----
print("\n" + "=" * 60)
print("Q2: Does the subject's name appear before the affected's name in the description?")
print("=" * 60)

subject_first = 0
affected_first = 0
neither_found = 0
both_at_same_pos = 0
sample_subject_first = []
sample_affected_first = []
sample_neither = []

# Sample 30000 rows for speed (full dataset is 2.9M)
import random
random.seed(42)
all_rows = []
for rows in pair_to_rows.values():
    all_rows.extend(rows)
sample = random.sample(all_rows, min(30000, len(all_rows)))

for row in sample:
    sub_name = id_to_name.get(row["subject_id"], "")
    aff_name = id_to_name.get(row["affected_id"], "")
    desc = row["description"]
    if not sub_name or not aff_name:
        continue
    sub_pos = desc.find(sub_name)
    aff_pos = desc.find(aff_name)
    if sub_pos == -1 or aff_pos == -1:
        neither_found += 1
        if len(sample_neither) < 5:
            sample_neither.append((row, sub_name, aff_name))
    elif sub_pos < aff_pos:
        subject_first += 1
        if len(sample_subject_first) < 5:
            sample_subject_first.append((row, sub_name, aff_name))
    elif aff_pos < sub_pos:
        affected_first += 1
        if len(sample_affected_first) < 5:
            sample_affected_first.append((row, sub_name, aff_name))
    else:
        both_at_same_pos += 1

total_classified = subject_first + affected_first + neither_found + both_at_same_pos
print(f"Sample size: {total_classified:,}")
print(f"  Subject's name appears BEFORE affected's:  {subject_first:,} ({100*subject_first/total_classified:.1f}%)")
print(f"  Affected's name appears BEFORE subject's:  {affected_first:,} ({100*affected_first/total_classified:.1f}%)")
print(f"  Neither name found in description:         {neither_found:,} ({100*neither_found/total_classified:.1f}%)")
print(f"  Both at same position (rare overlap):      {both_at_same_pos:,}")

print("\n--- Sample: subject-first ---")
for row, sub, aff in sample_subject_first:
    print(f"  [sub={sub}, aff={aff}]")
    print(f"    {row['description']}")

print("\n--- Sample: affected-first ---")
for row, sub, aff in sample_affected_first:
    print(f"  [sub={sub}, aff={aff}]")
    print(f"    {row['description']}")

if sample_neither:
    print("\n--- Sample: neither name found (synonym used?) ---")
    for row, sub, aff in sample_neither:
        print(f"  [sub={sub}, aff={aff}]")
        print(f"    {row['description']}")
