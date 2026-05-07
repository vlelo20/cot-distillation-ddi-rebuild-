"""
Step 3c: Deduplicate interactions and normalize drug ordering.

The previous script (03_parse_interactions.py) wrote 2,915,498 rows but every
two-way pair has identical descriptions on both sides — so half of those are
redundant. We deduplicate on (frozenset({drug_a, drug_b}), description).

Since the original 'subject_id'/'affected_id' fields don't carry directional
meaning (Q2 in the inspection script showed 50/50), we normalize by always
putting the lexicographically smaller drugbank_id first as drug_a. The
description remains the source of truth for actual directionality.

Input:
  processed_v2/interactions_full.jsonl  (2.9M rows, doubled)

Outputs:
  processed_v2/interactions_dedup.jsonl  (~1.46M rows, canonical)
  reports/03c_dedup.txt
"""
import json
from pathlib import Path
from tqdm import tqdm

ROOT = Path.home() / "ddiproject"
IN_PATH = ROOT / "processed_v2" / "interactions_full.jsonl"
OUT_PATH = ROOT / "processed_v2" / "interactions_dedup.jsonl"
REPORT = ROOT / "reports" / "03c_dedup.txt"


def main():
    print("Reading and deduplicating...")
    seen = {}  # (drug_a, drug_b, description) -> row dict, with drug_a < drug_b

    n_input = 0
    n_self_loop = 0

    with open(IN_PATH) as f:
        for line in tqdm(f, desc="rows", unit="row"):
            row = json.loads(line)
            n_input += 1
            sub = row["subject_id"]
            aff = row["affected_id"]
            desc = row["description"]

            if sub == aff:
                n_self_loop += 1
                continue

            # Canonical ordering: smaller ID first
            if sub < aff:
                drug_a, drug_b = sub, aff
            else:
                drug_a, drug_b = aff, sub

            key = (drug_a, drug_b, desc)
            if key not in seen:
                seen[key] = {
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "description": desc,
                }

    n_unique = len(seen)
    print(f"\nRead {n_input:,} rows. Unique interactions: {n_unique:,}")
    print(f"Compression ratio: {n_input / max(n_unique, 1):.2f}x")
    print(f"Self-loops dropped: {n_self_loop:,}")

    print(f"Writing {OUT_PATH.name}...")
    with open(OUT_PATH, "w") as f:
        for row in seen.values():
            f.write(json.dumps(row) + "\n")

    # Quick sanity stats
    drugs_a = {r["drug_a"] for r in seen.values()}
    drugs_b = {r["drug_b"] for r in seen.values()}
    distinct_drugs = drugs_a | drugs_b

    desc_counts = {}
    for r in seen.values():
        desc_counts[r["description"]] = desc_counts.get(r["description"], 0) + 1
    distinct_descriptions = len(desc_counts)
    most_common = sorted(desc_counts.items(), key=lambda x: -x[1])[:5]

    lines = [
        "Step 3c -- Deduplication",
        "=" * 60,
        f"Input rows:                    {n_input:,}",
        f"Self-loops removed:            {n_self_loop:,}",
        f"Unique interactions:           {n_unique:,}",
        f"Compression ratio:             {n_input / max(n_unique, 1):.2f}x",
        f"Distinct drugs in pairs:       {len(distinct_drugs):,}",
        f"Distinct description strings:  {distinct_descriptions:,}",
        "",
        "Top 5 most common descriptions (these will become labels in Step 4):",
    ]
    for desc, count in most_common:
        lines.append(f"  ({count:,}x) {desc[:120]}")

    lines.extend([
        "",
        f"Output: {OUT_PATH.name}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)",
    ])
    report = "\n".join(lines)
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
