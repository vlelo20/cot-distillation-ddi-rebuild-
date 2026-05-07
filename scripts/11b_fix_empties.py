"""
Step 11b: Fix the 6.85% of train rows with 0 retrieved examples.

The bug in Step 11: when S_A x S_B lookup found exactly one candidate (the
query itself), candidate_idxs was non-empty so the ATC and random fallbacks
were skipped. Then self-exclusion emptied the set and we returned [].

Fix: move self-exclusion BEFORE the fallback chain so fallbacks fire when
the only candidate was the query itself.

This script:
  1. Loads the existing retrieved_examples_train.json
  2. Identifies rows with empty retrieval
  3. Re-runs retrieval ONLY on those rows with corrected fallback ordering
  4. Patches them back into the cache
  5. Writes the fixed file

Total time: ~3-5 minutes (only re-running 79K rows, not all 1.16M).
"""
import json
import pickle
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "11b_fix_empties.txt"

K = 5
ATC_PREFIX_LEN = 5
FALLBACK_POOL_SIZE = 200


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    print("Loading existing cache and base data...")
    with open(PROCESSED / "retrieved_examples_train.json") as f:
        cache = json.load(f)
    with open(PROCESSED / "drug_profiles.json") as f:
        profiles = json.load(f)
    with open(PROCESSED / "drug_similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
    train_rows = load_jsonl(PROCESSED / "train.jsonl")

    print(f"  cache entries: {len(cache):,}")
    print(f"  train rows: {len(train_rows):,}")

    # Identify empty-retrieval rows
    empty_indices = [i for i, examples in enumerate(cache) if not examples]
    print(f"  rows with 0 examples: {len(empty_indices):,} "
          f"({100*len(empty_indices)/len(cache):.2f}%)")

    if not empty_indices:
        print("Nothing to fix. Exiting.")
        return

    # Build the indexes (same as Step 11)
    print("\nBuilding (drug, drug) -> train_idx index...")
    pair_to_idx = defaultdict(list)
    for i, r in enumerate(tqdm(train_rows, desc="indexing")):
        pair_to_idx[(r["drug_a"], r["drug_b"])].append(i)

    print("Building per-drug neighbor maps...")
    nbr_map = {}
    for did, nbrs in similarity.items():
        nbr_map[did] = dict(nbrs)
        nbr_map[did][did] = 1.0

    print("Indexing ATC prefixes...")
    drug_to_atc_prefixes = {}
    atc_prefix_to_train_idx = defaultdict(list)
    for did, prof in profiles.items():
        prefixes = set()
        for atc in prof.get("atc_codes", []):
            if len(atc) >= ATC_PREFIX_LEN:
                prefixes.add(atc[:ATC_PREFIX_LEN])
        drug_to_atc_prefixes[did] = prefixes
    for i, r in enumerate(train_rows):
        for did in (r["drug_a"], r["drug_b"]):
            for prefix in drug_to_atc_prefixes.get(did, set()):
                atc_prefix_to_train_idx[prefix].append(i)

    random.seed(42)
    fallback_pool = random.sample(range(len(train_rows)), FALLBACK_POOL_SIZE)

    def get_neighborhood(drug_id):
        if drug_id in nbr_map:
            return set(nbr_map[drug_id].keys())
        return {drug_id}

    def drug_sim(d1, d2):
        if d1 == d2:
            return 1.0
        if d1 in nbr_map and d2 in nbr_map[d1]:
            return nbr_map[d1][d2]
        if d2 in nbr_map and d1 in nbr_map[d2]:
            return nbr_map[d2][d1]
        return 0.0

    def pair_sim(a1, b1, a2, b2):
        s1 = drug_sim(a1, a2) * drug_sim(b1, b2)
        s2 = drug_sim(a1, b2) * drug_sim(b1, a2)
        return max(s1, s2)

    # Corrected retrieve function with self-exclusion BEFORE fallbacks
    def retrieve_for_query_fixed(query_a, query_b, query_idx_to_exclude):
        S_A = get_neighborhood(query_a)
        S_B = get_neighborhood(query_b)

        # Step 1: S_A x S_B lookup
        candidate_idxs = set()
        for x in S_A:
            for y in S_B:
                if (x, y) in pair_to_idx:
                    for idx in pair_to_idx[(x, y)]:
                        candidate_idxs.add(idx)
                if (y, x) in pair_to_idx:
                    for idx in pair_to_idx[(y, x)]:
                        candidate_idxs.add(idx)

        # Self-exclude FIRST so fallbacks fire if needed
        candidate_idxs.discard(query_idx_to_exclude)

        # Step 2: ATC fallback if empty
        if not candidate_idxs:
            for prefix in (drug_to_atc_prefixes.get(query_a, set())
                           | drug_to_atc_prefixes.get(query_b, set())):
                for idx in atc_prefix_to_train_idx.get(prefix, []):
                    candidate_idxs.add(idx)
            candidate_idxs.discard(query_idx_to_exclude)

        # Step 3: random fallback if still empty
        if not candidate_idxs:
            candidate_idxs = set(fallback_pool)
            candidate_idxs.discard(query_idx_to_exclude)

        if not candidate_idxs:
            return []

        scored = []
        for idx in candidate_idxs:
            cand = train_rows[idx]
            sim = pair_sim(query_a, query_b, cand["drug_a"], cand["drug_b"])
            scored.append((sim, idx))
        scored.sort(reverse=True)

        result = []
        for sim, idx in scored[:K]:
            cand = train_rows[idx]
            result.append({
                "drug_a": cand["drug_a"],
                "drug_b": cand["drug_b"],
                "description": cand["description"],
                "label": cand["label"],
                "similarity": float(sim),
            })
        return result

    # Re-run on empty rows only
    print(f"\nRe-running retrieval on {len(empty_indices):,} empty rows...")
    fixed_count = 0
    still_empty_count = 0
    fallback_path_counts = {"sasb": 0, "atc": 0, "random": 0, "empty": 0}

    for i in tqdm(empty_indices, desc="patching"):
        r = train_rows[i]
        examples = retrieve_for_query_fixed(r["drug_a"], r["drug_b"],
                                             query_idx_to_exclude=i)
        cache[i] = examples
        if examples:
            fixed_count += 1
            # Crude classification of which fallback path was used
            avg_sim = sum(e["similarity"] for e in examples) / len(examples)
            if avg_sim > 0.1:
                fallback_path_counts["sasb"] += 1
            elif avg_sim > 0.0:
                fallback_path_counts["atc"] += 1
            else:
                fallback_path_counts["random"] += 1
        else:
            still_empty_count += 1
            fallback_path_counts["empty"] += 1

    print(f"\nResults:")
    print(f"  Originally empty:    {len(empty_indices):,}")
    print(f"  Fixed:               {fixed_count:,}")
    print(f"  Still empty:         {still_empty_count:,}")
    print(f"  By fallback path:")
    for path, count in fallback_path_counts.items():
        print(f"    {path}: {count:,}")

    # Sanity check: verify lengths match
    assert len(cache) == len(train_rows), \
        f"Cache length mismatch: {len(cache)} vs {len(train_rows)}"

    # Save patched cache
    print("\nWriting patched cache...")
    with open(PROCESSED / "retrieved_examples_train.json", "w") as f:
        json.dump(cache, f)
    out_size_mb = (PROCESSED / "retrieved_examples_train.json").stat().st_size / 1e6
    print(f"  size: {out_size_mb:.1f} MB")

    # Verify final state
    final_empties = sum(1 for ex in cache if not ex)
    print(f"\nFinal empty count in cache: {final_empties:,} "
          f"({100*final_empties/len(cache):.3f}%)")

    # Report
    lines = [
        "Step 11b -- Fix empty retrieval rows",
        "=" * 60,
        f"Train rows total:           {len(cache):,}",
        f"Originally empty:           {len(empty_indices):,} "
        f"({100*len(empty_indices)/len(cache):.2f}%)",
        f"Fixed:                      {fixed_count:,}",
        f"Still empty after fix:      {still_empty_count:,}",
        f"Final empty rate:           {100*final_empties/len(cache):.3f}%",
        "",
        "Fallback path breakdown for re-run rows:",
        f"  S_A x S_B match:          {fallback_path_counts['sasb']:,}",
        f"  ATC overlap fallback:     {fallback_path_counts['atc']:,}",
        f"  Random pool fallback:     {fallback_path_counts['random']:,}",
        f"  Truly empty:              {fallback_path_counts['empty']:,}",
        "",
        f"Output: retrieved_examples_train.json ({out_size_mb:.1f} MB) -- patched in place",
    ]
    report = "\n".join(lines)
    print()
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
