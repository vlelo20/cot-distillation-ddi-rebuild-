"""
Step 11 (FAST REWRITE): Retrieved few-shot examples cache.

The original implementation used "candidate has at least one drug in query's
neighbor set" which let common drugs explode the candidate count to ~1M per
query. This version uses "candidate has BOTH drugs related to the query"
which is the condition that actually makes a candidate similar.

Algorithm:
  For query (A, B):
    1. Build set S_A of drugs in A's top-50 neighbors (plus A itself).
    2. Build set S_B of drugs in B's top-50 neighbors (plus B itself).
    3. For each training pair (X, Y), the pair is a candidate ONLY IF
       (X in S_A and Y in S_B) OR (X in S_B and Y in S_A).
    4. Score the at-most ~50*50 = 2500 candidates (typically far fewer
       because not every pair of drugs has a recorded interaction).

This is the correct restriction: a candidate's similarity is bounded above
by min(sim(A, candidate_drug_1), sim(B, candidate_drug_2)). If either drug
is outside the top-50 of its query counterpart, the pair similarity cannot
exceed the smallest sim in the top-50 cache, which is uniformly low.

We build a (drug, drug) -> [train_idx] index up front so the lookup in step 3
is O(|S_A| * |S_B|) per query, with each lookup being a small list.

Inputs:
  processed_v2/train.jsonl
  processed_v2/test.jsonl
  processed_v2/drug_similarity.pkl
  processed_v2/drug_profiles.json

Outputs:
  processed_v2/retrieved_examples_train.json
  processed_v2/retrieved_examples_test.json
  reports/11_retrieval.txt
"""
import json
import pickle
import random
import time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "11_retrieval.txt"

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
    print("Loading drug profiles, similarity, train/test...")
    with open(PROCESSED / "drug_profiles.json") as f:
        profiles = json.load(f)
    with open(PROCESSED / "drug_similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
    train_rows = load_jsonl(PROCESSED / "train.jsonl")
    test_rows = load_jsonl(PROCESSED / "test.jsonl")
    print(f"  drugs: {len(profiles):,}")
    print(f"  fingerprinted drugs: {len(similarity):,}")
    print(f"  train: {len(train_rows):,}")
    print(f"  test:  {len(test_rows):,}")

    # ---- Build (drug_a, drug_b) -> [train_idx] index, with both orderings keyed
    print("\nBuilding (drug, drug) -> train_idx index...")
    pair_to_idx = defaultdict(list)
    for i, r in enumerate(tqdm(train_rows, desc="indexing")):
        # Canonical pair already has drug_a < drug_b, but we want lookup
        # symmetric for the AxB / BxA matching.
        pair_to_idx[(r["drug_a"], r["drug_b"])].append(i)

    # ---- Build neighbor-to-similarity map for fast lookup
    print("Building per-drug neighbor maps...")
    nbr_map = {}  # drug_id -> {neighbor_id: similarity}
    for did, nbrs in similarity.items():
        nbr_map[did] = dict(nbrs)
        nbr_map[did][did] = 1.0  # include self with similarity 1 for completeness

    # ---- ATC prefix index for biotech fallback
    print("Indexing ATC prefixes (for biotech fallback)...")
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

    # ---- Random fallback pool
    random.seed(42)
    fallback_pool = random.sample(range(len(train_rows)), FALLBACK_POOL_SIZE)

    # ---- Helper: get S_A = set of drugs in A's neighborhood (top 50 + self)
    def get_neighborhood(drug_id):
        if drug_id in nbr_map:
            return set(nbr_map[drug_id].keys())
        return {drug_id}

    # ---- Helper: drug-drug sim lookup
    def drug_sim(d1, d2):
        if d1 == d2:
            return 1.0
        # Prefer d1's neighbor list
        if d1 in nbr_map and d2 in nbr_map[d1]:
            return nbr_map[d1][d2]
        if d2 in nbr_map and d1 in nbr_map[d2]:
            return nbr_map[d2][d1]
        return 0.0

    # ---- Pair similarity
    def pair_sim(a1, b1, a2, b2):
        s1 = drug_sim(a1, a2) * drug_sim(b1, b2)
        s2 = drug_sim(a1, b2) * drug_sim(b1, a2)
        return max(s1, s2)

    # ---- Retrieve K examples for one query row
    def retrieve_for_query(query_a, query_b, query_idx_to_exclude=None):
        S_A = get_neighborhood(query_a)
        S_B = get_neighborhood(query_b)

        # Find candidate training rows: pairs (X, Y) where {X, Y} matches
        # one drug from S_A and one from S_B (or vice versa).
        candidate_idxs = set()
        for x in S_A:
            for y in S_B:
                # Look up both orderings
                if (x, y) in pair_to_idx:
                    for idx in pair_to_idx[(x, y)]:
                        candidate_idxs.add(idx)
                if (y, x) in pair_to_idx:
                    for idx in pair_to_idx[(y, x)]:
                        candidate_idxs.add(idx)

        # Biotech fallback: ATC prefix overlap
        if not candidate_idxs:
            for prefix in drug_to_atc_prefixes.get(query_a, set()) | drug_to_atc_prefixes.get(query_b, set()):
                for idx in atc_prefix_to_train_idx.get(prefix, []):
                    candidate_idxs.add(idx)

        # Final fallback: random pool
        if not candidate_idxs:
            candidate_idxs = set(fallback_pool)

        # Exclude self
        if query_idx_to_exclude is not None:
            candidate_idxs.discard(query_idx_to_exclude)

        if not candidate_idxs:
            return []

        # Score candidates by pair similarity
        scored = []
        for idx in candidate_idxs:
            cand = train_rows[idx]
            sim = pair_sim(query_a, query_b, cand["drug_a"], cand["drug_b"])
            scored.append((sim, idx))
        scored.sort(reverse=True)

        # Take top K
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

    # ---- Process train
    print(f"\nRetrieving K={K} examples for {len(train_rows):,} train rows...")
    start = time.time()
    train_retrieved = []
    for i, r in enumerate(tqdm(train_rows, desc="train", smoothing=0.05)):
        examples = retrieve_for_query(r["drug_a"], r["drug_b"], query_idx_to_exclude=i)
        train_retrieved.append(examples)
    train_time = time.time() - start
    print(f"  train done in {train_time:.0f}s ({len(train_rows)/max(train_time,1):.0f} rows/sec)")

    # ---- Process test
    print(f"\nRetrieving K={K} examples for {len(test_rows):,} test rows...")
    start = time.time()
    test_retrieved = []
    for r in tqdm(test_rows, desc="test", smoothing=0.05):
        examples = retrieve_for_query(r["drug_a"], r["drug_b"])
        test_retrieved.append(examples)
    test_time = time.time() - start
    print(f"  test done in {test_time:.0f}s ({len(test_rows)/max(test_time,1):.0f} rows/sec)")

    # ---- Save
    print("\nWriting outputs...")
    with open(PROCESSED / "retrieved_examples_train.json", "w") as f:
        json.dump(train_retrieved, f)
    with open(PROCESSED / "retrieved_examples_test.json", "w") as f:
        json.dump(test_retrieved, f)

    # ---- Stats
    train_zero = sum(1 for r in train_retrieved if not r)
    test_zero = sum(1 for r in test_retrieved if not r)
    train_avg_sim = sum(e["similarity"] for ex in train_retrieved for e in ex) / max(sum(len(ex) for ex in train_retrieved), 1)
    test_avg_sim = sum(e["similarity"] for ex in test_retrieved for e in ex) / max(sum(len(ex) for ex in test_retrieved), 1)

    train_size_mb = (PROCESSED / "retrieved_examples_train.json").stat().st_size / 1e6
    test_size_mb = (PROCESSED / "retrieved_examples_test.json").stat().st_size / 1e6

    lines = [
        "Step 11 (FAST) -- Retrieved examples cache",
        "=" * 60,
        f"K (examples per row):              {K}",
        f"Train rows processed:              {len(train_rows):,}",
        f"  rows with 0 examples retrieved:  {train_zero:,}",
        f"  avg similarity of retrieved:     {train_avg_sim:.3f}",
        f"  time:                            {train_time:.0f}s",
        f"Test rows processed:               {len(test_rows):,}",
        f"  rows with 0 examples retrieved:  {test_zero:,}",
        f"  avg similarity of retrieved:     {test_avg_sim:.3f}",
        f"  time:                            {test_time:.0f}s",
        "",
        "Outputs:",
        f"  retrieved_examples_train.json  ({train_size_mb:.1f} MB)",
        f"  retrieved_examples_test.json   ({test_size_mb:.1f} MB)",
    ]
    report = "\n".join(lines)
    print()
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
