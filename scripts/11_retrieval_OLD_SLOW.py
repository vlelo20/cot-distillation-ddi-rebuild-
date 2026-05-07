"""
Step 11: Retrieved few-shot examples cache.

For every interaction (in train and test), find the K=5 most similar TRAINING
interactions to use as in-context examples for the teacher prompt.

Pair similarity for a query (A, B) and a candidate (X, Y):
  pair_sim = max(sim(A,X) * sim(B,Y), sim(A,Y) * sim(B,X))
This is symmetric in drug ordering and bounded in [0, 1].

To avoid O(N*M) comparisons (which would be 1.7 trillion), we use the precomputed
top-50 neighbor lists from Step 10. For query drugs A and B, candidates are
restricted to training pairs where one drug is in A's neighbors and the other is
in B's neighbors. This narrows from M=1.16M candidates to typically ~100-500.

For drugs without fingerprints (biotech), fall back: skip neighbor lookup, draw
candidates from training pairs sharing at least one ATC code or category. If
nothing matches, use a small set of random training examples per row.

Inputs:
  processed_v2/train.jsonl
  processed_v2/test.jsonl
  processed_v2/drug_similarity.pkl
  processed_v2/drug_profiles.json

Outputs:
  processed_v2/retrieved_examples_train.json  -- per-row list of 5 example dicts
  processed_v2/retrieved_examples_test.json
  reports/11_retrieval.txt
"""
import json
import pickle
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "11_retrieval.txt"

K = 5  # number of examples per query
MIN_SIM_FOR_NEIGHBOR_LOOKUP = 0.0  # use all top-50 neighbors regardless of similarity

# For biotech drugs, retrieve N candidates from training pairs sharing at least
# one ATC prefix (5-char code = chemical subgroup level)
ATC_PREFIX_LEN = 5

# Random fallback pool size when nothing better matches
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

    # ---- Build training-pair lookup: drug_id -> list of training row indices that include it
    print("\nIndexing training pairs by drug membership...")
    drug_to_train_idx = defaultdict(list)
    for i, r in enumerate(train_rows):
        drug_to_train_idx[r["drug_a"]].append(i)
        drug_to_train_idx[r["drug_b"]].append(i)

    # ---- Build ATC index for biotech fallback
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

    # ---- Random fallback pool (for drugs with neither fingerprint nor ATC)
    random.seed(42)
    fallback_pool = random.sample(range(len(train_rows)), FALLBACK_POOL_SIZE)

    # ---- Lookup helper: similarity between two drug ids
    def drug_sim(d1, d2):
        if d1 == d2:
            return 1.0
        # similarity is keyed by drug; look up d2 in d1's neighbor list
        for nid, sim in similarity.get(d1, []):
            if nid == d2:
                return sim
        for nid, sim in similarity.get(d2, []):
            if nid == d1:
                return sim
        return 0.0

    # ---- Pair similarity
    def pair_sim(a1, b1, a2, b2):
        return max(drug_sim(a1, a2) * drug_sim(b1, b2),
                   drug_sim(a1, b2) * drug_sim(b1, a2))

    # ---- Get candidate training row indices for a query (A, B)
    def get_candidates(query_a, query_b, query_idx_to_exclude=None):
        candidates = set()

        # Primary: training pairs where one drug is in query's drug-similarity neighbors
        for nid, _ in similarity.get(query_a, [])[:50]:
            for idx in drug_to_train_idx.get(nid, []):
                candidates.add(idx)
        for nid, _ in similarity.get(query_b, [])[:50]:
            for idx in drug_to_train_idx.get(nid, []):
                candidates.add(idx)

        # Also include exact-drug-match training pairs
        for idx in drug_to_train_idx.get(query_a, []):
            candidates.add(idx)
        for idx in drug_to_train_idx.get(query_b, []):
            candidates.add(idx)

        # Biotech fallback: ATC prefix match
        if not candidates:
            for prefix in drug_to_atc_prefixes.get(query_a, set()) | drug_to_atc_prefixes.get(query_b, set()):
                for idx in atc_prefix_to_train_idx.get(prefix, []):
                    candidates.add(idx)

        # Final fallback: random
        if not candidates:
            candidates = set(fallback_pool)

        # Exclude the query itself if it's a training row
        if query_idx_to_exclude is not None:
            candidates.discard(query_idx_to_exclude)
        return list(candidates)

    # ---- Retrieve K examples for one query row
    def retrieve_for_query(query_a, query_b, query_idx=None):
        candidates = get_candidates(query_a, query_b, query_idx)
        if not candidates:
            return []

        # Score each candidate
        scored = []
        for idx in candidates:
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

    # ---- Process train (each train row retrieves from training set, excluding itself)
    print(f"\nRetrieving K={K} examples for {len(train_rows):,} train rows...")
    train_retrieved = []
    for i, r in enumerate(tqdm(train_rows, desc="train")):
        examples = retrieve_for_query(r["drug_a"], r["drug_b"], query_idx=i)
        train_retrieved.append(examples)

    # ---- Process test
    print(f"\nRetrieving K={K} examples for {len(test_rows):,} test rows...")
    test_retrieved = []
    for r in tqdm(test_rows, desc="test"):
        examples = retrieve_for_query(r["drug_a"], r["drug_b"], query_idx=None)
        test_retrieved.append(examples)

    # ---- Save
    print("\nWriting outputs...")
    with open(PROCESSED / "retrieved_examples_train.json", "w") as f:
        json.dump(train_retrieved, f)
    with open(PROCESSED / "retrieved_examples_test.json", "w") as f:
        json.dump(test_retrieved, f)

    # ---- Stats
    train_zero_examples = sum(1 for r in train_retrieved if not r)
    test_zero_examples = sum(1 for r in test_retrieved if not r)
    train_avg_sim = sum(e["similarity"] for ex in train_retrieved for e in ex) / max(sum(len(ex) for ex in train_retrieved), 1)
    test_avg_sim = sum(e["similarity"] for ex in test_retrieved for e in ex) / max(sum(len(ex) for ex in test_retrieved), 1)

    train_size_mb = (PROCESSED / "retrieved_examples_train.json").stat().st_size / 1e6
    test_size_mb = (PROCESSED / "retrieved_examples_test.json").stat().st_size / 1e6

    lines = [
        "Step 11 -- Retrieved examples cache",
        "=" * 60,
        f"K (examples per row):              {K}",
        f"Train rows processed:              {len(train_rows):,}",
        f"  rows with 0 examples retrieved:  {train_zero_examples:,}",
        f"  avg similarity of retrieved:     {train_avg_sim:.3f}",
        f"Test rows processed:               {len(test_rows):,}",
        f"  rows with 0 examples retrieved:  {test_zero_examples:,}",
        f"  avg similarity of retrieved:     {test_avg_sim:.3f}",
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
