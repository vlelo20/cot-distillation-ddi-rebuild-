#!/usr/bin/env python3
"""
Step 11c: Pathway-based retrieval for DDI prompt generation.

Compares against Step 11/11b Tanimoto retrieval. Uses enzyme / transporter /
target / carrier overlap (with action roles) as the similarity signal instead
of structural fingerprints.

Output schema is identical to retrieved_examples_*.json (drop-in for Stage 1
teacher prompts).

Place this at /home/vian/ddiproject/step11c_pathway_retrieval.py and run from
that directory. Inputs and outputs default to /home/vian/ddiproject/processed_v2/.
"""

import argparse
import json
import logging
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# Category weights — enzymes are most predictive of DDI mechanism, then
# transporters (drug movement), targets (PD synergy), carriers (rare).
WEIGHTS: Dict[str, float] = {
    "enzyme": 1.0,
    "transporter": 0.7,
    "target": 0.5,
    "carrier": 0.3,
}

Feature = Tuple[str, str, str]  # (category, uniprot, action)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_drug_features(profiles_path: Path) -> Dict[str, Set[Feature]]:
    """Map drug_id -> set of (category, uniprot, action) tuples (humans only)."""
    log.info("Loading drug profiles...")
    with open(profiles_path) as f:
        profiles = json.load(f)

    cat_map = [
        ("enzymes", "enzyme"),
        ("transporters", "transporter"),
        ("targets", "target"),
        ("carriers", "carrier"),
    ]

    features: Dict[str, Set[Feature]] = {}
    n_with_features = 0
    for did, p in profiles.items():
        feats: Set[Feature] = set()
        for field, cat in cat_map:
            for entry in p.get(field, []) or []:
                if entry.get("organism") != "Humans":
                    continue
                up = entry.get("uniprot") or ""
                if not up:
                    continue
                actions = entry.get("actions") or ["unknown"]
                for action in actions:
                    feats.add((cat, up, str(action).lower()))
        features[did] = feats
        if feats:
            n_with_features += 1

    log.info(
        f"Loaded {len(profiles):,} drugs; "
        f"{n_with_features:,} ({100*n_with_features/len(profiles):.1f}%) have >=1 pathway feature"
    )
    nonzero = sorted(len(f) for f in features.values() if f)
    if nonzero:
        log.info(
            f"Features per drug (nonzero only): "
            f"min={nonzero[0]} median={nonzero[len(nonzero)//2]} "
            f"mean={sum(nonzero)/len(nonzero):.1f} max={nonzero[-1]}"
        )
    return features


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _w(f: Feature) -> float:
    return WEIGHTS.get(f[0], 0.0)


def drug_sim(fa: Set[Feature], fb: Set[Feature]) -> float:
    """Weighted Jaccard: sum w(intersection) / sum w(union)."""
    if not fa or not fb:
        return 0.0
    inter = fa & fb
    if not inter:
        return 0.0
    num = sum(_w(f) for f in inter)
    den = sum(_w(f) for f in fa | fb)
    return num / den if den > 0 else 0.0


def pair_sim(
    qa: Set[Feature], qb: Set[Feature],
    ca: Set[Feature], cb: Set[Feature],
) -> float:
    """
    Symmetric pair similarity: try both pairings (qa<->ca,qb<->cb) and
    (qa<->cb,qb<->ca), take the geometric mean of the better one.
    Geometric mean keeps the score in [0, 1] and stays linearly comparable
    to per-drug Jaccard.
    """
    s1 = drug_sim(qa, ca) * drug_sim(qb, cb)
    s2 = drug_sim(qa, cb) * drug_sim(qb, ca)
    s = s1 if s1 >= s2 else s2
    return s ** 0.5


# ---------------------------------------------------------------------------
# Inverted index
# ---------------------------------------------------------------------------

def build_pair_index(
    train_pairs: List[Tuple[str, str]],
    drug_features: Dict[str, Set[Feature]],
) -> Dict[Feature, List[int]]:
    """feature -> list of train pair indices that contain it on either drug."""
    log.info("Building inverted index over train pairs...")
    index: Dict[Feature, List[int]] = defaultdict(list)
    for i, (a, b) in enumerate(train_pairs):
        feats = drug_features.get(a, set()) | drug_features.get(b, set())
        for f in feats:
            index[f].append(i)
        if (i + 1) % 200_000 == 0:
            log.info(f"  indexed {i+1:,}/{len(train_pairs):,} pairs")
    n_post = sum(len(v) for v in index.values())
    log.info(
        f"Index: {len(index):,} unique features, {n_post:,} postings "
        f"(avg {n_post/max(1,len(index)):.1f} pairs/feature)"
    )
    return index


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query_a: str, query_b: str,
    drug_features: Dict[str, Set[Feature]],
    train_pairs: List[Tuple[str, str]],
    train_meta: List[dict],
    index: Dict[Feature, List[int]],
    k: int = 5,
    max_candidates: int = 2000,
    exclude_self: bool = True,
) -> List[dict]:
    qa = drug_features.get(query_a, set())
    qb = drug_features.get(query_b, set())
    qfeats = qa | qb
    if not qfeats:
        return []

    # Gather candidates with overlap counts; cap by overlap to keep scoring cheap.
    overlap: Counter = Counter()
    for f in qfeats:
        for idx in index.get(f, ()):
            overlap[idx] += 1

    if len(overlap) > max_candidates:
        candidates = [idx for idx, _ in overlap.most_common(max_candidates)]
    else:
        candidates = list(overlap.keys())

    # Pre-compute per-drug sims against unique candidate drugs (avoids redoing
    # the same Jaccard hundreds of times).
    unique_drugs: Set[str] = set()
    for idx in candidates:
        ca, cb = train_pairs[idx]
        unique_drugs.add(ca)
        unique_drugs.add(cb)

    sim_qa: Dict[str, float] = {}
    sim_qb: Dict[str, float] = {}
    for d in unique_drugs:
        df = drug_features.get(d, set())
        sim_qa[d] = drug_sim(qa, df)
        sim_qb[d] = drug_sim(qb, df)

    scored: List[Tuple[float, int]] = []
    for idx in candidates:
        ca, cb = train_pairs[idx]
        if exclude_self and (
            (ca == query_a and cb == query_b)
            or (ca == query_b and cb == query_a)
        ):
            continue
        s1 = sim_qa[ca] * sim_qb[cb]
        s2 = sim_qa[cb] * sim_qb[ca]
        s = (s1 if s1 >= s2 else s2) ** 0.5
        if s > 0:
            scored.append((s, idx))

    scored.sort(reverse=True)
    out: List[dict] = []
    for s, idx in scored[:k]:
        m = train_meta[idx]
        out.append({
            "drug_a": m["drug_a"],
            "drug_b": m["drug_b"],
            "description": m["description"],
            "label": m["label"],
            "similarity": s,
        })
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/home/vian/ddiproject/processed_v2"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/home/vian/ddiproject/processed_v2"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=2000,
                    help="Cap candidates per query (by feature overlap count) before scoring.")
    ap.add_argument("--target", choices=["train", "test", "both"], default="test")
    ap.add_argument("--pilot", type=int, default=None,
                    help="If set, retrieve for first N queries only.")
    args = ap.parse_args()

    t0 = time.time()
    drug_features = build_drug_features(args.data_dir / "drug_profiles.json")

    log.info("Loading train.jsonl...")
    train_meta: List[dict] = []
    with open(args.data_dir / "train.jsonl") as f:
        for line in f:
            train_meta.append(json.loads(line))
    train_pairs = [(m["drug_a"], m["drug_b"]) for m in train_meta]
    log.info(f"Loaded {len(train_pairs):,} train pairs")

    # Coverage diagnostic on the train set itself
    n_both = sum(
        1 for a, b in train_pairs
        if drug_features.get(a) and drug_features.get(b)
    )
    n_either = sum(
        1 for a, b in train_pairs
        if drug_features.get(a) or drug_features.get(b)
    )
    log.info(
        f"Train coverage: both drugs have features: {n_both:,} "
        f"({100*n_both/len(train_pairs):.1f}%); "
        f"at least one: {n_either:,} ({100*n_either/len(train_pairs):.1f}%)"
    )

    index = build_pair_index(train_pairs, drug_features)

    targets = ["train", "test"] if args.target == "both" else [args.target]
    for target in targets:
        log.info(f"=== Retrieval for {target} ===")
        if target == "train":
            queries = train_meta
            exclude_self = True
        else:
            queries = []
            with open(args.data_dir / "test.jsonl") as f:
                for line in f:
                    queries.append(json.loads(line))
            log.info(f"Loaded {len(queries):,} test queries")
            exclude_self = False

        if args.pilot:
            queries = queries[: args.pilot]
            log.info(f"Pilot mode: using first {len(queries):,} queries")

        results: List[List[dict]] = []
        empty = 0
        t_r = time.time()
        for i, q in enumerate(queries):
            top = retrieve(
                q["drug_a"], q["drug_b"],
                drug_features, train_pairs, train_meta, index,
                k=args.k, max_candidates=args.max_candidates,
                exclude_self=exclude_self,
            )
            if not top:
                empty += 1
            results.append(top)
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t_r
                rate = (i + 1) / elapsed
                eta = (len(queries) - i - 1) / rate
                log.info(
                    f"  {i+1:,}/{len(queries):,}  "
                    f"({rate:.1f}/s, ETA {eta:.0f}s, empties: {empty})"
                )

        suffix = f"pilot{args.pilot}" if args.pilot else "full"
        out_path = args.out_dir / f"retrieved_examples_pathway_{target}_{suffix}.json"
        log.info(
            f"Writing {out_path.name}: {len(results):,} entries, "
            f"{empty:,} empty ({100*empty/max(1,len(results)):.1f}%)"
        )
        with open(out_path, "w") as f:
            json.dump(results, f)
        log.info(f"  -> {out_path.stat().st_size / 1e6:.1f} MB")

    log.info(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
