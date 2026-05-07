"""
Step 10: Drug-drug Tanimoto similarity matrix.

For each drug with a Morgan fingerprint, find its top-K most similar other drugs
by Tanimoto coefficient. Store as a per-drug ranked neighbor list (not the full
dense matrix, which would be 860 MB at float32).

Tanimoto formula for binary fingerprints:
  T(A, B) = |A ∩ B| / |A ∪ B|
          = popcount(A & B) / popcount(A | B)

Vectorized using numpy: stack all fingerprints into a single (N, 2048) bool array,
then compute similarity in batches of M drugs at a time against the full set,
which keeps peak memory bounded while leveraging numpy's broadcasting.

Inputs:
  processed_v2/drug_fingerprints.pkl

Outputs:
  processed_v2/drug_similarity.pkl  -- {drug_id: [(neighbor_id, similarity), ...]}
                                        sorted desc by similarity, top-K per drug
  reports/10_similarity.txt
"""
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "10_similarity.txt"

TOP_K = 50      # cache top 50 neighbors per drug
BATCH_SIZE = 256  # drugs per batch for similarity computation


def main():
    print("Loading fingerprints...")
    with open(PROCESSED / "drug_fingerprints.pkl", "rb") as f:
        fps = pickle.load(f)

    drug_ids = sorted(fps.keys())
    n = len(drug_ids)
    print(f"  {n:,} drugs with fingerprints")

    # Stack into a single (n, 2048) bool array. Bool gives us efficient
    # bitwise ops without packing/unpacking, and at n=14617 it's ~30 MB.
    print("Building fingerprint matrix...")
    fp_matrix = np.zeros((n, 2048), dtype=bool)
    for i, did in enumerate(drug_ids):
        fp_matrix[i] = fps[did].astype(bool)

    # Precompute popcount per drug (number of bits set)
    popcounts = fp_matrix.sum(axis=1).astype(np.int32)
    print(f"  matrix: {fp_matrix.shape}, popcount range: {popcounts.min()} - {popcounts.max()}")

    # Compute similarity in batches. For batch B (rows i_start:i_end),
    # we compute Tanimoto vs all n drugs:
    #   intersection[i,j] = popcount(B[i] & fp_matrix[j])
    #   union[i,j]        = popcount(B[i] | fp_matrix[j])
    #                     = popcounts[i] + popcounts[j] - intersection[i,j]
    #   tanimoto[i,j]     = intersection / union
    #
    # Computing intersection via matrix multiplication on bool arrays cast to
    # int8 is the fastest pure-numpy approach: A @ B.T gives count of shared bits.

    print(f"\nComputing top-{TOP_K} neighbors per drug "
          f"(batch_size={BATCH_SIZE})...")

    fp_int8 = fp_matrix.astype(np.int8)
    neighbors = {}  # drug_id -> [(neighbor_id, similarity), ...]
    start = time.time()

    for batch_start in tqdm(range(0, n, BATCH_SIZE), desc="batches"):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch = fp_int8[batch_start:batch_end]  # (b, 2048)

        # Intersection counts: (b, n) matrix
        intersection = batch @ fp_int8.T  # int dot product

        # Union: popcounts[i] + popcounts[j] - intersection[i, j]
        # popcounts is shape (n,), intersection is (b, n)
        # we want union[i, j] = batch_popcounts[i] + popcounts[j] - intersection[i, j]
        batch_popcounts = popcounts[batch_start:batch_end][:, None]  # (b, 1)
        union = batch_popcounts + popcounts[None, :] - intersection  # (b, n)

        # Tanimoto, with safe divide (some all-zero fps could give 0/0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tanimoto = np.where(union > 0, intersection / union, 0.0)

        # For each drug in batch, find top-K (excluding self)
        for local_i in range(batch_end - batch_start):
            global_i = batch_start + local_i
            sims = tanimoto[local_i].copy()
            sims[global_i] = -1  # exclude self

            # Argpartition is faster than full sort for top-K
            top_idx = np.argpartition(sims, -TOP_K)[-TOP_K:]
            # Sort just those K by similarity descending
            top_idx_sorted = top_idx[np.argsort(-sims[top_idx])]

            neighbors[drug_ids[global_i]] = [
                (drug_ids[j], float(sims[j])) for j in top_idx_sorted
            ]

    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.1f}s ({n*n/2/elapsed:,.0f} pairs/sec)")

    # Save
    out_path = PROCESSED / "drug_similarity.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(neighbors, f, protocol=pickle.HIGHEST_PROTOCOL)
    out_size_mb = out_path.stat().st_size / 1e6

    # Sanity checks
    print("\nSanity checks:")
    # Aspirin should be similar to other NSAIDs / salicylates
    aspirin_id = "DB00945"
    if aspirin_id in neighbors:
        print(f"  Aspirin's top 5 most similar:")
        # Need drug names -- load profiles
        with open(PROCESSED / "drug_profiles.json") as f:
            profiles = json.load(f)
        for nid, sim in neighbors[aspirin_id][:5]:
            name = profiles.get(nid, {}).get("name", nid)
            print(f"    {nid}  sim={sim:.3f}  {name}")

    # Distribution of top-1 similarity
    top1_sims = [neighbors[d][0][1] for d in drug_ids if neighbors[d]]
    top1_sims = np.array(top1_sims)
    print(f"\n  Top-1 similarity distribution:")
    print(f"    median: {np.median(top1_sims):.3f}")
    print(f"    mean:   {np.mean(top1_sims):.3f}")
    print(f"    p10:    {np.percentile(top1_sims, 10):.3f}")
    print(f"    p90:    {np.percentile(top1_sims, 90):.3f}")
    print(f"    drugs with top-1 sim > 0.7: {int(np.sum(top1_sims > 0.7)):,}")
    print(f"    drugs with top-1 sim > 0.5: {int(np.sum(top1_sims > 0.5)):,}")

    # Report
    lines = [
        "Step 10 -- Drug-drug similarity",
        "=" * 60,
        f"Drugs in similarity index:    {n:,}",
        f"Top-K cached per drug:        {TOP_K}",
        f"Total (drug, neighbor) pairs: {n * TOP_K:,}",
        f"Compute time:                 {elapsed:.1f}s",
        "",
        "Top-1 similarity distribution:",
        f"  median: {np.median(top1_sims):.3f}",
        f"  mean:   {np.mean(top1_sims):.3f}",
        f"  p10:    {np.percentile(top1_sims, 10):.3f}",
        f"  p90:    {np.percentile(top1_sims, 90):.3f}",
        f"  drugs with top-1 > 0.7: {int(np.sum(top1_sims > 0.7)):,}",
        f"  drugs with top-1 > 0.5: {int(np.sum(top1_sims > 0.5)):,}",
        "",
        "Aspirin (DB00945) top-5 neighbors (sanity check):",
    ]
    if aspirin_id in neighbors:
        for nid, sim in neighbors[aspirin_id][:5]:
            name = profiles.get(nid, {}).get("name", nid)
            lines.append(f"  {nid}  sim={sim:.3f}  {name}")
    lines.extend([
        "",
        "Output:",
        f"  drug_similarity.pkl  ({out_size_mb:.1f} MB)",
    ])
    report = "\n".join(lines)
    print()
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
