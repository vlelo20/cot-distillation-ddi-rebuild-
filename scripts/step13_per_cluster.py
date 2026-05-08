#!/usr/bin/env python3
"""Per-cluster accuracy breakdown across 4 pilot conditions."""
import json
from collections import defaultdict
from pathlib import Path

PILOT = Path("pilot_scored.jsonl")
DOCUMENTED_IMPERFECT = {
    "vasoactivity", "gi_effects_increase", "thrombosis",
    "electrolyte_imbalance", "myopathy", "cns_depression",
}

# {(cluster, condition): {n, exact, cluster_match, outcome_aligned, wrong_dir}}
agg = defaultdict(lambda: {"n": 0, "exact": 0, "cluster_match": 0,
                           "outcome_aligned": 0, "wrong_dir": 0})
clusters_seen = set()
conditions_seen = set()

with PILOT.open() as f:
    for line in f:
        r = json.loads(line)
        cluster = r["gold_l2"]
        cond = r["condition"]
        clusters_seen.add(cluster)
        conditions_seen.add(cond)
        k = (cluster, cond)
        agg[k]["n"] += 1
        agg[k]["exact"] += int(r.get("exact_match", False))
        agg[k]["cluster_match"] += int(r.get("cluster_match", False))
        v = r.get("direction_verdict")
        # outcome-aligned (adjusted): correct OR ambiguous count as aligned;
        # incorrect = wrong-direction; missing/no_summary excluded from denom
        if v == "correct":
            agg[k]["outcome_aligned"] += 1
        elif v == "incorrect":
            agg[k]["wrong_dir"] += 1
        elif v == "ambiguous":
            agg[k]["outcome_aligned"] += 1  # adjusted metric

CONDS = sorted(conditions_seen)
print(f"Conditions: {CONDS}")
print(f"Clusters with data: {len(clusters_seen)}\n")

# Per-cluster table, sorted by total n desc
cluster_totals = defaultdict(int)
for (cl, _), v in agg.items():
    cluster_totals[cl] += v["n"]
ordered_clusters = sorted(clusters_seen, key=lambda c: -cluster_totals[c])

print(f"{'cluster':<35} {'cond':<18} {'n':>4} {'exact%':>7} {'clus%':>7} {'aln%':>7} {'wrng%':>7}")
print("-" * 95)
for cl in ordered_clusters:
    tag = " *" if cl in DOCUMENTED_IMPERFECT else ""
    for cond in CONDS:
        v = agg.get((cl, cond))
        if not v or v["n"] == 0:
            continue
        n = v["n"]
        ex = 100 * v["exact"] / n
        cm = 100 * v["cluster_match"] / n
        al = 100 * v["outcome_aligned"] / n
        wr = 100 * v["wrong_dir"] / n
        print(f"{cl+tag:<35} {cond:<18} {n:>4} {ex:>6.1f} {cm:>6.1f} {al:>6.1f} {wr:>6.1f}")
    print()

# Summary: documented imperfect vs rest, hint vs no-hint
print("\n=== DOCUMENTED IMPERFECT vs REST ===")
groups = {"imperfect": DOCUMENTED_IMPERFECT, "rest": clusters_seen - DOCUMENTED_IMPERFECT}
for gname, gset in groups.items():
    for cond in CONDS:
        tot = {"n": 0, "exact": 0, "cluster_match": 0,
               "outcome_aligned": 0, "wrong_dir": 0}
        for cl in gset:
            v = agg.get((cl, cond))
            if not v: continue
            for k_ in tot: tot[k_] += v[k_]
        if tot["n"] == 0: continue
        n = tot["n"]
        print(f"{gname:<10} {cond:<18} n={n:>3}  "
              f"exact={100*tot['exact']/n:5.1f}  "
              f"aln={100*tot['outcome_aligned']/n:5.1f}  "
              f"wrng={100*tot['wrong_dir']/n:5.1f}")
    print()

# Save JSON
out = {"per_cluster_condition": {f"{cl}|{co}": v for (cl, co), v in agg.items()},
       "documented_imperfect": sorted(DOCUMENTED_IMPERFECT),
       "clusters_seen": sorted(clusters_seen)}
Path("per_cluster_breakdown.json").write_text(json.dumps(out, indent=2))
print("\nWrote per_cluster_breakdown.json")
