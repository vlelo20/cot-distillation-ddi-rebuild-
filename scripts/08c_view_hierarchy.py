"""
Step 8c: Pretty-print the full hierarchy so you can review every cluster
and the templates that fall under it.

Output is grouped by l1 (PK vs PD), then sorted by cluster size descending.
For each cluster: cluster name, number of labels, and every template with
its label ID and how many interactions in the dataset use that label.
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"

with open(PROCESSED / "hierarchy_map.json") as f:
    hierarchy = json.load(f)
with open(PROCESSED / "label_distribution.json") as f:
    label_dist = json.load(f)

# Group by l1 -> l2 -> [(label, template, count)]
by_l1 = defaultdict(lambda: defaultdict(list))
for label, info in hierarchy.items():
    count = label_dist.get(label, 0)
    by_l1[info["l1"]][info["l2"]].append((int(label), info["template"], count))

# Print in order: PK first, then PD
for l1 in ["PK", "PD"]:
    if l1 not in by_l1:
        continue
    clusters = by_l1[l1]
    # Sort clusters by total interaction count (most populated first)
    cluster_total = {
        cname: sum(c for _, _, c in members)
        for cname, members in clusters.items()
    }
    sorted_clusters = sorted(clusters.items(), key=lambda x: -cluster_total[x[0]])

    print()
    print("#" * 78)
    print(f"#  {l1}  —  {len(clusters)} clusters, "
          f"{sum(cluster_total.values()):,} total interactions")
    print("#" * 78)

    for cname, members in sorted_clusters:
        members.sort(key=lambda x: -x[2])  # sort by count desc within cluster
        total = cluster_total[cname]
        print()
        print(f"--- {cname}  (n={len(members)} labels, "
              f"{total:,} interactions) ---")
        for lbl, template, count in members:
            print(f"  [{lbl:>3}]  n={count:>7,}  {template}")

# Summary
print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
total_labels = len(hierarchy)
pk_labels = sum(len(c) for c in by_l1.get("PK", {}).values())
pd_labels = sum(len(c) for c in by_l1.get("PD", {}).values())
print(f"Total labels:        {total_labels}")
print(f"  in PK clusters:    {pk_labels}")
print(f"  in PD clusters:    {pd_labels}")
print(f"Total clusters:      {sum(len(c) for c in by_l1.values())}")
