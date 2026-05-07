"""
Step 8b: Resolve the 10 unclassified templates by adding manual mappings.

After Step 8, 10 templates were not matched by the keyword rules. We assign
them by hand here based on pharmacological judgment:
  - QTc-prolonging activities -> qt_prolongation (same mechanism)
  - liver enzyme elevations  -> hepatotoxicity (subclinical hepatotoxicity)
  - hypocalcemia              -> electrolyte_imbalance (Ca2+ is an electrolyte)
  - GI motility reducing     -> gi_effects
  - Others get new clusters: antipsychotic_change, hyperthermia,
    urinary_retention, peripheral_neuropathy, photosensitivity,
    cardiac_depression

Inputs:
  processed_v2/hierarchy_map.json
  processed_v2/hierarchy_clusters.json

Outputs:
  same files, updated in place
  reports/08b_resolve.txt
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "08b_resolve.txt"

# Manual assignments for unclassified labels. Each entry: (label, l1, l2)
MANUAL = [
    ("65",  "PD", "qt_prolongation"),       # fold
    ("124", "PD", "antipsychotic_change"),   # new
    ("135", "PD", "gi_effects"),             # fold
    ("137", "PD", "hyperthermia"),           # new
    ("139", "PD", "urinary_retention"),      # new
    ("140", "PD", "peripheral_neuropathy"),  # new
    ("141", "PD", "photosensitivity"),       # new
    ("145", "PD", "hepatotoxicity"),         # fold
    ("146", "PD", "electrolyte_imbalance"),  # fold
    ("148", "PD", "cardiac_depression"),     # new
]


def main():
    with open(PROCESSED / "hierarchy_map.json") as f:
        hierarchy = json.load(f)

    print(f"Updating {len(MANUAL)} manual assignments...")
    for label, l1, l2 in MANUAL:
        if label not in hierarchy:
            print(f"  WARNING: label {label} not in hierarchy_map; skipping")
            continue
        old = hierarchy[label]
        hierarchy[label] = {"l1": l1, "l2": l2, "template": old["template"]}
        print(f"  [{label}] {l2}  | {old['template'][:60]}")

    # Rebuild clusters from updated hierarchy
    cluster_members = defaultdict(list)
    unclassified = []
    for label, info in hierarchy.items():
        cluster_members[info["l2"]].append(int(label))
        if info["l1"] == "UNCLASSIFIED":
            unclassified.append((label, info["template"]))
    for k in cluster_members:
        cluster_members[k] = sorted(cluster_members[k])

    # Write back
    with open(PROCESSED / "hierarchy_map.json", "w") as f:
        json.dump(hierarchy, f, indent=2)
    with open(PROCESSED / "hierarchy_clusters.json", "w") as f:
        json.dump(dict(cluster_members), f, indent=2)

    # Stats
    total = len(hierarchy)
    classified = total - len(unclassified)
    cluster_sizes = sorted(
        ((k, len(v)) for k, v in cluster_members.items() if k != "UNCLASSIFIED"),
        key=lambda x: -x[1]
    )

    lines = [
        "Step 8b -- Manual cluster assignments",
        "=" * 60,
        f"Total labels:        {total}",
        f"Classified:          {classified}  ({100*classified/total:.1f}%)",
        f"Still unclassified:  {len(unclassified)}",
        f"Distinct l2 clusters: {len(cluster_sizes)}",
        "",
        "Top 15 clusters after resolution:",
    ]
    for cluster, size in cluster_sizes[:15]:
        member_labels = cluster_members[cluster][:8]
        lines.append(f"  {cluster:30s}  n={size:>3}  labels={member_labels}{'...' if size > 8 else ''}")
    if unclassified:
        lines.append("")
        lines.append("Still unclassified (FIX THESE):")
        for label, template in unclassified:
            lines.append(f"  [{label}]  {template[:100]}")

    report = "\n".join(lines)
    print()
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
