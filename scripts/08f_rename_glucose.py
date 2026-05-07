"""
Step 8f: Rename glucose_increase/glucose_decrease to hyperglycemia/hypoglycemia.

The medical Greek prefixes "hyper-" and "hypo-" already encode direction, so
hyperglycemia/hypoglycemia is more domain-natural than glucose_increase/
glucose_decrease, and matches the existing pattern set by hyperkalemia/
hypokalemia from Step 8e.

Naming convention going forward:
  - Medical-prefix nouns (hyperkalemia, hypokalemia, hyperglycemia, hypoglycemia,
    hypertension, hypotension): stand alone, no suffix needed because the prefix
    encodes direction.
  - Neutral nouns (bleeding, qt_prolongation, cns_depression, nephrotoxicity,
    sedation, etc.): get _increase / _decrease suffix when both directions
    exist in the data.
  - Mechanism families (vasoactivity, thrombosis, myopathy): no suffix;
    documented as known imperfections per Step 8e.
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "08f_rename_glucose.txt"

RENAMES = {
    "glucose_increase": "hyperglycemia",
    "glucose_decrease": "hypoglycemia",
}


def main():
    # Backup
    shutil.copy(PROCESSED / "hierarchy_map.json",
                PROCESSED / "hierarchy_map.json.bak_step8f")
    shutil.copy(PROCESSED / "hierarchy_clusters.json",
                PROCESSED / "hierarchy_clusters.json.bak_step8f")

    with open(PROCESSED / "hierarchy_map.json") as f:
        hierarchy = json.load(f)

    print(f"Applying {len(RENAMES)} cluster renames...")
    affected_labels = []
    for label, info in hierarchy.items():
        if info["l2"] in RENAMES:
            old = info["l2"]
            new = RENAMES[old]
            info["l2"] = new
            affected_labels.append((label, old, new, info["template"]))

    # Rebuild cluster index from updated hierarchy
    cluster_members = defaultdict(list)
    for label, info in hierarchy.items():
        cluster_members[info["l2"]].append(int(label))
    for k in cluster_members:
        cluster_members[k] = sorted(cluster_members[k])

    # Write
    with open(PROCESSED / "hierarchy_map.json", "w") as f:
        json.dump(hierarchy, f, indent=2)
    with open(PROCESSED / "hierarchy_clusters.json", "w") as f:
        json.dump(dict(cluster_members), f, indent=2)

    # Report
    lines = [
        "Step 8f -- Rename glucose clusters",
        "=" * 60,
        f"Renames applied: {len(RENAMES)}",
        f"Total clusters: {len(cluster_members)}",
        f"Labels affected: {len(affected_labels)}",
        "",
        "Renames:",
    ]
    for old, new in RENAMES.items():
        lines.append(f"  {old}  ->  {new}")
    lines.append("")
    lines.append("Affected labels:")
    for label, old, new, tmpl in affected_labels:
        lines.append(f"  [{label}]  {old} -> {new}")
        lines.append(f"        '{tmpl[:80]}'")

    lines.extend([
        "",
        "Final cluster check (medical-prefix nouns vs polarity-suffixed):",
    ])
    medical_prefix = sorted([k for k in cluster_members
                             if k in {"hyperkalemia", "hypokalemia",
                                      "hyperglycemia", "hypoglycemia"}])
    polarity_suffix = sorted([k for k in cluster_members
                              if k.endswith("_increase") or k.endswith("_decrease")])
    lines.append(f"  Medical-prefix nouns ({len(medical_prefix)}): {medical_prefix}")
    lines.append(f"  Polarity-suffixed ({len(polarity_suffix)}):")
    for c in polarity_suffix:
        lines.append(f"    {c}")

    lines.extend([
        "",
        "Backups: hierarchy_map.json.bak_step8f, hierarchy_clusters.json.bak_step8f",
    ])
    report = "\n".join(lines)
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
