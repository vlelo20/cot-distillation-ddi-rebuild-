"""
Step 8: Build a principled pharmacological hierarchy over the 166 templates.

Replaces the broken coarse_category_map.json from the old pipeline (which
dumped 70+ labels into 'other'). The new hierarchy is two-level:
  level 1: PK (pharmacokinetic) vs PD (pharmacodynamic)
  level 2: mechanism cluster (e.g., PK.exposure_increase, PD.qt_prolongation)

This enables:
  - Hierarchical F1: a prediction is 'cluster-correct' if it picks any label
    in the same cluster as the truth, even if it picks the wrong template
  - Cluster-aware loss: within-cluster confusion penalized less than
    cross-cluster confusion
  - Mechanism-accuracy reporting separate from exact-match F1

Approach: rule-based keyword matching against template text. Templates that
match no rule are flagged OTHER for manual review.

Inputs:
  processed_v2/label_map.json

Outputs:
  processed_v2/hierarchy_map.json   -- {label: {"l1": ..., "l2": ...}}
  processed_v2/hierarchy_clusters.json  -- {cluster_id: [label, label, ...]}
  reports/08_hierarchy.txt
"""
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
LABEL_MAP = PROCESSED / "label_map.json"
REPORT = ROOT / "reports" / "08_hierarchy.txt"


# Rules ordered by specificity. First match wins.
# Each rule is (pattern, l1, l2). Pattern matched against lowercased template.
RULES = [
    # ----- PK: Absorption -----
    (r"absorption.*decrease|decrease.*absorption|reduced.*absorption|bioavailability.*decrease",
     "PK", "absorption_decrease"),
    (r"absorption.*increase|increase.*absorption|bioavailability.*increase",
     "PK", "absorption_increase"),

    # ----- PK: Metabolism -----
    (r"metabolism.*decrease|decrease.*metabolism",
     "PK", "metabolism_decrease"),
    (r"metabolism.*increase|increase.*metabolism",
     "PK", "metabolism_increase"),

    # ----- PK: Exposure (serum / excretion - same event, opposite directions) -----
    # Decreased excretion = higher serum = increased exposure
    (r"decrease.*excretion.*higher serum|excretion.*decrease.*higher serum",
     "PK", "exposure_increase"),
    (r"serum concentration.*increase|active metabolite.*increase",
     "PK", "exposure_increase"),
    # Increased excretion = lower serum = decreased exposure
    (r"increase.*excretion.*lower serum|excretion.*increase.*lower serum",
     "PK", "exposure_decrease"),
    (r"serum concentration.*decrease",
     "PK", "exposure_decrease"),
    (r"excretion.*decrease(?!.*serum)",
     "PK", "exposure_increase"),
    (r"excretion.*increase(?!.*serum)",
     "PK", "exposure_decrease"),

    # ----- PK: Distribution / Binding -----
    (r"protein binding",
     "PK", "protein_binding"),

    # ----- PD: Cardiac electrical -----
    (r"qtc prolongation|qt prolongation|torsade",
     "PD", "qt_prolongation"),
    (r"arrhythmia|arrhythmogenic",
     "PD", "arrhythmia"),
    (r"bradycard",
     "PD", "bradycardia"),
    (r"tachycard",
     "PD", "tachycardia"),
    (r"av block|atrioventricular blocking",
     "PD", "av_block"),

    # ----- PD: Hemodynamic -----
    (r"orthostatic hypotensive|hypotensive activities|hypotension",
     "PD", "hypotension"),
    (r"hypertensive activities|hypertension",
     "PD", "hypertension"),
    (r"vasodilatory|vasopressor|vasoconstricting|vasospastic",
     "PD", "vasoactivity"),

    # ----- PD: Hemostasis -----
    (r"bleeding|hemorrhage|bruising",
     "PD", "bleeding"),
    (r"anticoagulant",
     "PD", "anticoagulant_change"),
    (r"thrombogenic|thrombosis|thromboembolism|antiplatelet",
     "PD", "thrombosis"),
    (r"thrombocytopenia|neutropenia|myelosuppress",
     "PD", "myelosuppression"),

    # ----- PD: CNS -----
    (r"cns depress|sedation|somnolence|respiratory depression",
     "PD", "cns_depression"),
    (r"seizure|neuroexcitatory",
     "PD", "neuroexcitation"),
    (r"extrapyramidal",
     "PD", "extrapyramidal"),
    (r"neuromuscular block",
     "PD", "neuromuscular_blockade"),
    (r"serotonin syndrome|serotonergic",
     "PD", "serotonergic"),
    (r"neurotoxic|neuropsychiatric",
     "PD", "neurotoxic"),
    (r"sedative activities",
     "PD", "sedation"),

    # ----- PD: Metabolic / Endocrine / Electrolyte -----
    (r"hypoglycemi|hyperglycemi",
     "PD", "glycemic"),
    (r"hyperkalemi|hypokalemi",
     "PD", "potassium_imbalance"),
    (r"hyponatremi|hypercalcemi|electrolyte",
     "PD", "electrolyte_imbalance"),
    (r"thyroid function",
     "PD", "thyroid"),

    # ----- PD: Organ toxicity -----
    (r"nephrotoxic|renal failure",
     "PD", "nephrotoxicity"),
    (r"hepatotoxic|liver damage",
     "PD", "hepatotoxicity"),
    (r"cardiotoxic",
     "PD", "cardiotoxicity"),
    (r"myopathy|rhabdomyolysis|tendinopathy",
     "PD", "myopathy"),
    (r"ototoxic",
     "PD", "ototoxicity"),

    # ----- PD: GI -----
    (r"gastrointestinal bleeding",
     "PD", "bleeding"),  # group with bleeding
    (r"gastrointestinal irritation|gastrointestinal ulceration|ulceration|reduced gastrointestinal motility|constipation",
     "PD", "gi_effects"),

    # ----- PD: Immune -----
    (r"immunosuppress",
     "PD", "immunosuppression"),
    (r"hypersensitivity|angioedema",
     "PD", "hypersensitivity"),
    (r"infection",
     "PD", "infection"),
    (r"methemoglobinemia",
     "PD", "methemoglobinemia"),

    # ----- PD: Other / Functional -----
    (r"therapeutic efficacy.*decrease|efficacy.*decrease",
     "PD", "efficacy_decrease"),
    (r"therapeutic efficacy.*increase|efficacy.*increase",
     "PD", "efficacy_increase"),
    (r"adverse effects.*decrease",
     "PD", "adverse_effects_decrease"),
    (r"adverse effects.*increase",
     "PD", "adverse_effects_generic"),
    (r"analgesic",
     "PD", "analgesia"),
    (r"fluid retention|edema",
     "PD", "fluid_retention"),
    (r"dehydration",
     "PD", "dehydration"),
    (r"anticholinergic",
     "PD", "anticholinergic"),
    (r"sympathomimetic",
     "PD", "sympathomimetic"),
    (r"intracranial pressure|pseudotumor cerebri",
     "PD", "intracranial_pressure"),
    (r"jaw osteonecrosis|anti-angiogenesis",
     "PD", "osteonecrosis"),
    (r"reye's syndrome",
     "PD", "reye_syndrome"),
    (r"diuretic",
     "PD", "diuretic"),
    (r"stimulatory",
     "PD", "stimulant_change"),
    (r"sympatholytic",
     "PD", "sympatholytic"),
    (r"bronchodilatory",
     "PD", "bronchial"),
    (r"diagnostic agent",
     "PD", "diagnostic_efficacy"),
    (r"hypoglycemic activities",
     "PD", "glycemic"),
]


def classify(template):
    """Return (l1, l2) for a template, or ('UNCLASSIFIED', 'UNCLASSIFIED')."""
    t = template.lower()
    for pattern, l1, l2 in RULES:
        if re.search(pattern, t):
            return (l1, l2)
    return ("UNCLASSIFIED", "UNCLASSIFIED")


def main():
    with open(LABEL_MAP) as f:
        label_map = json.load(f)
    print(f"Classifying {len(label_map)} templates...")

    hierarchy = {}  # label -> {"l1": ..., "l2": ..., "template": ...}
    cluster_members = defaultdict(list)  # l2 -> [label, ...]
    unclassified = []

    for label, template in label_map.items():
        l1, l2 = classify(template)
        hierarchy[label] = {"l1": l1, "l2": l2, "template": template}
        cluster_members[l2].append(int(label))
        if l1 == "UNCLASSIFIED":
            unclassified.append((label, template))

    # Sort cluster member lists
    for k in cluster_members:
        cluster_members[k] = sorted(cluster_members[k])

    # Write outputs
    with open(PROCESSED / "hierarchy_map.json", "w") as f:
        json.dump(hierarchy, f, indent=2)
    with open(PROCESSED / "hierarchy_clusters.json", "w") as f:
        json.dump(dict(cluster_members), f, indent=2)

    # Stats
    total = len(label_map)
    classified = total - len(unclassified)
    pct = 100 * classified / total

    # Cluster size distribution
    cluster_sizes = sorted(
        ((k, len(v)) for k, v in cluster_members.items() if k != "UNCLASSIFIED"),
        key=lambda x: -x[1]
    )

    lines = [
        "Step 8 -- Pharmacological hierarchy",
        "=" * 60,
        f"Total labels:                {total}",
        f"Classified:                  {classified}  ({pct:.1f}%)",
        f"Unclassified (need review):  {len(unclassified)}",
        f"Distinct l2 clusters:        {len([k for k in cluster_members if k != 'UNCLASSIFIED'])}",
        "",
        "Cluster size distribution (top 20):",
    ]
    for cluster, size in cluster_sizes[:20]:
        member_labels = cluster_members[cluster][:10]
        lines.append(f"  {cluster:30s}  n={size:>3}  labels={member_labels}{'...' if size > 10 else ''}")

    if unclassified:
        lines.append("")
        lines.append("UNCLASSIFIED templates (review these and add rules):")
        for label, template in unclassified:
            lines.append(f"  [{label:>3}]  {template[:100]}")

    lines.extend([
        "",
        "Outputs:",
        f"  hierarchy_map.json       ({(PROCESSED / 'hierarchy_map.json').stat().st_size / 1e3:.1f} KB)",
        f"  hierarchy_clusters.json  ({(PROCESSED / 'hierarchy_clusters.json').stat().st_size / 1e3:.1f} KB)",
    ])
    report = "\n".join(lines)
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
