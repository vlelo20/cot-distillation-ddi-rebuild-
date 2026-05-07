"""
Step 8d: Enrich hierarchy with polarity, affected_drug_role, and secondary_tags.

Adds three new fields to every entry in hierarchy_map.json:

  polarity: "increase" | "decrease" | "n/a"
    The pharmacological direction. Critical for distinguishing
    "increase bleeding risk" from "decrease bleeding risk" -- these are
    NOT near-misses, they are clinically dangerous opposite errors.

  affected_drug_role: "drug1" | "drug2" | "both" | "unknown"
    Which drug in the pair experiences the change.
    - drug1: "The metabolism of #Drug1 can be decreased..."
    - drug2: "#Drug1 may increase the X activities of #Drug2"
    - both:  "The risk or severity of X can be increased when..."

  secondary_tags: [cluster, ...]
    For compound templates that describe multiple outcomes (e.g.,
    "renal failure, hyperkalemia, AND hypertension"), list additional
    clusters this template also belongs to.

Also splits direction-mixed clusters into _increase / _decrease variants
where both directions have meaningful representation.

Inputs:
  processed_v2/hierarchy_map.json
  processed_v2/hierarchy_clusters.json

Outputs:
  same files, updated in place (backup made first)
  reports/08d_enrich.txt
"""
import json
import re
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "08d_enrich.txt"


# -------------------------------------------------------------------
# Polarity classification
# -------------------------------------------------------------------

# Templates explicitly tagged as decrease (manually identified from analysis)
DECREASE_LABELS = {
    "10",   # decrease antihypertensive activities
    "16",   # serum concentration decreased
    "23",   # decrease in absorption
    "44",   # decrease hypoglycemic activities
    "46",   # decrease anticoagulant activities
    "49",   # decrease sedative activities
    "61",   # protein binding decreased
    "64",   # decrease stimulatory activities
    "74",   # bioavailability decreased
    "85",   # decrease effectiveness as diagnostic agent
    "93",   # decrease cardiotoxic activities
    "98",   # decrease bronchodilatory activities
    "103",  # absorption decreased
    "107",  # excretion increased -> exposure decreased outcome
    "13",   # increase excretion rate -> lower serum -> exposure decreased
    "114",  # decrease hypertensive activities
    "116",  # adverse effects decreased
    "134",  # hypertension decreased
    "138",  # decrease neuromuscular blocking activities
    "142",  # QTc prolongation decreased
    "156",  # decrease nephrotoxic activities
}

# Templates with no clear pharmacological direction
NA_LABELS = {
    "39",   # infection (no inherent direction in template)
    "85",   # diagnostic effectiveness (already marked decrease above)
}


def classify_polarity(label, template):
    if label in DECREASE_LABELS:
        return "decrease"
    if label in NA_LABELS:
        return "n/a"
    # Exposure_increase has labels like 1 ("decrease excretion -> higher serum")
    # whose surface verb is "decrease" but pharmacological outcome is "increase"
    if label in {"1", "26"}:  # decrease excretion, decrease excretion rate
        return "increase"  # because the OUTCOME is increased exposure
    return "increase"  # default for everything else


# -------------------------------------------------------------------
# Affected drug role classification
# -------------------------------------------------------------------

def classify_role(template):
    """
    Returns "drug1" | "drug2" | "both" based on template grammar.
    """
    # "The [property] of #Drug1 can be [verb]ed..." -> Drug1
    if re.match(r"^The (metabolism|serum concentration|protein binding|excretion|absorption|bioavailability|therapeutic efficacy) of #Drug1", template):
        return "drug1"
    # "The serum concentration of the active metabolites of #Drug1..." -> Drug1
    if re.match(r"^The serum concentration of (the active metabolites of |dextroamphetamine, an active metabolite of )#Drug1", template):
        return "drug1"
    # "The risk of a hypersensitivity reaction to #Drug1 is increased..." -> Drug1
    if re.match(r"^The risk of (a )?hypersensitivity reaction to #Drug1", template):
        return "drug1"
    # "The risk or severity of X can be [verb]ed when #Drug1 is combined with #Drug2" -> both
    if re.match(r"^The risk or severity", template):
        return "both"
    # "#Drug1 can cause a [change] in absorption of #Drug2..." -> Drug2
    if re.match(r"^#Drug1 can cause", template):
        return "drug2"
    # "#Drug1 may [verb] the X activities of #Drug2" -> Drug2
    if re.match(r"^#Drug1 may (increase|decrease) ", template) and "#Drug2" in template:
        return "drug2"
    # "#Drug1 may decrease effectiveness of #Drug2" -> Drug2
    if re.match(r"^#Drug1 may decrease effectiveness of #Drug2", template):
        return "drug2"
    return "unknown"


# -------------------------------------------------------------------
# Secondary tags for compound templates
# -------------------------------------------------------------------

# Manually identified compound templates with their secondary clusters.
# Primary cluster stays as-is in l2; secondary_tags lists additional concepts.
SECONDARY_TAGS = {
    "35":  ["nephrotoxicity", "potassium_imbalance"],   # renal failure + hyperkalemia + hypertension (already in)
    "40":  ["myopathy"],                                 # myopathy + rhabdomyolysis + myoglobinuria (one umbrella)
    "67":  ["sedation", "cns_depression"],               # sedation + somnolence + CNS depression
    "86":  ["nephrotoxicity", "potassium_imbalance"],    # renal failure + hypotension (in) + hyperkalemia
    "88":  ["sedation", "cns_depression", "respiratory_depression"],  # multi-event
    "92":  ["sedation", "cns_depression"],               # sedation + CNS depression (already in cns_depression)
    "122": ["sedation"],                                 # sedation + somnolence
    "130": ["bradycardia", "av_block"],                  # ventricular arrhythmias + bradycardia + heart block
    "144": ["electrolyte_imbalance"],                    # hypertension + hyponatremia + water intoxication
    "147": ["gi_effects"],                               # GI bleeding + peptic ulcer (already bleeding)
    "149": ["urinary_retention"],                        # urinary retention + constipation (already gi)
    "151": ["urinary_retention", "gi_effects"],          # urinary retention + reduced GI motility + constipation
    "155": ["nephrotoxicity"],                           # bleeding + nephrotoxicity + GI bleeding
    "162": ["hypotension"],                              # hypotensive + electrolyte
    "165": ["nephrotoxicity"],                           # ototoxicity + nephrotoxicity (already nephro)
    "118": [],   # jaw osteonecrosis + anti-angiogenesis (treated as one mechanism)
}


# -------------------------------------------------------------------
# Cluster splits by polarity
# -------------------------------------------------------------------

# Clusters where both increase and decrease have meaningful representation
# get split into _increase and _decrease variants. Singletons-with-opposite stay.
SPLIT_BY_POLARITY = {
    "anticoagulant_change",   # 31 increase, 46 decrease
    "sedation",                # 47 increase, 49 decrease
    "cardiotoxicity",          # 110 increase, 93 decrease
    "neuromuscular_blockade",  # 43,59 increase, 138 decrease
    "protein_binding",         # 153 increase, 61 decrease
    "qt_prolongation",         # 8,65,96,104 increase, 142 decrease
    "nephrotoxicity",          # 21,81,83,90,165 increase, 156 decrease
    "glycemic",                # 17,27,29 increase, 44 decrease
    "hypertension",            # multiple increase, 10,114,134 decrease
    "gi_effects",              # don't split here; gi_effects is a heterogeneous bucket
    "hypotension",             # all increase
}


def main():
    # Backup
    for fname in ["hierarchy_map.json", "hierarchy_clusters.json"]:
        src = PROCESSED / fname
        dst = PROCESSED / f"{fname}.bak_step8d"
        shutil.copy(src, dst)

    with open(PROCESSED / "hierarchy_map.json") as f:
        hierarchy = json.load(f)

    print(f"Enriching {len(hierarchy)} entries...")

    # Pass 1: assign polarity, role, secondary_tags to every entry
    for label, info in hierarchy.items():
        template = info["template"]
        info["polarity"] = classify_polarity(label, template)
        info["affected_drug_role"] = classify_role(template)
        info["secondary_tags"] = SECONDARY_TAGS.get(label, [])

    # Pass 2: split clusters by polarity where appropriate
    splits_applied = 0
    for label, info in hierarchy.items():
        l2 = info["l2"]
        polarity = info["polarity"]
        if l2 in SPLIT_BY_POLARITY and polarity in {"increase", "decrease"}:
            # Don't split exposure clusters; they're already polarity-split by design
            new_l2 = f"{l2}_{polarity}"
            info["l2"] = new_l2
            splits_applied += 1

    # Rebuild cluster index
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

    # Stats
    polarity_counts = defaultdict(int)
    role_counts = defaultdict(int)
    multitag_count = 0
    role_unknown = []
    for label, info in hierarchy.items():
        polarity_counts[info["polarity"]] += 1
        role_counts[info["affected_drug_role"]] += 1
        if info["secondary_tags"]:
            multitag_count += 1
        if info["affected_drug_role"] == "unknown":
            role_unknown.append((label, info["template"]))

    cluster_sizes = sorted(
        [(k, len(v)) for k, v in cluster_members.items()],
        key=lambda x: -x[1]
    )

    lines = [
        "Step 8d -- Hierarchy enrichment",
        "=" * 60,
        f"Total labels:                       {len(hierarchy)}",
        f"Polarity-split applied:             {splits_applied}",
        f"Multi-tag (compound) templates:     {multitag_count}",
        f"Total clusters after split:         {len(cluster_members)}",
        "",
        "Polarity distribution:",
    ]
    for k, v in sorted(polarity_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Affected drug role distribution:")
    for k, v in sorted(role_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v}")

    if role_unknown:
        lines.append("")
        lines.append("Roles still unknown (review manually):")
        for label, tmpl in role_unknown:
            lines.append(f"  [{label}] {tmpl[:90]}")

    lines.extend([
        "",
        "Top 20 clusters after split:",
    ])
    for cluster, size in cluster_sizes[:20]:
        members_preview = cluster_members[cluster][:6]
        lines.append(f"  {cluster:35s}  n={size:>3}  labels={members_preview}{'...' if size > 6 else ''}")

    lines.extend([
        "",
        "Backup files written:",
        f"  hierarchy_map.json.bak_step8d",
        f"  hierarchy_clusters.json.bak_step8d",
    ])
    report = "\n".join(lines)
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
