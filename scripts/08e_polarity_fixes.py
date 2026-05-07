"""
Step 8e: Targeted polarity fixes after AI review.

Two real polarity bugs identified:
  1. potassium_imbalance lumps hyperkalemia and hypokalemia (opposite directions)
  2. glycemic_increase lumps hyperglycemia and hypoglycemia (opposite directions)
  3. hypertension_decrease lumps "decreased antihypertensive activity" (BP up) with
     "decreased hypertension risk" (BP down) -- opposite outcomes labeled the same way.

We split potassium and glycemic by physiological direction (high/low), and
re-classify the hypertension labels by their actual outcome direction (BP up/down).

Other concerns raised by the review (vasoactivity is broad, gi_effects is broad,
thrombosis is a mechanism umbrella) are left alone -- splitting them further would
fragment the hierarchy without fixing actual polarity errors. They are documented
as known imperfections.
"""
import json
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "08e_polarity_fixes.txt"

# Label -> (new_l2, new_polarity) overrides
# Outcome direction is what matters: hyperkalemia = K+ up, hypokalemia = K+ down.
OVERRIDES = {
    # Potassium: split hyper/hypo
    "18":  ("hyperkalemia", "increase"),       # "hyperkalemia risk increased"
    "32":  ("hyperkalemia", "increase"),       # "increase hyperkalemic activities"
    "38":  ("hypokalemia", "increase"),        # "hypokalemia risk increased" (K low)
    "94":  ("hypokalemia", "increase"),        # "increase hypokalemic activities"

    # Glycemic: regroup by outcome direction
    # 17: "hypoglycemia risk increased" -> glucose DOWN
    # 27: "hyperglycemia risk increased" -> glucose UP
    # 29: "increase hypoglycemic activities" -> glucose DOWN
    # 44: "decrease hypoglycemic activities" -> glucose UP
    "17": ("glucose_decrease", "increase"),    # risk of low glucose going up
    "27": ("glucose_increase", "increase"),    # risk of high glucose going up
    "29": ("glucose_decrease", "increase"),    # increase the activity that lowers glucose
    "44": ("glucose_increase", "increase"),    # decrease the activity that lowers glucose -> glucose goes up

    # Hypertension: clarify by outcome
    # 9:   "hypertension risk increased" -> BP UP        (currently hypertension_increase) OK
    # 10:  "decrease antihypertensive activities" -> BP UP (currently hypertension_decrease) WRONG
    # 53:  "increase hypertensive activities" -> BP UP    (currently hypertension_increase) OK
    # 60:  "increase antihypertensive activities" -> BP DOWN (currently hypertension_increase) WRONG
    # 114: "decrease hypertensive activities" -> BP DOWN  (currently hypertension_decrease) OK
    # 134: "hypertension risk decreased" -> BP DOWN       (currently hypertension_decrease) OK
    "10":  ("bp_increase", "increase"),
    "60":  ("bp_decrease", "increase"),
    "9":   ("bp_increase", "increase"),
    "53":  ("bp_increase", "increase"),
    "114": ("bp_decrease", "increase"),
    "134": ("bp_decrease", "increase"),
    "35":  ("bp_increase", "increase"),  # renal failure + hyperkalemia + hypertension (in)
    "121": ("bp_increase", "increase"),  # renal failure + hypertension
    "144": ("bp_increase", "increase"),  # hypertension + hyponatremia + water intoxication
}

KNOWN_IMPERFECTIONS = """
Documented as known imperfections (not patched):
  - vasoactivity (n=6): mixes vasoconstriction, vasodilation, vasopressor, vasospastic.
    Splitting cleanly would require per-label surgery; left as a mechanism family.
  - gi_effects_increase (n=9): bundles irritation, constipation, ulceration, motility.
    Different endpoints but mechanistically related (anticholinergic, NSAID, opioid drugs).
    Left consolidated to support CoT mechanism reasoning.
  - thrombosis (n=4): umbrella over thrombosis, thromboembolism, thrombogenic activity,
    antiplatelet activity. Antiplatelet (label 82) is technically opposite-direction but
    in the same mechanistic family (platelet/coagulation).
  - electrolyte_imbalance (n=5): bundles hyponatremia, hypercalcemia, hypocalcemia, generic
    electrolyte imbalance. Specific electrolytes vary but all are "imbalance increases."
  - myopathy (n=4): includes tendinopathy as adjacent musculoskeletal toxicity.
  - cns_depression (n=8): includes respiratory_depression (117) and serotonergic+CNS (163)
    as compound templates. Handled via secondary_tags rather than splitting.
"""


def main():
    # Backup
    shutil.copy(PROCESSED / "hierarchy_map.json",
                PROCESSED / "hierarchy_map.json.bak_step8e")
    shutil.copy(PROCESSED / "hierarchy_clusters.json",
                PROCESSED / "hierarchy_clusters.json.bak_step8e")

    with open(PROCESSED / "hierarchy_map.json") as f:
        hierarchy = json.load(f)

    print(f"Applying {len(OVERRIDES)} cluster overrides...")
    changes = []
    for label, (new_l2, new_polarity) in OVERRIDES.items():
        if label not in hierarchy:
            print(f"  WARNING: label {label} not in hierarchy")
            continue
        old_l2 = hierarchy[label]["l2"]
        old_polarity = hierarchy[label]["polarity"]
        hierarchy[label]["l2"] = new_l2
        hierarchy[label]["polarity"] = new_polarity
        changes.append((label, old_l2, new_l2, old_polarity, new_polarity,
                        hierarchy[label]["template"]))

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

    # Report
    cluster_sizes = sorted(
        ((k, len(v)) for k, v in cluster_members.items()),
        key=lambda x: -x[1]
    )

    lines = [
        "Step 8e -- Targeted polarity fixes",
        "=" * 60,
        f"Overrides applied: {len(changes)}",
        f"Total clusters: {len(cluster_members)}",
        "",
        "Changes:",
    ]
    for label, old_l2, new_l2, old_pol, new_pol, tmpl in changes:
        if old_l2 != new_l2 or old_pol != new_pol:
            lines.append(f"  [{label}]  {old_l2} -> {new_l2}")
            lines.append(f"        polarity: {old_pol} -> {new_pol}")
            lines.append(f"        '{tmpl[:80]}'")

    lines.append("")
    lines.append("New / changed clusters:")
    new_clusters = {"hyperkalemia", "hypokalemia",
                    "glucose_increase", "glucose_decrease",
                    "bp_increase", "bp_decrease"}
    for cname in new_clusters:
        if cname in cluster_members:
            members = cluster_members[cname]
            lines.append(f"  {cname:25s}  n={len(members)}  labels={members}")

    lines.append(KNOWN_IMPERFECTIONS)
    lines.extend([
        "",
        "Backup files:",
        "  hierarchy_map.json.bak_step8e",
        "  hierarchy_clusters.json.bak_step8e",
    ])
    report = "\n".join(lines)
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
