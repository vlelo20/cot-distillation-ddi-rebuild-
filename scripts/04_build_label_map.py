"""
Step 4: Extract templates from interaction descriptions and build label_map.

For each interaction, we replace the actual drug names with #Drug1 and #Drug2
placeholders. The resulting templates are grouped, counted, and assigned
integer labels.

This fixes the text/label_text contamination bug: templates are derived FROM
descriptions rather than applied TO them, so the description and the
templated label can never disagree.

Synonyms: about 0.8% of descriptions use a non-canonical drug name
(e.g., "Florbetaben (18F)" vs canonical "Florbetaben F-18"). We try canonical
names first, then fall back to all known synonyms/brands/secondary IDs.
Rows where neither works are flagged as extraction failures and excluded.

Filtering: a template must appear at least MIN_COUNT times across the corpus
to qualify for a label. This drops one-off curator phrasings.

Inputs:
  processed_v2/interactions_dedup.jsonl
  processed_v2/drug_profiles.json
  processed_v2/drug_synonyms.json

Outputs:
  processed_v2/interactions_labeled.jsonl  -- one row per interaction with template + label
  processed_v2/label_map.json              -- int_label (str) -> template
  processed_v2/label_distribution.json     -- int_label (str) -> count
  reports/04_build_label_map.txt           -- stats and examples
"""
import json
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "04_build_label_map.txt"

INTERACTIONS = PROCESSED / "interactions_dedup.jsonl"
PROFILES = PROCESSED / "drug_profiles.json"
SYNONYMS = PROCESSED / "drug_synonyms.json"

# Minimum count for a template to qualify as a label.
# Templates with fewer occurrences are likely curator one-offs or extraction noise.
MIN_COUNT = 50


def normalize_for_match(s):
    """Lowercase + strip, for case-insensitive matching."""
    return s.strip().lower() if s else ""


def build_name_index(profiles, synonyms):
    """
    Build drug_id -> sorted list of all known names (canonical + synonyms +
    brand names + secondary IDs), longest first.

    Sorting longest-first matters: if "Lepirudin" and "Lepirudin recombinant"
    are both candidates, we must try the longer one first or we'll match
    "Lepirudin" inside "Lepirudin recombinant" and miss the right substitution.
    """
    name_index = {}
    for did, prof in profiles.items():
        candidates = set()
        if prof.get("name"):
            candidates.add(prof["name"])
        for s in prof.get("synonyms", []):
            if s:
                candidates.add(s)
        for b in prof.get("brand_names", []):
            if b:
                candidates.add(b)
        # We don't add secondary_ids (BTD/BIOD codes) since those don't
        # appear in description prose.
        # Sort by length descending so longer names match first.
        name_index[did] = sorted(candidates, key=lambda x: -len(x))
    return name_index


def extract_template(description, drug_a_id, drug_b_id, name_index):
    """
    Replace drug_a's name with #Drug1 and drug_b's name with #Drug2 in the
    description. Try canonical name first, then synonyms/brands.

    Returns (template, success_bool, debug_info).
    debug_info indicates which name variant matched (for diagnostics).
    """
    # We don't know a priori which drug appears first in the description.
    # The description's first-occurring drug is #Drug1 by DrugBank's convention
    # in templates. We'll try both orderings and pick the one that produces
    # the most placeholders.

    a_names = name_index.get(drug_a_id, [])
    b_names = name_index.get(drug_b_id, [])

    if not a_names or not b_names:
        return description, False, "missing_names"

    # Try ordering: a -> #Drug1, b -> #Drug2
    template_v1, ok_v1, info_v1 = _try_substitute(description, a_names, b_names)
    # Try ordering: b -> #Drug1, a -> #Drug2
    template_v2, ok_v2, info_v2 = _try_substitute(description, b_names, a_names)

    # Prefer the ordering where both placeholders ended up in the right relative
    # positions in the description (i.e., #Drug1 appears before #Drug2).
    # If both work, pick the one matching DrugBank's "first-mentioned-is-Drug1" pattern.
    def is_canonical_order(template):
        if "#Drug1" not in template or "#Drug2" not in template:
            return False
        return template.find("#Drug1") < template.find("#Drug2")

    if ok_v1 and is_canonical_order(template_v1):
        return template_v1, True, f"v1_{info_v1}"
    if ok_v2 and is_canonical_order(template_v2):
        return template_v2, True, f"v2_{info_v2}"
    if ok_v1:
        return template_v1, True, f"v1_{info_v1}_noncanonical"
    if ok_v2:
        return template_v2, True, f"v2_{info_v2}_noncanonical"
    return description, False, "no_match"


def _try_substitute(description, drug1_names, drug2_names):
    """
    Replace one of drug1_names with #Drug1, one of drug2_names with #Drug2.
    Returns (template, success, info).
    """
    matched1_name = None
    matched2_name = None
    template = description

    # First pass: replace drug1 with placeholder
    for name in drug1_names:
        # Word-boundary match, case-insensitive
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        if pattern.search(template):
            template = pattern.sub("#Drug1", template, count=1)
            matched1_name = name
            break

    # Second pass: replace drug2 with placeholder
    for name in drug2_names:
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        if pattern.search(template):
            template = pattern.sub("#Drug2", template, count=1)
            matched2_name = name
            break

    success = (matched1_name is not None) and (matched2_name is not None)
    if matched1_name == drug1_names[0] and matched2_name == drug2_names[0]:
        info = "canonical"
    elif success:
        info = "synonym"
    else:
        info = f"miss_{matched1_name is None}_{matched2_name is None}"
    return template, success, info


def main():
    print("Loading drug profiles...")
    with open(PROFILES) as f:
        profiles = json.load(f)
    with open(SYNONYMS) as f:
        synonyms = json.load(f)

    print("Building name index (canonical + synonyms + brands)...")
    name_index = build_name_index(profiles, synonyms)
    total_names = sum(len(v) for v in name_index.values())
    print(f"  {len(name_index):,} drugs, {total_names:,} total name variants")

    print("\nExtracting templates from descriptions...")
    template_counts = Counter()
    extraction_results = []  # list of dicts to write to interactions_labeled
    n_success = 0
    n_canonical = 0
    n_synonym = 0
    n_failure = 0
    failure_examples = []

    with open(INTERACTIONS) as f:
        for line in tqdm(f, desc="rows", unit="row"):
            row = json.loads(line)
            template, ok, info = extract_template(
                row["description"], row["drug_a"], row["drug_b"], name_index
            )
            if ok:
                n_success += 1
                if "canonical" in info:
                    n_canonical += 1
                elif "synonym" in info:
                    n_synonym += 1
                template_counts[template] += 1
                extraction_results.append({
                    "drug_a": row["drug_a"],
                    "drug_b": row["drug_b"],
                    "description": row["description"],
                    "template": template,
                })
            else:
                n_failure += 1
                if len(failure_examples) < 10:
                    failure_examples.append(row)

    print(f"\nExtraction results:")
    print(f"  Success:           {n_success:,}")
    print(f"    via canonical:   {n_canonical:,}")
    print(f"    via synonym:     {n_synonym:,}")
    print(f"  Failure:           {n_failure:,}")
    print(f"  Unique templates:  {len(template_counts):,}")

    # Filter: only keep templates with >= MIN_COUNT occurrences
    qualifying = {t: c for t, c in template_counts.items() if c >= MIN_COUNT}
    print(f"\nTemplates with >={MIN_COUNT} occurrences: {len(qualifying):,}")
    coverage = sum(qualifying.values())
    print(f"Coverage: {coverage:,} / {n_success:,} interactions ({100*coverage/n_success:.2f}%)")

    # Sort by frequency desc, assign integer labels (1-indexed)
    sorted_templates = sorted(qualifying.items(), key=lambda x: -x[1])
    label_map = {str(i + 1): t for i, (t, _) in enumerate(sorted_templates)}
    label_dist = {str(i + 1): c for i, (_, c) in enumerate(sorted_templates)}
    template_to_label = {t: str(i + 1) for i, (t, _) in enumerate(sorted_templates)}

    # Write outputs
    with open(PROCESSED / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(PROCESSED / "label_distribution.json", "w") as f:
        json.dump(label_dist, f, indent=2)

    # Write labeled interactions (only those with a qualifying label)
    print("Writing labeled interactions...")
    n_written = 0
    with open(PROCESSED / "interactions_labeled.jsonl", "w") as fout:
        for r in extraction_results:
            label = template_to_label.get(r["template"])
            if label is None:
                continue  # template didn't qualify
            r["label"] = int(label)
            r["label_text"] = r["description"]  # by construction, identical
            fout.write(json.dumps(r) + "\n")
            n_written += 1
    print(f"  wrote {n_written:,} labeled interactions")

    # Anomaly check: any templates that don't contain both placeholders?
    suspicious = [(lbl, t) for lbl, t in label_map.items()
                  if "#Drug1" not in t or "#Drug2" not in t]

    # Report
    counts = list(label_dist.values())
    lines = [
        "Step 4 -- Label map construction",
        "=" * 60,
        f"Interactions processed:       {n_success + n_failure:,}",
        f"  Successful extractions:     {n_success:,}",
        f"    via canonical name:       {n_canonical:,}",
        f"    via synonym/brand:        {n_synonym:,}",
        f"  Failed extractions:         {n_failure:,}",
        "",
        f"Unique templates extracted:   {len(template_counts):,}",
        f"Templates with >={MIN_COUNT}:           {len(qualifying):,}",
        f"Coverage of labeled rows:     {100*coverage/n_success:.2f}%",
        f"Labeled interactions written: {n_written:,}",
        "",
        "Class distribution (label counts):",
        f"  largest class:   {max(counts):,}",
        f"  smallest class:  {min(counts):,}",
        f"  imbalance ratio: {max(counts) / min(counts):,.0f}:1",
        f"  classes <50:     {sum(1 for c in counts if c < 50)}",
        f"  classes <100:    {sum(1 for c in counts if c < 100)}",
        "",
        "Top 15 most frequent labels:",
    ]
    for lbl, t in list(label_map.items())[:15]:
        c = label_dist[lbl]
        lines.append(f"  [{lbl:>3}]  n={c:>6,}  {t[:90]}")
    lines.extend([
        "",
        "Bottom 15 least frequent (qualifying) labels:",
    ])
    for lbl, t in list(label_map.items())[-15:]:
        c = label_dist[lbl]
        lines.append(f"  [{lbl:>3}]  n={c:>6,}  {t[:90]}")

    if suspicious:
        lines.append("")
        lines.append(f"WARNING: {len(suspicious)} templates missing #Drug1 or #Drug2:")
        for lbl, t in suspicious[:5]:
            lines.append(f"  [{lbl}]  {t[:100]}")

    if failure_examples:
        lines.append("")
        lines.append("Sample extraction failures (debug):")
        for r in failure_examples[:5]:
            a_name = profiles.get(r["drug_a"], {}).get("name", r["drug_a"])
            b_name = profiles.get(r["drug_b"], {}).get("name", r["drug_b"])
            lines.append(f"  drug_a={a_name}  drug_b={b_name}")
            lines.append(f"    desc: {r['description'][:130]}")

    lines.extend([
        "",
        f"Outputs:",
        f"  label_map.json              ({(PROCESSED / 'label_map.json').stat().st_size / 1e3:.1f} KB)",
        f"  label_distribution.json     ({(PROCESSED / 'label_distribution.json').stat().st_size / 1e3:.1f} KB)",
        f"  interactions_labeled.jsonl  ({(PROCESSED / 'interactions_labeled.jsonl').stat().st_size / 1e6:.1f} MB)",
    ])
    report = "\n".join(lines)
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
