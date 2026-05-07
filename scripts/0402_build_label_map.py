"""
Step 4 (FAST): Extract templates from interaction descriptions and build label_map.

Optimized over the previous version: builds one alternation regex per drug,
matches once on a lowercase copy of the description, and avoids per-row
re.compile() calls. Expected runtime: 1-3 minutes instead of 2 hours.

Inputs:
  processed_v2/interactions_dedup.jsonl
  processed_v2/drug_profiles.json
  processed_v2/drug_synonyms.json

Outputs:
  processed_v2/interactions_labeled.jsonl
  processed_v2/label_map.json
  processed_v2/label_distribution.json
  reports/04_build_label_map.txt
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

MIN_COUNT = 50  # template must appear at least this many times to qualify as a label


def build_drug_patterns(profiles):
    """
    For each drug, build:
      - a sorted list of all candidate names (canonical + synonyms + brands),
        longest first (so 'Lepirudin recombinant' is tried before 'Lepirudin')
      - a single compiled regex that matches any of those names, case-insensitive
        and word-bounded, but tolerant of common edge cases like trailing
        parentheticals
    Returns: {drug_id: (sorted_name_list, compiled_pattern)}
    """
    patterns = {}
    for did, prof in profiles.items():
        names = set()
        if prof.get("name"):
            names.add(prof["name"])
        for s in prof.get("synonyms", []):
            if s:
                names.add(s)
        for b in prof.get("brand_names", []):
            if b:
                names.add(b)

        if not names:
            patterns[did] = ([], None)
            continue

        # Sort longest first so the alternation greedily matches the longest name.
        # (Python re alternation tries left-to-right, so longer names must come first.)
        sorted_names = sorted(names, key=lambda x: -len(x))

        # Escape each name for regex, then join with alternation.
        # Use word boundaries that tolerate parenthetical names like "(R)-warfarin".
        # We use lookbehind/lookahead instead of \b because \b doesn't fire next to
        # parens or hyphens.
        escaped = [re.escape(n) for n in sorted_names]
        # (?<!\w) ... (?!\w) is a word boundary that fires correctly around
        # punctuation like parens and hyphens.
        big_pattern = r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)"
        try:
            compiled = re.compile(big_pattern, re.IGNORECASE)
        except re.error:
            # Fall back to simple alternation if anything weird (shouldn't happen)
            compiled = None
        patterns[did] = (sorted_names, compiled)
    return patterns


def extract_template(description, drug_a_id, drug_b_id, patterns):
    """
    Find first occurrence of either drug in the description, replace it with
    #Drug1, then find the other drug and replace it with #Drug2.
    Returns (template, success, info).
    """
    a_names, a_pat = patterns.get(drug_a_id, ([], None))
    b_names, b_pat = patterns.get(drug_b_id, ([], None))
    if a_pat is None or b_pat is None:
        return description, False, "missing_pattern"

    # Find first match position for each drug
    m_a = a_pat.search(description)
    m_b = b_pat.search(description)
    if m_a is None or m_b is None:
        return description, False, "no_match"

    # The drug that appears earlier in the text becomes #Drug1
    if m_a.start() < m_b.start():
        first_pat, second_pat = a_pat, b_pat
    elif m_b.start() < m_a.start():
        first_pat, second_pat = b_pat, a_pat
    else:
        # Identical start (extremely rare overlap, e.g., when one name is prefix
        # of the other). Pick longer-name drug as the first match.
        if m_a.end() >= m_b.end():
            first_pat, second_pat = a_pat, b_pat
        else:
            first_pat, second_pat = b_pat, a_pat

    # Replace first occurrence of #Drug1, then first occurrence of #Drug2
    template = first_pat.sub("#Drug1", description, count=1)
    template = second_pat.sub("#Drug2", template, count=1)

    if "#Drug1" in template and "#Drug2" in template:
        return template, True, "ok"
    return description, False, "substitution_failed"


def main():
    print("Loading drug profiles...")
    with open(PROFILES) as f:
        profiles = json.load(f)
    with open(SYNONYMS) as f:
        synonyms = json.load(f)

    print("Building per-drug regex patterns...")
    patterns = build_drug_patterns(profiles)
    n_with_pattern = sum(1 for (_, p) in patterns.values() if p is not None)
    total_names = sum(len(names) for (names, _) in patterns.values())
    print(f"  {n_with_pattern:,} drugs have patterns; {total_names:,} total name variants")

    print("\nExtracting templates from descriptions...")
    template_counts = Counter()
    extraction_results = []
    n_success = 0
    n_failure = 0
    failure_examples = []

    with open(INTERACTIONS) as f:
        for line in tqdm(f, desc="rows", unit="row"):
            row = json.loads(line)
            template, ok, info = extract_template(
                row["description"], row["drug_a"], row["drug_b"], patterns
            )
            if ok:
                n_success += 1
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
                    failure_examples.append((row, info))

    print(f"\nExtraction results:")
    print(f"  Success:           {n_success:,}")
    print(f"  Failure:           {n_failure:,}")
    print(f"  Unique templates:  {len(template_counts):,}")

    qualifying = {t: c for t, c in template_counts.items() if c >= MIN_COUNT}
    print(f"\nTemplates with >={MIN_COUNT} occurrences: {len(qualifying):,}")
    coverage = sum(qualifying.values())
    if n_success > 0:
        print(f"Coverage: {coverage:,} / {n_success:,} ({100*coverage/n_success:.2f}%)")

    sorted_templates = sorted(qualifying.items(), key=lambda x: -x[1])
    label_map = {str(i + 1): t for i, (t, _) in enumerate(sorted_templates)}
    label_dist = {str(i + 1): c for i, (_, c) in enumerate(sorted_templates)}
    template_to_label = {t: str(i + 1) for i, (t, _) in enumerate(sorted_templates)}

    with open(PROCESSED / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(PROCESSED / "label_distribution.json", "w") as f:
        json.dump(label_dist, f, indent=2)

    print("Writing labeled interactions...")
    n_written = 0
    with open(PROCESSED / "interactions_labeled.jsonl", "w") as fout:
        for r in extraction_results:
            label = template_to_label.get(r["template"])
            if label is None:
                continue
            r["label"] = int(label)
            r["label_text"] = r["description"]
            fout.write(json.dumps(r) + "\n")
            n_written += 1
    print(f"  wrote {n_written:,} labeled interactions")

    suspicious = [(lbl, t) for lbl, t in label_map.items()
                  if "#Drug1" not in t or "#Drug2" not in t]

    counts = list(label_dist.values())
    lines = [
        "Step 4 (FAST) -- Label map construction",
        "=" * 60,
        f"Interactions processed:       {n_success + n_failure:,}",
        f"  Successful extractions:     {n_success:,}",
        f"  Failed extractions:         {n_failure:,}",
        "",
        f"Unique templates extracted:   {len(template_counts):,}",
        f"Templates with >={MIN_COUNT}:           {len(qualifying):,}",
        f"Coverage of labeled rows:     {100*coverage/max(n_success,1):.2f}%",
        f"Labeled interactions written: {n_written:,}",
        "",
        "Class distribution:",
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
    lines.append("")
    lines.append("Bottom 15 least frequent (qualifying) labels:")
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
        for r, info in failure_examples[:5]:
            a_name = profiles.get(r["drug_a"], {}).get("name", r["drug_a"])
            b_name = profiles.get(r["drug_b"], {}).get("name", r["drug_b"])
            lines.append(f"  [{info}] drug_a={a_name}  drug_b={b_name}")
            lines.append(f"    desc: {r['description'][:130]}")

    lines.extend([
        "",
        "Outputs:",
        f"  label_map.json              ({(PROCESSED / 'label_map.json').stat().st_size / 1e3:.1f} KB)",
        f"  label_distribution.json     ({(PROCESSED / 'label_distribution.json').stat().st_size / 1e3:.1f} KB)",
        f"  interactions_labeled.jsonl  ({(PROCESSED / 'interactions_labeled.jsonl').stat().st_size / 1e6:.1f} MB)",
    ])
    report = "\n".join(lines)
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
