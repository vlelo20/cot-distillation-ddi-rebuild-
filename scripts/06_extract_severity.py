"""
Step 6: Extract per-interaction severity from XML.

DrugBank 6.0 stores a 'severity' field per drug-interaction. Extract it,
match it to the (drug_a, drug_b, description) keys we already have, and
write a severity_map.json that maps interaction-key -> severity.

This enables:
  - Severity-weighted F1 in evaluation
  - Filtering training data by severity (e.g., 'major only' subset)
  - Per-class severity profile reporting

Inputs:
  raw/drugbank_full.xml
  processed_v2/interactions_dedup.jsonl  (canonical interactions)

Outputs:
  processed_v2/severity_map.json   -- {drug_a||drug_b||desc_hash: severity}
  processed_v2/interactions_with_severity.jsonl  -- labeled interactions augmented with severity
  reports/06_severity.txt
"""
import json
import hashlib
from pathlib import Path
from lxml import etree
from tqdm import tqdm
from collections import Counter

ROOT = Path.home() / "ddiproject"
XML = ROOT / "raw" / "drugbank_full.xml"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "06_severity.txt"
NS = {"db": "http://www.drugbank.ca"}


def text(elem, path):
    if elem is None:
        return ""
    f = elem.find(path, NS)
    return (f.text or "").strip() if f is not None and f.text else ""


def make_key(drug_a, drug_b, description):
    """Canonical interaction key: smaller-id first, then description."""
    if drug_a > drug_b:
        drug_a, drug_b = drug_b, drug_a
    h = hashlib.md5(description.encode("utf-8")).hexdigest()[:12]
    return f"{drug_a}||{drug_b}||{h}"


def main():
    print(f"Streaming severity from {XML}...")
    severity_by_key = {}  # canonical key -> severity
    severity_counts = Counter()
    interactions_with_severity = 0
    interactions_total = 0

    drug_tag = f"{{{NS['db']}}}drug"
    context = etree.iterparse(str(XML), events=("end",), tag=drug_tag)

    pbar = tqdm(desc="scanning", unit="drug")
    for event, elem in context:
        parent = elem.getparent()
        if parent is None or not parent.tag.endswith("drugbank"):
            elem.clear()
            continue

        subject_id = ""
        for did in elem.findall("db:drugbank-id", NS):
            if did.get("primary") == "true":
                subject_id = (did.text or "").strip()
                break
        if not subject_id:
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            continue

        di_container = elem.find("db:drug-interactions", NS)
        if di_container is not None:
            for di in di_container.findall("db:drug-interaction", NS):
                affected_id = text(di, "db:drugbank-id")
                description = text(di, "db:description")
                severity = text(di, "db:severity")
                if not affected_id or not description:
                    continue
                interactions_total += 1
                if severity:
                    interactions_with_severity += 1
                    severity_counts[severity] += 1
                    key = make_key(subject_id, affected_id, description)
                    # Use first observation; both sides should agree
                    if key not in severity_by_key:
                        severity_by_key[key] = severity

        pbar.update(1)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    pbar.close()

    pct = 100 * interactions_with_severity / max(interactions_total, 1)
    print(f"\nInteractions scanned: {interactions_total:,}")
    print(f"With severity field:  {interactions_with_severity:,} ({pct:.1f}%)")
    print(f"Distinct keys:        {len(severity_by_key):,}")
    print(f"Severity distribution: {dict(severity_counts)}")

    # Save the key -> severity map
    with open(PROCESSED / "severity_map.json", "w") as f:
        json.dump(severity_by_key, f)
    print(f"Wrote severity_map.json ({(PROCESSED / 'severity_map.json').stat().st_size / 1e6:.1f} MB)")

    # Augment interactions_labeled.jsonl with severity (where available)
    print("\nMerging severity into labeled interactions...")
    labeled_path = PROCESSED / "interactions_labeled.jsonl"
    out_path = PROCESSED / "interactions_with_severity.jsonl"
    n_total = 0
    n_with_sev = 0
    sev_dist_in_labeled = Counter()
    with open(labeled_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            r = json.loads(line)
            n_total += 1
            key = make_key(r["drug_a"], r["drug_b"], r["description"])
            sev = severity_by_key.get(key)
            r["severity"] = sev
            if sev:
                n_with_sev += 1
                sev_dist_in_labeled[sev] += 1
            fout.write(json.dumps(r) + "\n")

    pct_labeled = 100 * n_with_sev / max(n_total, 1)
    print(f"Labeled rows: {n_total:,}")
    print(f"With severity assigned: {n_with_sev:,} ({pct_labeled:.1f}%)")
    print(f"Severity distribution in labeled set: {dict(sev_dist_in_labeled)}")

    # Write report
    lines = [
        "Step 6 -- Severity extraction",
        "=" * 60,
        f"Raw interactions in XML:        {interactions_total:,}",
        f"With severity field populated:  {interactions_with_severity:,} ({pct:.1f}%)",
        "",
        "Severity distribution in raw XML:",
    ]
    for sev, count in severity_counts.most_common():
        lines.append(f"  {sev}: {count:,}")
    lines.extend([
        "",
        f"Labeled interactions:           {n_total:,}",
        f"With severity merged in:        {n_with_sev:,} ({pct_labeled:.1f}%)",
        "",
        "Severity distribution in labeled set:",
    ])
    for sev, count in sev_dist_in_labeled.most_common():
        lines.append(f"  {sev}: {count:,}")
    lines.extend([
        "",
        "Outputs:",
        f"  severity_map.json                  ({(PROCESSED / 'severity_map.json').stat().st_size / 1e6:.1f} MB)",
        f"  interactions_with_severity.jsonl   ({out_path.stat().st_size / 1e6:.1f} MB)",
    ])
    report = "\n".join(lines)
    print()
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
