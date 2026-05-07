"""
Step 3: Extract and deduplicate DDIs from XML.

Each interaction in DrugBank appears in BOTH drugs' DrugCards
(drug A's card mentions B; drug B's card mentions A).
We keep direction by storing the subject (whose card we found it in)
and the affected drug, but deduplicate on the
(subject, affected, description) triple so we count each interaction
exactly once.

Input:
  ~/ddiproject/raw/drugbank_full.xml

Outputs:
  processed_v2/interactions_full.jsonl    -- one row per unique interaction
  reports/03_parse_interactions.txt       -- counts and anomalies
"""
import json
from pathlib import Path
from lxml import etree
from tqdm import tqdm

ROOT = Path.home() / "ddiproject"
XML = ROOT / "raw" / "drugbank_full.xml"
OUT_DIR = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "03_parse_interactions.txt"

NS = {"db": "http://www.drugbank.ca"}


def text(elem, path):
    if elem is None:
        return ""
    f = elem.find(path, NS)
    return (f.text or "").strip() if f is not None and f.text else ""


def main():
    print(f"Streaming interactions from {XML} ({XML.stat().st_size / 1e9:.2f} GB)")
    print("This will take 1-3 minutes.")

    seen = {}  # (subject_id, affected_id, description) -> row dict
    parsed_drugs = 0
    raw_interactions = 0
    drugs_with_interactions = 0
    affected_id_unknown = 0

    drug_tag = f"{{{NS['db']}}}drug"
    context = etree.iterparse(str(XML), events=("end",), tag=drug_tag)

    pbar = tqdm(desc="scanning drugs", unit="drug")
    for event, elem in context:
        parent = elem.getparent()
        if parent is None or not parent.tag.endswith("drugbank"):
            elem.clear()
            continue

        # Subject = primary DrugBank ID of the drug whose card we're reading
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

        # Walk the drug-interactions block
        di_container = elem.find("db:drug-interactions", NS)
        if di_container is not None:
            interactions_here = 0
            for di in di_container.findall("db:drug-interaction", NS):
                affected_id = text(di, "db:drugbank-id")
                description = text(di, "db:description")

                if not description:
                    continue
                if not affected_id:
                    affected_id_unknown += 1
                    continue

                raw_interactions += 1
                interactions_here += 1
                key = (subject_id, affected_id, description)
                if key not in seen:
                    seen[key] = {
                        "subject_id": subject_id,
                        "affected_id": affected_id,
                        "description": description,
                    }
            if interactions_here > 0:
                drugs_with_interactions += 1

        parsed_drugs += 1
        pbar.update(1)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    pbar.close()

    print(f"\nWriting {len(seen):,} unique interactions...")
    out_path = OUT_DIR / "interactions_full.jsonl"
    with open(out_path, "w") as f:
        for row in seen.values():
            f.write(json.dumps(row) + "\n")

    # Anomaly checks
    self_loops = sum(1 for r in seen.values() if r["subject_id"] == r["affected_id"])

    out_degree = {}
    for r in seen.values():
        out_degree[r["subject_id"]] = out_degree.get(r["subject_id"], 0) + 1
    top_drugs = sorted(out_degree.items(), key=lambda x: -x[1])[:10]

    subjects = {r["subject_id"] for r in seen.values()}
    affecteds = {r["affected_id"] for r in seen.values()}

    lines = [
        "Step 3 -- Interaction extraction",
        "=" * 60,
        f"Drugs scanned:                  {parsed_drugs:,}",
        f"Drugs with at least 1 DDI:      {drugs_with_interactions:,}",
        f"Raw <drug-interaction> tags:    {raw_interactions:,}",
        f"Unique interactions (dedup):    {len(seen):,}",
        f"Dedup ratio:                    {raw_interactions / max(len(seen), 1):.2f}x",
        f"Affected drug missing:          {affected_id_unknown:,}",
        f"Self-loops (anomaly):           {self_loops:,}",
        f"Distinct drugs as subject:      {len(subjects):,}",
        f"Distinct drugs as affected:     {len(affecteds):,}",
        "",
        "Top 10 drugs by outgoing-interaction count:",
    ]
    for d, c in top_drugs:
        lines.append(f"  {d}: {c:,}")
    lines.extend([
        "",
        f"Output: {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)",
    ])
    report = "\n".join(lines)
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
