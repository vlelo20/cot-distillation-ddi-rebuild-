"""
Step 2: Stream-parse DrugBank 6.0 XML and extract clean drug profiles.

Outputs (in ~/ddiproject/processed_v2/):
  drug_profiles.json    -- per-drug enriched record
  drug_synonyms.json    -- name index for retrieval
  drug_id_order.json    -- stable ordering of drug IDs

Cleaning principle: keep everything that helps reasoning (synonyms, mechanism,
metabolism, targets, enzymes, transporters), drop bibliographic/regulatory/
spectral/formulation noise.

No group filter -- all drugs included regardless of approved/withdrawn/
investigational/experimental status. The 'groups' field is preserved so
downstream code can filter if needed.
"""
import json
from pathlib import Path
from lxml import etree
from tqdm import tqdm

ROOT = Path.home() / "ddiproject"
XML = ROOT / "raw" / "drugbank_full.xml"
OUT_DIR = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "02_parse_drugs.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

NS = {"db": "http://www.drugbank.ca"}


def text(elem, path):
    if elem is None:
        return ""
    f = elem.find(path, NS)
    return (f.text or "").strip() if f is not None and f.text else ""


def all_text(elem, path):
    if elem is None:
        return []
    return [(e.text or "").strip() for e in elem.findall(path, NS) if e.text]


def truncate(s, max_words=200):
    if not s:
        return ""
    words = s.split()
    if len(words) <= max_words:
        return s
    return " ".join(words[:max_words]) + " [...]"


def parse_targets(drug_elem, kind):
    container = drug_elem.find(f"db:{kind}", NS)
    if container is None:
        return []
    out = []
    item_tag = kind[:-1]
    for item in container.findall(f"db:{item_tag}", NS):
        name = text(item, "db:name")
        organism = text(item, "db:organism")
        actions = all_text(item, "db:actions/db:action")
        uniprot = ""
        poly = item.find("db:polypeptide", NS)
        if poly is not None:
            for eid in poly.findall("db:external-identifiers/db:external-identifier", NS):
                if text(eid, "db:resource") == "UniProtKB":
                    uniprot = text(eid, "db:identifier")
                    break
        out.append({
            "uniprot": uniprot,
            "name": name,
            "actions": actions,
            "organism": organism,
        })
    return out


def parse_drug(drug_elem):
    db_ids = drug_elem.findall("db:drugbank-id", NS)
    primary_id = ""
    secondary_ids = []
    for did in db_ids:
        if did.get("primary") == "true":
            primary_id = (did.text or "").strip()
        else:
            secondary_ids.append((did.text or "").strip())
    if not primary_id:
        return None

    name = text(drug_elem, "db:name")
    drug_type = drug_elem.get("type", "")

    synonyms = []
    syn_container = drug_elem.find("db:synonyms", NS)
    if syn_container is not None:
        for syn in syn_container.findall("db:synonym", NS):
            if syn.text:
                synonyms.append(syn.text.strip())

    brand_names = set()
    products = drug_elem.find("db:products", NS)
    if products is not None:
        for prod in products.findall("db:product", NS):
            pname = text(prod, "db:name")
            if pname:
                brand_names.add(pname)
    intl = drug_elem.find("db:international-brands", NS)
    if intl is not None:
        for ib in intl.findall("db:international-brand", NS):
            iname = text(ib, "db:name")
            if iname:
                brand_names.add(iname)

    groups = all_text(drug_elem, "db:groups/db:group")

    atc_codes = []
    atc_container = drug_elem.find("db:atc-codes", NS)
    if atc_container is not None:
        for atc in atc_container.findall("db:atc-code", NS):
            code = atc.get("code", "")
            if code:
                atc_codes.append(code)

    categories = []
    cat_container = drug_elem.find("db:categories", NS)
    if cat_container is not None:
        for cat in cat_container.findall("db:category", NS):
            cname = text(cat, "db:category")
            if cname:
                categories.append(cname)

    smiles = ""
    inchi_key = ""
    cprops = drug_elem.find("db:calculated-properties", NS)
    if cprops is not None:
        for prop in cprops.findall("db:property", NS):
            kind = text(prop, "db:kind")
            if kind == "SMILES":
                smiles = text(prop, "db:value")
            elif kind == "InChIKey":
                inchi_key = text(prop, "db:value")

    return {
        "drugbank_id": primary_id,
        "secondary_ids": secondary_ids,
        "name": name,
        "type": drug_type,
        "groups": groups,
        "synonyms": synonyms,
        "brand_names": sorted(brand_names),
        "smiles": smiles,
        "inchi_key": inchi_key,
        "description": truncate(text(drug_elem, "db:description"), 200),
        "indication": truncate(text(drug_elem, "db:indication"), 200),
        "pharmacodynamics": truncate(text(drug_elem, "db:pharmacodynamics"), 200),
        "mechanism_of_action": truncate(text(drug_elem, "db:mechanism-of-action"), 200),
        "metabolism": truncate(text(drug_elem, "db:metabolism"), 150),
        "route_of_elimination": truncate(text(drug_elem, "db:route-of-elimination"), 80),
        "half_life": truncate(text(drug_elem, "db:half-life"), 60),
        "atc_codes": atc_codes,
        "categories": categories,
        "targets": parse_targets(drug_elem, "targets"),
        "enzymes": parse_targets(drug_elem, "enzymes"),
        "transporters": parse_targets(drug_elem, "transporters"),
        "carriers": parse_targets(drug_elem, "carriers"),
    }


def main():
    print(f"Parsing {XML} ({XML.stat().st_size / 1e9:.2f} GB)")
    print("This will take 3-8 minutes.")

    profiles = {}
    drug_id_order = []
    synonyms_index = {}
    counts = {
        "total": 0,
        "by_type": {},
        "by_group": {},
        "with_smiles": 0,
        "with_targets": 0,
        "with_enzymes": 0,
        "with_metabolism_text": 0,
        "with_moa_text": 0,
    }

    drug_tag = f"{{{NS['db']}}}drug"
    context = etree.iterparse(str(XML), events=("end",), tag=drug_tag)

    pbar = tqdm(desc="parsing drugs", unit="drug")
    for event, elem in context:
        parent = elem.getparent()
        if parent is None or not parent.tag.endswith("drugbank"):
            elem.clear()
            continue

        profile = parse_drug(elem)
        if profile:
            pid = profile["drugbank_id"]
            profiles[pid] = profile
            drug_id_order.append(pid)
            synonyms_index[pid] = {
                "primary_name": profile["name"],
                "synonyms": profile["synonyms"],
                "brand_names": profile["brand_names"],
                "secondary_ids": profile["secondary_ids"],
            }
            counts["total"] += 1
            t = profile["type"]
            counts["by_type"][t] = counts["by_type"].get(t, 0) + 1
            for g in profile["groups"]:
                counts["by_group"][g] = counts["by_group"].get(g, 0) + 1
            if profile["smiles"]:
                counts["with_smiles"] += 1
            if profile["targets"]:
                counts["with_targets"] += 1
            if profile["enzymes"]:
                counts["with_enzymes"] += 1
            if profile["metabolism"]:
                counts["with_metabolism_text"] += 1
            if profile["mechanism_of_action"]:
                counts["with_moa_text"] += 1
            pbar.update(1)

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    pbar.close()

    print("Writing outputs...")
    with open(OUT_DIR / "drug_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)
    with open(OUT_DIR / "drug_synonyms.json", "w") as f:
        json.dump(synonyms_index, f, indent=2)
    with open(OUT_DIR / "drug_id_order.json", "w") as f:
        json.dump(drug_id_order, f, indent=2)

    total = counts["total"]
    lines = [
        "Step 2 -- Drug profile extraction",
        "=" * 60,
        f"Total drugs parsed: {total:,}",
        "",
        "By type:",
    ]
    for t, c in sorted(counts["by_type"].items(), key=lambda x: -x[1]):
        lines.append(f"  {t or '(blank)'}: {c:,}")
    lines.append("")
    lines.append("By group:")
    for g, c in sorted(counts["by_group"].items(), key=lambda x: -x[1]):
        lines.append(f"  {g}: {c:,}")
    lines.extend([
        "",
        "Coverage of high-value fields:",
        f"  with SMILES:           {counts['with_smiles']:,}  ({100*counts['with_smiles']/total:.1f}%)",
        f"  with targets:          {counts['with_targets']:,}  ({100*counts['with_targets']/total:.1f}%)",
        f"  with enzymes:          {counts['with_enzymes']:,}  ({100*counts['with_enzymes']/total:.1f}%)",
        f"  with mechanism text:   {counts['with_moa_text']:,}  ({100*counts['with_moa_text']/total:.1f}%)",
        f"  with metabolism text:  {counts['with_metabolism_text']:,}  ({100*counts['with_metabolism_text']/total:.1f}%)",
        "",
        "Outputs:",
        f"  drug_profiles.json    ({(OUT_DIR / 'drug_profiles.json').stat().st_size / 1e6:.1f} MB)",
        f"  drug_synonyms.json    ({(OUT_DIR / 'drug_synonyms.json').stat().st_size / 1e6:.1f} MB)",
        f"  drug_id_order.json    ({(OUT_DIR / 'drug_id_order.json').stat().st_size / 1e6:.1f} MB)",
    ])
    report = "\n".join(lines)
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
