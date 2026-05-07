"""
Step 6b: Diagnose where severity actually lives in the XML.

The previous step found 0 <severity> children of <drug-interaction>. Either:
  (a) DrugBank moved the field to a different element name
  (b) It's nested deeper (e.g., inside an <evidence> block)
  (c) The academic XML export doesn't include severity at all
  (d) It's an attribute rather than a child element

We dump the full XML structure of the first 5 drug-interaction elements
encountered to see what's actually in there.
"""
from pathlib import Path
from lxml import etree

ROOT = Path.home() / "ddiproject"
XML = ROOT / "raw" / "drugbank_full.xml"
NS = {"db": "http://www.drugbank.ca"}


def main():
    drug_tag = f"{{{NS['db']}}}drug"
    context = etree.iterparse(str(XML), events=("end",), tag=drug_tag)

    interactions_dumped = 0
    target_dumps = 5

    for event, elem in context:
        parent = elem.getparent()
        if parent is None or not parent.tag.endswith("drugbank"):
            elem.clear()
            continue

        di_container = elem.find("db:drug-interactions", NS)
        if di_container is not None:
            for di in di_container.findall("db:drug-interaction", NS):
                if interactions_dumped >= target_dumps:
                    break
                # Print the whole element so we see every child + attribute
                print("=" * 70)
                print(f"DRUG-INTERACTION #{interactions_dumped + 1}")
                print("=" * 70)
                print(f"Element attributes: {dict(di.attrib)}")
                print()
                print("Children:")
                for child in di:
                    tag = etree.QName(child).localname
                    text_preview = (child.text or "").strip()[:80] if child.text else "(empty)"
                    attrs = dict(child.attrib) if child.attrib else None
                    print(f"  <{tag}>  text={text_preview!r}  attrs={attrs}")
                    # If this child has its own children, list them
                    for grandchild in child:
                        gtag = etree.QName(grandchild).localname
                        gtext = (grandchild.text or "").strip()[:60] if grandchild.text else "(empty)"
                        print(f"      <{gtag}>  text={gtext!r}")
                print()
                # Also dump the full XML of this element for absolute clarity
                xml_str = etree.tostring(di, pretty_print=True).decode("utf-8")
                print("Full XML:")
                print(xml_str[:1200])
                print()
                interactions_dumped += 1

            if interactions_dumped >= target_dumps:
                break

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if interactions_dumped >= target_dumps:
            break

    if interactions_dumped == 0:
        print("ERROR: No drug-interaction elements found at all.")
    else:
        print(f"Dumped {interactions_dumped} drug-interaction examples.")
        print()
        print("Look for any of: 'severity', 'evidence', 'level', 'risk', 'major',")
        print("'moderate', 'minor' in the output above.")


if __name__ == "__main__":
    main()
