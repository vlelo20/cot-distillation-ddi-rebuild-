#!/usr/bin/env python3
"""Scan hierarchy_map.json for polarity-tagging inconsistencies.

Two checks:
  (A) intra-cluster mixed polarity — same l2, different polarity tags
      (this is unambiguously a bug)
  (B) name-vs-tag mismatches against expected polarity per cluster type:
      - PD outcome clusters (efficacy, bp, hypotension, bleeding, etc.):
        polarity should match the surface name suffix
      - PK exposure-affecting clusters (metabolism, excretion):
        polarity should be INVERTED from the surface name
        (less metabolism -> more exposure -> polarity=increase)
      - Absorption / exposure: polarity matches name suffix
        (less absorption -> less exposure)
"""
import json
from collections import defaultdict, Counter
from pathlib import Path

HIER = Path("../processed_v2/hierarchy_map.json")
hier = json.loads(HIER.read_text())

# Group by l2
clusters = defaultdict(list)  # l2 -> list of (label_id, l1, polarity, template)
for label_id, rec in hier.items():
    clusters[rec.get("l2")].append((
        label_id, rec.get("l1"), rec.get("polarity"),
        rec.get("template", "")[:80]
    ))

# --- Check A: intra-cluster mixed polarity ---
print("="*72)
print("CHECK A · intra-cluster mixed polarity (unambiguous bugs)")
print("="*72)
mixed = []
for l2, members in sorted(clusters.items()):
    pols = Counter(m[2] for m in members)
    if len(pols) > 1:
        mixed.append((l2, pols, members))
        print(f"\n  {l2}  ({len(members)} labels, polarities={dict(pols)})")
        for label_id, l1, polarity, tpl in members:
            print(f"    Y={label_id:>4} l1={l1:<3} pol={polarity:<8}  {tpl}")
if not mixed:
    print("  None found.")

# --- Check B: name-vs-tag mismatches by cluster type ---
# Define expectations
PK_INVERT = {"metabolism_decrease", "metabolism_increase",
             "excretion_decrease", "excretion_increase"}
# absorption_*, exposure_*, protein_binding_* polarity matches name suffix

def expected_polarity(l2):
    if l2 is None: return None
    if l2 in PK_INVERT:
        # name says decrease -> exposure increases -> polarity=increase
        return "increase" if l2.endswith("_decrease") else "decrease"
    if l2.endswith("_increase"):
        return "increase"
    if l2.endswith("_decrease"):
        return "decrease"
    return None  # unsuffixed clusters (bleeding, thrombosis, etc.) -- skip

print("\n" + "="*72)
print("CHECK B · name-vs-tag mismatches (expected based on cluster naming)")
print("="*72)
mismatches = []
for l2, members in sorted(clusters.items()):
    exp = expected_polarity(l2)
    if exp is None:
        continue
    for label_id, l1, polarity, tpl in members:
        if polarity != exp and polarity not in (None, "n/a"):
            mismatches.append((l2, label_id, l1, polarity, exp, tpl))

if mismatches:
    print(f"\n  {len(mismatches)} mismatches found:\n")
    print(f"  {'cluster':<28} {'Y':>4} {'l1':<3} {'got':<10} {'expected':<10} template")
    print("  " + "-"*98)
    for l2, lid, l1, pol, exp, tpl in mismatches:
        print(f"  {l2:<28} {lid:>4} {l1:<3} {pol:<10} {exp:<10} {tpl}")
else:
    print("  None found.")

# --- Cluster-level overview ---
print("\n" + "="*72)
print("OVERVIEW · all clusters with their polarity tags")
print("="*72)
print(f"  {'cluster':<32} {'n':>3} {'polarities (count)'}")
print("  " + "-"*72)
for l2 in sorted(clusters):
    members = clusters[l2]
    pols = Counter(m[2] for m in members)
    print(f"  {l2:<32} {len(members):>3} {dict(pols)}")

# Save bug report
out = {
    "intra_cluster_mixed": [
        {"l2": l2, "polarities": dict(pols),
         "members": [{"label_id": lid, "l1": l1, "polarity": p, "template": t}
                     for lid, l1, p, t in mems]}
        for l2, pols, mems in mixed
    ],
    "name_vs_tag_mismatches": [
        {"l2": l2, "label_id": lid, "l1": l1,
         "got_polarity": pol, "expected_polarity": exp, "template": tpl}
        for l2, lid, l1, pol, exp, tpl in mismatches
    ],
}
Path("polarity_scan_report.json").write_text(json.dumps(out, indent=2))
print("\nWrote polarity_scan_report.json")
print(f"\nSummary: {len(mixed)} mixed-polarity clusters, "
      f"{len(mismatches)} name-vs-tag mismatches.")
