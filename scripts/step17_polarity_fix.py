#!/usr/bin/env python3
"""Apply polarity tag corrections from step16 scan, with guardrails."""
import json, shutil, sys
from pathlib import Path
from datetime import datetime

HIER = Path("../processed_v2/hierarchy_map.json")
BAK  = HIER.with_suffix(".json.bak_step17_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

# Each fix asserts the EXPECTED current value before overwriting.
FIXES = {
    "5":   {"expect": "increase", "set": "decrease",
            "reason": "efficacy_decrease: efficacy goes down"},
    "7":   {"expect": "increase", "set": "decrease",
            "reason": "metabolism_increase: more metab -> less exposure (inversion)"},
    "60":  {"expect": "increase", "set": "decrease",
            "reason": "bp_decrease: more antihypertensive -> BP down"},
    "114": {"expect": "increase", "set": "decrease",
            "reason": "bp_decrease: less hypertensive -> BP down"},
    "134": {"expect": "increase", "set": "decrease",
            "reason": "bp_decrease: hypertension risk decreased"},
}

hier = json.loads(HIER.read_text())

# Sweep: infection cluster with polarity=n/a -> increase
infection_fixes = {k: v for k, v in hier.items()
                   if v.get("l2") == "infection" and v.get("polarity") == "n/a"}
for k in infection_fixes:
    FIXES[k] = {"expect": "n/a", "set": "increase",
                "reason": f"infection: risk increased (template: {hier[k].get('template','')[:60]})"}

# Pre-flight: check every expected old value
errors = []
for k, fix in FIXES.items():
    if k not in hier:
        errors.append(f"Y={k}: not found in hierarchy_map")
        continue
    actual = hier[k].get("polarity")
    if actual != fix["expect"]:
        errors.append(f"Y={k}: expected polarity={fix['expect']!r}, got {actual!r} - file may have been modified")

if errors:
    print("ABORT - pre-flight asserts failed:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)

print(f"Pre-flight OK ({len(FIXES)} fixes, all expected old values match)")
print(f"Backing up to {BAK.name}")
shutil.copy(HIER, BAK)

# Apply
changes = []
for k, fix in FIXES.items():
    old = hier[k].get("polarity")
    hier[k]["polarity"] = fix["set"]
    changes.append({"label_id": k, "l2": hier[k].get("l2"),
                    "old_polarity": old, "new_polarity": fix["set"],
                    "reason": fix["reason"]})
    print(f"  Y={k:>4} {hier[k].get('l2'):<22} {old:<8} -> {fix['set']:<8}  {fix['reason']}")

HIER.write_text(json.dumps(hier, indent=2))
Path("polarity_fix_log.json").write_text(json.dumps(
    {"timestamp": datetime.now().isoformat(), "backup": BAK.name, "changes": changes},
    indent=2))
print(f"\nApplied {len(changes)} fixes. Log: polarity_fix_log.json")
