#!/usr/bin/env python3
"""Patch gold_polarity (and gold_l2 if needed) in pilot files from corrected hierarchy."""
import json, shutil
from pathlib import Path
from datetime import datetime

HIER = Path("../processed_v2/hierarchy_map.json")
hier = json.loads(HIER.read_text())

FILES = ["pilot_prompts.jsonl", "pilot_traces.jsonl"]
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

for fname in FILES:
    p = Path(fname)
    if not p.exists():
        print(f"  SKIP {fname} (not found)")
        continue
    bak = p.with_suffix(f".jsonl.bak_step17b_{ts}")
    shutil.copy(p, bak)
    print(f"  Backed up {fname} -> {bak.name}")

    n_total = 0
    n_changed = 0
    out_lines = []
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            n_total += 1
            label_id = str(r.get("label"))
            rec = hier.get(label_id, {})
            new_pol = rec.get("polarity")
            new_l2  = rec.get("l2")
            old_pol = r.get("gold_polarity")
            old_l2  = r.get("gold_l2")
            if new_pol is not None and old_pol != new_pol:
                r["gold_polarity"] = new_pol
                n_changed += 1
            if new_l2 is not None and old_l2 != new_l2:
                r["gold_l2"] = new_l2
            out_lines.append(json.dumps(r))
    p.write_text("\n".join(out_lines) + "\n")
    print(f"  {fname}: {n_total} records, {n_changed} polarity values updated")

print("\nDone. Now re-run score:")
print("  python ../scripts/step12_pilot_2x2.py --data-dir ../processed_v2 --out-dir . score")
