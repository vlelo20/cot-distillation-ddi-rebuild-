#!/usr/bin/env python3
"""Confusion analysis: when predictions miss, where do they land?"""
import json
from collections import defaultdict, Counter
from pathlib import Path

PILOT = Path("pilot_scored.jsonl")
HIER  = Path("../processed_v2/hierarchy_map.json")

hier = json.loads(HIER.read_text())
def label_to_l2(label_id):
    rec = hier.get(str(label_id))
    return rec["l2"] if rec else None

by_cond = defaultdict(lambda: {"n":0,"hit":0,"miss_within":0,"miss_cross":0})
confusion = defaultdict(Counter)

with PILOT.open() as f:
    for line in f:
        r = json.loads(line)
        cond = r["condition"]
        gold_l2 = r["gold_l2"]
        pred_label = r.get("pred_label")
        pred_l2 = label_to_l2(pred_label) if pred_label is not None else None
        d = by_cond[cond]
        d["n"] += 1
        if r.get("exact_match"):
            d["hit"] += 1
        elif r.get("cluster_match"):
            d["miss_within"] += 1
        else:
            d["miss_cross"] += 1
            if pred_l2 and pred_l2 != gold_l2:
                confusion[gold_l2][pred_l2] += 1

print(f"{'condition':<18} {'n':>4} {'hit%':>6} {'within%':>8} {'cross%':>7}")
print("-"*50)
for cond in sorted(by_cond):
    d = by_cond[cond]
    n = d["n"]
    print(f"{cond:<18} {n:>4} {100*d['hit']/n:>5.1f} "
          f"{100*d['miss_within']/n:>7.1f} {100*d['miss_cross']/n:>6.1f}")

print("\n=== TOP CROSS-CLUSTER CONFUSIONS (gold -> predicted, summed across all 4 conditions) ===")
flat = []
for gold, ctr in confusion.items():
    for pred, n in ctr.items():
        flat.append((n, gold, pred))
flat.sort(reverse=True)
for n, gold, pred in flat[:20]:
    print(f"  {n:>3}x  {gold:<30} -> {pred}")

out = {"by_condition": {k: dict(v) for k,v in by_cond.items()},
       "confusion_pairs": [{"count":n,"gold_l2":g,"pred_l2":p} for n,g,p in flat]}
Path("confusion_analysis.json").write_text(json.dumps(out, indent=2))
print("\nWrote confusion_analysis.json")
