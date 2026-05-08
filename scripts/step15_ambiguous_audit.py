#!/usr/bin/env python3
"""Sample ambiguous traces stratified by condition for manual audit."""
import json
import random
from pathlib import Path

random.seed(42)
SAMPLE_PER_CONDITION = 8  # 32 total

amb = {}  # condition -> list of records
with open("pilot_scored.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r.get("direction_verdict") == "ambiguous":
            amb.setdefault(r["condition"], []).append(r)

print("Ambiguous trace counts per condition:")
for c in sorted(amb):
    print(f"  {c}: {len(amb[c])}")

samples = []
for cond in sorted(amb):
    pool = amb[cond]
    n = min(SAMPLE_PER_CONDITION, len(pool))
    samples.extend(random.sample(pool, n))

print(f"\nSampled {len(samples)} ambiguous traces ({SAMPLE_PER_CONDITION}/condition)")

# Write to a readable file for you to skim
with open("ambiguous_audit_sample.txt", "w") as out:
    for i, r in enumerate(samples, 1):
        out.write(f"\n{'='*80}\n")
        out.write(f"#{i}  condition={r['condition']}  pair={r['pair_uid']}\n")
        out.write(f"gold_l2={r['gold_l2']}  gold_polarity={r['gold_polarity']}\n")
        out.write(f"direction_extracted={r.get('direction_extracted')}  "
                  f"evidence={r.get('direction_evidence')}  "
                  f"source={r.get('direction_source')}\n")
        out.write(f"label_text: {r['label_text'][:200]}\n")
        # Pull just the Summary section for readability
        trace = r.get("trace","")
        summary_start = trace.find("## Summary")
        if summary_start >= 0:
            summary_end = trace.find("## Classification", summary_start)
            if summary_end < 0:
                summary_end = summary_start + 800
            out.write(f"\n[SUMMARY ONLY]\n{trace[summary_start:summary_end].strip()}\n")
        else:
            out.write(f"\n[NO SUMMARY HEADER FOUND - last 600 chars]\n{trace[-600:]}\n")

print("Wrote ambiguous_audit_sample.txt")
print(f"Quick read it: less ambiguous_audit_sample.txt")
