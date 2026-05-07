"""
src/step11d_direction_scorer.py

Direction-aware evaluation scorer for teacher CoT traces.

For each generated trace, extracts the direction-of-effect signal from
the ## Summary section (or full text fallback) and compares it against
the gold polarity stored in hierarchy_map[label]['polarity'].

Returns one of:
  - "correct"     — extracted direction matches gold polarity
  - "incorrect"   — extracted direction is opposite of gold
  - "ambiguous"   — both directions found, or gold polarity is n/a
  - "missing"    — no direction phrasing found in trace
  - "no_summary" — trace has no ## Summary section at all

Usage as a module:
    from step11d_direction_scorer import score_trace, score_traces

    result = score_trace(trace_text, gold_polarity="increase")
    # -> {"verdict": "correct", "extracted": "increase", "gold": "increase",
    #     "evidence": ["increases", "raised"], ...}

Usage as a CLI:
    python scripts/step11d_direction_scorer.py traces.jsonl --out scored.jsonl

Inputs (CLI mode): a JSONL file where each row has at least
    {"trace": "...", "label": <int>}
Outputs: same rows with "direction_score" field appended.

Methodological notes:

The scorer prefers signal from the ## Summary section because that's where
the prompt asks the teacher to commit to a direction explicitly. Falling
back to full-text scan is more permissive but risks scoring "increase" from
a passage like "increase in metabolism leading to decreased levels".

Polarity in the hierarchy reflects the PHYSIOLOGICAL OUTCOME, not the
surface verb (e.g., "decrease excretion" has gold polarity=increase because
exposure goes up). The scorer therefore looks for outcome-level direction
words, not just any verb in the trace.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ── Patterns ─────────────────────────────────────────────────────────────────

# Word-boundary regexes for direction signals. Order matters only for
# evidence display — verdict logic uses set membership.
INCREASE_PATTERNS = [
    r"\bincrease(?:s|d|ing)?\b",
    r"\bhigher\b",
    r"\belevate(?:s|d|ing)?\b",
    r"\braise(?:s|d|ing)?\b",
    r"\brise(?:s|n|ing)?\b",
    r"\brose\b",
    r"\benhance(?:s|d|ing)?\b",
    r"\bpotentiate(?:s|d|ing)?\b",
    r"\bamplif(?:y|ies|ied|ying)\b",
    r"\baugment(?:s|ed|ing)?\b",
    r"\bgreater\s+(?:risk|exposure|levels?|concentrations?|effect|activity)\b",
    r"\bmore\s+(?:risk|exposure|drug|effect|activity)\b",
    r"\baccumulat(?:e|es|ed|ing|ion)\b",
    r"\bsupratherapeutic\b",
    r"\btoxic\s+(?:levels?|concentrations?)\b",
]
DECREASE_PATTERNS = [
    r"\bdecrease(?:s|d|ing)?\b",
    r"\blower(?:s|ed|ing)?\b",
    r"\breduce(?:s|d|ing)?\b",
    r"\bdiminish(?:es|ed|ing)?\b",
    r"\bsuppress(?:es|ed|ing)?\b",
    r"\battenuate(?:s|d|ing)?\b",
    r"\binhibit(?:s|ed|ing)?\s+(?:the\s+)?(?:therapeutic|clinical)\s+effect\b",
    r"\bless\s+(?:effective|exposure|drug|effect|activity)\b",
    r"\bsmaller\s+(?:exposure|effect|levels?)\b",
    r"\bsubtherapeutic\b",
    r"\bloss\s+of\s+(?:efficacy|effect)\b",
    r"\bclear(?:s|ed|ance)\s+(?:more\s+)?(?:rapidly|quickly|fast)",
]

# Sentinel for the Summary section. Tolerant to extra whitespace and casing.
SUMMARY_HEADER_RE = re.compile(r"^\s*##\s*summary\s*$", re.IGNORECASE | re.MULTILINE)
NEXT_HEADER_RE = re.compile(r"^\s*##\s*\S", re.MULTILINE)

INCREASE_RE = re.compile("|".join(INCREASE_PATTERNS), re.IGNORECASE)
DECREASE_RE = re.compile("|".join(DECREASE_PATTERNS), re.IGNORECASE)


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_summary(trace: str) -> Optional[str]:
    """Pull out the ## Summary section from a trace. Returns None if absent."""
    m = SUMMARY_HEADER_RE.search(trace)
    if not m:
        return None
    start = m.end()
    rest = trace[start:]
    nxt = NEXT_HEADER_RE.search(rest)
    end = nxt.start() if nxt else len(rest)
    return rest[:end].strip() or None


def extract_direction(text: str) -> Tuple[Optional[str], List[str]]:
    """
    Look for direction-of-effect signal in the given text.

    Returns (verdict, evidence) where verdict is:
      "increase" — only increase-words found
      "decrease" — only decrease-words found
      "mixed"    — both found
      None       — neither found
    and evidence is a list of matched strings (lowercased, deduped).
    """
    inc = [m.group(0).lower() for m in INCREASE_RE.finditer(text)]
    dec = [m.group(0).lower() for m in DECREASE_RE.finditer(text)]
    inc_uniq = sorted(set(inc))
    dec_uniq = sorted(set(dec))
    has_inc, has_dec = bool(inc_uniq), bool(dec_uniq)

    if has_inc and has_dec:
        return "mixed", inc_uniq + dec_uniq
    if has_inc:
        return "increase", inc_uniq
    if has_dec:
        return "decrease", dec_uniq
    return None, []


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_trace(trace: str, gold_polarity: str) -> Dict:
    """
    Score a single trace against the gold polarity.

    Returns a dict with:
      verdict      ∈ {correct, incorrect, ambiguous, missing, no_summary}
      gold         the gold polarity passed in
      extracted    direction extracted from trace (or None / "mixed")
      source       "summary" | "fallback_fulltext" | "none"
      evidence     list of matched phrases
    """
    out = {"gold": gold_polarity, "extracted": None, "source": "none", "evidence": []}

    # Gold polarity n/a → scorer cannot adjudicate (e.g., infection risk labels)
    if gold_polarity not in ("increase", "decrease"):
        out["verdict"] = "ambiguous"
        out["reason"] = "gold_polarity_not_directional"
        return out

    summary = extract_summary(trace)
    if summary is None:
        # Try fallback: scan the whole trace
        ext, ev = extract_direction(trace)
        out["source"] = "fallback_fulltext" if ext else "none"
        if ext is None:
            out["verdict"] = "no_summary"
            return out
        out["extracted"], out["evidence"] = ext, ev
    else:
        ext, ev = extract_direction(summary)
        if ext is None:
            # Summary present but had no direction signal — fall back to full text
            ext_full, ev_full = extract_direction(trace)
            out["source"] = "fallback_fulltext" if ext_full else "summary"
            out["extracted"], out["evidence"] = ext_full, ev_full
        else:
            out["source"] = "summary"
            out["extracted"], out["evidence"] = ext, ev

    extracted = out["extracted"]
    if extracted is None:
        out["verdict"] = "missing"
    elif extracted == "mixed":
        out["verdict"] = "ambiguous"
    elif extracted == gold_polarity:
        out["verdict"] = "correct"
    else:
        out["verdict"] = "incorrect"
    return out


def score_traces(
    traces: List[Dict],
    hierarchy_map: Dict,
    trace_field: str = "trace",
    label_field: str = "label",
) -> List[Dict]:
    """Score a list of trace records. Each record must have trace_field and label_field."""
    scored = []
    for rec in traces:
        gold = hierarchy_map.get(str(rec[label_field]), {}).get("polarity", "unspecified")
        result = score_trace(rec[trace_field], gold)
        scored.append({**rec, "direction_score": result})
    return scored


def summarize_scores(scored: List[Dict]) -> Dict:
    """Aggregate verdicts from a scored batch into a small report."""
    verdicts = Counter(r["direction_score"]["verdict"] for r in scored)
    sources = Counter(r["direction_score"]["source"] for r in scored)
    n = len(scored)
    decisive = verdicts["correct"] + verdicts["incorrect"]
    accuracy = verdicts["correct"] / decisive if decisive else 0.0
    return {
        "n": n,
        "verdict_counts": dict(verdicts),
        "verdict_pct": {k: round(100 * v / max(1, n), 1) for k, v in verdicts.items()},
        "source_counts": dict(sources),
        "directional_accuracy": round(accuracy, 4),
        "directional_accuracy_note": "correct / (correct + incorrect)",
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, nargs="?",
                    help="JSONL with trace records (each must have 'trace' and 'label'). "
                         "Not required when --smoke is passed.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSONL path. Defaults to input.scored.jsonl")
    ap.add_argument("--hierarchy", type=Path,
                    default=Path("processed_v2/hierarchy_map.json"))
    ap.add_argument("--trace-field", default="trace")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--smoke", action="store_true",
                    help="Run an internal self-test instead of reading input.")
    args = ap.parse_args()

    if args.smoke:
        _self_test()
        return

    if args.input is None:
        ap.error("input is required unless --smoke is given")

    log.info(f"Loading hierarchy from {args.hierarchy}")
    with open(args.hierarchy) as f:
        hierarchy = json.load(f)

    log.info(f"Reading traces from {args.input}")
    traces = []
    with open(args.input) as f:
        for line in f:
            traces.append(json.loads(line))
    log.info(f"  {len(traces):,} traces loaded")

    scored = score_traces(traces, hierarchy,
                          trace_field=args.trace_field,
                          label_field=args.label_field)

    out_path = args.out or args.input.with_suffix(".scored.jsonl")
    with open(out_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    log.info(f"Wrote {out_path}")

    summary = summarize_scores(scored)
    log.info("Summary:")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")


# ── Self-test ────────────────────────────────────────────────────────────────

def _self_test():
    """Quick assertions on synthetic traces. Run via --smoke."""
    cases = [
        # (description, trace, gold, expected_verdict)
        ("clear increase, summary present",
         "## Reasoning\nStuff happens.\n## Summary\nDrug A's exposure increases substantially.\n## Classification\nY=1 -- ...",
         "increase", "correct"),
        ("clear decrease, summary present",
         "## Reasoning\nStuff.\n## Summary\nThe therapeutic effect of B is reduced and clearance is faster.\n",
         "decrease", "correct"),
        ("wrong direction in summary",
         "## Summary\nDrug levels decrease markedly.\n",
         "increase", "incorrect"),
        ("mixed in summary",
         "## Summary\nMetabolism increases, leading to lower drug exposure.\n",
         "decrease", "ambiguous"),
        ("no summary section, fallback finds clean signal",
         "Drug A serum levels are higher and exposure increases.",
         "increase", "correct"),
        ("no summary, fallback finds mixed signal (PK reversal phrasing)",
         "Drug A clearance is reduced and serum levels are higher.",
         "increase", "ambiguous"),
        ("no direction word anywhere",
         "## Summary\nThe interaction has been characterized in clinical trials.\n",
         "increase", "missing"),
        ("gold n/a",
         "## Summary\nIncreased risk of opportunistic infection.\n",
         "n/a", "ambiguous"),
        ("summary header tolerant to whitespace",
         "##  SUMMARY  \nDrug levels rise.\n##Classification\nY=1",
         "increase", "correct"),
    ]
    fail = 0
    for desc, trace, gold, expected in cases:
        result = score_trace(trace, gold)
        ok = result["verdict"] == expected
        flag = "✓" if ok else "✗"
        print(f"  {flag}  {desc}: got {result['verdict']!r} expected {expected!r}")
        if not ok:
            fail += 1
            print(f"      detail: {result}")
    print(f"\n{len(cases) - fail}/{len(cases)} self-test cases passed.")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
