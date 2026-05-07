"""
src/step12_pilot_2x2.py

Step 12 — 2x2 pilot harness for the DDI CoT distillation paper.

Compares two retrieval methods (Tanimoto vs pathway) crossed with two prompt
configurations (no hierarchy hints vs hierarchy hints), giving four conditions:

    A: Tanimoto + no_hints
    B: Pathway  + no_hints
    C: Tanimoto + hints
    D: Pathway  + hints

For each sampled query pair we generate a teacher trace under all four conditions,
then score with three tiers: exact label match, l2 cluster match, and direction
accuracy (on the adjudicable subset).

Three modes (run separately so SLURM failures don't waste prompt-build work):

    --prepare    Stratified sample of 500 test pairs, build all 4 prompts each.
                 CPU-only; runs on a login node. Output: pilot_prompts.jsonl
                 plus a coverage audit report. Fully deterministic given seed.

    --generate   Load pilot_prompts.jsonl, batch-generate Qwen3-4B traces with
                 vLLM. GPU job. Resume-safe: skips rows already in the output.
                 Output: pilot_traces.jsonl

    --score      Load pilot_traces.jsonl, score each trace (direction via
                 step11d, exact label via Y= regex, cluster via hierarchy).
                 CPU-only. Output: pilot_scored.jsonl + summary tables.

Each mode writes a manifest JSON next to its output with the run metadata
(git commit, model revision, library versions, parameters, file paths) so
that any pilot_*.jsonl can be traced back to the exact harness state that
produced it.

Usage:
    python scripts/step12_pilot_2x2.py --prepare
    sbatch run_step12_generate.sh    # --generate inside the SLURM script
    python scripts/step12_pilot_2x2.py --score
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make scripts/ importable so we can use teacher_prompt_v3 and step11d.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from teacher_prompt_v3 import build_teacher_prompt, TEACHER_SYSTEM_PROMPT
from step11d_direction_scorer import score_trace

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_DATA = Path("/scratch/vlelo/ddiproject/processed_v2")
DEFAULT_OUT  = Path("/scratch/vlelo/ddiproject/pilot")
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_BUDGET = 500
DEFAULT_FLOOR = 3
DEFAULT_HEAD_CAP = 15
DEFAULT_SEED = 42

# Greedy / high-consistency decoding (NOT bit-exact deterministic — hardware
# and kernel non-determinism can still produce small variation).
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS  = 1024


# ── Manifest ─────────────────────────────────────────────────────────────────

def _git_commit(script_path: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(script_path.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_manifest(mode: str, out_dir: Path, params: dict, files: dict) -> Path:
    versions: Dict[str, str] = {}
    for pkg in ("torch", "vllm", "transformers"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    manifest = {
        "mode": mode,
        "script": str(THIS_DIR / "step12_pilot_2x2.py"),
        "git_commit": _git_commit(THIS_DIR),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": versions,
        "params": params,
        "files": {k: str(v) for k, v in files.items()},
    }
    path = out_dir / f"manifest_{mode}_{int(time.time())}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest written: {path}")
    return path


# ── PREPARE: stratified sampling and prompt building ────────────────────────

def stratified_allocation(
    pool_by_cluster: Dict[str, list],
    budget: int = DEFAULT_BUDGET,
    floor: int = DEFAULT_FLOOR,
    head_cap: int = DEFAULT_HEAD_CAP,
) -> Dict[str, int]:
    """
    Strategy B: floor=3 per cluster, then distribute remainder proportionally
    to cluster pool size, with each cluster capped at head_cap. Tail clusters
    with fewer than floor available take what they have.

    Allocation is iterative: at each step we add 1 unit to the cluster with
    the largest 'deficit' (proportional target minus current allocation),
    among clusters that still have room (under head_cap and pool not exhausted).

    Stops when budget exhausted OR all clusters are saturated. In the latter
    case the pilot will be slightly below `budget` and we log a warning.
    """
    pool_size = {c: len(p) for c, p in pool_by_cluster.items()}
    total_pool = sum(pool_size.values())
    if total_pool == 0:
        raise ValueError("Empty overlap pool — no clusters have eligible pairs.")

    alloc = {c: min(floor, pool_size[c]) for c in pool_by_cluster}
    if sum(alloc.values()) > budget:
        raise ValueError(
            f"Floor allocation ({sum(alloc.values())}) already exceeds "
            f"budget ({budget}). Reduce floor or raise budget."
        )

    def cap_for(c: str) -> int:
        return min(head_cap, pool_size[c])

    remaining = budget - sum(alloc.values())
    while remaining > 0:
        eligible = [c for c in pool_by_cluster if alloc[c] < cap_for(c)]
        if not eligible:
            log.warning(
                f"All clusters saturated; allocation = {sum(alloc.values())}/"
                f"{budget}. Increase head_cap or expand overlap pool."
            )
            break
        # Pick cluster with largest unmet share relative to proportional target.
        def deficit(c: str) -> float:
            target = budget * pool_size[c] / total_pool
            return target - alloc[c]
        eligible.sort(key=deficit, reverse=True)
        alloc[eligible[0]] += 1
        remaining -= 1

    return alloc


def coverage_audit(
    test_rows: List[dict],
    tan_cache: list,
    pwy_cache: list,
) -> Dict[str, int]:
    """Coverage on the FULL test pool (not just the overlap subset)."""
    n = len(test_rows)
    pwy_only = tan_only = both = neither = 0
    for tan, pwy in zip(tan_cache, pwy_cache):
        t_ok, p_ok = bool(tan), bool(pwy)
        if t_ok and p_ok: both += 1
        elif t_ok and not p_ok: tan_only += 1
        elif p_ok and not t_ok: pwy_only += 1
        else: neither += 1
    return {
        "n_test_pairs": n,
        "both_methods_nonempty": both,
        "tanimoto_only": tan_only,
        "pathway_only": pwy_only,
        "neither": neither,
        "tanimoto_total_coverage": both + tan_only,
        "pathway_total_coverage":  both + pwy_only,
    }


def cmd_prepare(args: argparse.Namespace) -> None:
    data = args.data_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading test set, retrieval caches, hierarchy...")
    test_rows = [json.loads(l) for l in open(data / "test.jsonl")]
    log.info(f"  test rows: {len(test_rows):,}")

    with open(data / "retrieved_examples_test.json") as f:
        tan_cache = json.load(f)
    log.info(f"  Tanimoto cache: {len(tan_cache):,}")

    pwy_path = data / args.pathway_cache
    with open(pwy_path) as f:
        pwy_cache = json.load(f)
    log.info(f"  pathway cache:  {len(pwy_cache):,}  ({pwy_path.name})")

    if not (len(test_rows) == len(tan_cache) == len(pwy_cache)):
        # Pilot caches may be shorter than full test set — that's fine, we'll
        # operate on the prefix that's available across all three.
        n = min(len(test_rows), len(tan_cache), len(pwy_cache))
        log.warning(
            f"Length mismatch — using prefix of {n:,} "
            f"(test={len(test_rows)}, tan={len(tan_cache)}, pwy={len(pwy_cache)})"
        )
        test_rows = test_rows[:n]
        tan_cache = tan_cache[:n]
        pwy_cache = pwy_cache[:n]

    with open(data / "drug_profiles.json") as f:
        profiles = json.load(f)
    with open(data / "hierarchy_map.json") as f:
        hierarchy = json.load(f)

    # ── Coverage audit on the full pool ──
    audit = coverage_audit(test_rows, tan_cache, pwy_cache)
    log.info("Coverage audit (full test pool):")
    for k, v in audit.items():
        log.info(f"  {k}: {v:,}")

    # ── Build overlap-eligible pool, bucket by gold l2 cluster ──
    pool_by_cluster: Dict[str, list] = defaultdict(list)
    for i, (row, tan, pwy) in enumerate(zip(test_rows, tan_cache, pwy_cache)):
        if not tan or not pwy:
            continue
        l2 = hierarchy.get(str(row["label"]), {}).get("l2")
        if l2 is None:
            continue
        pool_by_cluster[l2].append({
            "row_idx": i,
            "row": row,
            "tan": tan,
            "pwy": pwy,
            "l2": l2,
        })

    n_overlap = sum(len(v) for v in pool_by_cluster.values())
    log.info(f"\nOverlap-eligible pool (both retrievals nonempty + has l2 cluster): "
             f"{n_overlap:,} pairs across {len(pool_by_cluster):,} clusters")

    # ── Allocate ──
    alloc = stratified_allocation(
        pool_by_cluster, budget=args.budget,
        floor=args.floor, head_cap=args.head_cap,
    )
    total_alloc = sum(alloc.values())
    log.info(f"Allocation total: {total_alloc} (target {args.budget})")
    sat = [c for c in alloc if alloc[c] == min(args.head_cap, len(pool_by_cluster[c]))]
    log.info(f"  saturated clusters (at head_cap or pool empty): {len(sat)}")

    # ── Sample within each cluster ──
    rng = random.Random(args.seed)
    sampled = []
    for cluster, pairs in pool_by_cluster.items():
        k = alloc[cluster]
        if k == 0:
            continue
        # Sort by row_idx so the universe is deterministic, then sample.
        sorted_pairs = sorted(pairs, key=lambda p: p["row_idx"])
        picked = rng.sample(sorted_pairs, k=min(k, len(sorted_pairs)))
        sampled.extend(picked)
    sampled.sort(key=lambda p: (p["l2"], p["row_idx"]))
    log.info(f"Sampled {len(sampled)} pairs.")

    # ── Build 4 prompts per sampled pair ──
    conditions = [
        # condition_id, retrieval, hints, build_kwargs
        ("A_tan_nohints",  "tanimoto", False, dict(use_hierarchy_hints=False, use_cluster_count_hint=False)),
        ("B_pwy_nohints",  "pathway",  False, dict(use_hierarchy_hints=False, use_cluster_count_hint=False)),
        ("C_tan_hints",    "tanimoto", True,  dict(use_hierarchy_hints=True,  use_cluster_count_hint=True)),
        ("D_pwy_hints",    "pathway",  True,  dict(use_hierarchy_hints=True,  use_cluster_count_hint=True)),
    ]

    out_path = out / "pilot_prompts.jsonl"
    n_written = 0
    with open(out_path, "w") as f:
        for sp in sampled:
            row = sp["row"]
            for cond_id, retrieval, hints, kw in conditions:
                neighbors = sp["tan"] if retrieval == "tanimoto" else sp["pwy"]
                prompt_text = build_teacher_prompt(
                    row, profiles, retrieved_examples=neighbors,
                    processed_dir=str(data),
                    use_prodrug_warning=True,   # constant across conditions
                    use_no_pathway_note=True,   # constant across conditions
                    **kw,
                )
                gold = hierarchy.get(str(row["label"]), {})
                rec = {
                    "pair_uid": f"{row['drug_a']}__{row['drug_b']}__{row['label']}",
                    "condition": cond_id,
                    "retrieval": retrieval,
                    "hints": hints,
                    "row_idx": sp["row_idx"],
                    "drug_a": row["drug_a"],
                    "drug_b": row["drug_b"],
                    "label": int(row["label"]),
                    "label_text": row.get("description", ""),
                    "gold_l2": gold.get("l2"),
                    "gold_polarity": gold.get("polarity", "unspecified"),
                    "gold_role": gold.get("affected_drug_role"),
                    "n_retrieved": len(neighbors) if neighbors else 0,
                    "prompt_text": prompt_text,
                    "prompt_chars": len(prompt_text),
                }
                f.write(json.dumps(rec) + "\n")
                n_written += 1
    log.info(f"Wrote {n_written:,} prompts -> {out_path}")

    # ── Audit + allocation report ──
    audit_path = out / "pilot_prepare_audit.json"
    with open(audit_path, "w") as f:
        json.dump({
            "coverage_full_test_pool": audit,
            "overlap_pool_size": n_overlap,
            "n_clusters_in_overlap_pool": len(pool_by_cluster),
            "allocation_per_cluster": dict(sorted(alloc.items())),
            "n_sampled_pairs": len(sampled),
            "n_prompts": n_written,
            "seed": args.seed,
            "budget": args.budget,
            "floor": args.floor,
            "head_cap": args.head_cap,
            "saturated_clusters": sat,
        }, f, indent=2)
    log.info(f"Audit/allocation report: {audit_path}")

    write_manifest("prepare", out, params={
        "budget": args.budget, "floor": args.floor, "head_cap": args.head_cap,
        "seed": args.seed, "pathway_cache": str(pwy_path.name),
    }, files={
        "prompts": out_path, "audit": audit_path,
    })


# ── GENERATE: vLLM batch inference ──────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    prompts_path = out / "pilot_prompts.jsonl"
    traces_path  = out / "pilot_traces.jsonl"

    if not prompts_path.exists():
        raise SystemExit(f"Missing {prompts_path}; run --prepare first.")

    # Resume safety: skip pair_uid+condition combinations already in output.
    done = set()
    if traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                rec = json.loads(line)
                done.add((rec["pair_uid"], rec["condition"]))
        log.info(f"Resuming: {len(done):,} traces already present, will skip.")

    pending = []
    with open(prompts_path) as f:
        for line in f:
            rec = json.loads(line)
            if (rec["pair_uid"], rec["condition"]) in done:
                continue
            pending.append(rec)
    log.info(f"Pending: {len(pending):,} prompts to generate.")
    if not pending:
        log.info("Nothing to do.")
        return

    log.info(f"Loading vLLM with model={args.model} (bf16)"
             + (f", max_model_len={args.max_model_len}" if args.max_model_len else "")
             + ")")
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        # gpu_memory_utilization left at vLLM default (0.9). On A100 40GB
        # for a 9B model in bf16 (~18 GB) that leaves ~18 GB for KV cache.
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # Build chat-format messages list. vLLM's llm.chat handles the chat template.
    messages_list = [
        [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user",   "content": rec["prompt_text"]},
        ]
        for rec in pending
    ]

    chat_kwargs = {}
    if args.no_thinking:
        # Qwen3/3.5 chat template reads this and skips emitting <think>...</think>.
        # Requires vLLM >= 0.6 (chat_template_kwargs support); vLLM >= 0.9 strongly
        # recommended for full Qwen3.5 architecture support.
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        log.info("Thinking mode DISABLED via chat_template_kwargs.")

    log.info(f"Generating {len(messages_list):,} traces "
             f"(greedy/high-consistency, T={args.temperature}, "
             f"max_tokens={args.max_tokens}, seed={args.seed})")
    t0 = time.time()
    outputs = llm.chat(messages_list, sampling_params=sampling, **chat_kwargs)
    elapsed = time.time() - t0
    log.info(f"Generation finished: {elapsed:.1f}s for {len(outputs)} traces "
             f"({len(outputs)/elapsed:.2f}/s)")

    # Write traces
    with open(traces_path, "a") as f:
        for rec, out_obj in zip(pending, outputs):
            gen = out_obj.outputs[0]
            trace = gen.text
            n_in  = len(out_obj.prompt_token_ids) if out_obj.prompt_token_ids is not None else None
            n_out = len(gen.token_ids) if gen.token_ids is not None else None
            scrubbed = {k: v for k, v in rec.items() if k != "prompt_text"}
            scrubbed.update({
                "trace": trace,
                "prompt_tokens": n_in,
                "generated_tokens": n_out,
                "finish_reason": gen.finish_reason,
            })
            f.write(json.dumps(scrubbed) + "\n")
    log.info(f"Wrote traces -> {traces_path}")

    write_manifest("generate", out, params={
        "model": args.model, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "seed": args.seed,
        "max_model_len": args.max_model_len,
        "thinking_disabled": args.no_thinking,
        "n_generated_now": len(pending), "elapsed_seconds": elapsed,
    }, files={"prompts": prompts_path, "traces": traces_path})


# ── SCORE: parse traces, score each, write summary tables ───────────────────

CLASSIFICATION_HEADER_RE = re.compile(
    r"^\s*##\s*classification\b", re.IGNORECASE | re.MULTILINE,
)
LABEL_RE = re.compile(r"\bY\s*=\s*(\d+)")


def parse_predicted_label(trace: str) -> Optional[int]:
    """Extract Y=N from the ## Classification section, or anywhere as fallback."""
    m = CLASSIFICATION_HEADER_RE.search(trace)
    section = trace[m.end():] if m else trace
    lm = LABEL_RE.search(section)
    if lm:
        return int(lm.group(1))
    # Fallback: scan whole trace
    lm = LABEL_RE.search(trace)
    return int(lm.group(1)) if lm else None


def cmd_score(args: argparse.Namespace) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    traces_path = out / "pilot_traces.jsonl"
    scored_path = out / "pilot_scored.jsonl"
    summary_path = out / "pilot_summary.json"

    if not traces_path.exists():
        raise SystemExit(f"Missing {traces_path}; run --generate first.")

    with open(args.data_dir / "hierarchy_map.json") as f:
        hierarchy = json.load(f)

    by_cond: Dict[str, list] = defaultdict(list)
    n = 0
    with open(traces_path) as fin, open(scored_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            n += 1
            trace = rec.get("trace", "")
            gold_label = int(rec["label"])
            gold_l2 = rec.get("gold_l2")
            gold_pol = rec.get("gold_polarity", "unspecified")

            pred_label = parse_predicted_label(trace)
            pred_l2 = (
                hierarchy.get(str(pred_label), {}).get("l2")
                if pred_label is not None else None
            )
            dir_score = score_trace(trace, gold_pol)

            scored = {**rec,
                "pred_label": pred_label,
                "pred_l2": pred_l2,
                "exact_match": (pred_label == gold_label),
                "cluster_match": (pred_l2 == gold_l2 and gold_l2 is not None),
                "direction_verdict": dir_score["verdict"],
                "direction_extracted": dir_score.get("extracted"),
                "direction_evidence": dir_score.get("evidence", []),
                "direction_source": dir_score.get("source"),
            }
            fout.write(json.dumps(scored) + "\n")
            by_cond[rec["condition"]].append(scored)

    log.info(f"Scored {n:,} traces -> {scored_path}")

    # ── Summary tables ──
    summary = {
        "total_traces": n,
        "conditions": {},
    }
    log.info("\n=== 2x2 pilot results ===\n")
    header = f"{'condition':<18} {'n':>5} {'exact':>8} {'cluster':>8} {'dir(adj)':>10} {'dir(adj/n)':>10}"
    log.info(header)
    log.info("-" * len(header))
    for cond, recs in sorted(by_cond.items()):
        nc = len(recs)
        exact = sum(r["exact_match"] for r in recs) / nc
        cluster = sum(r["cluster_match"] for r in recs) / nc
        verdicts = Counter(r["direction_verdict"] for r in recs)
        decisive = verdicts["correct"] + verdicts["incorrect"]
        dir_acc_adj = (verdicts["correct"] / decisive) if decisive else 0.0
        dir_acc_all = verdicts["correct"] / nc
        log.info(f"{cond:<18} {nc:>5} {exact:>8.3f} {cluster:>8.3f} "
                 f"{dir_acc_adj:>10.3f} {dir_acc_all:>10.3f}")
        summary["conditions"][cond] = {
            "n": nc,
            "exact_match": round(exact, 4),
            "cluster_match": round(cluster, 4),
            "direction_accuracy_adjudicable": round(dir_acc_adj, 4),
            "direction_accuracy_all": round(dir_acc_all, 4),
            "verdict_counts": dict(verdicts),
            "verdict_pct": {k: round(100 * v / nc, 1) for k, v in verdicts.items()},
            "n_unparseable_label": sum(1 for r in recs if r["pred_label"] is None),
            "median_prompt_tokens": _median([r.get("prompt_tokens") or 0 for r in recs]),
            "median_generated_tokens": _median([r.get("generated_tokens") or 0 for r in recs]),
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSummary -> {summary_path}")

    write_manifest("score", out, params={}, files={
        "traces": traces_path, "scored": scored_path, "summary": summary_path,
    })


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    n = len(xs)
    return float(xs[n//2]) if n % 2 else float((xs[n//2-1] + xs[n//2]) / 2)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir",  type=Path, default=DEFAULT_OUT)
    sub = ap.add_subparsers(dest="mode", required=True)

    pp = sub.add_parser("prepare", help="Sample + build prompts")
    pp.add_argument("--pathway-cache", default="retrieved_examples_pathway_test_pilot500_v2.json")
    pp.add_argument("--budget",   type=int, default=DEFAULT_BUDGET)
    pp.add_argument("--floor",    type=int, default=DEFAULT_FLOOR)
    pp.add_argument("--head-cap", type=int, default=DEFAULT_HEAD_CAP)
    pp.add_argument("--seed",     type=int, default=DEFAULT_SEED)

    gp = sub.add_parser("generate", help="vLLM batch inference")
    gp.add_argument("--model",       default=DEFAULT_MODEL)
    gp.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    gp.add_argument("--max-tokens",  type=int,   default=DEFAULT_MAX_TOKENS)
    gp.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    gp.add_argument("--max-model-len", type=int, default=None,
                    help="Cap context length passed to vLLM. Reduces KV-cache "
                         "memory dramatically when the model's native context "
                         "(e.g. 262K for Qwen3.5) is far larger than needed. "
                         "Our prompts are ~3-4K tokens; 8192 is comfortable.")
    gp.add_argument("--no-thinking", action="store_true",
                    help="Pass enable_thinking=False through the chat template. "
                         "Required for Qwen3.5 family unless you want <think>...</think> "
                         "blocks before the structured output.")

    sp = sub.add_parser("score", help="Score traces (direction + label + cluster)")

    args = ap.parse_args()
    if args.mode == "prepare":
        cmd_prepare(args)
    elif args.mode == "generate":
        cmd_generate(args)
    elif args.mode == "score":
        cmd_score(args)


if __name__ == "__main__":
    main()
