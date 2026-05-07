"""
src/step12_visualize.py

Produce paper-ready figures from pilot_scored.jsonl.

Outputs PNG files to <out_dir>/figures/:
  fig_metrics_2x2.png       - main 2x2 metrics bar chart (exact / cluster / dir-adj / dir-overall)
  fig_verdicts_stacked.png  - stacked verdict distribution per condition
  fig_alignment_focus.png   - outcome-polarity alignment focus chart

Usage:
    python scripts/step12_visualize.py \
        --scored /scratch/vlelo/ddiproject/pilot/pilot_scored.jsonl \
        --out    /scratch/vlelo/ddiproject/pilot/figures
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np


CONDITION_ORDER = ["A_tan_nohints", "B_pwy_nohints", "C_tan_hints", "D_pwy_hints"]
CONDITION_LABELS = {
    "A_tan_nohints": "A · Tanimoto, no hints",
    "B_pwy_nohints": "B · Pathway, no hints",
    "C_tan_hints":   "C · Tanimoto, hints",
    "D_pwy_hints":   "D · Pathway, hints",
}
# Color palette: muted, two pairs to show retrieval vs hints structure
CONDITION_COLORS = {
    "A_tan_nohints": "#9bb7d4",   # light blue
    "B_pwy_nohints": "#d4a59b",   # light terracotta
    "C_tan_hints":   "#3e6594",   # dark blue
    "D_pwy_hints":   "#94403e",   # dark terracotta
}
VERDICT_ORDER = ["correct", "incorrect", "ambiguous", "missing", "no_summary"]
VERDICT_COLORS = {
    "correct":    "#5b9c6a",
    "incorrect":  "#c4544a",
    "ambiguous":  "#e0b15c",
    "missing":    "#9c9c9c",
    "no_summary": "#5c5c5c",
}


def load_rows(path: Path) -> list:
    return [json.loads(l) for l in open(path)]


def metrics_per_condition(rows: list) -> dict:
    out = {}
    for cond in CONDITION_ORDER:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        n = len(sub)
        exact = sum(r["exact_match"] for r in sub) / n
        cluster = sum(r["cluster_match"] for r in sub) / n
        v = Counter(r["direction_verdict"] for r in sub)
        decisive = v["correct"] + v["incorrect"]
        dir_adj = (v["correct"] / decisive) if decisive else 0.0
        dir_all = v["correct"] / n
        out[cond] = {
            "n": n,
            "exact": exact, "cluster": cluster,
            "dir_adj": dir_adj, "dir_all": dir_all,
            "verdicts": v,
        }
    return out


def fig_metrics_2x2(metrics: dict, out_path: Path) -> None:
    """Four subplots, each with bars for the four conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    panels = [
        ("exact",   "Exact label match",                "Accuracy"),
        ("cluster", "Cluster (l2) match",               "Accuracy"),
        ("dir_adj", "Outcome-polarity alignment\n(adjudicable)", "Alignment rate"),
        ("dir_all", "Outcome-polarity alignment\n(over all traces)",       "Rate"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        conds   = [c for c in CONDITION_ORDER if c in metrics]
        values  = [metrics[c][key] for c in conds]
        colors  = [CONDITION_COLORS[c] for c in conds]
        labels  = [CONDITION_LABELS[c] for c in conds]
        bars = ax.bar(range(len(conds)), values, color=colors,
                      edgecolor="white", linewidth=1.2)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([l.replace(" · ", "\n") for l in labels],
                           fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle("2×2 Pilot: retrieval method × hierarchy hints (n=254 stratified)",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_verdicts_stacked(metrics: dict, out_path: Path) -> None:
    """Stacked horizontal bars showing verdict distribution per condition."""
    conds = [c for c in CONDITION_ORDER if c in metrics]
    n_cond = len(conds)
    bar_height = 0.55

    fig, ax = plt.subplots(figsize=(10, 1 + 0.9 * n_cond), constrained_layout=True)

    for i, cond in enumerate(conds):
        v = metrics[cond]["verdicts"]
        n = metrics[cond]["n"]
        left = 0
        for verdict in VERDICT_ORDER:
            count = v.get(verdict, 0)
            if count == 0:
                continue
            frac = count / n
            ax.barh(i, frac, left=left, height=bar_height,
                    color=VERDICT_COLORS[verdict],
                    edgecolor="white", linewidth=1)
            if frac >= 0.04:
                ax.text(left + frac / 2, i,
                        f"{verdict}\n{count}",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if verdict in ("incorrect", "no_summary") else "black")
            left += frac

    ax.set_yticks(range(n_cond))
    ax.set_yticklabels([CONDITION_LABELS[c] for c in conds])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Fraction of traces")
    ax.set_title("Direction-verdict distribution by condition",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [plt.Rectangle((0,0), 1, 1, color=VERDICT_COLORS[v])
                      for v in VERDICT_ORDER]
    ax.legend(legend_handles, VERDICT_ORDER, loc="lower right",
              ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(1, -0.18))

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_alignment_focus(metrics: dict, out_path: Path) -> None:
    """Focused chart on the 'hints crush direction errors' finding."""
    conds = [c for c in CONDITION_ORDER if c in metrics]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)

    incorrect_pct = []
    for c in conds:
        v = metrics[c]["verdicts"]
        n = metrics[c]["n"]
        incorrect_pct.append(100 * v.get("incorrect", 0) / n)

    colors = [CONDITION_COLORS[c] for c in conds]
    bars = ax.bar(range(len(conds)), incorrect_pct, color=colors,
                  edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, incorrect_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([CONDITION_LABELS[c].replace(" · ", "\n") for c in conds],
                       fontsize=10)
    ax.set_ylabel("Wrong-direction traces (% of total)")
    ax.set_title("Hierarchy hints reduce wrong-direction commitments\n"
                 "(scorer-flagged; manual audit indicates most are surface-phrasing artifacts)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(incorrect_pct) * 1.25 + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path,
                    default=Path("/scratch/vlelo/ddiproject/pilot/pilot_scored.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("/scratch/vlelo/ddiproject/pilot/figures"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.scored)
    metrics = metrics_per_condition(rows)

    print(f"Loaded {len(rows):,} scored rows.")
    print(f"Conditions: {list(metrics.keys())}")
    print(f"Output dir: {args.out}\n")

    fig_metrics_2x2(metrics,    args.out / "fig_metrics_2x2.png")
    fig_verdicts_stacked(metrics, args.out / "fig_verdicts_stacked.png")
    fig_alignment_focus(metrics,  args.out / "fig_alignment_focus.png")

    # Print summary table to stdout for the talk
    print("\nFinal numeric table:")
    print(f"  {'condition':<24} {'n':>4} {'exact':>7} {'cluster':>8} {'dir(adj)':>9} {'dir(all)':>9} {'wrong-dir%':>11}")
    for c in CONDITION_ORDER:
        if c not in metrics: continue
        m = metrics[c]
        wrong_pct = 100 * m["verdicts"].get("incorrect", 0) / m["n"]
        print(f"  {c:<24} {m['n']:>4} {m['exact']:>7.3f} {m['cluster']:>8.3f} "
              f"{m['dir_adj']:>9.3f} {m['dir_all']:>9.3f} {wrong_pct:>10.1f}%")


if __name__ == "__main__":
    main()
