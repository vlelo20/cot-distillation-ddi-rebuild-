"""
src/teacher_prompt_v3.py

Improved teacher prompt builder for the rebuilt DDI dataset.

Differences from v2:
  - Fixes _format_drug_profile (was joining dicts as strings -> TypeError)
  - Fixes _extract_pathway_nodes (was putting dicts in sets -> TypeError)
  - Surfaces enzyme/transporter/target ACTIONS (substrate/inhibitor/inducer/etc.)
    in drug profiles, since action is the DDI mechanism signal
  - Filters pathway entries to organism='Humans' (was implicit but not enforced)
  - Adds use_cluster_count_hint toggle: a more conservative alternative to
    use_cluster_siblings that mentions the cluster size and asks the teacher
    to focus on what distinguishes the gold template, without showing sibling
    text verbatim. Mutually exclusive with use_cluster_siblings (siblings wins
    if both are on)
  - Default config flipped to conservative: count_hint ON, siblings OFF.
    Sibling-text mode is now an ablation, not the main condition
  - System prompt explicitly names this as gold-label-derived supervision
    (privileged-information distillation, named honestly)
  - Output instructions include anti-parroting clauses for Reasoning and
    Summary sections (Classification line is mechanical and stays verbatim)
  - No-pathway note softened: 'no shared annotated nodes in DrugBank' rather
    than 'no PK pathway connecting them' (absence of annotation != absence
    of connection)
  - Smoke test prints length + head + tail per variant by default; pass
    --full to dump full prompts

Builds on the Rameen Jafri (RJ) prompt design. Adds:
  - polarity, affected_drug_role, l2 mechanism cluster, secondary_tags
  - cluster siblings (or cluster count hint) for within-family disambiguation

Keeps from RJ:
  - drug profiles with raised truncation caps (8/5/5)
  - prodrug warning for prodrug pairs
  - no-shared-pathway note when drugs have no overlapping nodes
  - structured output format (## Reasoning / ## Summary / ## Classification)

Drops from RJ:
  - severity sections (DrugBank 6.0 academic XML lacks severity)

Schema notes:
  - Our train.jsonl uses drug_a/drug_b column names (her code uses
    drug1_id/drug2_id). Both supported via _g() helper
  - hierarchy_map.json keys are stringified labels
  - drug_profiles.json entries for enzymes/transporters/targets/carriers are
    lists of dicts: {uniprot, name, actions: [...], organism}
"""
from pathlib import Path
import json
import functools


# ── System prompt ─────────────────────────────────────────────────────────────

TEACHER_SYSTEM_PROMPT = (
    "You are an expert pharmacologist specialising in drug-drug interactions. "
    "You will be given two drugs, their pharmacological profiles, and "
    "STRUCTURED METADATA DERIVED FROM THE GOLD INTERACTION LABEL: the gold "
    "label and its template text, the mechanism domain (PK/PD), the mechanism "
    "cluster, the direction of effect (polarity), and which drug is affected. "
    "Your task is to produce a faithful mechanistic rationale that is "
    "consistent with this gold-label-derived metadata. The rationale you "
    "generate will be used to train a smaller student model that, at "
    "inference time, sees only the drug pair and profiles — not these "
    "structured anchors. "
    "Your reasoning should commit to the specific direction of effect "
    "indicated (which drug's levels or activity changes, in which direction) "
    "and identify the dominant mechanism (enzyme, transporter, receptor, or "
    "pharmacodynamic). "
    "Structure your response exactly as:\n\n"
    "## Reasoning\n"
    "[Numbered steps explaining the mechanism. Identify each drug's relevant "
    "pharmacology, the point of interaction, and the resulting effect. Do "
    "not restate the label text verbatim — explain the pharmacology in your "
    "own words.]\n\n"
    "## Summary\n"
    "[2-3 sentence summary that explicitly states the direction of effect and "
    "which drug experiences the change. Again, do not restate the label "
    "text verbatim.]\n\n"
    "## Classification\n"
    "Y={label} -- \"{label_text}\""
)


# ── Hierarchy / label-map / prodrug loading (cached) ──────────────────────────

@functools.lru_cache(maxsize=1)
def _load_hierarchy(processed_dir: str) -> dict:
    path = Path(processed_dir) / "hierarchy_map.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _load_clusters(processed_dir: str) -> dict:
    path = Path(processed_dir) / "hierarchy_clusters.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _load_label_map(processed_dir: str) -> dict:
    path = Path(processed_dir) / "label_map.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


@functools.lru_cache(maxsize=1)
def _load_prodrug_ids(processed_dir: str) -> set:
    path = Path(processed_dir) / "prodrug_ids.json"
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    return set(data.keys() if isinstance(data, dict) else data)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _g(row, *keys, default=""):
    """Get the first present key from a row dict (handles schema differences)."""
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _polarity_human(polarity: str) -> str:
    return {
        "increase": "INCREASE (the effect goes up — more drug, stronger activity, higher risk)",
        "decrease": "DECREASE (the effect goes down — less drug, weaker activity, lower risk)",
        "n/a":      "no inherent direction (e.g., infection risk, diagnostic effect)",
    }.get(polarity, "unspecified")


def _role_human(role: str, name1: str, name2: str) -> str:
    return {
        "drug1": f"Drug 1 ({name1}) — its pharmacology is what changes",
        "drug2": f"Drug 2 ({name2}) — its pharmacology is what changes",
        "both":  "Both drugs share the effect (e.g., combined risk of an event)",
    }.get(role, "unspecified")


def _format_pathway_entries(entries, cap: int, humans_only: bool = True) -> list:
    """
    Format enzyme/transporter/target/carrier entries as human-readable strings.
    Each entry is a dict: {uniprot, name, actions: [...], organism}.
    Returns up to `cap` strings like 'Cytochrome P450 3A4 (substrate/inhibitor)'.
    """
    out = []
    for e in (entries or []):
        if humans_only and e.get("organism") != "Humans":
            continue
        name = e.get("name") or e.get("uniprot") or "?"
        actions = e.get("actions") or []
        if actions:
            out.append(f"{name} ({'/'.join(actions)})")
        else:
            out.append(name)
        if len(out) >= cap:
            break
    return out


def _format_drug_profile(profile: dict) -> str:
    """Format a drug profile compactly. Caps from RJ Fix 3 (8/5/5)."""
    if not profile:
        return "  (no detailed profile available)"

    lines = []
    if profile.get("description"):
        lines.append(f"  Description: {profile['description'][:300]}")
    if profile.get("mechanism_of_action"):
        lines.append(f"  Mechanism: {profile['mechanism_of_action'][:200]}")

    enzymes = _format_pathway_entries(profile.get("enzymes"), cap=8)
    if enzymes:
        lines.append(f"  Key enzymes: {'; '.join(enzymes)}")

    transporters = _format_pathway_entries(profile.get("transporters"), cap=5)
    if transporters:
        lines.append(f"  Transporters: {'; '.join(transporters)}")

    targets = _format_pathway_entries(profile.get("targets"), cap=5)
    if targets:
        lines.append(f"  Targets: {'; '.join(targets)}")

    if profile.get("smiles"):
        lines.append(f"  SMILES: {profile['smiles'][:200]}")

    return "\n".join(lines) if lines else "  (no detailed profile available)"


def _extract_pathway_nodes(profile: dict) -> dict:
    """
    Return sets of UniProt IDs per category (humans only).
    Used for shared-pathway detection — role doesn't matter for 'do they share
    a touchpoint', only identity does.
    """
    def _ids(entries):
        return {
            e["uniprot"] for e in (entries or [])
            if e.get("organism") == "Humans" and e.get("uniprot")
        }
    return {
        "enzymes":      _ids(profile.get("enzymes")),
        "transporters": _ids(profile.get("transporters")),
        "targets":      _ids(profile.get("targets")),
    }


# ── Main prompt builder ───────────────────────────────────────────────────────

def build_teacher_prompt(
    row: dict,
    profiles: dict,
    retrieved_examples: list = None,
    processed_dir: str = "processed_v2",
    use_hierarchy_hints: bool = True,
    use_cluster_siblings: bool = False,
    use_cluster_count_hint: bool = True,
    use_prodrug_warning: bool = True,
    use_no_pathway_note: bool = True,
) -> str:
    """
    Build the teacher prompt user message.

    Default configuration is the conservative main-paper setting:
      hierarchy_hints ON, cluster_count_hint ON, cluster_siblings OFF.
    Sibling-text mode (use_cluster_siblings=True) is available as a
    diagnostic ablation but is not the recommended main condition because
    showing verbatim sibling text invites paraphrase-leakage.

    Toggles:
      use_hierarchy_hints:    polarity, affected_drug_role, l2 cluster, secondary_tags
      use_cluster_siblings:   show other label texts in the same cluster (verbose; ablation only).
      use_cluster_count_hint: mention cluster size + ask teacher to differentiate,
                              without showing sibling text. Ignored if
                              use_cluster_siblings is True.
      use_prodrug_warning:    flag prodrugs (data/processed/prodrug_ids.json)
      use_no_pathway_note:    flag pairs with no shared pathway nodes
    """
    hierarchy   = _load_hierarchy(processed_dir)   if use_hierarchy_hints   else {}
    clusters    = _load_clusters(processed_dir)    if (use_cluster_siblings or use_cluster_count_hint) else {}
    label_map   = _load_label_map(processed_dir)
    prodrug_ids = _load_prodrug_ids(processed_dir) if use_prodrug_warning   else set()

    parts = []

    drug1_id   = _g(row, "drug1_id", "drug_a")
    drug2_id   = _g(row, "drug2_id", "drug_b")
    drug1_name = _g(row, "drug1_name", default=profiles.get(drug1_id, {}).get("name") or drug1_id)
    drug2_name = _g(row, "drug2_name", default=profiles.get(drug2_id, {}).get("name") or drug2_id)
    label      = int(row.get("label", 0))
    label_text = _g(row, "label_text", "description")

    # ── Retrieved examples ────────────────────────────────────────────────────
    if retrieved_examples:
        for i, ex in enumerate(retrieved_examples[:5], 1):
            ex_d1 = _g(ex, "drug1_id", "drug_a")
            ex_d2 = _g(ex, "drug2_id", "drug_b")
            ex_n1 = _g(ex, "drug1_name", default=profiles.get(ex_d1, {}).get("name") or ex_d1)
            ex_n2 = _g(ex, "drug2_name", default=profiles.get(ex_d2, {}).get("name") or ex_d2)
            ex_label = int(ex.get("label", 0))
            ex_text  = _g(ex, "label_text", "description")
            ex_sim   = ex.get("similarity")

            p1 = profiles.get(ex_d1, {})
            p2 = profiles.get(ex_d2, {})

            sim_str = f" (similarity: {ex_sim:.2f})" if ex_sim is not None else ""
            parts.append(f"--- Example {i}{sim_str} ---")
            parts.append(f"Drug 1: {ex_n1} ({ex_d1})")
            parts.append(_format_drug_profile(p1))
            parts.append(f"Drug 2: {ex_n2} ({ex_d2})")
            parts.append(_format_drug_profile(p2))
            parts.append(f"Interaction: Y={ex_label} -- \"{ex_text}\"")
            parts.append("")

    # ── Query pair ────────────────────────────────────────────────────────────
    parts.append("--- Your turn ---")
    p1 = profiles.get(drug1_id, {})
    p2 = profiles.get(drug2_id, {})

    parts.append(f"Drug 1: {drug1_name} ({drug1_id})")
    parts.append(_format_drug_profile(p1))
    parts.append(f"Drug 2: {drug2_name} ({drug2_id})")
    parts.append(_format_drug_profile(p2))
    parts.append("")
    parts.append(f"Known interaction: Y={label} -- \"{label_text}\"")

    # ── Hierarchy hints ───────────────────────────────────────────────────────
    if use_hierarchy_hints and str(label) in hierarchy:
        h = hierarchy[str(label)]
        cluster   = h.get("l2", "unknown")
        l1        = h.get("l1", "unknown")
        polarity  = h.get("polarity", "unspecified")
        role      = h.get("affected_drug_role", "unspecified")
        secondary = h.get("secondary_tags", [])

        domain_desc = (
            "pharmacokinetic — ADME effects on drug exposure"
            if l1 == "PK"
            else "pharmacodynamic — combined effects on physiology"
        )
        parts.append("")
        parts.append("Mechanism context:")
        parts.append(f"  Domain: {l1} ({domain_desc})")
        parts.append(f"  Mechanism cluster: {cluster}")
        parts.append(f"  Direction of effect: {_polarity_human(polarity)}")
        parts.append(f"  Affected drug: {_role_human(role, drug1_name, drug2_name)}")
        if secondary:
            parts.append(f"  Compound mechanism — also involves: {', '.join(secondary)}")

    # ── Cluster siblings OR count hint (mutually exclusive) ───────────────────
    if str(label) in hierarchy:
        cluster_name = hierarchy[str(label)].get("l2")
        sibling_labels = clusters.get(cluster_name, []) if cluster_name else []

        if use_cluster_siblings and len(sibling_labels) > 1:
            parts.append("")
            parts.append(f"Other templates in the {cluster_name} cluster (mechanistically related):")
            for sib in sibling_labels:
                marker = "  -> " if sib == label else "     "
                tag    = " (THIS ONE)" if sib == label else ""
                parts.append(f"{marker}Y={sib}{tag}: \"{label_map.get(sib, '?')}\"")
            parts.append(
                "  Your reasoning should explain why THIS template applies to "
                "the query pair rather than its siblings."
            )
        elif use_cluster_count_hint and len(sibling_labels) > 1:
            # Conservative variant: don't show sibling text, just signal that
            # within-cluster disambiguation is required.
            parts.append("")
            parts.append(
                f"This label is one of {len(sibling_labels)} mechanistically "
                f"related templates in the '{cluster_name}' cluster. Focus your "
                f"reasoning on what distinguishes this specific template "
                f"(direction of effect, which drug is affected, mechanism "
                f"specifics) — not on generic cluster-level claims."
            )

    # ── Prodrug warning (RJ Fix 2) ────────────────────────────────────────────
    if use_prodrug_warning and prodrug_ids:
        warnings = []
        for did, dname in [(drug1_id, drug1_name), (drug2_id, drug2_name)]:
            if did in prodrug_ids:
                warnings.append(
                    f"  Note: {dname} is a PRODRUG — pharmacologically inactive until "
                    f"converted to its active form by an enzyme. Inhibition of the "
                    f"activating enzyme DECREASES active drug levels (the opposite of "
                    f"a normal drug). Reason about activation, not elimination."
                )
        if warnings:
            parts.append("")
            parts.append("Prodrug context:")
            parts.extend(warnings)

    # ── No-shared-pathway note (RJ Fix 5) ─────────────────────────────────────
    if use_no_pathway_note and p1 and p2:
        n1 = _extract_pathway_nodes(p1)
        n2 = _extract_pathway_nodes(p2)
        shared = (
            (n1["enzymes"]      & n2["enzymes"])      |
            (n1["transporters"] & n2["transporters"]) |
            (n1["targets"]      & n2["targets"])
        )
        has_data = (
            any(n1[k] for k in ("enzymes", "transporters", "targets")) or
            any(n2[k] for k in ("enzymes", "transporters", "targets"))
        )
        if not shared and has_data:
            parts.append("")
            parts.append(
                "Pathway note: no shared enzymes, transporters, or targets are "
                "annotated for both drugs in DrugBank. Avoid assuming a direct "
                "shared pharmacokinetic pathway unless supported by the profiles "
                "above. Pharmacodynamic mechanisms (both drugs acting on the same "
                "receptor or physiological system) are more likely the relevant "
                "frame for this pair."
            )

    # ── Final instruction ─────────────────────────────────────────────────────
    parts.append("")
    parts.append(
        "Explain step-by-step the pharmacological mechanisms behind this "
        "drug-drug interaction. Make explicit:\n"
        "  - which drug's pharmacology changes\n"
        "  - the direction of that change (increases / decreases)\n"
        "  - the molecular point of interaction (enzyme, transporter, receptor, etc.)\n"
        "Then provide a 2-3 sentence Summary that states the direction explicitly. "
        "End with the Classification line in the exact format shown."
    )

    return "\n".join(parts)


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test(processed_dir: str = "processed_v2", show_full: bool = False) -> None:
    """
    Build prompts for the first test row under several toggle configurations.
    Prints length + head + tail per variant by default; pass show_full=True to
    dump full prompts (only useful for one-off debugging).

    Run from /scratch/vlelo/ddiproject:
        python scripts/teacher_prompt_v3.py
        python scripts/teacher_prompt_v3.py processed_v2 --full
    """
    base = Path(processed_dir)
    print(f"Smoke test using {base.resolve()}")

    with open(base / "drug_profiles.json") as f:
        profiles = json.load(f)
    with open(base / "test.jsonl") as f:
        first_row = json.loads(f.readline())
    rt_path = base / "retrieved_examples_pathway_test_pilot500_v2.json"
    retrieved = []
    if rt_path.exists():
        with open(rt_path) as f:
            retrieved = json.load(f)[0]  # neighbors for test row 0

    print(f"\nQuery row: {first_row.get('drug_a')} + {first_row.get('drug_b')}")
    print(f"Label: {first_row.get('label')} — {first_row.get('description', '')[:80]}")
    print(f"Retrieved neighbors: {len(retrieved)}")

    configs = [
        ("Main (defaults: count_hint ON, siblings OFF)",
         dict()),
        ("Ablation: siblings ON (count_hint ignored)",
         dict(use_cluster_siblings=True, use_cluster_count_hint=False)),
        ("Ablation: no hierarchy hints at all",
         dict(use_hierarchy_hints=False, use_cluster_siblings=False,
              use_cluster_count_hint=False)),
    ]

    for name, kwargs in configs:
        print(f"\n{'='*72}\n=== {name} ===\n{'='*72}")
        try:
            prompt = build_teacher_prompt(
                first_row, profiles, retrieved_examples=retrieved,
                processed_dir=str(base), **kwargs,
            )
            n_chars = len(prompt)
            n_lines = prompt.count("\n") + 1
            print(f"Built OK: {n_chars:,} chars, {n_lines} lines")
            if show_full:
                print(prompt)
            else:
                head = prompt[:500]
                tail = prompt[-400:]
                print(f"\n--- head (first 500 chars) ---\n{head}")
                print(f"\n--- tail (last 400 chars) ---\n{tail}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    show_full = "--full" in args
    args = [a for a in args if a != "--full"]
    pd = args[0] if args else "processed_v2"
    _smoke_test(pd, show_full=show_full)
