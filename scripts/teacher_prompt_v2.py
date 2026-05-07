"""
src/teacher_prompt_v2.py

Improved teacher prompt builder for the rebuilt DDI dataset.

Builds on the Rameen Jafri (RJ) prompt design with additions that exploit
the hierarchy schema this dataset has and her dataset doesn't:

  - polarity: explicit direction-of-effect tag (increase / decrease / n/a)
  - affected_drug_role: which drug in the pair experiences the change
  - l2 mechanism cluster: groups mechanistically equivalent templates
  - secondary_tags: additional clusters for compound templates
  - cluster siblings: nudges teacher to commit to a specific template within
    a mechanism family

Also keeps her good ideas:
  - drug profiles with raised truncation caps (8/5/5)
  - prodrug warning for prodrug pairs
  - no-shared-pathway note when drugs have no overlapping nodes
  - structured output format (## Reasoning / ## Summary / ## Classification)

Drops her severity sections — DrugBank 6.0 academic XML doesn't have severity,
and faking it via a rule-based classifier introduces unverified noise.

Schema notes:
  - Our train.jsonl uses drug_a/drug_b column names (her code uses
    drug1_id/drug2_id). We accept both via _g() helper for compatibility.
  - hierarchy_map.json keys are stringified labels.
"""
from pathlib import Path
import json
import functools


# ── System prompt ─────────────────────────────────────────────────────────────

TEACHER_SYSTEM_PROMPT = (
    "You are an expert pharmacologist specialising in drug-drug interactions. "
    "Given two drugs, their pharmacological profiles, and structured information "
    "about the known interaction, explain the underlying mechanisms step by step. "
    "Your reasoning should commit to a specific direction of effect (which drug's "
    "levels or activity changes, in which direction) and identify the dominant "
    "mechanism (enzyme, transporter, receptor, or pharmacodynamic). "
    "Structure your response exactly as:\n\n"
    "## Reasoning\n"
    "[Numbered steps explaining the mechanism. Identify each drug's relevant "
    "pharmacology, the point of interaction, and the resulting effect.]\n\n"
    "## Summary\n"
    "[2-3 sentence summary that explicitly states the direction of effect and "
    "which drug experiences the change.]\n\n"
    "## Classification\n"
    "Y={label} -- \"{label_text}\""
)


# ── Hierarchy loading ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_hierarchy(processed_dir: str) -> dict:
    """Load hierarchy_map.json once per process."""
    path = Path(processed_dir) / "hierarchy_map.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _load_clusters(processed_dir: str) -> dict:
    """Load hierarchy_clusters.json once per process."""
    path = Path(processed_dir) / "hierarchy_clusters.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _load_label_map(processed_dir: str) -> dict:
    """Load label_map.json (int -> template) once per process."""
    path = Path(processed_dir) / "label_map.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


@functools.lru_cache(maxsize=1)
def _load_prodrug_ids(processed_dir: str) -> set:
    """Load prodrug ID set if available; empty set otherwise."""
    path = Path(processed_dir) / "prodrug_ids.json"
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    return set(data.keys())


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
        "n/a": "no inherent direction (e.g., infection risk, diagnostic effect)",
    }.get(polarity, "unspecified")


def _role_human(role: str, name1: str, name2: str) -> str:
    return {
        "drug1": f"Drug 1 ({name1}) — its pharmacology is what changes",
        "drug2": f"Drug 2 ({name2}) — its pharmacology is what changes",
        "both": f"Both drugs share the effect (e.g., combined risk of an event)",
    }.get(role, "unspecified")


def _format_drug_profile(profile: dict) -> str:
    """Format a drug profile compactly. Caps from RJ Fix 3 (8/5/5)."""
    lines = []
    if profile.get("description"):
        lines.append(f"  Description: {profile['description'][:300]}")
    if profile.get("mechanism_of_action"):
        lines.append(f"  Mechanism: {profile['mechanism_of_action'][:200]}")
    if profile.get("enzymes"):
        lines.append(f"  Key enzymes: {'; '.join(profile['enzymes'][:8])}")
    if profile.get("transporters"):
        lines.append(f"  Transporters: {'; '.join(profile['transporters'][:5])}")
    if profile.get("targets"):
        lines.append(f"  Targets: {'; '.join(profile['targets'][:5])}")
    if profile.get("smiles"):
        lines.append(f"  SMILES: {profile['smiles'][:200]}")
    return "\n".join(lines) if lines else "  (no detailed profile available)"


def _extract_pathway_nodes(profile: dict) -> dict:
    """Return enzymes/transporters/targets as sets, for shared-node checking."""
    return {
        "enzymes": set(profile.get("enzymes", []) or []),
        "transporters": set(profile.get("transporters", []) or []),
        "targets": set(profile.get("targets", []) or []),
    }


# ── Main prompt builder ───────────────────────────────────────────────────────

def build_teacher_prompt(
    row: dict,
    profiles: dict,
    retrieved_examples: list = None,
    processed_dir: str = "processed_v2",
    use_hierarchy_hints: bool = True,
    use_cluster_siblings: bool = True,
    use_prodrug_warning: bool = True,
    use_no_pathway_note: bool = True,
) -> str:
    """
    Build the teacher prompt user message.

    Toggles:
      use_hierarchy_hints:    polarity, affected_drug_role, l2 cluster, secondary_tags
      use_cluster_siblings:   show other label texts in the same cluster
      use_prodrug_warning:    flag prodrugs (data/processed/prodrug_ids.json)
      use_no_pathway_note:    flag pairs with no shared pathway nodes

    Each toggle isolates a contribution for ablation. Default ON for all.
    """
    hierarchy = _load_hierarchy(processed_dir) if use_hierarchy_hints else {}
    clusters = _load_clusters(processed_dir) if use_cluster_siblings else {}
    label_map = _load_label_map(processed_dir)
    prodrug_ids = _load_prodrug_ids(processed_dir) if use_prodrug_warning else set()

    parts = []

    # Resolve fields with schema flexibility
    drug1_id = _g(row, "drug1_id", "drug_a")
    drug2_id = _g(row, "drug2_id", "drug_b")
    drug1_name = _g(row, "drug1_name", default=drug1_id)
    drug2_name = _g(row, "drug2_name", default=drug2_id)
    label = int(row.get("label", 0))
    label_text = _g(row, "label_text", "description")

    # ── Retrieved examples ────────────────────────────────────────────────────
    if retrieved_examples:
        for i, ex in enumerate(retrieved_examples[:5], 1):
            ex_d1 = _g(ex, "drug1_id", "drug_a")
            ex_d2 = _g(ex, "drug2_id", "drug_b")
            ex_n1 = _g(ex, "drug1_name", default=ex_d1)
            ex_n2 = _g(ex, "drug2_name", default=ex_d2)
            ex_label = int(ex.get("label", 0))
            ex_text = _g(ex, "label_text", "description")
            ex_sim = ex.get("similarity")

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

    # ── Hierarchy hints (the new contribution) ────────────────────────────────
    if use_hierarchy_hints and str(label) in hierarchy:
        h = hierarchy[str(label)]
        cluster = h.get("l2", "unknown")
        l1 = h.get("l1", "unknown")
        polarity = h.get("polarity", "unspecified")
        role = h.get("affected_drug_role", "unspecified")
        secondary = h.get("secondary_tags", [])

        parts.append("")
        parts.append("Mechanism context:")
        parts.append(f"  Domain: {l1} ({'pharmacokinetic — ADME effects on drug exposure' if l1 == 'PK' else 'pharmacodynamic — combined effects on physiology'})")
        parts.append(f"  Mechanism cluster: {cluster}")
        parts.append(f"  Direction of effect: {_polarity_human(polarity)}")
        parts.append(f"  Affected drug: {_role_human(role, drug1_name, drug2_name)}")
        if secondary:
            parts.append(f"  Compound mechanism — also involves: {', '.join(secondary)}")

    # ── Cluster siblings (helps disambiguate within mechanism family) ────────
    if use_cluster_siblings and str(label) in hierarchy:
        cluster_name = hierarchy[str(label)].get("l2")
        sibling_labels = clusters.get(cluster_name, [])
        # Only show siblings if there are multiple templates in this cluster,
        # because that's exactly when confusion is most likely.
        if len(sibling_labels) > 1:
            parts.append("")
            parts.append(f"Other templates in the {cluster_name} cluster (mechanistically related):")
            for sib in sibling_labels:
                if sib == label:
                    parts.append(f"  -> Y={sib} (THIS ONE): \"{label_map.get(sib, '?')}\"")
                else:
                    parts.append(f"     Y={sib}: \"{label_map.get(sib, '?')}\"")
            parts.append(
                "  Your reasoning should explain why THIS template applies to "
                "the query pair rather than its siblings."
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
            (n1["enzymes"] & n2["enzymes"]) |
            (n1["transporters"] & n2["transporters"]) |
            (n1["targets"] & n2["targets"])
        )
        has_data = (
            any(n1[k] for k in ("enzymes", "transporters", "targets")) or
            any(n2[k] for k in ("enzymes", "transporters", "targets"))
        )
        if not shared and has_data:
            parts.append("")
            parts.append(
                "Pathway note: these two drugs share NO common enzymes, "
                "transporters, or targets in DrugBank. There is no direct "
                "pharmacokinetic pathway connecting them. Focus your reasoning "
                "on pharmacodynamic mechanisms (do both drugs act on the same "
                "receptor or physiological system?). Do not invoke CYP enzyme "
                "reasoning unless the profiles above explicitly show CYP "
                "involvement for both drugs."
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
