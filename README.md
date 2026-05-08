# DDI Chain-of-Thought Distillation — Data Rebuild + Hierarchy-Guided Teacher Prompting

This repository contains my contribution to a chain-of-thought distillation
project for drug-drug interaction (DDI) classification. The overall goal of
the project is to distill reasoning from a large teacher model into a smaller
student. My work targets the **teacher-side stage** (Stage 1 in the larger
pipeline): producing higher-fidelity rationale traces by rebuilding the
training data from raw sources and adding hierarchy-guided supervision to the
teacher prompt.

This is a complement to (not a replacement of) prior student-side work in the
same project (Compact CoT, classification-weighted loss, aggressive class
balancing).

---

## Why this work exists

The base paper for this project reports a 4.1-point Macro-F1 gap between
chain-of-thought distillation (`C_compact = 0.8783`) and a label-only baseline
(`B = 0.9196`). The paper attributes much of the residual gap to the model
"reasoning itself into a pharmacologically adjacent but incorrect class" —
direction-confusion errors concentrated in CYP-related labels where surface
verbs and physiological outcomes have opposite signs (for example, "decrease
metabolism" implies "increase exposure"). The paper's interventions all
operate **after** the teacher trace exists: they reshape gradient flow during
student fine-tuning.

This repository attacks the upstream problem: can we improve the **teacher
trace itself** so the student has cleaner supervision to learn from?
Specifically:

1. Is the dataset everyone uses (TDC-derived DrugBank 5.1.17, 129 classes)
   actually clean enough for fine-grained CoT distillation work? **No, it
   contains three contamination defects** (documented below).
2. Is structural similarity (Tanimoto) the right retrieval signal for DDI
   few-shot examples? **Pathway-overlap retrieval surfaces more
   mechanistically-relevant neighbors**, especially for tail labels.
3. Can we tell the teacher *what direction the interaction goes* via
   structured metadata derived from a pharmacological hierarchy, before it
   writes the rationale? **Yes — and it assists in eliminating wrong-direction
   commitments.

---

## What's new in this repository

| Contribution | Stage | What changed |
|---|---|---|
| **Rebuilt 166-class dataset** from raw DrugBank 6.0 XML | Data | Three contamination defects fixed at source |
| **71-cluster pharmacological hierarchy** with polarity / role / secondary tags | Data | Outcome-level direction, not surface verb |
| **Pathway retrieval with asymmetric handling** (Step 11c v2) | Retrieval | Empty rate 31% → 3% via single-sided fallback |
| **Hierarchy-guided teacher prompt v3** | Prompting | Label-conditioned structured supervision (privileged-info distillation, named honestly) |
| **Direction-aware scorer** with five verdict categories | Eval | Outcome-polarity alignment as a first-class metric |
| **2×2 pilot harness** with prepare / generate / score modes | Eval | Stratified sampling + reproducibility manifests |

Each is justified below with evidence from the pilot.

---

## Contribution 1 · Rebuilt data layer

The TDC-derived DrugBank 5.1.17 dataset commonly used for DDI classification
work has three defects that meaningfully affect a CoT distillation study:

| Defect | Symptom | Fix |
|---|---|---|
| Text / label_text contamination | Rows where the text described one interaction but the label referenced another | Extract templates *from* descriptions, not apply them post-hoc |
| Broken coarse_category | 70+ of 129 labels lumped into "other" while mechanistically equivalent labels were in separate buckets | Build a fresh 71-cluster hierarchy (Contribution 2) |
| Class count mismatch | Inherited preprocessing has 129; actual extractable from full DrugBank: **166** | Re-extract templates with `MIN_COUNT=50`, achieve 99.85% coverage |

**Evidence from rebuild:**
- 19,857 drugs parsed
- 2,915,498 raw interaction tags → 1,458,020 deduplicated unique pairs (matches the DrugBank 6.0 paper's 1.4M figure exactly)
- 1,162,084 train / 290,522 test (80/20 stratified, seed=42)
- 166 templates covering 99.85% of pairs
- Class imbalance reduced from 8,837:1 (TDC version) to 3,637:1

Reproducibility: all 17 numbered pipeline steps are in `scripts/` and run
sequentially from raw `drugbank_full.xml`. `PIPELINE.md` documents each step's
inputs, outputs, and timing.

---

## Contribution 2 · Pharmacological hierarchy

Each label has a 6-field schema:

```
{
  "l1": "PK" | "PD",                       # mechanism domain
  "l2": "<mechanism cluster name>",         # 71 clusters total
  "template": "<label text with #Drug1, #Drug2 placeholders>",
  "polarity": "increase" | "decrease" | "n/a",
  "affected_drug_role": "drug1" | "drug2" | "both",
  "secondary_tags": [<additional clusters for compound templates>]
}
```

**Polarity reflects physiological outcome, not surface verb.** Example: a
label like "decrease excretion" has `polarity=increase` because exposure goes
up. This decision matters because the C_seq → B gap concentrates in labels
where surface verb and outcome have opposite signs.

Cluster shape: 71 clusters total. 38 are singletons (one label only — typically
self-explanatory like `adverse_effects_generic`); 33 contain 2+ labels.
Cluster sizes range from 1 to 12, with `hypotension_increase` (n=12) and
`gi_effects_increase` (n=9) the largest. Six clusters carry documented
imperfections (`vasoactivity`, `gi_effects_increase`, `thrombosis`,
`electrolyte_imbalance`, `myopathy`, `cns_depression`) — kept consolidated
because mechanism heterogeneity within them is real and not cleanly
separable. The pilot's per-cluster breakdown (Step 13 in `PIPELINE.md`)
only powered two of these six (`cns_depression` n=15, `thrombosis` n=9);
the other four were either absent from the eligible pool (`vasoactivity`,
`gi_effects_increase`) or appeared with n=1 (`myopathy`,
`electrolyte_imbalance`), so the imperfect-cluster hypothesis remains
**untested for four of six** in this pilot.

---

## Contribution 3 · Hierarchy-guided teacher prompt (v3)

The teacher prompt is **label-conditioned with structured supervision**:
during teacher generation it sees the gold label, label text, mechanism
domain, cluster name, polarity, and affected-drug role. The student at
inference receives only drug profiles. This is privileged-information
distillation, named explicitly in the system prompt to avoid implicit
methodological misreading.

Key toggles for ablation:

- `use_hierarchy_hints` — polarity / role / cluster / secondary_tags
- `use_cluster_count_hint` — *conservative* default: tells the teacher
  "this is one of N templates in cluster X; differentiate" without showing
  sibling templates verbatim (avoids paraphrase-leakage)
- `use_cluster_siblings` — verbose alternative that shows sibling labels
  (kept for ablation; **not** the default main condition)
- `use_prodrug_warning`, `use_no_pathway_note` — kept constant across the
  2×2 conditions to avoid confounding (these are auxiliary fixes preserved
  from prior collaborator work; see Acknowledgements)

The prompt enforces structured output:

```
## Reasoning
[numbered mechanism steps]

## Summary
[2-3 sentences with explicit direction commitment]

## Classification
Y={label} -- "{label_text}"
```

This structure is what the direction-aware scorer (Contribution 5) parses.

**Evidence the toggles work:**

| Configuration | Prompt size for sample query (label 1, 5-template cluster) |
|---|---|
| Ablation: no hierarchy hints | 2,004 chars |
| Main (count_hint on, siblings off) | 2,562 chars |
| Ablation: siblings on | 3,058 chars |

Smoke test verified the system prompt is honest about gold-label-derived
supervision, the structured output instructions include anti-parroting clauses
on Reasoning and Summary, and the no-pathway note is epistemically softened
("no shared annotated nodes in DrugBank" rather than "no PK pathway").

---

## Contribution 4 · Asymmetric pathway retrieval (Step 11c v2)

Pathway-based retrieval (drug-drug similarity computed over shared enzyme,
transporter, target, and carrier annotations from DrugBank) is conceptually
introduced by collaborator work (see Acknowledgements). My contribution is an
independent implementation against the rebuilt 166-class dataset, plus an
**asymmetric-handling fix** that materially improves coverage.

**Problem:** the natural pair-similarity formula `sim(qa,ca) × sim(qb,cb)`
returns zero whenever any one of the four drug feature sets is empty —
which happens for ~70% of test queries (since only 42.9% of drugs have any
human pathway annotations). On the first pilot test (n=500), this produced
a 31.4% empty rate.

**Fix:** when a query has only one drug with features, fall back to
single-sided similarity weighted by an explicit penalty
(`asymmetric_penalty=0.5`, configurable). Full-pair matches still rank
above asymmetric ones at equal strength.

**Evidence — empty rate before / after on identical pilot set:**

| Bucket | Before fix (v1) | After fix (v2) |
|---|---|---|
| Both drugs lack features | 11 | 15 |
| Asymmetric (one has, one doesn't) | 122 | 0 |
| Both have features but no train overlap | 24 | 0 |
| **Total empty / 500** | **156 (31.4%)** | **15 (3.0%)** |

**Top-1 stability v1 → v2: 88.7%.** Asymmetric matches did not displace
strong full-pair matches in the top-k; they filled gaps. The remaining 3.0%
empties are pairs where neither drug has any pathway annotation in DrugBank
— genuinely uncoverable by pathway retrieval, and we explicitly fall back
to Tanimoto retrieval for these in the pilot harness.

---

## Contribution 5 · 2×2 pilot + three-tier evaluation

The pilot harness measures four conditions on a stratified subset:

|   | No hints | + Hierarchy hints |
|---|---|---|
| Tanimoto retrieval | A | C |
| Pathway retrieval  | B | D |

**Stratification**: 254 pairs across 42 mechanism clusters with floor=3 and
head-cap=15, sampled from the 485-pair "both retrievals nonempty" subset of
the first 500 test rows. Auxiliary fixes (prodrug warning, no-pathway note)
constant across all four conditions.

**Three-tier evaluation:**

1. **Exact label match** — predicted Y equals gold Y
2. **Cluster match** — predicted label's `l2` cluster equals gold `l2` cluster
3. **Outcome-polarity alignment** — direction extracted from `## Summary`
   matches gold polarity (correct / incorrect / ambiguous / missing /
   no_summary)

The third tier specifically tests whether the teacher commits to the right
*outcome direction* in its rationale, regardless of whether it picks the
exact gold template.

### Pilot results

Generated with **Qwen3-8B** in bf16 on Nibi (Compute Canada H100), 1016
prompts initial run + 193-prompt targeted regeneration after polarity-tag
fixes (Step 17 in `PIPELINE.md`), greedy decoding (temperature=0,
seed=42), max_tokens=4096 in the regenerated subset (3072 in initial run),
thinking mode disabled via `chat_template_kwargs={"enable_thinking": False}`.

| Condition | Exact match | Cluster match | Outcome-polarity alignment | Wrong-direction (% of all traces) |
|---|---|---|---|---|
| A · Tanimoto, no hints | 96.5% | 96.5% | 86.2% | 13.8% |
| B · Pathway, no hints  | 98.0% | 98.0% | 90.9% | 9.1%  |
| C · Tanimoto, hints    | 98.8% | 98.8% | **99.2%** | **0.8%** |
| D · Pathway, hints     | 98.8% | 98.8% | **98.5%** | **1.6%** |

**Headline:** hierarchy hints reduce scorer-flagged wrong-direction
commitments roughly 10–15× (13.8% → 0.8% Tanimoto, 9.1% → 1.6% pathway).
Pathway retrieval also gives a consistent baseline gain at no-hints
(+1.5pt exact, +4.7pt outcome alignment); the retrieval gain shrinks once
hints lift alignment toward the ceiling, as expected when two
non-orthogonal signals both push in the same direction.

The numbers above reflect a **post-correction** state. The original pilot
ran against a hierarchy schema containing six polarity-tagging bugs (see
`PIPELINE.md` Step 16). Audit + fix + targeted regeneration of the 193
affected traces (72 buggy-hint cases + 138 truncated cases at
max_tokens=3072, with 17-trace overlap) reaffirmed the headline rather
than weakening it: under correct hints the wrong-direction failure mode
nearly disappears.

**Schema-sensitivity finding (methodological):** between the bug
discovery and the regeneration we observed an intermediate state in
which the affected hint conditions exhibited *elevated* wrong-direction
rates (e.g. `efficacy_decrease` C/D rose from 0% to 40%/27% under
incorrect hints baked into the prompts). This confirmed that the
teacher's outcome commitment tracks the privileged supervision signal
faithfully — which is the desired behavior of label-conditioned
distillation, but means **schema correctness propagates 1:1 into trace
correctness in this pipeline**. We treat this as a structural property
of hierarchy-guided teacher prompting that any downstream user should
account for: a wrong polarity tag in the schema becomes a wrong polarity
trace in the dataset.

### Per-cluster breakdown — where the wrong-direction signal lives

The pilot covers 42 mechanism clusters; the aggregate 13.8% / 9.1%
no-hint wrong-direction rate is **not uniform** across them. The signal
concentrates in clusters where surface verb and physiological outcome
disagree:

| Cluster (n=15 each) | A · tan, no hints | B · pwy, no hints | C · tan, hints | D · pwy, hints |
|---|---|---|---|---|
| `metabolism_decrease` | **80.0%** | **46.7%** | **6.7%** | **6.7%** |
| `bp_increase`         | **33.3%** | **33.3%** | **0.0%** | **6.7%** |
| `efficacy_decrease`†  | 0.0% | 0.0% | **0.0%** | **0.0%** |
| `metabolism_increase`†| **53.3%** | **13.3%** | **0.0%** | **0.0%** |

† clusters whose polarity tag was corrected at Step 16 and whose C/D
prompts were regenerated with the corrected hint at Step 17. The post-
fix C/D rates of 0% in these clusters are the cleanest demonstration of
the "correct hint → correct trace" effect; their A/B rates report
no-hint baseline, which `efficacy_decrease` already passes
trivially because the surface label text gives the answer (the hint
mostly matters when surface and outcome conflict).

The remaining 38 clusters in the pilot show ≤20% wrong-direction in any
condition; most show 0%.

**Audit and scorer caveats:**
- A 32-trace stratified manual audit of the "ambiguous" verdict bucket
  (~50–60% of traces in each condition) classified ~78% as benign PK-
  reversal scoring artifacts, ~22% as token-budget truncations of
  otherwise-correct reasoning, and ~6% as schema-tagging bugs — the last
  category is what surfaced Step 16. A 32-trace audit is not large
  enough to support a strong prevalence claim about the ambiguous
  bucket; the qualitative finding (PK reversal dominates) is reported
  honestly with that caveat.
- After bumping `max_tokens` to 4096 in the targeted regeneration, 40
  of 193 regenerated traces still hit the length limit. Truncation thus
  remains a partial confound on the exact-match tier, even though the
  outcome-alignment tier (which only requires the Summary section to
  contain the correct polarity) is largely robust.

**Cross-check on the documented imperfect-cluster list:** of the six
clusters tagged in `hierarchy_clusters.json` as known-imperfect, only
two had pilot coverage adequate for inference (`cns_depression` n=15;
`thrombosis` n=9). Both showed 0% wrong-direction across all four
conditions. The other four (`vasoactivity`, `gi_effects_increase`,
`myopathy`, `electrolyte_imbalance`) were absent from the pool or had
n=1. The imperfect-cluster hypothesis remains **untested for four of
six**; future full-scale runs should target them explicitly.

### How this fits the existing paper

The base paper diagnoses the C_seq → B gap as drift into pharmacologically
adjacent but direction-incorrect labels (e.g. CYP3A4 inhibition pairs
clustered in Y=47 / Y=49 / Y=73 in the older labelmap). Hierarchy hints
attack this drift **at the teacher** — the rationale commits to the correct
outcome direction before the student ever sees it. This is a teacher-side
intervention complementary to (not competitive with) the paper's
student-side fixes. **No claim of student fine-tuning improvement is made
in this pilot** — that's future work.

---

## Reproducibility

```bash
# Prerequisites: Python 3.11, torch, transformers, vllm, RDKit
# DrugBank 6.0 academic XML (obtain from DrugBank under their license)

# 1. Rebuild the data layer (see PIPELINE.md for step details)
python scripts/02_parse_drugs.py
python scripts/03_parse_interactions.py
python scripts/03c_dedup_interactions.py
python scripts/04_build_label_map.py
python scripts/05_train_test_split.py
# ... continue through step 11b for retrieval caches
python scripts/08_build_hierarchy.py
python scripts/08b_resolve_unclassified.py
python scripts/08d_enrich_hierarchy.py
python scripts/08e_polarity_fixes.py
python scripts/08f_rename_glucose.py
python scripts/09_fingerprints.py
python scripts/10_similarity.py
python scripts/11_retrieval.py
python scripts/11b_fix_empties.py

# 2. Build pathway retrieval cache
python scripts/step11c_pathway_retrieval_v2.py \
    --target test --pilot 500

# 3. Run 2×2 pilot
python scripts/step12_pilot_2x2.py prepare \
    --data-dir processed_v2 --out-dir pilot

python scripts/step12_pilot_2x2.py generate \
    --data-dir processed_v2 --out-dir pilot \
    --model Qwen/Qwen3-8B --max-model-len 8192 \
    --max-tokens 2048 --no-thinking

python scripts/step12_pilot_2x2.py score \
    --data-dir processed_v2 --out-dir pilot

# 4. Generate figures
python scripts/step12_visualize.py \
    --scored pilot/pilot_scored.jsonl \
    --out pilot/figures
```

Every mode of the pilot harness writes a manifest JSON next to its outputs
(`manifest_prepare_*.json`, `manifest_generate_*.json`,
`manifest_score_*.json`) recording git commit, library versions, parameters,
input file paths, and timestamps. The pilot run for the headline numbers
above is reproducible from these manifests.

---

## File map

```
scripts/
├── 02–11b_*.py                          # Data layer pipeline (DrugBank → train/test/retrieval)
├── 08–08f_*.py                          # Hierarchy construction (cluster → polarity → renames)
├── step11c_pathway_retrieval.py         # v1: pathway retrieval, naive scoring
├── step11c_pathway_retrieval_v2.py      # v2: asymmetric handling (Contribution 4)
├── step11d_direction_scorer.py          # Direction-aware regex scorer with 9-case self-test
├── teacher_prompt_v2.py                 # Reference: prior collaborator-style prompt
├── teacher_prompt_v3.py                 # Hierarchy-guided prompt (Contribution 3)
├── step12_pilot_2x2.py                  # 2×2 harness: prepare / generate / score (Contribution 5)
├── step12_visualize.py                  # Figures from scored output
├── step13_per_cluster.py                # Per-cluster wrong-direction breakdown
├── step14_confusion.py                  # When predictions miss, where do they land?
├── step15_ambiguous_audit.py            # Stratified sampling of "ambiguous"-verdict traces
├── step16_polarity_scan.py              # Hierarchy schema consistency check
├── step17_polarity_fix.py               # Apply schema corrections (with backup + asserts)
└── step17b_patch_polarity_in_traces.py  # Patch cached gold_polarity in pilot files

processed_v2/                            # Public artifacts only (small files)
├── label_map.json                       # 166-class label → template
├── hierarchy_map.json                   # per-label hierarchy schema (Contribution 2)
├── hierarchy_clusters.json              # cluster → label-list (inverse)
└── label_distribution.json              # class counts

pilot/
├── pilot_prompts.jsonl                  # 1016 prompts (4 conditions × 254 pairs)
├── pilot_scored.jsonl                   # per-trace verdicts (sanitized)
├── pilot_summary.json                   # aggregate metrics for the table above
├── per_cluster_breakdown.json           # per-cluster wrong-direction by condition (Step 13)
├── confusion_analysis.json              # cross-cluster confusion among misses (Step 14)
├── ambiguous_audit_sample.txt           # 32-trace stratified audit (Step 15)
├── polarity_scan_report.json            # schema consistency check output (Step 16)
├── polarity_fix_log.json                # changes applied to hierarchy_map.json (Step 17)
├── regen_keys.txt                       # keys for the 72 polarity-affected prompts
├── regen_keys_v2.txt                    # keys for the 193 regenerated prompts (polarity + truncation)
├── manifest_*.json                      # full reproducibility provenance
└── figures/
    ├── fig_metrics_2x2.png
    ├── fig_verdicts_stacked.png
    └── fig_alignment_focus.png

PIPELINE.md                              # Full reproducibility log of the data rebuild
```

Bulk artifacts (`drug_profiles.json`, `train.jsonl`, `test.jsonl`, raw
DrugBank XML, retrieval caches, individual teacher traces) are **not**
included in this repository because the source DrugBank XML is licensed and
the derived files are large. They are reproducible from the scripts above
given a DrugBank academic license.

---

## Acknowledgements

Pathway-based retrieval as a concept was independently developed by
collaborator Rameen Jafri in contemporaneous work on the same overall
project. The implementation in this repo (`step11c_pathway_retrieval_v2.py`)
is a from-scratch reimplementation against my rebuilt 166-class dataset,
with the asymmetric-handling extension. The auxiliary prompt fixes preserved
in `teacher_prompt_v3.py` (prodrug warning, no-shared-pathway note, and the
raised drug-profile truncation caps) also originate from her work and are
credited in inline comments.

What is independently mine in this repo: the data-layer rebuild from raw
DrugBank 6.0, the 71-cluster hierarchy with outcome-level polarity / role /
secondary-tag schema, the asymmetric-handling extension to pathway retrieval,
the hierarchy-guided teacher prompt design (label-conditioned structured
supervision, conservative count-hint default, anti-parroting clauses,
gold-label-derived supervision named honestly in the system prompt), the
direction-aware scorer, the 2×2 pilot harness with three-tier evaluation,
and the pilot result above.

The base paper this work extends — including the C_compact intervention,
classification-weighted loss, aggressive class balancing, and the C_seq
diagnostic analysis identifying CYP-related drift as the residual gap — is
not my work; this repo is a teacher-side complement to those student-side
contributions.

---

## License

MIT. See `LICENSE`.

---

## Citation

If you use this work, please cite the underlying paper (TBD) and reference
this repository at the relevant commit hash.
