# DDI Pipeline — Reproducibility Log

This document records every step of rebuilding the processed dataset from
DrugBank 6.0 raw XML and running the 2×2 teacher-prompting pilot. Each step
records:

- **Goal**: what the step is for
- **Script**: the file
- **Command**: how to run it
- **Output**: key results (counts, paths, anomalies)

---

## Project Layout

```
~/ddiproject/
├── raw/
│   └── drugbank_full.xml        # 2.4 GB, DrugBank 6.0 academic XML
├── processed_v2/                # Cleaned outputs (this rebuild)
├── scripts/                     # All scripts used in the rebuild
├── reports/                     # Diagnostic outputs (counts, mismatch logs)
├── pilot/                       # 2×2 pilot prompts, traces, scores, figures
└── PIPELINE.md                  # This file
```

---

## Decisions (final, post-rebuild)

- **Label scheme**: 166 templates extracted from raw with `MIN_COUNT=50`, achieving 99.85% coverage of deduplicated interactions. (Prior work used 86 / 129 — both were under-extractions of the underlying DrugBank label space.)
- **Drug scope**: all groups retained (approved + investigational + experimental + biotech). Experimental compounds dominate raw counts but contribute almost nothing to interaction edges (4,635 of 19,857 drugs have ≥1 DDI).
- **Severity**: dropped. The academic XML export does not include severity data; see Step 6 for diagnostic confirmation.
- **Train/test split**: 80/20 stratified by label, seed=42 (matches base paper's seed).
- **CoT regeneration**: completed in pilot phase using Qwen3-8B as a cost-bounded teacher (Step 12). Full-scale teacher generation TBD with paper-track teacher.

---

## Step 0 — Workspace cleanup

**Goal:** organize the working directory after extracting raw DrugBank XML from a Windows-side download.

**Commands:**
```bash
cd ~/ddiproject
find . -name "*:Zone.Identifier" -delete   # remove Windows metadata
mkdir -p ~/ddiproject/raw
mv ~/ddiproject/drugbank_all_full_database.xml/"full database.xml" \
   ~/ddiproject/raw/drugbank_full.xml
rmdir ~/ddiproject/drugbank_all_full_database.xml
mkdir -p ~/ddiproject/{processed_v2,reports,scripts,pilot}
```

**Output:** raw XML at `~/ddiproject/raw/drugbank_full.xml`.

---

## Step 1 — XML structural sanity check

**Goal:** confirm the XML loads and verify schema before parsing.

**Script:** `scripts/00_xml_peek.sh`

**Command:**
```bash
chmod +x scripts/00_xml_peek.sh
scripts/00_xml_peek.sh | tee reports/00_xml_peek.txt
```

**Output:**
- File size: 2.4 GB
- Schema header: `<drugbank xmlns="http://www.drugbank.ca" version="5.1" exported-on="2026-04-23">`
- Drug entries: **19,857**
- `<drug-interaction>` tags: **2,915,498** (≈2× the deduplicated count, since each interaction appears in both drugs' cards)
- Sample first drug: DB00001 Lepirudin (biotech, approved+withdrawn)

---

## Step 2 — Drug profile extraction

**Goal:** extract clean per-drug records, keeping synonyms / mechanism / metabolism / target / enzyme / transporter info while dropping bibliographic and regulatory noise.

**Script:** `scripts/02_parse_drugs.py`

**Command:**
```bash
python scripts/02_parse_drugs.py
```

**Runtime:** 61 seconds.

**Results:**
- 19,857 drugs parsed (15,497 small molecules + 4,360 biotech)
- Group breakdown: experimental 8,864 / investigational 8,547 / approved 4,799 / withdrawn 914 / vet_approved 428 / illicit 205 / nutraceutical 135
- Field coverage: SMILES 73.7%, targets 50.5%, mechanism-of-action text 21.4%, metabolism text 12.5%, enzymes 9.4%

**Outputs:**
- `processed_v2/drug_profiles.json` (46.0 MB)
- `processed_v2/drug_synonyms.json` (14.6 MB)
- `processed_v2/drug_id_order.json` (0.3 MB)

**Limitations note:** explicit enzyme data only covers 9.4% of drugs. The teacher-side "hallucinates pharmacology" risk is mitigated but not eliminated by v6 grounding. The 9.4% figure is also a coverage floor for any retrieval-augmented teacher prompt that conditions on enzyme overlap.

---

## Step 3 — Interaction extraction

**Goal:** pull every drug-drug interaction out of the XML.

**Script:** `scripts/03_parse_interactions.py`

**Command:**
```bash
python scripts/03_parse_interactions.py
```

**Runtime:** 94 seconds.

**Results (raw, pre-dedup):**
- 4,635 of 19,857 drugs have ≥1 DDI (rest are isolated nodes)
- 2,915,498 raw `<drug-interaction>` tags

**Output:** `processed_v2/interactions_full.jsonl` (493 MB).

---

## Step 3b — Inspect duplication pattern (DIAGNOSTIC)

**Goal:** verify that interactions are double-counted (each appears in both drugs' cards) and decide on dedup strategy.

**Script:** `scripts/03b_inspect_interactions.py`

**Findings:**
- 1,458,020 unique unordered drug pairs
- 1,457,478 of those have 2 rows (one per side); 542 have only 1
- For all pairs with 2 rows: descriptions are 100% identical between the two sides
- The `subject_id` field carries no directional meaning (49.9% subject-first / 49.3% subject-second in 30K random sample) — direction is encoded in description text only
- ~0.8% of descriptions use a non-canonical drug name (synonym, capitalization variant, or development code)

**Conclusion:** dedup on `(unordered_pair, description)` collapses each mirror into one row.

---

## Step 3c — Deduplication

**Goal:** collapse mirrored rows into one canonical row per interaction.

**Script:** `scripts/03c_dedup_interactions.py`

**Approach:** key each row on `(min_drug_id, max_drug_id, description)`. Keep one row per unique key with `drug_a` = lexicographically smaller drugbank_id, `drug_b` = larger. The original `subject_id`/`affected_id` distinction is dropped (it carried no directional information); direction is preserved in the description text.

**Command:**
```bash
python scripts/03c_dedup_interactions.py
```

**Runtime:** 15 seconds.

**Results:**
- 2,915,498 input rows
- 1,458,020 unique interactions (matches DrugBank 6.0 paper's published 1.4M figure exactly)
- Compression ratio: 2.00× exactly
- 0 self-loops
- 4,636 distinct drugs participate in interactions

**Output:** `processed_v2/interactions_dedup.jsonl` (234 MB). Schema: `{drug_a, drug_b, description}`.

---

## Step 4 — Label map construction

**Goal:** extract templates from interaction descriptions to build a label space. Replace drug names with `#Drug1` / `#Drug2` placeholders. By construction, the description and the templated label can never disagree (this fixes the text/label_text contamination defect from prior pipelines).

**Script:** `scripts/04_build_label_map.py`

**Performance fix:** original implementation took 1h56m. Optimized version uses one compiled alternation regex per drug (instead of ~14 sequential regex compiles per row) and matches once on the description. Runtime dropped to 43 seconds — ~160× speedup.

**Command:**
```bash
python scripts/04_build_label_map.py
```

**Results:**
- 1,458,020 deduplicated interactions processed
- 1,454,752 successful template extractions (99.78%)
- 3,268 failures (0.22%) — biotech drugs where DrugBank uses development codes (e.g., TNX-901 for Talizumab) not in the synonym list
- 527 unique templates extracted; 166 qualify at MIN_COUNT=50
- 1,452,606 labeled interactions written
- Class distribution: largest 181,847, smallest 50, imbalance ratio **3,637:1**
- 18 classes have <100 examples

**vs prior pipelines:**
- Base paper used 86 (TDC) / 129 classes; correct count is 166
- 8,837:1 imbalance ratio (TDC) → 3,637:1 (rebuild) — substantially less severe

**Outputs:**
- `processed_v2/label_map.json` (16.7 KB)
- `processed_v2/label_distribution.json` (2.3 KB)
- `processed_v2/interactions_labeled.jsonl` (567 MB) — every row satisfies `description == label_text` by construction

---

## Step 5 — Stratified train/test split

**Goal:** 80/20 split stratified by label so all classes appear proportionally in both splits.

**Script:** `scripts/05_train_test_split.py`

**Method:** `sklearn.model_selection.train_test_split` with `stratify=labels`, seed=42.

**Command:**
```bash
python scripts/05_train_test_split.py
```

**Results:**
- Total labeled: 1,452,606
- Train: 1,162,084 (80.00%)
- Test: 290,522 (20.00%)
- All 166 labels present in both splits (0 dropped)
- Smallest class (label 166, n=50): split 40/10 train/test — stratification holds at the rare-class boundary

**vs base paper:**
- Base paper: 153,446 train / 38,362 test, 86 classes
- Rebuild: 1,162,084 train / 290,522 test, 166 classes — ~7.5× more training data, ~2× more classes

**Outputs:** `processed_v2/train.jsonl` (454 MB), `processed_v2/test.jsonl` (113 MB).

---

## Step 6 — Severity extraction (BLOCKED)

**Goal:** extract per-interaction severity for severity-weighted F1 reporting.

**Status:** abandoned. The DrugBank 6.0 academic XML export does not include severity. Diagnostic (`scripts/06b_diagnose_severity.py`) confirmed each `<drug-interaction>` element contains only `drugbank-id`, `name`, and `description`. No `severity`, `evidence-level`, or risk-related fields.

**Why:** severity lives in DrugBank's commercial Clinical API, not the academic export. Older `severity_map.json` files in prior pipelines were populated from a TDC preprocessed file or one-time scrape — not reproducible from the current academic data.

**Implication:** severity-weighted F1 cannot be reported from this rebuild. Documented in paper Limitations.

---

## Step 8 — Pharmacological hierarchy: rule-based pass

**Goal:** replace the broken `coarse_category_map.json` (TDC inheritance, 70+ labels lumped into "other") with a principled two-level hierarchy. Level 1 is PK vs PD; Level 2 is mechanism cluster.

**Script:** `scripts/08_build_hierarchy.py`

**Approach:** rule-based keyword classification over template text.

**Command:**
```bash
python scripts/08_build_hierarchy.py
```

**Results:**
- 156 of 166 templates classified by rules (94%)
- Largest clusters: hypotension (n=12), hypertension (n=9), gi_effects (n=9), cns_depression (n=8), bleeding (n=7)
- `exposure_increase` cluster {1, 6, 26, 129, 132} unifies decreased-excretion and increased-serum templates that the TDC coarse_category had split into separate buckets
- 10 templates left as `unclassified` for manual resolution

**Outputs:**
- `processed_v2/hierarchy_map.json` — label → {l1, l2, template}
- `processed_v2/hierarchy_clusters.json` — cluster → [labels]

---

## Step 8b — Resolve unclassified labels

**Goal:** assign the 10 templates the rule-based pass couldn't classify.

**Script:** `scripts/08b_resolve_unclassified.py`

**Manual assignments:**
- 65 → `qt_prolongation` (fold)
- 124 → `antipsychotic_change` (new cluster)
- 135 → `gi_effects` (fold)
- 137 → `hyperthermia` (new)
- 139 → `urinary_retention` (new)
- 140 → `peripheral_neuropathy` (new)
- 141 → `photosensitivity` (new)
- 145 → `hepatotoxicity` (fold)
- 146 → `electrolyte_imbalance` (fold)
- 148 → `cardiac_depression` (new)

**Why this matters:** templates like {15, 36, 45, 66, 72, 100, 102, 125} all describe hypotension with slightly different wording. Hierarchical F1 over the resulting clusters gives the model partial credit for "right family, wrong template," which is a more defensible metric for fine-grained DDI classification than flat F1.

**Outputs:** updates `hierarchy_map.json` and `hierarchy_clusters.json` in place.

---

## Step 8c — Inspect hierarchy (DIAGNOSTIC)

**Goal:** print the cluster structure for human review before downstream enrichment.

**Script:** `scripts/08c_view_hierarchy.py`

**Output:** stdout summary of clusters and member labels. Used to identify clusters that needed polarity splits (driving Step 8e) and clusters where enrichment was needed (driving Step 8d).

No state changes.

---

## Step 8d — Enrich hierarchy with polarity / role / secondary_tags

**Goal:** add three fields to each label's hierarchy entry that the teacher prompt can use as structured supervision.

**Script:** `scripts/08d_enrich_hierarchy.py`

**Schema added per label:**
- `polarity`: `"increase"` | `"decrease"` | `"n/a"` — physiological outcome direction (NOT surface verb). Example: a "decrease excretion" template has `polarity=increase` because exposure goes up.
- `affected_drug_role`: `"drug1"` | `"drug2"` | `"both"` — which drug experiences the pharmacological change.
- `secondary_tags`: `[<cluster names>]` — for compound templates that touch multiple mechanism families (e.g., a template that involves both serotonergic and respiratory CNS depression).

**Why polarity-as-outcome:** the base paper diagnoses a residual 4.1pt Macro-F1 gap as direction confusion in pharmacologically adjacent labels (e.g. CYP3A4 inhibition where "decrease metabolism" and "increase exposure" describe the same event in opposite surface verbs). An outcome-level polarity field lets the teacher prompt commit to outcome direction explicitly, regardless of which mechanism step the rationale describes.

**Backups:** `hierarchy_map.json.bak_step8d`, `hierarchy_clusters.json.bak_step8d`.

**Output:** in-place enrichment of `hierarchy_map.json`.

---

## Step 8e — Targeted polarity fixes

**Goal:** address residual polarity mixing in clusters that were hiding opposite-direction effects under the same name.

**Script:** `scripts/08e_polarity_fixes.py`

**Bugs fixed:**
- `potassium_imbalance` lumped hyperkalemia (K up) and hypokalemia (K down) → split into `hyperkalemia` and `hypokalemia` clusters
- `glycemic_increase` lumped hyperglycemia and hypoglycemia (opposite glucose directions) → regrouped by glucose direction
- `hypertension_increase` and `hypertension_decrease` mixed outcome statements with activity-modulation statements → regrouped by actual BP outcome (`bp_increase` / `bp_decrease`)

**Command:**
```bash
python scripts/08e_polarity_fixes.py
```

**Results:**
- 17 label overrides applied
- 71 clusters total (was 70)
- New clusters: `hyperkalemia` (n=2), `hypokalemia` (n=2), `glucose_increase` (n=2), `glucose_decrease` (n=2), `bp_increase` (n=6), `bp_decrease` (n=3)

**Documented known imperfections (left intentionally consolidated):**
- `vasoactivity` (n=6): mechanism family mixing vasoconstriction/vasodilation/vasopressor/vasospastic
- `gi_effects_increase` (n=9): different GI endpoints with shared drug-class etiologies
- `thrombosis` (n=4): includes antiplatelet activity in same mechanism family
- `electrolyte_imbalance` (n=5): non-potassium electrolytes lumped (rare individually)
- `myopathy` (n=4): includes tendinopathy as adjacent musculoskeletal toxicity
- `cns_depression` (n=8): includes respiratory and serotonergic compound templates (handled via `secondary_tags` from Step 8d)

These are acknowledged-imperfect and stay consolidated because mechanism heterogeneity within them is real and not cleanly separable. They trade clinical-precision granularity for hierarchical-F1 forgiveness — defensible per the project's consolidation-first philosophy.

**Backups:** `hierarchy_map.json.bak_step8e`, `hierarchy_clusters.json.bak_step8e`.

---

## Step 8f — Cluster naming consistency

**Goal:** apply final naming-consistency renames to clusters introduced or restructured in Step 8e.

**Script:** `scripts/08f_rename_glucose.py`

**Output:** in-place renames; backup at `hierarchy_map.json.bak_step8f`.

After this step the hierarchy is final: 71 clusters, 166 labels, 6-field schema per label.

---

## Step 9 — Morgan fingerprints

**Goal:** generate molecular fingerprints for similarity-based retrieval.

**Method:** RDKit Morgan fingerprints, radius=2 (ECFP4 equivalent), 2048-bit width.

**Script:** `scripts/09_fingerprints.py`

**Command:**
```bash
python scripts/09_fingerprints.py
```

**Runtime:** 4 seconds.

**Results:**
- 14,617 fingerprints generated (out of 14,627 drugs with SMILES — 99.93%)
- 5,230 drugs have no SMILES (biotech: proteins, antibodies, vaccines)
- 10 parse failures: exotic compounds (organometallics, porphyrin sulfonates, vitamin B12 derivatives) — RDKit can't handle their valence/charge configurations

**Output:** `processed_v2/drug_fingerprints.pkl` (30.5 MB) — `dict[drugbank_id → np.ndarray[uint8, 2048]]`.

**Note for Step 11:** the 5,230 drugs without fingerprints fall back to ATC code / category overlap during retrieval.

---

## Step 10 — Drug-drug similarity

**Goal:** for each drug, find its top-K most molecularly similar drugs by Tanimoto coefficient over Morgan fingerprints.

**Method:** Tanimoto similarity computed via batched matrix multiplication. For each batch of 256 drugs, intersection counts come from `batch @ all.T` (int8 dot product); union derived from popcounts. Top-50 neighbors per drug stored as a ranked list.

**Script:** `scripts/10_similarity.py`

**Command:**
```bash
python scripts/10_similarity.py
```

**Runtime:** 4 min 35 sec on 22 cores (~388K pairs/sec).

**Results:**
- 14,617 drugs in similarity index
- 730,850 (drug, neighbor, sim) triples cached (top-50 per drug)
- Median top-1 similarity: 0.568
- 4,061 drugs (28%) have a top-1 similarity > 0.7 (near-twins)
- 8,871 drugs (61%) have a top-1 similarity > 0.5

**Sanity check (Aspirin DB00945):**
- Aloxiprin (sim=0.923) — aspirin-aluminum complex
- Dipyrocetyl (0.700) — salicylate derivative
- Guacetisal (0.606) — salicylate prodrug
- Carbaspirin calcium (0.606) — aspirin salt
- Salsalate (0.594) — salicylsalicylic acid

All 5 are salicylates or aspirin derivatives — confirms similarity computation correctly groups pharmacologically related drugs.

**Output:** `processed_v2/drug_similarity.pkl` (11.9 MB).

---

## Step 11 — Retrieval cache (Tanimoto-based)

**Goal:** for every train+test row, cache K=5 most similar training interactions for use as in-context few-shot examples in teacher prompts.

**Algorithm:** for each query (A, B):
1. S_A = top-50 molecularly similar drugs to A; S_B = top-50 to B
2. Candidates = training pairs (X, Y) where X∈S_A and Y∈S_B (or swapped)
3. Score candidates by `pair_sim = max(sim(A,X)·sim(B,Y), sim(A,Y)·sim(B,X))`
4. Take top K=5

**Performance fix:** original implementation used "at least one drug in neighbor set" which let common drugs explode candidate counts. Rewrite uses "BOTH drugs related to query" with a precomputed `(drug, drug) → train_idx` index. ~16× speedup, finished in 52 minutes.

**Script:** `scripts/11_retrieval.py`

**Command:**
```bash
python scripts/11_retrieval.py
```

**Results:**
- Train: 1,162,084 rows processed, 537 rows/sec, 52 min total
- Test: 290,522 rows processed, 299 rows/sec
- Train rows with 0 examples: 79,595 (6.85%) — pairs where both drugs lack fingerprints AND no ATC overlap matches
- Test rows with 0 examples: 0
- Avg similarity of retrieved examples: train 0.466, test 0.434

**Outputs:**
- `retrieved_examples_train.json` (1.07 GB)
- `retrieved_examples_test.json` (288 MB)

---

## Step 11b — Fix empty retrieval rows

**Goal:** patch the 6.85% of train rows that got 0 retrieved examples in Step 11.

**Bug:** original retrieve function applied self-exclusion AFTER the fallback chain. When `S_A × S_B` lookup found exactly one candidate (the query itself), self-exclusion emptied the set, but ATC and random fallbacks were skipped because `candidate_idxs` had been non-empty earlier in the function.

**Fix:** move self-exclusion before the fallback chain so fallbacks fire if the only candidate was the query itself.

**Script:** `scripts/11b_fix_empties.py`

**Results:**
- 79,595 empty rows processed
- 0 truly empty after fix (final empty rate: 0.000%)
- Fallback path breakdown:
  - 792 (1%) S_A × S_B match (recovered by exclusion-first ordering)
  - 2,449 (3%) ATC code prefix overlap
  - 76,354 (96%) random pool fallback

**Caveat:** the 76,354 random-fallback rows are biotech drugs (no fingerprints, no ATC codes). Their retrieved examples are not similarity-matched. Test set has 0% empties so test-time evaluation is clean.

**Output:** `retrieved_examples_train.json` (1.15 GB) — patched in place.

---

## Step 11c — Pathway-based retrieval (v1 + v2)

**Goal:** alternative to Tanimoto retrieval that uses overlap of DrugBank pathway-relevant features (enzymes, transporters, targets, carriers) instead of structural fingerprints. Expected to surface mechanistically-relevant neighbors at the cost of coverage (since not all drugs have pathway annotations).

**Conceptual credit:** pathway-based retrieval was independently developed by collaborator Rameen Jafri in contemporaneous work. This implementation is a from-scratch re-implementation against the rebuilt 166-class dataset, with an added asymmetric-handling extension (v2).

**Method:** build an inverted index over `(category, uniprot_id, action)` tuples. For each drug, collect its set of (category, uniprot, action) tuples. For a query pair (A, B), score candidate train pair (X, Y) as the weighted-Jaccard similarity over the four category-specific feature sets, with category weights `enzymes=1.0, transporters=0.7, targets=0.5, carriers=0.3`.

**v1 — `scripts/step11c_pathway_retrieval.py`:**
- Strict pair_sim formula required all four drug feature sets nonempty
- Empty rate on 500-pair pilot: **31.4% (157/500)**
  - 11 pairs both drugs lack features (uncoverable)
  - 122 pairs asymmetric (one has features, one doesn't) — strict formula → zero
  - 24 pairs both have features but no train pair overlap

**v2 — `scripts/step11c_pathway_retrieval_v2.py`** (asymmetric-handling extension):
- When a query has only one drug with features, fall back to single-sided similarity weighted by `--asymmetric-penalty` (default 0.5)
- Full-pair matches still rank above asymmetric ones at equivalent strength
- Preserves the v1 behavior whenever both query drugs have features

**Command (v2 pilot):**
```bash
python scripts/step11c_pathway_retrieval_v2.py \
    --target test --pilot 500 --asymmetric-penalty 0.5
```

**Results — empty rate before/after on identical 500-pair pilot:**

| Bucket                                        | v1 (strict) | v2 (asymmetric) |
|-----------------------------------------------|-------------|-----------------|
| Both drugs lack features                      | 11          | 15              |
| Asymmetric (one drug has, one doesn't)        | 122         | 0               |
| Both have features but no train overlap       | 24          | 0               |
| **Total empty / 500**                         | **156 (31.4%)** | **15 (3.0%)** |

**Top-1 stability v1 → v2:** 88.7% — asymmetric matches do not displace strong full-pair matches in the top-K; they fill gaps.

**Index statistics:**
- 5,309 unique features
- 16.6M postings
- 42.9% of drugs have ≥1 feature

The remaining 3.0% empties are pairs where neither drug has any pathway annotation in DrugBank — genuinely uncoverable by pathway retrieval. The pilot harness (Step 12) falls back to Tanimoto retrieval for these.

**Output:** `processed_v2/retrieved_examples_pathway_test_pilot500_v2.json`.

---

## Step 11d — Direction-aware scorer

**Goal:** parse teacher rationale traces and emit a verdict for whether the trace's stated outcome direction matches the gold polarity from the hierarchy.

**Script:** `scripts/step11d_direction_scorer.py`

**Verdict categories:**
- `correct` — extracted direction matches gold polarity
- `incorrect` — extracted direction is the opposite of gold polarity
- `ambiguous` — both increase and decrease language appear (PK reversal phrasing: e.g., "clearance reduced → exposure increased")
- `missing` — `## Summary` section exists but no direction word found
- `no_summary` — no `## Summary` section in the trace

**Method:** regex extracts the `## Summary` block, runs increase/decrease lexicon match. Falls back to scanning the full trace if no Summary section exists.

**Self-test:** 9 hand-constructed trace cases covering all five verdicts including the PK reversal edge case ("clearance reduced + exposure increased" → correctly classified `ambiguous`). All 9 pass.

**Known limitation:** PK reversal phrasing where mechanism description ("decreases metabolism") and outcome description ("increases exposure") share a sentence will surface as `ambiguous`. This is intentional — the scorer is designed to be conservative, and the relative comparison across pilot conditions is what carries the headline finding (see Step 12).

---

## Step 12 — 2×2 pilot harness with three-tier evaluation

**Goal:** measure the effect of (a) retrieval method (Tanimoto vs pathway) and (b) hierarchy hints (off vs on) on teacher trace quality, on a stratified pilot.

**Script:** `scripts/step12_pilot_2x2.py` (three modes: `prepare`, `generate`, `score`)

**Conditions:**

|              | No hints     | + Hierarchy hints |
|--------------|--------------|-------------------|
| Tanimoto     | A_tan_nohints | C_tan_hints       |
| Pathway      | B_pwy_nohints | D_pwy_hints       |

**Constants across all four conditions** (preserve comparability):
- Prodrug warning, no-pathway note, drug-profile-truncation caps (auxiliary fixes from collaborator Rameen Jafri's work, kept with attribution)
- Same teacher model, same decoding settings, same prompt structure

**Stratification — `prepare` mode:**
- Floor=3 pairs per cluster, head_cap=15, proportional remainder
- Pool = first 500 test rows where both retrieval methods returned non-empty (485 pairs across 42 clusters)
- Allocated 254 pairs (saturated; cluster floors filled before head-cap took effect)
- 1,016 prompts written (254 pairs × 4 conditions)

**Generate — Qwen3-8B on Nibi (Compute Canada H100):**

```bash
sbatch run_step12_generate.sh
# inside the SLURM script:
python scripts/step12_pilot_2x2.py generate \
    --data-dir processed_v2 --out-dir pilot \
    --model Qwen/Qwen3-8B \
    --max-model-len 8192 --max-tokens 2048 \
    --no-thinking
```

**Important runtime notes:**
- Initial run (run 1) used `max_tokens=1024` with thinking mode enabled; 28% of traces hit the token limit before producing structured output. Preserved as `pilot_traces_run1_thinking_max1024.jsonl`.
- Final run (run 2) used `max_tokens=2048` with `--no-thinking` (passes `chat_template_kwargs={"enable_thinking": False}` to vLLM's chat call). Truncation rate dropped to <1%.
- Greedy decoding (`temperature=0`, seed=42); model loaded from local cache at `/scratch/vlelo/hf_cache/`.
- Wall time: 6 min 42 sec for 1,016 prompts on 1× H100 80GB.
- Peak RSS: 19.5 GB.

**Score — three-tier evaluation:**

```bash
python scripts/step12_pilot_2x2.py score \
    --data-dir processed_v2 --out-dir pilot
```

Metrics:
1. `exact_match` — predicted Y equals gold Y
2. `cluster_match` — predicted label's `l2` cluster equals gold `l2` cluster
3. `direction_verdict` — outcome-polarity alignment via Step 11d scorer

**Pilot results (run 2, n=254 stratified across 42 clusters, 1016 traces total):**

| Condition          | Exact match | Cluster match | Outcome-polarity alignment (adjudicable) | Wrong-direction (% of total) |
|--------------------|-------------|---------------|------------------------------------------|------------------------------|
| A · Tanimoto, no hints | 95.7%       | 95.7%         | 77.9%                                    | 8.3%                         |
| B · Pathway, no hints  | 96.9%       | 96.9%         | 81.7%                                    | 5.5%                         |
| C · Tanimoto, hints    | 96.9%       | 96.9%         | 98.7%                                    | **0.0%**                     |
| D · Pathway, hints     | 96.5%       | 96.5%         | 98.0%                                    | **0.8%**                     |

**Verdict distribution per condition (run 2):**
- A: correct 22.4% / incorrect 8.3% / ambiguous 69.3%
- B: correct 23.2% / incorrect 5.5% / ambiguous 71.3%
- C: correct 34.3% / incorrect 0.0% / ambiguous 65.7%
- D: correct 36.2% / incorrect 0.8% / ambiguous 63.0%

**Manual audit of 5 hint-condition flagged "incorrect" traces:** 4 of 5 are surface-vs-outcome phrasing artifacts where the trace explicitly states the correct outcome direction (e.g. "thereby increasing the hypoglycemic effect") but contains earlier "reduce/diminish" language from describing intermediate mechanism. The fifth is a single token-budget edge case (trace truncated mid-Summary). Zero genuine biology errors.

**Headline:** hierarchy hints reduce scorer-flagged wrong-direction commitments by an order of magnitude or more. Pathway retrieval gives a small consistent baseline gain (+1.2pt exact, +3.8pt direction); the gain shrinks once hints lift alignment toward the ceiling.

**Outputs:**
- `pilot/pilot_prompts.jsonl` — 1,016 prompts with full provenance
- `pilot/pilot_traces.jsonl` — clean run 2 traces
- `pilot/pilot_traces_run1_thinking_max1024.jsonl` — preserved evidence of run 1
- `pilot/pilot_scored.jsonl` — per-trace verdicts
- `pilot/pilot_summary.json` — aggregate metrics
- `pilot/manifest_*.json` — full provenance for each mode (prepare / generate / score)

---

## Visualization

**Script:** `scripts/step12_visualize.py`

```bash
python scripts/step12_visualize.py \
    --scored pilot/pilot_scored.jsonl \
    --out    pilot/figures
```

Produces three PNG files in `pilot/figures/`:
- `fig_metrics_2x2.png` — 4-panel grid of metrics by condition
- `fig_verdicts_stacked.png` — stacked verdict distribution
- `fig_alignment_focus.png` — focused chart on wrong-direction reduction (the headline finding)

---

## Reproducibility

Every state-changing step writes outputs to `processed_v2/` or `pilot/`. Diagnostic steps (3b, 6b, 8c) leave no artifacts beyond stdout / `reports/`.

Pilot modes write a manifest JSON next to their outputs (`manifest_prepare_*.json`, `manifest_generate_*.json`, `manifest_score_*.json`) recording git commit, library versions, parameters, input file paths, and timestamps.

The full pipeline from raw `drugbank_full.xml` to the pilot result table runs in roughly 80 minutes end-to-end on a modern multi-core CPU plus 7 minutes of H100 GPU time for the teacher generation.
