# DDI Pipeline — Reproducibility Log

This document records every step of rebuilding the processed dataset from
DrugBank 6.0 raw XML, including every script run, key parameters, and outputs.

It is updated as we go. Each step has:
- **Goal**: what the step is for
- **Script**: the exact code (or a path to the file)
- **Command**: how to run it
- **Output**: key results (counts, paths, anomalies)

---

## Project Layout
~/ddiproject/
├── raw/
│   └── drugbank_full.xml          # 2.4 GB, DrugBank 6.0 (exported 2026-04-23)
├── processed_v2/                   # New cleaned outputs (this rebuild)
├── processed-20260424T134040Z-3-001/
│   └── processed/                  # Old processed files (keep as reference)
├── scripts/                        # All scripts used in the rebuild
├── reports/                        # Audit outputs (counts, mismatch logs, etc.)
└── PIPELINE.md                     # This file

---

## Decisions

- Label scheme: TDC-style 129-class templates (preserves comparability with prior work)
- Drug scope: TBD — pending decision (approved + investigational vs. all groups)
- Severity: TBD — pending decision (metadata only vs. weighted F1 vs. filter)
- Train/test: re-derive 80/20 stratified split; keep `interactions_full.jsonl` as canonical immutable source
- CoT regeneration: deferred until data layer is clean

---

## Step 0 — Workspace cleanup

**Goal:** remove Windows metadata files that came across via WSL copy.

**Commands:**
```bash
cd ~/ddiproject
find . -name "*:Zone.Identifier" -delete
mkdir -p ~/ddiproject/raw
mv ~/ddiproject/drugbank_all_full_database.xml/"full database.xml" \
   ~/ddiproject/raw/drugbank_full.xml
rmdir ~/ddiproject/drugbank_all_full_database.xml
mkdir -p ~/ddiproject/{processed_v2,reports,scripts}
```

**Output:**
- Workspace organized
- Raw XML at `~/ddiproject/raw/drugbank_full.xml`

---

## Step 1 — XML structural sanity check

**Goal:** confirm the XML loads, count drugs and interactions, verify schema.

**Script:** `scripts/00_xml_peek.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
XML=~/ddiproject/raw/drugbank_full.xml

echo "File size:"
du -h "$XML"
echo

echo "First 50 lines:"
head -50 "$XML"
echo

echo "Counting <drug> entries:"
grep -c '<drug ' "$XML" || true
grep -c '<drug type=' "$XML" || true
echo

echo "Counting <drug-interaction> tags:"
grep -c '<drug-interaction>' "$XML" || true
```

**Command:**
```bash
chmod +x ~/ddiproject/scripts/00_xml_peek.sh
~/ddiproject/scripts/00_xml_peek.sh | tee ~/ddiproject/reports/00_xml_peek.txt
```

**Output:**
- File size: **2.4 GB**
- Schema: `<drugbank xmlns="http://www.drugbank.ca" version="5.1" exported-on="2026-04-23">`
- Drug entries: **19,857**
- `<drug-interaction>` tags: **2,915,498** (≈2× the deduplicated count, since each interaction appears in both drugs' cards)
- Sample first drug: DB00001 Lepirudin (biotech, approved+withdrawn) — schema matches expectations

---

cat >> ~/ddiproject/PIPELINE.md << 'EOF'

## Step 2 — Drug profile extraction (DONE)

**Goal:** extract clean per-drug records from raw XML, dropping bibliographic / regulatory / spectral noise; keep synonyms and mechanism/metabolism/target/enzyme/transporter info for downstream reasoning.

**Script:** `scripts/02_parse_drugs.py` (275 lines)

**Run:**
```bash
python scripts/02_parse_drugs.py
```

**Runtime:** 61 seconds.

**Results:**
- 19,857 drugs parsed (15,497 small molecules + 4,360 biotech)
- Group breakdown: experimental 8,864 / investigational 8,547 / approved 4,799 / withdrawn 914 / vet_approved 428 / illicit 205 / nutraceutical 135
- Field coverage: SMILES 73.7%, targets 50.5%, mechanism-of-action text 21.4%, metabolism text 12.5%, enzymes 9.4%
- No filtering applied — all groups retained

**Outputs:**
- `processed_v2/drug_profiles.json` (46.0 MB)
- `processed_v2/drug_synonyms.json` (14.6 MB)
- `processed_v2/drug_id_order.json` (0.3 MB)

**Note for paper Limitations:** explicit enzyme data only covers 9.4% of drugs in DrugBank, so the "teacher hallucinates pharmacology" problem is mitigated but not eliminated by the v6 grounding. The 9.4% figure is also a coverage floor for any retrieval-augmented teacher prompt.

EOF
cat >> ~/ddiproject/PIPELINE.md << 'EOF'

cat >> ~/ddiproject/PIPELINE.md << 'EOF'

## Step 3 — Interaction extraction (DONE)

**Goal:** pull every drug-drug interaction out of the XML. Each drug's `<drug-interactions>` block lists every other drug it interacts with, but each interaction is stored in BOTH drugs' cards — so the raw count is doubled.

**Script:** `scripts/03_parse_interactions.py`

**Run:**
```bash
python scripts/03_parse_interactions.py
```

**Runtime:** 94 seconds.

**Results (raw, pre-dedup):**
- 19,857 drugs scanned
- 4,635 drugs have at least 1 DDI (the rest are isolated nodes — experimental/biotech drugs with no curated interactions)
- 2,915,498 raw `<drug-interaction>` tags extracted
- All rows kept at this stage (dedup handled in Step 3c after diagnosis)

**Output:** `processed_v2/interactions_full.jsonl` (493 MB)

---

## Step 3b — Inspect duplication pattern (DIAGNOSTIC)

**Goal:** verify that interactions are double-counted in the raw file (each interaction appears in both drugs' cards), and decide on the correct dedup strategy.

**Script:** `scripts/03b_inspect_interactions.py`

**Run:**
```bash
python scripts/03b_inspect_interactions.py
```

**Findings:**
- 1,458,020 unique unordered drug pairs
- 1,457,478 of those have 2 rows (one from each drug's card); 542 have only 1 row
- For all pairs with 2 rows: descriptions are 100% identical between the two sides
- The `subject_id` field carries no directional meaning: in a 30K random sample, subject's name appeared before affected's in 49.9% of descriptions and after in 49.3% — essentially a coin flip
- ~0.8% of descriptions use a non-canonical drug name (synonym, capitalization variant, or development code)

**Conclusion:** every interaction is stored exactly twice. Direction is encoded in the description text, not in the subject/affected fields. Dedup on `(unordered_pair, description)` collapses each mirror into one row.

---

## Step 3c — Deduplication (DONE)

**Goal:** collapse mirrored rows into one canonical row per interaction.

**Script:** `scripts/03c_dedup_interactions.py`

**Approach:** key each row on `(min_drug_id, max_drug_id, description)`. For each unique key, keep one row with `drug_a` = lexicographically smaller drugbank_id, `drug_b` = larger. The original `subject_id` / `affected_id` distinction is dropped since it carried no directional information; direction is preserved in the description text.

**Run:**
```bash
python scripts/03c_dedup_interactions.py
```

**Runtime:** 15 seconds.

**Results:**
- 2,915,498 input rows
- 1,458,020 unique interactions after dedup (matches the DrugBank 6.0 paper's published 1.4M figure)
- Compression ratio: 2.00x exactly
- 0 self-loops
- 4,636 distinct drugs participate in interactions

**Output:** `processed_v2/interactions_dedup.jsonl` (234 MB) — replaces `interactions_full.jsonl` for all downstream steps. Schema: `{drug_a, drug_b, description}`.

EOF

## Step 4 — Label map construction (DONE)

**Goal:** extract templates from interaction descriptions to build a label space. Replace actual drug names with #Drug1 / #Drug2 placeholders. By construction, the description and the templated label can never disagree (fixes the text/label_text contamination from the old pipeline).

**Script:** `scripts/04_build_label_map.py`

**Performance fix:** original implementation took 1h56m. Optimized version uses one compiled alternation regex per drug (instead of ~14 sequential regex compiles per row) and matches once on the description. Runtime dropped to 43 seconds — ~160x speedup.

**Run:**
```bash
python scripts/04_build_label_map.py
```

**Results:**
- 1,458,020 deduplicated interactions processed
- 1,454,752 successful template extractions (99.78%)
- 3,268 failures (0.22%) — biotech drugs where DrugBank uses development codes (e.g., TNX-901 for Talizumab, GS-5745 for Andecaliximab) that aren't in the synonym list
- 527 unique templates extracted
- 166 templates qualify at MIN_COUNT=50
- 1,452,606 labeled interactions written
- Class distribution: largest 181,847, smallest 50, imbalance 3,637:1
- 18 classes have <100 examples

**vs paper figures:**
- Paper used 86 classes (TDC) / teammate's older processed had 129. New count is 166.
- Paper's 8,837:1 imbalance ratio is now 3,637:1 — substantially less severe.

**Outputs:**
- `processed_v2/label_map.json` (16.7 KB) — int label → template
- `processed_v2/label_distribution.json` (2.3 KB) — int label → count
- `processed_v2/interactions_labeled.jsonl` (567 MB) — interactions with labels assigned. By construction, `description == label_text` for every row (fixes text/label_text contamination).

EOF

"""
cat >> ~/ddiproject/PIPELINE.md << 'EOF'

## Step 5 — Stratified train/test split (DONE)

**Goal:** split labeled interactions 80/20 stratified by label so all classes appear proportionally in both splits.

**Script:** `scripts/05_train_test_split.py`

**Run:**
```bash
python scripts/05_train_test_split.py
```

**Method:** sklearn train_test_split with stratify=labels, seed=42 (matches paper's seed).

**Results:**
- Total labeled: 1,452,606
- Train: 1,162,084 (80.00%)
- Test: 290,522 (20.00%)
- All 166 labels present in both splits (0 dropped)
- Smallest class (label 166, n=50): split 40/10 train/test — stratification holds at the rare-class boundary

**vs paper:**
- Paper: 153,446 train / 38,362 test, 86 classes
- New: 1,162,084 train / 290,522 test, 166 classes
- ~7.5x more training data, ~2x more classes

**Outputs:**
- `processed_v2/train.jsonl` (454 MB)
- `processed_v2/test.jsonl` (113 MB)

EOF

## Step 6 — Severity extraction (BLOCKED)

**Goal:** extract per-interaction severity (major/moderate/minor) from XML for severity-weighted F1 reporting.

**Status:** abandoned. The DrugBank 6.0 academic XML export does not include the severity field. Diagnosis (`scripts/06b_diagnose_severity.py`) confirmed each `<drug-interaction>` element contains only `drugbank-id`, `name`, and `description` children. No `severity`, `evidence-level`, or risk-related fields are present.

**Why:** severity is a feature of DrugBank's Clinical API (commercial), not the academic XML export. The teammate's older `severity_map.json` was probably populated from a TDC preprocessed file or a one-time web scrape and is not reproducible from the current academic data.

**Implication for paper:** severity-weighted F1 cannot be reported. Add a note to Limitations or Methodology acknowledging this. The dataset's "Unknown" severity in the older processed files reflected the same data-availability gap.

**Action taken:** removed empty `severity_map.json` and `interactions_with_severity.jsonl`.


## Step 8 — Pharmacological hierarchy (DONE)

**Goal:** replace the broken coarse_category_map.json with a principled two-level hierarchy that supports hierarchical F1 evaluation and cluster-aware loss.

**Approach:** rule-based keyword classification (Step 8) + manual resolution of edge cases (Step 8b). Level 1 is PK vs PD; level 2 is mechanism cluster.

**Scripts:**
- `scripts/08_build_hierarchy.py` — rule-based pass
- `scripts/08b_resolve_unclassified.py` — manual assignments

**Results:**
- 166 templates classified across 61 mechanism clusters
- 94% classified by rules, remaining 10 resolved manually with pharmacological judgment
- Largest clusters: hypotension (n=12), hypertension (n=9), gi_effects (n=9), cns_depression (n=8), bleeding (n=7)
- exposure_increase cluster {1, 6, 26, 129, 132} unifies decreased-excretion and increased-serum templates that the old coarse_category split into separate buckets

**Manual assignments (Step 8b):**
- 65 → qt_prolongation (fold)
- 124 → antipsychotic_change (new)
- 135 → gi_effects (fold)
- 137 → hyperthermia (new)
- 139 → urinary_retention (new)
- 140 → peripheral_neuropathy (new)
- 141 → photosensitivity (new)
- 145 → hepatotoxicity (fold)
- 146 → electrolyte_imbalance (fold)
- 148 → cardiac_depression (new)

**Why this matters:** templates like {15, 36, 45, 66, 72, 100, 102, 125} all describe hypotension with slightly different wording. A model that picks any of these for a true hypotension interaction is mechanistically correct; flat F1 punishes it identically to picking a completely unrelated label. Hierarchical F1 over these clusters gives a fairer metric and a more defensible paper claim.

**Outputs:**
- `processed_v2/hierarchy_map.json` — label → {l1, l2, template}
- `processed_v2/hierarchy_clusters.json` — cluster name → [label, ...]


## Step 8e — Targeted polarity fixes (DONE)

**Goal:** address residual polarity mixing in clusters that hide opposite-direction effects under the same name.

**Bugs fixed:**
- `potassium_imbalance` lumped hyperkalemia (K up) and hypokalemia (K down) -> split into separate clusters
- `glycemic_increase` lumped hyperglycemia and hypoglycemia (opposite glucose directions) -> regrouped by glucose direction (`glucose_increase` / `glucose_decrease`)
- `hypertension_increase` and `hypertension_decrease` mixed outcome statements with activity-modulation statements -> regrouped by actual BP outcome (`bp_increase` / `bp_decrease`)

**Script:** `scripts/08e_polarity_fixes.py`

**Run:**
```bash
python scripts/08e_polarity_fixes.py
```

**Results:**
- 17 label overrides applied
- 71 clusters total (was 70)
- New clusters: `hyperkalemia` (n=2), `hypokalemia` (n=2), `glucose_increase` (n=2), `glucose_decrease` (n=2), `bp_increase` (n=6), `bp_decrease` (n=3)

**Documented known imperfections (left alone):**
- `vasoactivity` (n=6): mechanism family mixing vasoconstriction/vasodilation/vasopressor/vasospastic
- `gi_effects_increase` (n=9): different GI endpoints with shared drug-class etiologies
- `thrombosis` (n=4): includes antiplatelet activity in same mechanism family
- `electrolyte_imbalance` (n=5): non-potassium electrolytes lumped (rare individually)
- `myopathy` (n=4): includes tendinopathy as adjacent musculoskeletal toxicity
- `cns_depression` (n=8): includes respiratory and serotonergic compound templates (handled via secondary_tags from Step 8d)

These are documented because they trade off perfect clinical precision for hierarchical-F1 forgiveness in mechanistically related but distinct endpoints. Acceptable per the project's consolidation-first philosophy.

**Backups:** `hierarchy_map.json.bak_step8e`, `hierarchy_clusters.json.bak_step8e`


## Step 9 — Morgan fingerprints (DONE)

**Goal:** generate molecular fingerprints for similarity-based retrieval.

**Method:** RDKit Morgan fingerprints, radius=2 (ECFP4 equivalent), 2048-bit width. Standard for drug-drug structural similarity.

**Script:** `scripts/09_fingerprints.py`

**Run:**
```bash
cat >> ~/ddiproject/PIPELINE.md << 'EOF'

## Step 9 — Morgan fingerprints (DONE)

**Goal:** generate molecular fingerprints for similarity-based retrieval.

**Method:** RDKit Morgan fingerprints, radius=2 (ECFP4 equivalent), 2048-bit width. Standard for drug-drug structural similarity.

**Script:** `scripts/09_fingerprints.py`

**Run:**
```bash
python scripts/09_fingerprints.py
```

**Runtime:** 4 seconds.

**Results:**
- 14,617 fingerprints generated (out of 14,627 drugs with SMILES — 99.93%)
- 5,230 drugs have no SMILES (biotech: proteins, antibodies, vaccines)
- 10 parse failures: exotic compounds (organometallics, porphyrin sulfonates, vitamin B12 derivatives) — RDKit can't handle their valence/charge configurations

**Output:** `processed_v2/drug_fingerprints.pkl` (30.5 MB) — dict mapping drugbank_id to numpy uint8 array of length 2048.

**Note for Step 10/11:** the 5,230 drugs without fingerprints will use ATC code / category overlap as similarity fallback during retrieval.


## Step 10 — Drug-drug similarity (DONE)

**Goal:** for each drug, find its top-K most molecularly similar drugs by Tanimoto coefficient over Morgan fingerprints. Used in Step 11 for retrieval-augmented few-shot example caching.

**Method:** Tanimoto similarity computed via batched matrix multiplication. For each batch of 256 drugs, intersection counts come from `batch @ all.T` (int8 dot product); union derived from popcounts. Top-50 neighbors per drug stored as a ranked list.

**Script:** `scripts/10_similarity.py`

**Run:**
```bash
**Runtime:** 4 min 35 sec on 22 cores (~388K pairs/sec).

**Results:**
- 14,617 drugs in similarity index
- Top-50 neighbors cached per drug (730,850 pairs total)
- Median top-1 similarity: 0.568
- 4,061 drugs (28%) have a top-1 similarity > 0.7 (near-twins exist in the database)
- 8,871 drugs (61%) have a top-1 similarity > 0.5

**Sanity check (Aspirin DB00945):**
- Aloxiprin (sim=0.923) — aspirin-aluminum complex
- Dipyrocetyl (0.700) — salicylate derivative
- Guacetisal (0.606) — salicylate prodrug
- Carbaspirin calcium (0.606) — aspirin salt
- Salsalate (0.594) — salicylsalicylic acid

All 5 are salicylates or aspirin derivatives, confirming the similarity computation correctly groups pharmacologically related drugs.

**Output:** `processed_v2/drug_similarity.pkl` (11.9 MB) — `{drugbank_id: [(neighbor_id, tanimoto), ...]}` sorted descending, top-50 per drug.


## Step 11 — Retrieval cache (DONE)

**Goal:** for every train+test row, cache K=5 most similar training interactions for use as in-context few-shot examples in teacher prompts.

**Algorithm:** for each query (A, B):
1. S_A = top-50 molecularly similar drugs to A; S_B = top-50 to B
2. Candidates = training pairs (X, Y) where X∈S_A and Y∈S_B (or swapped)
3. Score candidates by pair_sim = max(sim(A,X)·sim(B,Y), sim(A,Y)·sim(B,X))
4. Take top K=5

**Performance fix:** original implementation used "at least one drug in neighbor set" which let common drugs explode candidate counts. Rewrite uses "BOTH drugs related to query" with a precomputed (drug, drug) → train_idx index. ~16x speedup, finished in 52 minutes.

**Script:** `scripts/11_retrieval.py`

**Run:**
```bash
python scripts/11_retrieval.py
```

**Results:**
- Train: 1,162,084 rows processed, 537 rows/sec, 52 min total
- Test: 290,522 rows processed, 299 rows/sec
- Train rows with 0 examples: 79,595 (6.85%) — pairs where both drugs lack fingerprints AND no ATC overlap matches (rare biotech-only combinations)
- Test rows with 0 examples: 0
- Avg similarity of retrieved examples: train 0.466, test 0.434

**Caveat:** retrieval uses Tanimoto structural similarity, which a teammate's pathway-retrieval analysis showed has ~80% mechanistic overlap rate (MOR) vs ~99% for pathway-based retrieval. Step 11b will add pathway retrieval as an alternative for comparison.

**Outputs:**
- `retrieved_examples_train.json` (1.07 GB) — list of 5 example dicts per train row
- `retrieved_examples_test.json` (288 MB) — same for test rows


## Step 11b — Fix empty retrieval rows (DONE)

**Goal:** patch the 6.85% of train rows that got 0 retrieved examples in Step 11.

**Bug:** original retrieve function applied self-exclusion AFTER the fallback chain. When S_A × S_B lookup found exactly one candidate (the query itself), self-exclusion emptied the set, but ATC and random fallbacks were skipped because candidate_idxs had been non-empty earlier in the function.

**Fix:** move self-exclusion before the fallback chain so fallbacks fire if the only candidate was the query itself.

**Script:** `scripts/11b_fix_empties.py`

**Run:**
```bash
python scripts/11b_fix_empties.py
```

**Results:**
- 79,595 empty rows processed
- 0 truly empty after fix (final empty rate: 0.000%)
- Fallback path breakdown:
  - 792 (1%) S_A × S_B match (recovered by exclusion-first ordering)
  - 2,449 (3%) ATC code prefix overlap
  - 76,354 (96%) random pool fallback

**Caveat:** the 76,354 random-fallback rows are biotech drugs (no fingerprints, no ATC codes). Their retrieved examples are not similarity-matched. The teacher will reason about these pairs primarily from drug-profile information rather than retrieval context. Test set has 0% empties so evaluation metrics are clean.

**Output:** `retrieved_examples_train.json` (1.15 GB) — patched in place.

