"""
Step 5: Stratified 80/20 train/test split of labeled interactions.

Stratified by label so that every class appears in both splits with
proportional representation. Uses sklearn.model_selection.train_test_split
with stratify=labels.

Inputs:
  processed_v2/interactions_labeled.jsonl
  processed_v2/label_map.json

Outputs:
  processed_v2/train.jsonl
  processed_v2/test.jsonl
  reports/05_split.txt
"""
import json
import random
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "05_split.txt"

INPUT = PROCESSED / "interactions_labeled.jsonl"
LABEL_MAP = PROCESSED / "label_map.json"

TEST_FRACTION = 0.2
SEED = 42  # paper used seed 42; preserve for reproducibility


def main():
    print(f"Loading {INPUT}...")
    rows = []
    with open(INPUT) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"  {len(rows):,} labeled interactions")

    with open(LABEL_MAP) as f:
        label_map = json.load(f)
    print(f"  {len(label_map):,} labels in label_map")

    labels = [r["label"] for r in rows]
    label_counts = Counter(labels)
    print(f"  smallest class: {min(label_counts.values())} examples")
    print(f"  largest class:  {max(label_counts.values()):,} examples")

    # Stratified split
    print(f"\nSplitting {1-TEST_FRACTION:.0%}/{TEST_FRACTION:.0%} stratified by label, seed={SEED}...")
    train_rows, test_rows = train_test_split(
        rows,
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=labels,
    )
    print(f"  train: {len(train_rows):,}")
    print(f"  test:  {len(test_rows):,}")

    # Shuffle within each split for good measure (stratify preserves label
    # ordering by class otherwise)
    random.seed(SEED)
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    print("\nWriting outputs...")
    with open(PROCESSED / "train.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(PROCESSED / "test.jsonl", "w") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    # Sanity checks
    train_label_counts = Counter(r["label"] for r in train_rows)
    test_label_counts = Counter(r["label"] for r in test_rows)

    train_labels_set = set(train_label_counts.keys())
    test_labels_set = set(test_label_counts.keys())
    map_labels_set = {int(k) for k in label_map.keys()}

    in_train_not_test = train_labels_set - test_labels_set
    in_test_not_train = test_labels_set - train_labels_set
    in_map_not_train = map_labels_set - train_labels_set
    in_map_not_test = map_labels_set - test_labels_set

    # Find smallest classes in test
    smallest_test = sorted(test_label_counts.items(), key=lambda x: x[1])[:5]

    lines = [
        "Step 5 -- Train/test split",
        "=" * 60,
        f"Total labeled interactions: {len(rows):,}",
        f"Train size:                 {len(train_rows):,}  ({100*len(train_rows)/len(rows):.1f}%)",
        f"Test size:                  {len(test_rows):,}  ({100*len(test_rows)/len(rows):.1f}%)",
        f"Seed:                       {SEED}",
        f"Stratified by label:        yes",
        "",
        "Label coverage:",
        f"  Labels in train:           {len(train_labels_set)}",
        f"  Labels in test:            {len(test_labels_set)}",
        f"  Labels in map:             {len(map_labels_set)}",
        f"  In train but not test:     {len(in_train_not_test)}  {sorted(in_train_not_test)[:10]}",
        f"  In test but not train:     {len(in_test_not_train)}  {sorted(in_test_not_train)[:10]}",
        f"  In map but not train:      {len(in_map_not_train)}",
        f"  In map but not test:       {len(in_map_not_test)}",
        "",
        "Smallest classes in TEST set (sanity check that rare labels survive):",
    ]
    for lbl, count in smallest_test:
        train_count = train_label_counts.get(lbl, 0)
        template = label_map.get(str(lbl), "(missing)")
        lines.append(f"  label {lbl:>3}  train={train_count:>3}  test={count:>3}  {template[:70]}")
    lines.extend([
        "",
        "Outputs:",
        f"  train.jsonl  ({(PROCESSED / 'train.jsonl').stat().st_size / 1e6:.1f} MB)",
        f"  test.jsonl   ({(PROCESSED / 'test.jsonl').stat().st_size / 1e6:.1f} MB)",
    ])
    report = "\n".join(lines)
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
