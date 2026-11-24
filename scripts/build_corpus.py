#!/usr/bin/env python3
"""Build the committed corpus (data/corpus.csv) from the raw AG News train set.

Provenance
----------
Source : AG News Topic Classification dataset (Zhang et al., 2015),
         train split, 120,000 rows.
Mirror : https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
Sample : stratified -- 100 documents per class, seeded shuffle (seed=42).

Cleaning
--------
- ``\\`` sequences (raw line-break markers in AG News) -> single space
- HTML entities unescaped (html.unescape)
- trailing wire-service tags "(Reuters)" / "(AP)" / "(AFP)" stripped from titles
- whitespace collapsed

Usage
-----
    python scripts/build_corpus.py --raw raw/train.csv --out data/corpus.csv

Deterministic: same input file -> byte-identical output.
"""
from __future__ import annotations

import argparse
import csv
import html
import random
import re
from collections import defaultdict
from pathlib import Path

LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
PER_CLASS = 100
SEED = 42
WIRE_TAG = re.compile(r"\s*\((Reuters|AP|AFP|AP Online|UPI)\)\s*$")


def clean_text(raw: str) -> str:
    """Normalize one AG News text field."""
    text = raw.replace("\\", " ")          # raw line-break markers
    text = html.unescape(text)             # &quot; &lt; ...
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text.strip()


def load_rows(raw_path: Path) -> list[tuple[int, str, str, int]]:
    """Read the raw CSV -> (row_index, title, text, class_id)."""
    rows = []
    with raw_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, (class_id, title, text) in enumerate(reader):
            rows.append((i, clean_text(title), clean_text(text), int(class_id)))
    return rows


def stratified_sample(rows, per_class: int, seed: int):
    """Pick `per_class` docs from each class, deterministically."""
    rng = random.Random(seed)
    by_class: dict[int, list] = defaultdict(list)
    for row in rows:
        by_class[row[3]].append(row)

    sampled = []
    for class_id in sorted(by_class):
        pool = sorted(by_class[class_id])          # stable order first
        rng.shuffle(pool)                          # then seeded shuffle
        sampled.extend(pool[:per_class])

    # Stable final ordering: by class, then original row index.
    sampled.sort(key=lambda r: (r[3], r[0]))
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="path to AG News train.csv")
    parser.add_argument("--out", type=Path, required=True, help="output corpus.csv path")
    args = parser.parse_args()

    rows = load_rows(args.raw)
    sampled = stratified_sample(rows, PER_CLASS, SEED)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "text", "label"])
        for n, (_, title, text, class_id) in enumerate(sampled, start=1):
            title = WIRE_TAG.sub("", title).strip()
            writer.writerow([f"doc-{n:04d}", title, text, LABELS[class_id]])

    counts = defaultdict(int)
    for (_, _, _, c) in sampled:
        counts[c] += 1
    print(f"wrote {len(sampled)} docs -> {args.out}")
    print("per-class:", {LABELS[c]: n for c, n in sorted(counts.items())})


if __name__ == "__main__":
    main()
