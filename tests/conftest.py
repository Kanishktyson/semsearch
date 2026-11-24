"""Shared fixtures: a tiny deterministic corpus so most tests stay fast and
hermetic. One smoke test exercises the real committed corpus."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from semsearch.config import SearchConfig
from semsearch.corpus import Corpus, load_corpus
from semsearch.embedder import HashingEmbedder
from semsearch.indexer import BruteForceIndex
from semsearch.search import SearchService

MINI_DOCS = [
    ("doc-0001", "Oil prices surge on supply fears", "Crude oil jumped above fifty dollars as traders worried about winter supply."),
    ("doc-0002", "Tennis star wins opening match", "The top seed cruised through the first round of the open tournament in straight sets."),
    ("doc-0003", "Company reports record quarterly profit", "Quarterly earnings beat analyst expectations, driven by strong retail sales."),
    ("doc-0004", "Oil refinery blast injures workers", "Several workers were hurt when a fuel storage tank exploded at the refinery."),
    ("doc-0005", "Stocks rally as investors buy shares", "Investors pushed stock indexes higher for a third straight session."),
]


@pytest.fixture()
def mini_csv(tmp_path: Path) -> Path:
    path = tmp_path / "mini.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "text", "label"])
        for i, (did, title, text) in enumerate(MINI_DOCS):
            writer.writerow([did, title, text, ["Business", "Sports", "Business", "Business", "Business"][i]])
    return path


@pytest.fixture()
def mini_service(mini_csv: Path, tmp_path: Path) -> SearchService:
    cfg = SearchConfig(corpus_path=mini_csv, cache_dir=tmp_path / "cache",
                       embedder_name="hashing", hashing_dim=1024)
    return SearchService.from_config(cfg)


@pytest.fixture()
def mini_corpus(mini_csv: Path) -> Corpus:
    return load_corpus(mini_csv)


@pytest.fixture()
def mini_index(mini_corpus: Corpus) -> BruteForceIndex:
    matrix = HashingEmbedder(dim=128).embed([d.combined_text for d in mini_corpus.docs])
    return BruteForceIndex(matrix, [d.id for d in mini_corpus.docs])
