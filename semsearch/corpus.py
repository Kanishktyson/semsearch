"""Corpus loading and the Document model."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """One searchable document. `label` is the AG News topic — kept for
    filtering/analytics, never used by the search algorithm itself."""

    id: str
    title: str
    text: str
    label: str

    @property
    def combined_text(self) -> str:
        """What gets embedded: title twice (titles carry the most signal)."""
        return f"{self.title}. {self.title}. {self.text}"


@dataclass(frozen=True)
class Corpus:
    docs: tuple[Document, ...]

    def __post_init__(self) -> None:
        ids = [d.id for d in self.docs]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate document ids in corpus")

    def __len__(self) -> int:
        return len(self.docs)

    @property
    def index_of(self) -> dict[str, int]:
        """doc id -> positional index (used by the vector index)."""
        return {d.id: i for i, d in enumerate(self.docs)}


def load_corpus(path: Path) -> Corpus:
    """Read corpus.csv (id, title, text, label). Raises a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found at {path} — build it with: python scripts/build_corpus.py"
        )
    with path.open(encoding="utf-8") as f:
        docs = tuple(
            Document(id=row["id"], title=row["title"], text=row["text"], label=row["label"])
            for row in csv.DictReader(f)
        )
    if not docs:
        raise ValueError(f"corpus at {path} is empty")
    return Corpus(docs=docs)
