"""Central configuration. Env vars override defaults so the same code runs
locally (sbert), in tests (tiny in-memory corpus) and in Docker (hashing)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SearchConfig:
    """All knobs in one place. Immutable on purpose — services treat it as read-only."""

    corpus_path: Path = Path("data/corpus.csv")
    queries_path: Path = Path("eval/queries.jsonl")
    embedder_name: str = "hashing"          # "hashing" | "sbert"
    hashing_dim: int = 512
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: Path = Path(".semsearch_cache")
    default_k: int = 5
    max_k: int = 50
    batch_size: int = 64
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls, root: Path | None = None) -> SearchConfig:
        """Build config from environment variables (all optional).

        SEMSEARCH_EMBEDDER   hashing | sbert      (default: hashing)
        SEMSEARCH_CORPUS     path to corpus.csv   (default: data/corpus.csv)
        SEMSEARCH_CACHE_DIR  embedding cache dir  (default: .semsearch_cache)
        """
        root = root or Path.cwd()
        return cls(
            corpus_path=Path(os.environ.get("SEMSEARCH_CORPUS", root / "data/corpus.csv")),
            queries_path=root / "eval/queries.jsonl",
            embedder_name=os.environ.get("SEMSEARCH_EMBEDDER", "hashing"),
            hashing_dim=int(os.environ.get("SEMSEARCH_HASHING_DIM", "512")),
            cache_dir=Path(os.environ.get("SEMSEARCH_CACHE_DIR", ".semsearch_cache")),
        )
