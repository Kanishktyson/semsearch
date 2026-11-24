"""SearchService — the orchestrator. Owns corpus + embedder + index and is the
only object the API and the eval harness talk to."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .config import SearchConfig
from .corpus import Corpus, Document, load_corpus
from .embedder import Embedder, get_embedder
from .indexer import BruteForceIndex


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    title: str
    text: str
    label: str
    score: float  # cosine similarity in [-1, 1]


class SearchService:
    def __init__(self, corpus: Corpus, embedder: Embedder,
                 index: BruteForceIndex, default_k: int = 5, max_k: int = 50) -> None:
        self.corpus = corpus
        self.embedder = embedder
        self.index = index
        self.default_k = default_k
        self.max_k = max_k

    # ── construction ──────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: SearchConfig) -> SearchService:
        """Build everything from config: load corpus, embed (with disk cache), index."""
        corpus = load_corpus(cfg.corpus_path)
        embedder = get_embedder(cfg.embedder_name, cfg.hashing_dim, cfg.sbert_model)
        matrix = cls._embed_corpus(corpus, embedder, cfg)
        index = BruteForceIndex(matrix, [d.id for d in corpus.docs])
        return cls(corpus, embedder, index, cfg.default_k, cfg.max_k)

    @staticmethod
    def _embed_corpus(corpus: Corpus, embedder: Embedder, cfg: SearchConfig) -> np.ndarray:
        """Embed all docs once, cache to disk keyed by (embedder, corpus hash).

        Re-embedding 400 docs is cheap; re-embedding 100k docs is not. The
        cache key binds the vectors to the exact corpus bytes + embedder name,
        so any change to either invalidates it automatically.
        """
        cache_dir = cfg.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        corpus_bytes = "".join(d.combined_text for d in corpus.docs).encode("utf-8")
        key = hashlib.sha256(f"{embedder.name}:{hashlib.sha256(corpus_bytes).hexdigest()}".encode()).hexdigest()[:24]
        npz_path = cache_dir / f"embeddings-{key}.npz"
        meta_path = cache_dir / f"embeddings-{key}.json"

        if npz_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["embedder"] == embedder.name and meta["count"] == len(corpus):
                data = np.load(npz_path)
                if data["matrix"].shape == (len(corpus), embedder.dim):
                    return data["matrix"]

        matrix = embedder.embed([d.combined_text for d in corpus.docs])
        np.savez_compressed(npz_path, matrix=matrix)
        meta_path.write_text(json.dumps({"embedder": embedder.name, "count": len(corpus)}))
        return matrix

    # ── queries ───────────────────────────────────────────────

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        """Embed the query, exact top-k cosine search, hydrate results."""
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        k = min(k if k is not None else self.default_k, self.max_k, self.index.size)

        query_vec = self.embedder.embed([query])[0]
        doc_ids, scores = self.index.search(query_vec, k)
        by_id = self.corpus.index_of

        results = []
        for doc_id, score in zip(doc_ids, scores, strict=False):
            doc: Document = self.corpus.docs[by_id[doc_id]]
            results.append(SearchResult(doc.id, doc.title, doc.text, doc.label, float(score)))
        return results
