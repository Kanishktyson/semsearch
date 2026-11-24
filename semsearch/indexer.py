"""Exact brute-force cosine index.

Why not ANN (HNSW / FAISS)? At this scale, exact search is the right answer:
- 400 docs x 384 dims = a single 400x384 matrix multiply — microseconds.
- Brute force is EXACT; ANN trades recall for speed you don't need yet.
- Switching threshold: past ~100k docs / latency SLOs, swap in FAISS behind
  the same interface. The rest of the codebase should not notice.

Design note: vectors arrive L2-normalized (embedder's contract), so cosine
similarity degenerates to a dot product — one matmul, no per-pair sqrt.
"""
from __future__ import annotations

import numpy as np


class BruteForceIndex:
    """Immutable vector store + exact top-k cosine search."""

    def __init__(self, matrix: np.ndarray, doc_ids: list[str]) -> None:
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2-D (n_docs x dim)")
        if matrix.shape[0] != len(doc_ids):
            raise ValueError("matrix rows and doc_ids length must match")
        norms = np.linalg.norm(matrix, axis=1)
        # Contract: rows are unit-norm OR exactly zero (a doc with no known
        # tokens has no signal — cosine 0 against everything, never ranks).
        bad = (norms > 1.001) | ((norms < 0.999) & (norms > 1e-8))
        if np.any(bad):
            raise ValueError("index expects L2-normalized rows (embedder contract)")
        self._matrix = matrix
        self.doc_ids = doc_ids
        self.dim = matrix.shape[1]

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def search(self, query_vec: np.ndarray, k: int) -> tuple[list[str], np.ndarray]:
        """Return (doc_ids, cosine scores) of the top-k, sorted best-first.

        argpartition is O(n) instead of a full O(n log n) sort — only k
        candidates are fully ordered.
        """
        k = min(k, len(self.doc_ids))
        sims = self._matrix @ query_vec                      # cosine, thanks to normalization
        top = np.argpartition(-sims, k - 1)[:k]
        order = top[np.argsort(-sims[top])]
        return [self.doc_ids[i] for i in order], sims[order]
