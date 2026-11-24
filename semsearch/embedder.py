"""Embedding layer — turns text into L2-normalized vectors.

Two implementations behind one protocol:

- HashingEmbedder: feature hashing of uni+bigrams. Zero downloads, fully
  deterministic, surprisingly decent lexical search. Powers tests and CI.
- SentenceTransformerEmbedder: all-MiniLM-L6-v2 (384-dim). Real semantic
  matching. Optional install (`pip install -e '.[sbert]'`) — torch is heavy,
  so it is deliberately NOT a core dependency.

Both return unit-norm vectors, which makes cosine similarity a plain dot
product (see indexer.py).
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")


class Embedder(Protocol):
    """Anything that can vectorize text. `name` keys the embedding cache."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _bucket(token: str, dim: int) -> tuple[int, float]:
    """Deterministic feature hashing: token -> (bucket index, ±1 sign).

    The sign trick (Weinberger et al., 2009) halves collision damage —
    colliding features partially cancel instead of always adding up.
    """
    digest = hashlib.md5(token.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:8], "little") % dim
    sign = 1.0 if digest[8] & 1 else -1.0
    return idx, sign


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str]) -> list[str]:
    """Unigrams + bigrams: bigrams carry local word order ('oil price' != 'price oil')."""
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]


class HashingEmbedder:
    """Hashing-trick TF-IDF-ish vectorizer. No downloads, no state, deterministic."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            grams = _ngrams(_tokenize(text))
            counts: dict[str, int] = {}
            for g in grams:
                counts[g] = counts.get(g, 0) + 1
            for gram, tf in counts.items():
                idx, sign = _bucket(gram, self.dim)
                # sublinear TF: a word repeated 10x matters ~2.3x, not 10x
                vectors[row, idx] += sign * (1.0 + math.log(tf))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers. Lazy import + lazy
    model load so the rest of the codebase never pays for torch at import time."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.name = f"sbert:{model_name.split('/')[-1]}"
        self._model = None  # loaded on first embed() call

    @property
    def dim(self) -> int:
        return 384  # all-MiniLM-L6-v2 output size; st models are fixed-dim

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, deferred

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,     # cosine == dot product, like the baseline
            show_progress_bar=False,
        ).astype(np.float32)


def get_embedder(name: str, hashing_dim: int = 512,
                 sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """Factory — the single place that maps a config string to an implementation."""
    if name == "hashing":
        return HashingEmbedder(dim=hashing_dim)
    if name == "sbert":
        return SentenceTransformerEmbedder(model_name=sbert_model)
    raise ValueError(f"unknown embedder {name!r} (expected 'hashing' or 'sbert')")
