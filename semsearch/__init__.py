"""semsearch — semantic search over a real news corpus, from first principles.

Layers (each one swappable via protocols):

    corpus    -> documents, loaded from CSV
    embedder  -> text -> L2-normalized vector (hashing baseline / MiniLM)
    index     -> vector index (exact brute-force cosine)
    search    -> orchestrates the three above
    api       -> FastAPI service on top of search
    evaluation-> recall@k / MRR against a hand-labeled query set
"""

__version__ = "0.1.0"
