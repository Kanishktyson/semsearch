# Architecture

```
              ┌────────────────────────────────────────────────────┐
              │                    SearchService                   │
              │  (orchestration — the only object callers touch)   │
              └──────┬──────────────────┬──────────────────┬───────┘
                 loads        embeds query      top-k + hydrate
                 │                 │                 │
        ┌────────▼───────┐  ┌──────▼───────┐  ┌──────▼────────┐
        │     Corpus     │  │   Embedder   │  │ BruteForce    │
        │  data/corpus   │  │  (Protocol)  │  │    Index      │
        │  .csv → Doc    │  │ hashing-512  │  │ exact cosine  │
        │  dataclass     │  │ MiniLM-384   │  │ dot product   │
        └────────────────┘  └──────┬───────┘  └───────────────┘
                                   │
                          ┌────────▼────────┐
                          │  disk cache     │
                          │ npz keyed by    │
                          │ sha256(model +  │
                          │  corpus bytes)  │
                          └─────────────────┘
```

## Layer contracts

Each layer is a small class with one job. Two of them are `Protocol`s (structural typing), so swapping implementations requires **zero changes** anywhere else:

| Layer | Contract | Implementations |
|---|---|---|
| `Embedder` | `name: str`, `dim: int`, `embed(texts) -> unit-norm matrix` | `HashingEmbedder`, `SentenceTransformerEmbedder` |
| Index | rows = L2-normalized (or zero), `search(vec, k) -> (ids, scores)` best-first | `BruteForceIndex` (exact) |

**The unit-norm contract is the linchpin:** because *both* doc and query vectors are unit length, cosine similarity collapses to a plain dot product — one matrix multiply, no per-pair sqrt, no separate "cosine mode".

## The hashing baseline (why it exists)

Feature hashing (Weinberger et al., 2009) maps each token/bigram to one of 512 buckets via `md5(token)`, with a **±1 sign** — colliding features partially cancel instead of always inflating, halving collision damage. On top: unigrams + bigrams (bigrams carry word order: "oil price" ≠ "price oil"), sublinear TF (`1 + log tf`), L2 norm.

It exists for three deliberate reasons:

1. **Tests and CI run on it.** No model download, no torch, deterministic bit-for-bit — the suite stays fast and hermetic, and the eval job can assert the committed metrics reproduce exactly.
2. **It's an honest baseline.** Every semantic-search claim should be measured *against* something. Its failure modes (see [EVAL.md](EVAL.md)) are the textbook lexical failures — that's the repo's teaching payload.
3. **It's a real fallback.** Air-gapped / low-memory deployments get useful lexical search with zero dependencies beyond numpy.

Known limitation, documented on purpose: feature hashing has no IDF, so at low dims (128) random ±1 collision noise can outweigh signal for small corpora; 512+ is the sane operating point here.

## Why brute-force exact search (and when to switch)

At 400 docs × 384 dims, "search" is one `400×384 @ 384` multiply — **~150k FLOPs, microseconds**. Exact beats ANN on every axis at this scale:

- **Recall:** exact = 1.0 by definition; HNSW trades recall for speed you cannot use.
- **Simplicity:** no graph construction, no index serialization format, no parameter tuning (ef/M/efConstruction).
- **Correctness confidence:** one matmul is auditable; an ANN graph is not.

Switch trigger (implemented as: swap `BruteForceIndex` for a FAISS/HNSW class behind the same interface): **~100k+ docs or a latency SLO that matmul misses.** Top-k uses `argpartition` (O(n) partial select) instead of a full sort, so even the brute-force path scales gracefully to tens of thousands of docs.

## Embedding cache

`SearchService._embed_corpus` caches the doc matrix to `.semsearch_cache/embeddings-<key>.npz` where `key = sha256(embedder_name + sha256(corpus_text_bytes))[:24]`. Properties worth defending:

- **Content-addressed** — edit one corpus row → key changes → automatic invalidation. No manual cache-busting, no stale-index bugs.
- **Binds model + data** — an embedder swap also changes the key (same corpus, different vectors).
- **Shape-checked on load** — a corrupted/truncated cache (wrong shape or count) is ignored and re-embedded, never trusted blindly.

## Config & environment

`SearchConfig.from_env()` — every knob has a code default, overridable by env var (`SEMSEARCH_EMBEDDER`, `SEMSEARCH_CORPUS`, `SEMSEARCH_CACHE_DIR`, `SEMSEARCH_HASHING_DIM`). Same code runs three ways with no branches scattered around the codebase: local + sbert, CI + hashing, Docker + hashing. The config dataclass is frozen — services treat configuration as immutable at runtime.

## API design notes

- Service built lazily on first request behind a lock — importing `semsearch.api` in tests is instant; the model loads only when an endpoint needs it.
- `q` is required and min-length-1; `k` is bounded `[1, 50]` at the FastAPI layer *and* clamped again in the service — validation lives with the boundary, invariants live with the domain.
- Errors are typed: unknown doc → 404 with detail; bad params → 422 from pydantic; empty query → domain `ValueError` (the service does not trust the HTTP layer to be the only caller).
