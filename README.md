# 🔎 semsearch

**Semantic search over a real news corpus — built from first principles, measured against an honest baseline.**

[![CI](https://github.com/Kanishktyson/semsearch/actions/workflows/ci.yml/badge.svg)](https://github.com/Kanishktyson/semsearch/actions/workflows/ci.yml)

Most search demos show you a model and say "trust me". This one shows the math, the baseline it beats, and the hand-labeled eval set that proves it — with every number reproducible from the repo.

## The result

| Embedder | recall@1 | recall@5 | MRR@10 | p50 | p95 |
|---|---|---|---|---|---|
| `hashing-512` — lexical baseline | 0.700 | 0.489 | 0.733 | 0.07 ms | 0.13 ms |
| `all-MiniLM-L6-v2` — semantic | **0.950** | **0.915** | **0.975** | 7.5 ms | 12.8 ms |

The gap *is* the lesson: query **"united nations peacekeeping troops deployment"** finds *"UN seeks more I Coast troops"* (MRR 0.00 → 1.00). The two texts share almost no vocabulary. Meaning beats keywords — and here's the evidence, not the assertion.

## Try it (30 seconds)

```bash
git clone https://github.com/Kanishktyson/semsearch.git && cd semsearch
pip install -r requirements.txt          # numpy + fastapi only, no torch
uvicorn semsearch.api:app --port 8000
```

```console
$ curl "localhost:8000/search?q=united%20nations%20peacekeeping%20troops&k=2"
{
  "query": "united nations peacekeeping troops",
  "embedder": "hashing-512",
  "results": [
    {"id": "doc-0098", "title": "UN seeks more I Coast troops", "label": "World", "score": 0.41, "snippet": "The UN mission says it needs extra troops..."},
    ...
  ]
}
```

Real semantic embeddings (the 0.95 row): `pip install -e '.[sbert]'`, then `SEMSEARCH_EMBEDDER=sbert make run`.

Docker (torch-free image, ~1 min build): `make docker-build && docker run -p 8000:8000 semsearch`

## Architecture

```
              ┌────────────────────────────────────────────────────┐
              │                    SearchService                   │
              │  (orchestration — the only object callers touch)   │
              └──────┬──────────────────┬──────────────────┬───────┘
                 loads        embeds query      top-k + hydrate
                 │                 │                 │
        ┌────────▼───────┐  ┌──────▼───────┐  ┌──────▼────────┐
        │     Corpus     │  │   Embedder   │  │ BruteForce    │
        │  400 AG News   │  │  (Protocol)  │  │    Index      │
        │  docs (real)   │  │ hashing-512  │  │ exact cosine  │
        │                │  │ MiniLM-384   │  │  = matmul     │
        └────────────────┘  └──────────────┘  └───────────────┘
```

Design decisions worth defending in an interview:

- **Pluggable embedders behind one Protocol.** Tests and CI run on a deterministic offline baseline; the headline model is optional. Swapping implementations touches zero other files.
- **Exact brute-force cosine, on purpose.** 400×384 matmul = microseconds; ANN (HNSW/FAISS) trades recall for speed that doesn't exist yet at this scale. Documented switch trigger: ~100k docs or a latency SLO.
- **Content-addressed embedding cache.** `sha256(model + corpus bytes)` → edit one doc and the cache invalidates itself. No stale-index bugs possible.
- **Eval as a first-class citizen.** 20 hand-labeled queries / 47 relevance labels, every label decided after *reading the article*. Committed metrics; CI re-runs the eval and fails if numbers drift.
- **Validation at the boundary, invariants in the domain.** FastAPI checks `q`/`k`; the service independently clamps `k` and rejects empty queries — it doesn't trust HTTP to be its only caller.

## Evaluation

Hand-labeled set, strict borderline rules, full methodology in [`docs/EVAL.md`](docs/EVAL.md) — including what the baseline gets wrong and why (vocabulary mismatch, the classic lexical failure), and the honest limitations (single labeler, 400-doc corpus, short docs).

Reproduce: `make eval` → writes `eval/metrics.json`, CI asserts it matches the committed numbers.

## The corpus

400 real AG News articles (World / Sports / Business / Sci-Tech, 100 each), stratified sample of the 120k-row train set, `seed=42`. Built by [`scripts/build_corpus.py`](scripts/build_corpus.py) — byte-identical output every run. Provenance documented in-file.

## What's inside

```
semsearch/           corpus.py  embedder.py  indexer.py  search.py  evaluation.py  api.py  config.py
tests/               39 tests — unit per layer, API via TestClient, metric-math via scripted embedder
scripts/             build_corpus.py · run_eval.py · bench.py (p50/p95 latency)
ui/                  Streamlit demo (pip install -e '.[ui]'; streamlit run ui/app.py)
eval/                queries.jsonl (labels) · metrics.json (baseline) · metrics-sbert.json (reference)
docs/                ARCHITECTURE.md · EVAL.md
.github/workflows/   ci.yml — ruff · pytest+coverage(≥88) × py3.10–3.12 · eval-repro check · docker build+smoke
```

## Commands

| | |
|---|---|
| `make dev-install` | core + pytest/ruff/coverage |
| `make test` | full suite (torch-free, ~1s) |
| `make lint` | ruff |
| `make eval` | eval set → metrics.json (`EMBEDDER=sbert make eval` for MiniLM) |
| `make bench` | latency p50/p95 |
| `make run` / `make ui` | API on :8000 / Streamlit demo |
| `make docker-build` | torch-free image + healthcheck |

## Roadmap

- [ ] FAISS ANN index behind the same interface (with a recall-vs-speed benchmark vs exact)
- [ ] Query expansion / hybrid scoring (lexical + semantic blend)
- [ ] Second annotator for the eval set + inter-annotator agreement
- [ ] Chunked indexing for long documents

## Honest limitations

Single-author labels · 400-doc corpus (differences within noise; the *pattern* is the finding) · short docs (no chunking needed yet) · MiniLM runs on CPU only here. Full list: [`docs/EVAL.md`](docs/EVAL.md).

## License

MIT — see [LICENSE](LICENSE).
