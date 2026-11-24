# Evaluation

Everything here is reproducible: `make eval` (hashing) or `EMBEDDER=sbert make eval`.

## The corpus

- **Source:** [AG News Topic Classification](https://huggingface.co/datasets/ag_news) (Zhang et al., 2015), train split — 120,000 real news titles + descriptions across World / Sports / Business / Sci-Tech.
- **Subset:** stratified 100 docs per class = 400 docs, seeded (`seed=42`), built by [`scripts/build_corpus.py`](../scripts/build_corpus.py) — same input file always produces a byte-identical corpus. The subset ships in-repo at [`data/corpus.csv`](../data/corpus.csv) so nobody needs to re-download 120k rows to reproduce anything.

## The query set

20 queries, 47 relevance labels, hand-labeled by the author ([`eval/queries.jsonl`](../queries.jsonl)):

1. Candidate relevant docs were surfaced with an independent lexical scorer (not either production embedder).
2. Every candidate's **actual text was read** before labeling — no title-only judgments.
3. Borderline rules: a doc counts as relevant only if a reasonable user issuing that query would want it. When in doubt → excluded (strict labels).
4. Deliberate query design: some queries share vocabulary with targets ("quarterly earnings profit report"), others deliberately don't ("united nations peacekeeping troops deployment" → `doc-0098` says "UN mission… needs extra troops to maintain order" — semantic match, zero word overlap except *troops*).

## Metric definitions

For a query with relevant set `R`, retrieved list `D1..Dk`:

| Metric | Formula | What it tells you |
|---|---|---|
| **recall@1** | fraction of queries whose top-1 is relevant | "Can I trust the first result?" |
| **recall@5** | `\|R ∩ top5\| / \|R\|`, macro-averaged | "Do I get most of what exists in a glance?" |
| **MRR@10** | mean of `1/rank(first relevant)` | "How far do I have to scroll?" |

## Results (2026-11-24)

| Embedder | recall@1 | recall@5 | MRR@10 | p50 latency | p95 latency |
|---|---|---|---|---|---|
| `hashing-512` (lexical baseline) | 0.700 | 0.489 | 0.733 | 0.07 ms | 0.13 ms |
| `all-MiniLM-L6-v2` (semantic) | **0.950** | **0.915** | **0.975** | 7.5 ms | 12.8 ms |

- Committed: [`eval/metrics.json`](../metrics.json) (hashing — regenerated exactly in CI), [`eval/metrics-sbert.json`](../metrics-sbert.json) (reference run).
- Latency = end-to-end search (embed query + exact index), warm cache, single CPU.

## What the numbers say

The failure modes of the baseline are exactly the textbook lexical-search failures:

- **"united nations peacekeeping troops deployment"** — hashing MRR 0.00, MiniLM 1.00. The best doc (`doc-0098`, "UN seeks more I Coast troops") shares almost no query vocabulary.
- **"airline bankruptcy protection"** — hashing finds the Delta story only at low rank; the Yukos bankruptcy story (*oil company*, not airline) drags lexical matching off target, while MiniLM ranks both bankruptcies sensibly.
- **"quarterly earnings profit report"** (5 relevant docs) — hashing recall@5 collapses because it fixates on whichever 1–2 docs share the most tokens; MiniLM recovers 4–5 of 5.

## Known limitations (honest)

- **Single-author labels.** One labeler, strict-but-subjective borderline rules. A second annotator + kappa score would strengthen this.
- **400-doc corpus** — recall@5 differences of a point or two are within noise at this size; the *gap pattern* (baseline fails on vocabulary mismatch) is the robust finding, not the decimals.
- **AG News docs are short** (title + ~1 sentence). Real search corpora are longer; chunking becomes necessary and is out of scope here.
- The eval is not a benchmark leaderboard entry — it is a portfolio-grade demonstration of methodology.
