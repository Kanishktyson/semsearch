#!/usr/bin/env python3
"""Run the evaluation suite and (optionally) write metrics.json.

    python scripts/run_eval.py                       # default embedder (hashing)
    python scripts/run_eval.py --embedder sbert      # needs `pip install -e '.[sbert]'`
    python scripts/run_eval.py --out eval/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semsearch.config import SearchConfig  # noqa: E402
from semsearch.evaluation import evaluate, load_queries  # noqa: E402
from semsearch.search import SearchService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", default="hashing", choices=["hashing", "sbert"])
    parser.add_argument("--out", type=Path, default=None, help="write metrics JSON here")
    args = parser.parse_args()

    cfg = SearchConfig.from_env(ROOT)
    cfg = SearchConfig(**{**cfg.__dict__, "embedder_name": args.embedder})
    svc = SearchService.from_config(cfg)
    queries = load_queries(cfg.queries_path)

    metrics = evaluate(svc, queries)
    per_query = metrics.pop("per_query")

    print(f"\nembedder: {metrics['embedder']}   docs: {metrics['n_docs']}   queries: {metrics['n_queries']}")
    print(f"  recall@1  : {metrics['recall_at_1']:.3f}")
    print(f"  recall@5  : {metrics['recall_at_5']:.3f}")
    print(f"  MRR@10    : {metrics['mrr_at_10']:.3f}")
    print("\n  worst queries (MRR):")
    for d in sorted(per_query, key=lambda x: -(x["first_relevant_rank"] or 99))[:5]:
        rr = 1 / d["first_relevant_rank"] if d["first_relevant_rank"] else 0.0
        print(f"    {rr:.2f}  {d['query']}")
    print("\n  best queries (MRR):")
    for d in sorted(per_query, key=lambda x: x["first_relevant_rank"] or 99)[:5]:
        rr = 1 / d["first_relevant_rank"] if d["first_relevant_rank"] else 0.0
        print(f"    {rr:.2f}  {d['query']}")

    if args.out:
        payload = {**metrics, "per_query": per_query, "generated": "2026-11-24",
                   "recall_at_1": round(metrics["recall_at_1"], 4),
                   "recall_at_5": round(metrics["recall_at_5"], 4),
                   "mrr_at_10": round(metrics["mrr_at_10"], 4)}
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
