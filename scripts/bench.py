#!/usr/bin/env python3
"""Latency benchmark: p50/p95 end-to-end search time (embed query + index)."""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semsearch.config import SearchConfig  # noqa: E402
from semsearch.evaluation import load_queries  # noqa: E402
from semsearch.search import SearchService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="number of timed queries")
    args = parser.parse_args()

    svc = SearchService.from_config(SearchConfig.from_env(ROOT))
    queries = [q.query for q in load_queries(ROOT / "eval" / "queries.jsonl")]

    rng = random.Random(42)
    times = []
    for _ in range(args.n):
        q = rng.choice(queries)
        t0 = time.perf_counter()
        svc.search(q, k=5)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(0.95 * len(times)) - 1]
    print(f"embedder={svc.embedder.name}  n={args.n}")
    print(f"  p50 = {p50:.2f} ms")
    print(f"  p95 = {p95:.2f} ms")


if __name__ == "__main__":
    main()
