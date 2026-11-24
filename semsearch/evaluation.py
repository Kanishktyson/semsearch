"""Evaluation: recall@k and MRR@10 against the hand-labeled query set
(eval/queries.jsonl). Methodology and metric definitions: docs/EVAL.md."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .search import SearchService


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_ids: frozenset[str]

    @classmethod
    def from_json(cls, obj: dict) -> EvalQuery:
        if not obj.get("query") or not obj.get("relevant_ids"):
            raise ValueError("each eval entry needs 'query' and 'relevant_ids'")
        return cls(query=obj["query"], relevant_ids=frozenset(obj["relevant_ids"]))


@dataclass
class QueryResult:
    query: str
    n_relevant: int
    hits_at_1: int
    hits_at_5: int
    first_relevant_rank: int | None  # 1-based; None = not in top-10
    retrieved: list[str] = field(default_factory=list)


def load_queries(path) -> list[EvalQuery]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(EvalQuery.from_json(json.loads(line)))
    if not queries:
        raise ValueError(f"no queries found in {path}")
    return queries


def evaluate(service: SearchService, queries: list[EvalQuery]) -> dict:
    """Macro-averaged recall@1 / recall@5 and MRR@10 over all queries."""
    details: list[QueryResult] = []

    for eq in queries:
        top10 = service.search(eq.query, k=10)
        top_ids = [r.doc_id for r in top10]
        hits1 = sum(1 for i in top_ids[:1] if i in eq.relevant_ids)
        hits5 = sum(1 for i in top_ids[:5] if i in eq.relevant_ids)
        rank = next((n + 1 for n, i in enumerate(top_ids[:10]) if i in eq.relevant_ids), None)
        details.append(QueryResult(eq.query, len(eq.relevant_ids), hits1, hits5, rank, top_ids[:5]))

    n = len(details)
    recall1 = sum(d.hits_at_1 for d in details) / sum(min(d.n_relevant, 1) for d in details)
    recall5 = sum(d.hits_at_5 for d in details) / sum(d.n_relevant for d in details)
    mrr10 = sum((1.0 / d.first_relevant_rank) if d.first_relevant_rank else 0.0 for d in details) / n

    return {
        "n_queries": n,
        "n_docs": len(service.corpus),
        "embedder": service.embedder.name,
        "recall_at_1": recall1,
        "recall_at_5": recall5,
        "mrr_at_10": mrr10,
        "per_query": [asdict(d) for d in details],
    }
