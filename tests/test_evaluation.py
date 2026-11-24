from pathlib import Path

import numpy as np
import pytest

from semsearch.corpus import Corpus, Document
from semsearch.evaluation import EvalQuery, evaluate, load_queries
from semsearch.indexer import BruteForceIndex
from semsearch.search import SearchService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_eval_query_from_json_and_validation():
    eq = EvalQuery.from_json({"query": "oil", "relevant_ids": ["doc-0001"]})
    assert eq.relevant_ids == frozenset(["doc-0001"])
    with pytest.raises(ValueError):
        EvalQuery.from_json({"query": ""})
    with pytest.raises(ValueError):
        EvalQuery.from_json({"query": "x", "relevant_ids": []})


def test_real_query_set_loads():
    queries = load_queries(PROJECT_ROOT / "eval" / "queries.jsonl")
    assert len(queries) == 20
    assert all(q.relevant_ids for q in queries)


class ScriptedEmbedder:
    """Metric-math test double: corpus vectors are one-hot rows (assigned in
    call order); query vectors come from a caller-supplied {query: vector} map.

    Rank positions are therefore hand-controlled and deterministic — this
    tests the METRICS precisely, independent of embedding quality. (Embedding
    quality itself is asserted in test_embedder.py and via the committed
    eval numbers.) Six docs so a relevant doc CAN rank below the top-5."""

    name, dim = "scripted", 6

    def __init__(self, query_map: dict[str, list[float]]) -> None:
        self.query_map = query_map

    def embed(self, texts: list[str]) -> np.ndarray:
        if len(texts) > 1:  # the corpus: positional one-hot rows
            return np.eye(self.dim, dtype=np.float32)
        vec = np.array(self.query_map.get(texts[0], [0.0] * self.dim), dtype=np.float32)
        return vec[None, :]


def _service(query_map: dict[str, list[float]]) -> SearchService:
    ids = [f"doc-{i:04d}" for i in range(1, 7)]
    docs = tuple(Document(id=i, title=f"title {i}", text="t", label="L") for i in ids)
    emb = ScriptedEmbedder(query_map)
    return SearchService(Corpus(docs=docs), emb, BruteForceIndex(np.eye(6, dtype=np.float32), ids))


def test_evaluate_perfect_metrics_when_top1_relevant():
    # 'oil prices' targets docs 0001+0004 (both relevant; ties among the
    # remaining zero-scored docs only affect irrelevant positions);
    # 'quarterly profit' targets 0003 exactly.
    svc = _service({
        "oil prices": [0.8, 0, 0, 0.7, 0, 0],
        "quarterly profit": [0, 0, 1.0, 0, 0, 0],
    })
    queries = [
        EvalQuery.from_json({"query": "oil prices", "relevant_ids": ["doc-0001", "doc-0004"]}),
        EvalQuery.from_json({"query": "quarterly profit", "relevant_ids": ["doc-0003"]}),
    ]
    m = evaluate(svc, queries)
    assert m["recall_at_1"] == 1.0
    assert m["recall_at_5"] == 1.0
    assert m["mrr_at_10"] == 1.0
    assert m["n_queries"] == 2


def test_mrr_penalizes_lower_rank():
    # relevant doc (0003) is outranked by two distractors -> first relevant at
    # rank 3 -> reciprocal rank = 1/3, still inside top-5
    svc = _service({"mixed query": [0.9, 0, 0.5, 0.8, 0, 0]})
    queries = [EvalQuery.from_json({"query": "mixed query", "relevant_ids": ["doc-0003"]})]
    m = evaluate(svc, queries)
    assert m["mrr_at_10"] == pytest.approx(1 / 3)
    assert m["recall_at_1"] == 0.0
    assert m["recall_at_5"] == 1.0


def test_wrong_doc_first_gives_zero_recall1():
    # query lands squarely on doc-0002; relevance says doc-0006 -> the top-1
    # is wrong, and the relevant doc ranks last of six -> MRR = 1/6
    svc = _service({"tennis open": [0.05, 1.0, 0.04, 0.03, 0.02, 0.01]})
    queries = [EvalQuery.from_json({"query": "tennis open", "relevant_ids": ["doc-0006"]})]
    m = evaluate(svc, queries)
    assert m["recall_at_1"] == 0.0
    assert m["mrr_at_10"] == pytest.approx(1 / 6)
    assert m["recall_at_5"] == 0.0  # rank 6 = outside top-5


def test_relevant_doc_below_top5_hurts_recall5():
    svc = _service({"deep hit": [0.9, 0.8, 0.7, 0.6, 0.5, 0.1]})
    queries = [EvalQuery.from_json({"query": "deep hit", "relevant_ids": ["doc-0006"]})]
    m = evaluate(svc, queries)
    assert m["recall_at_5"] == 0.0
    assert m["mrr_at_10"] == pytest.approx(1 / 6)
