import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from semsearch.config import SearchConfig
from semsearch.search import SearchService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_retrieves_relevant_docs_top2(mini_service):
    hits = mini_service.search("oil prices supply", k=2)
    top2 = {h.doc_id for h in hits[:2]}
    assert top2 == {"doc-0001", "doc-0004"}  # both oil docs; tennis must not rank top-2


def test_results_sorted_by_score_desc(mini_service):
    hits = mini_service.search("quarterly profit earnings", k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert len(hits) == 5


def test_empty_query_rejected(mini_service):
    with pytest.raises(ValueError, match="non-empty"):
        mini_service.search("   ")


def test_k_clamped_to_index_size(mini_service):
    assert len(mini_service.search("oil", k=100)) == 5


def test_search_result_fields(mini_service):
    hit = mini_service.search("tennis star wins", k=1)[0]
    assert hit.doc_id == "doc-0002"
    assert hit.label == "Sports"
    assert hit.title == "Tennis star wins opening match"
    assert isinstance(hit.score, float)


def test_embedding_cache_created_and_reused(mini_csv, tmp_path):
    cache = tmp_path / "cache"
    cfg = SearchConfig(corpus_path=mini_csv, cache_dir=cache,
                       embedder_name="hashing", hashing_dim=1024)

    svc1 = SearchService.from_config(cfg)
    npz_files = list(cache.glob("embeddings-*.npz"))
    assert len(npz_files) == 1
    meta = json.loads(list(cache.glob("embeddings-*.json"))[0].read_text())
    assert meta["embedder"] == "hashing-1024"
    assert meta["count"] == 5

    svc2 = SearchService.from_config(cfg)  # same key -> served from cache
    assert len(list(cache.glob("embeddings-*.npz"))) == 1
    assert np.allclose(svc1.index._matrix, svc2.index._matrix)


def test_cache_invalidated_when_corpus_changes(mini_csv, tmp_path):
    cfg = SearchConfig(corpus_path=mini_csv, cache_dir=tmp_path / "c2",
                       embedder_name="hashing", hashing_dim=1024)
    SearchService.from_config(cfg)
    # mutate corpus -> cache key must change -> second cache file appears
    with mini_csv.open("a", encoding="utf-8") as f:
        f.write('doc-0006,"Extra doc about oil rigs","more oil supply text",Business\n')
    SearchService.from_config(cfg)
    assert len(list((tmp_path / "c2").glob("embeddings-*.npz"))) == 2


def test_real_corpus_smoke_end_to_end(tmp_path):
    """The committed 400-doc corpus searches end-to-end with the offline embedder."""
    corpus_copy = tmp_path / "corpus.csv"
    shutil.copy(PROJECT_ROOT / "data" / "corpus.csv", corpus_copy)
    cfg = SearchConfig(corpus_path=corpus_copy, cache_dir=tmp_path / "cache",
                       embedder_name="hashing", hashing_dim=1024)
    svc = SearchService.from_config(cfg)
    hits = svc.search("oil prices surge", k=3)
    assert len(hits) == 3
    assert all(h.label in {"Business", "World", "Sports", "Sci/Tech"} for h in hits)
